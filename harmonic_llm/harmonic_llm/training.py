"""
Training pipeline: tokenizer, dataset, and a real training loop.

This is not a toy: it tokenises real text, batches it, trains with AdamW +
cosine schedule + gradient clipping, evaluates held-out loss and perplexity, and
reports throughput. It is what the benchmark script drives.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """
    A byte-level tokenizer.

    Deliberately simple and dependency-free: every byte is a token, so any text
    round-trips exactly and there is no vocabulary to train or ship. Vocab is
    256 bytes plus a small set of specials. This is the right choice for a
    benchmark harness -- it isolates the *model's* quality from the tokenizer's,
    and it cannot silently corrupt the evaluation the way a mismatched BPE can.
    """

    PAD, BOS, EOS = 256, 257, 258
    VOCAB_SIZE = 259

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        body = [i for i in ids if i < 256]
        return bytes(body).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.VOCAB_SIZE


class BPETokenizer:
    """
    A byte-level BPE tokenizer.

    Byte-level BPE learns merges over raw UTF-8 bytes, so:

    * it round-trips *any* text exactly (the base alphabet is all 256 bytes --
      there is no out-of-vocabulary character and no unknown token), and
    * it still compresses real text far below one-token-per-byte by merging
      frequent byte sequences into single tokens.

    Ids are laid out as: ``0..255`` raw bytes, then one id per learned merge, then
    the specials PAD/BOS/EOS at the top -- so ``BOS``/``EOS`` move with the vocab
    size and never collide with a byte or a merge. The interface (``encode`` /
    ``decode`` / ``__len__`` / ``PAD``/``BOS``/``EOS``) matches
    :class:`ByteTokenizer`, so datasets, generation and the CLI accept either.
    """

    # Whitespace-vs-non-whitespace pretokeniser. Every character falls in exactly
    # one class, so the concatenation of the chunks is the original text -- which
    # is what preserves exact round-tripping. Merges never cross a chunk boundary.
    _PAT = re.compile(r"\s+|\S+")
    _N_SPECIALS = 3

    def __init__(self, merges: Optional[List[Tuple[int, int]]] = None):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._build()

    def _build(self) -> None:
        # Base byte alphabet, then expand each merge into the bytes it stands for.
        self.id2bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.ranks: Dict[Tuple[int, int], int] = {}
        self.pair2id: Dict[Tuple[int, int], int] = {}
        for idx, (a, b) in enumerate(self.merges):
            new_id = 256 + idx
            self.ranks[(a, b)] = idx
            self.pair2id[(a, b)] = new_id
            self.id2bytes[new_id] = self.id2bytes[a] + self.id2bytes[b]
        self.base_size = 256 + len(self.merges)
        self.PAD = self.base_size
        self.BOS = self.base_size + 1
        self.EOS = self.base_size + 2
        self.VOCAB_SIZE = self.base_size + self._N_SPECIALS

    # -- BPE core -----------------------------------------------------------
    @staticmethod
    def _merge_pair(symbols: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
        out: List[int] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    def _bpe(self, byte_ids: List[int]) -> List[int]:
        symbols = list(byte_ids)
        while len(symbols) >= 2:
            # Merge the adjacent pair with the best (lowest) learned rank first.
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (symbols[i], symbols[i + 1])
            if best_pair is None:
                break
            symbols = self._merge_pair(symbols, best_pair, self.pair2id[best_pair])
        return symbols

    # -- public API ---------------------------------------------------------
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids: List[int] = []
        for chunk in self._PAT.findall(text):
            ids.extend(self._bpe(list(chunk.encode("utf-8"))))
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for i in ids:
            piece = self.id2bytes.get(i)
            if piece is not None:            # specials (>= base_size) are skipped
                out.extend(piece)
        return out.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.VOCAB_SIZE

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, corpus: str, vocab_size: int = 1024) -> "BPETokenizer":
        """
        Learn a BPE vocabulary of size ``vocab_size`` from ``corpus``.

        Standard word-frequency BPE: pretokenise, count adjacent byte pairs
        weighted by word frequency, greedily merge the most frequent pair (ties
        broken deterministically by pair value so training is reproducible), and
        repeat until the target vocab size or no mergeable pair remains.

        Uses **incremental** pair counting: rather than rescanning the whole
        corpus every merge (O(merges x corpus), which is minutes for a few
        thousand merges), it keeps a live pair->count table and a pair->words
        index, and after each merge only touches the words that actually
        contained the merged pair. Output is identical to the naive method (same
        greedy choice, same deterministic tie-break), just far faster.
        """
        from collections import Counter, defaultdict

        num_merges = max(0, int(vocab_size) - 256 - cls._N_SPECIALS)
        word_freq = Counter(
            chunk.encode("utf-8") for chunk in cls._PAT.findall(corpus)
        )
        # Parallel arrays: symbol lists + their frequencies (index = word id).
        words: List[List[int]] = [list(w) for w in word_freq]
        freqs: List[int] = [word_freq[w] for w in word_freq]

        pair_counts: Counter = Counter()
        where: Dict[Tuple[int, int], set] = defaultdict(set)  # pair -> word indices
        for i, syms in enumerate(words):
            f = freqs[i]
            for a, b in zip(syms, syms[1:]):
                pair_counts[(a, b)] += f
                where[(a, b)].add(i)

        merges: List[Tuple[int, int]] = []
        for _ in range(num_merges):
            if not pair_counts:
                break
            # Most frequent pair; deterministic tie-break on the pair itself.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] <= 0:
                break
            new_id = 256 + len(merges)
            merges.append(best)

            for i in list(where[best]):
                syms = words[i]
                f = freqs[i]
                # Retract this word's current pair contributions...
                for a, b in zip(syms, syms[1:]):
                    pc = pair_counts.get((a, b), 0) - f
                    if pc <= 0:
                        pair_counts.pop((a, b), None)
                    else:
                        pair_counts[(a, b)] = pc
                    w = where.get((a, b))
                    if w is not None:
                        w.discard(i)
                # ...merge, and re-add the new adjacency counts.
                merged = cls._merge_pair(syms, best, new_id)
                words[i] = merged
                for a, b in zip(merged, merged[1:]):
                    pair_counts[(a, b)] += f
                    where[(a, b)].add(i)

            pair_counts.pop(best, None)
            where.pop(best, None)

        return cls(merges=merges)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        data = {"version": 1, "merges": [list(m) for m in self.merges]}
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls(merges=[tuple(m) for m in data["merges"]])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LanguageModelDataset(Dataset):
    """
    Contiguous next-token-prediction dataset.

    The corpus is tokenised once into one long stream, then chopped into
    overlapping windows of ``seq_len + 1`` so each sample yields an (input,
    target) pair shifted by one position -- the standard causal-LM objective.
    """

    def __init__(self, text: str, tokenizer: ByteTokenizer, seq_len: int,
                 stride: Optional[int] = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

        stream = tokenizer.encode(text, add_special=False)
        if len(stream) < seq_len + 1:
            # Repeat short corpora so at least one window exists.
            reps = (seq_len + 1) // max(1, len(stream)) + 1
            stream = stream * reps
        self.data = torch.tensor(stream, dtype=torch.long)

        self.starts = list(range(0, len(self.data) - seq_len - 1, self.stride))
        if not self.starts:
            self.starts = [0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        chunk = self.data[s: s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    log_every: int = 20
    eval_every: int = 100
    device: str = "cpu"
    seed: int = 42
    # -- checkpointing -------------------------------------------------------
    checkpoint_dir: Optional[str] = None   # where to write checkpoints (None = off)
    save_every: int = 0                     # steps between checkpoints (0 = only final)
    resume_from: Optional[str] = None       # path to a checkpoint to resume from
    # -- wall-clock budget ---------------------------------------------------
    max_seconds: Optional[float] = None     # stop training after this many seconds
    log_fn: Optional[Any] = None            # optional callable(step, total, loss, lr)


@dataclass
class TrainResult:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_perplexities: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    total_steps: int = 0
    final_val_loss: float = float("inf")
    final_val_ppl: float = float("inf")
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
CHECKPOINT_FORMAT_VERSION = 1


def _config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
    """Normalise a model config (ModelConfig | dict | None) to a plain dict."""
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)
    return None


def save_checkpoint(path: str, model, optimizer=None, step: int = 0, epoch: int = 0,
                    config: Any = None, tokenizer_meta: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """
    Persist a resumable training checkpoint.

    Writes model weights, optimizer state, the step/epoch counters, the model
    config, tokenizer descriptor, and RNG states so a run can be picked up
    exactly where it left off. The write is atomic (temp file + ``os.replace``)
    so an interrupt mid-save never leaves a truncated, unloadable checkpoint in
    place -- the whole point of checkpointing is that a crash loses nothing.
    """
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "config": _config_to_dict(config),
        "tokenizer_meta": tokenizer_meta,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }
    if extra:
        payload["extra"] = extra

    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on POSIX/NT
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` into a dict."""
    # weights_only=False: the payload carries non-tensor metadata (config dict,
    # RNG states) that the safe loader would reject.
    return torch.load(str(path), map_location=map_location, weights_only=False)


def load_model_state(model, state_dict: Dict[str, Any]) -> None:
    """
    Load ``state_dict`` into ``model``, tolerating dynamically-grown parameters.

    Some parameters in this architecture start empty (shape ``[0]``) and only
    materialise to full size the first time they are used during training -- the
    two-tier fine-tune adapters (``_ft_rows`` / ``_ft_cols``) are the notable
    case. A checkpoint taken after training therefore holds sizes a freshly-built
    model has not allocated yet, and a strict ``load_state_dict`` would reject
    them. Reallocate those tensors to match the checkpoint before copying, so a
    crash-and-resume restores into a clean model correctly.
    """
    # Reach the live Parameter/buffer objects (state_dict() hands back detached
    # views, so resizing those would not update the parameter's own shape).
    live: Dict[str, Any] = dict(model.named_parameters())
    live.update(dict(model.named_buffers()))
    for k, v in state_dict.items():
        p = live.get(k)
        if p is not None and p.shape != v.shape:
            p.data = p.data.new_zeros(v.shape)
    model.load_state_dict(state_dict)


def _restore_rng(rng: Optional[Dict[str, Any]]) -> None:
    if not rng:
        return
    torch_state = rng.get("torch")
    if torch_state is not None:
        # RNG state must be a CPU ByteTensor regardless of map_location.
        torch.set_rng_state(torch_state.cpu() if hasattr(torch_state, "cpu") else torch_state)
    py_state = rng.get("python")
    if py_state is not None:
        # JSON/torch round-trips can turn the tuple's nested list back into a
        # list; random.setstate needs the inner element to be a tuple.
        try:
            random.setstate(py_state)
        except (TypeError, ValueError):
            pass


def _lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    """Linear warmup then cosine decay -- the standard LM schedule."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    """
    Run the model and return per-position logits, shape (B, T, V).

    The head now emits logits for every position in a single forward pass, so
    the causal-LM objective costs one forward per batch. (An earlier version of
    the head only produced the final position, which forced an O(T) loop.)
    """
    out = model(x, start_pos=0)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() == 2:
        # Defensive: an older head shape (B, V) -- expand to a single position.
        logits = logits.unsqueeze(1)
    return logits


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, max_batches: int = 8) -> Dict[str, float]:
    """Held-out loss and perplexity."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = _forward_logits(model, x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    return {"loss": mean_loss, "perplexity": math.exp(min(20.0, mean_loss))}


def train(model, train_ds: Dataset, val_ds: Dataset,
          cfg: TrainConfig = TrainConfig(), config: Any = None,
          tokenizer_meta: Optional[Dict[str, Any]] = None) -> TrainResult:
    """
    Train the model. Returns a full result record for benchmarking.

    If ``cfg.checkpoint_dir`` is set, checkpoints are written every
    ``cfg.save_every`` steps (and always at the end) via :func:`save_checkpoint`.
    If ``cfg.resume_from`` points at a checkpoint, model + optimizer + step/epoch
    counters + RNG are restored and training continues from there.

    ``config`` (the model's ``ModelConfig``) and ``tokenizer_meta`` are recorded
    into checkpoints so a resume can sanity-check it is reloading a compatible
    model/tokenizer.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = model.to(device)
    model.train()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    total_steps = max(1, cfg.epochs * len(train_loader))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # -- resume ------------------------------------------------------------
    start_epoch = 0
    step = 0
    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from, map_location=device)
        load_model_state(model, ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        # Saved epoch is the one in progress when the checkpoint was taken; it is
        # restarted from the top (data is reshuffled), so no partial epoch is
        # silently skipped.
        start_epoch = int(ckpt.get("epoch", 0))
        _restore_rng(ckpt.get("rng"))

    def _checkpoint(name: str, epoch: int):
        if not cfg.checkpoint_dir:
            return
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, name),
            model, optimizer=opt, step=step, epoch=epoch,
            config=config, tokenizer_meta=tokenizer_meta,
        )

    result = TrainResult()
    tokens_seen = 0
    t0 = time.time()
    stop = False

    for epoch in range(start_epoch, cfg.epochs):
        if stop:
            break
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            lr = _lr_at(step, cfg, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            logits = _forward_logits(model, x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            tokens_seen += y.numel()
            step += 1

            if step % cfg.log_every == 0:
                result.train_losses.append(loss.item())
                result.steps.append(step)
                if cfg.log_fn is not None:
                    cfg.log_fn(step, total_steps, loss.item(), lr)

            if step % cfg.eval_every == 0 or step == total_steps:
                ev = evaluate(model, val_loader, device)
                result.val_losses.append(ev["loss"])
                result.val_perplexities.append(ev["perplexity"])

            if cfg.save_every and step % cfg.save_every == 0:
                _checkpoint(f"step_{step}.pt", epoch)

            # Wall-clock budget: stop cleanly (a final checkpoint is written
            # below) once the time limit is hit, mid-epoch if necessary.
            if cfg.max_seconds is not None and (time.time() - t0) >= cfg.max_seconds:
                stop = True
                break

    _checkpoint("final.pt", cfg.epochs)

    elapsed = time.time() - t0
    final = evaluate(model, val_loader, device)
    result.total_steps = step
    result.wall_time = elapsed
    result.tokens_per_sec = tokens_seen / max(1e-9, elapsed)
    result.final_val_loss = final["loss"]
    result.final_val_ppl = final["perplexity"]
    return result


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100,
                  temperature: float = 0.8, top_k: int = 0, top_p: float = 0.0,
                  device: str = "cpu") -> str:
    """
    Prompt -> text, tying a tokenizer around ``model.generate``.

    Encodes ``prompt`` (with a leading BOS but *no* trailing EOS -- appending EOS
    would tell the model the sequence is already finished), samples continuation
    ids, and decodes only the newly generated ids back to text. The tokenizer's
    EOS, when it exposes one, is used as the stop id.
    """
    model = model.to(device)
    # The model's embedding only spans ``vocab_size`` rows; a tokenizer may carry
    # special ids (BOS/EOS) above that (e.g. ByteTokenizer's 257/258 against a
    # 256-vocab model). Only feed / stop on specials the model can actually index.
    vocab = getattr(getattr(model, "args", None), "vocab_size", None)

    def _in_vocab(tok):
        return tok is not None and (vocab is None or 0 <= tok < vocab)

    ids = tokenizer.encode(prompt, add_special=False)
    bos = getattr(tokenizer, "BOS", None)
    if _in_vocab(bos):
        ids = [bos] + ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos = getattr(tokenizer, "EOS", None)
    if not _in_vocab(eos):
        eos = None
    out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k, top_p=top_p,
                         eos_token_id=eos)
    new_ids = out[0, input_ids.size(1):].tolist()
    return tokenizer.decode(new_ids)
