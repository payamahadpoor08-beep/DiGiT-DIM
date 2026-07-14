"""
Training pipeline: tokenizer, dataset, and a real training loop.

This is not a toy: it tokenises real text, batches it, trains with AdamW +
cosine schedule + gradient clipping, evaluates held-out loss and perplexity, and
reports throughput. It is what the benchmark script drives.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
          cfg: TrainConfig = TrainConfig()) -> TrainResult:
    """
    Train the model. Returns a full result record for benchmarking.
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

    result = TrainResult()
    step = 0
    tokens_seen = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
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

            if step % cfg.eval_every == 0 or step == total_steps:
                ev = evaluate(model, val_loader, device)
                result.val_losses.append(ev["loss"])
                result.val_perplexities.append(ev["perplexity"])

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
