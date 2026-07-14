#!/usr/bin/env python3
"""
Train Harmonic-LLM on Persian (fa.wikipedia) text, end to end.

Pipeline (all reused from the package):
  1. train a BPE tokenizer on a sample of the corpus          (BPETokenizer)
  2. build a medium model                                     (ModelConfig/build_model)
  3. time-bounded training with checkpointing                 (train + TrainConfig)
  4. report perplexity vs the uniform-random baseline         (evaluate)
  5. generate Persian text before vs after                    (generate_text)

Honest scope: this is a CPU run. The goal is to demonstrate the model measurably
*learns Persian statistics* (perplexity far below chance) and that the whole
pipeline -- tokenizer, training, checkpoint/resume, generation -- works on real
text. It is not, and cannot on this hardware be, a capable assistant.

Usage:
    python scripts/train_persian.py --corpus data/fa_corpus.txt --max-minutes 45
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import harmonic_llm as hl
from harmonic_llm.training import (
    BPETokenizer, LanguageModelDataset, TrainConfig, train, evaluate,
    generate_text, save_checkpoint, load_checkpoint, load_model_state,
)


PROMPTS = ["ایران کشوری در", "زبان فارسی", "تاریخ جهان", "علم و دانش"]


def _fmt(n):
    return f"{n:,}"


def build_tokenizer(corpus: str, vocab_size: int, sample_mb: float,
                    out_path: str) -> BPETokenizer:
    sample = corpus[: int(sample_mb * 1_000_000)]
    print(f"[bpe] training on {len(sample)/1e6:.1f} MB sample, vocab={vocab_size} ...")
    t0 = time.time()
    tok = BPETokenizer.train(sample, vocab_size=vocab_size)
    tok.save(out_path)
    print(f"[bpe] {len(tok)} tokens ({len(tok.merges)} merges) in {time.time()-t0:.0f}s "
          f"-> {out_path}")
    return tok


def make_datasets(corpus: str, tok: BPETokenizer, seq_len: int, train_mb: float):
    train_text = corpus[: int(train_mb * 1_000_000)]
    # Hold out a contiguous tail for validation (unseen during training).
    val_text = corpus[int(train_mb * 1_000_000): int(train_mb * 1_000_000) + 200_000]
    if len(val_text) < seq_len * 4:
        val_text = train_text[-200_000:]
    print(f"[data] encoding train={len(train_text)/1e6:.1f}MB val={len(val_text)/1e6:.2f}MB ...")
    t0 = time.time()
    train_ds = LanguageModelDataset(train_text, tok, seq_len=seq_len)
    val_ds = LanguageModelDataset(val_text, tok, seq_len=seq_len)
    print(f"[data] {_fmt(len(train_ds))} train / {_fmt(len(val_ds))} val windows "
          f"({len(train_ds.data)} train tokens) in {time.time()-t0:.0f}s")
    return train_ds, val_ds


def sample_generations(model, tok, device, max_new=60, temperature=0.7):
    outs = []
    for p in PROMPTS:
        cont = generate_text(model, tok, p, max_new_tokens=max_new,
                             temperature=temperature, top_k=40, device=device)
        outs.append((p, cont))
    return outs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train Harmonic-LLM on Persian text")
    ap.add_argument("--corpus", default="data/fa_corpus.txt")
    ap.add_argument("--run-dir", default="runs/persian")
    ap.add_argument("--max-minutes", type=float, default=45.0)
    # tokenizer
    ap.add_argument("--vocab-size", type=int, default=6000)
    ap.add_argument("--bpe-mb", type=float, default=2.0)
    # data
    ap.add_argument("--train-mb", type=float, default=4.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    # model. Defaults are sized for a ~40 min CPU run (the HarmonicFlow bank makes
    # each step expensive, so a smaller model trained for more steps demonstrates
    # learning far better here than a large model barely trained). Scale up on GPU.
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=40)
    ap.add_argument("--flows", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true",
                    help="resume from runs/<dir>/final.pt if present")
    args = ap.parse_args(argv)

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    tok_path = os.path.join(args.run_dir, "tokenizer.bpe.json")

    if not os.path.exists(args.corpus):
        print(f"corpus not found: {args.corpus}\nRun: python scripts/fetch_persian.py",
              file=sys.stderr)
        return 2
    corpus = Path(args.corpus).read_text(encoding="utf-8")
    print(f"[corpus] {len(corpus)/1e6:.1f} MB, {corpus.count(chr(10))} lines")

    # 1. tokenizer (reuse if already trained for this run)
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        print(f"[bpe] reusing {tok_path}: {len(tok)} tokens")
    else:
        tok = build_tokenizer(corpus, args.vocab_size, args.bpe_mb, tok_path)

    # 2. datasets
    train_ds, val_ds = make_datasets(corpus, tok, args.seq_len, args.train_mb)

    # 3. model
    cfg = hl.ModelConfig(
        dim=args.dim, n_layers=args.layers, n_heads=args.heads,
        head_dim=args.head_dim, vocab_size=len(tok), max_seq_len=args.seq_len,
        max_batch_size=args.batch_size, n_flows=args.flows,
        window_size=min(256, args.seq_len), use_moe_attention=True,
    )
    model = hl.build_model(cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"flows={cfg.n_flows} vocab={cfg.vocab_size} -> {_fmt(n_params)} params "
          f"({n_params/1e6:.1f}M)")

    uniform_ppl = float(len(tok))          # chance perplexity == vocab size
    print(f"[baseline] uniform-random perplexity = {uniform_ppl:.0f} (= vocab size)")

    # "before" generations (untrained)
    print("\n[before training] samples:")
    for p, c in sample_generations(model, tok, device):
        print(f"  «{p}» -> {c[:80]!r}")

    # calibrate steps/sec on a few steps to size the LR horizon to the time budget
    from torch.utils.data import DataLoader
    cal_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    import torch.nn.functional as F
    t0 = time.time(); n_cal = 0
    for x, y in cal_loader:
        opt.zero_grad(set_to_none=True)
        out = model(x, start_pos=0)
        logits = out[0] if isinstance(out, tuple) else out
        F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)).backward()
        opt.step(); n_cal += 1
        if n_cal >= 5:
            break
    sps = n_cal / max(1e-6, time.time() - t0)
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    budget_steps = int(sps * args.max_minutes * 60)
    epochs = max(1, math.ceil(budget_steps / steps_per_epoch))
    print(f"\n[calibrate] ~{sps:.2f} steps/s -> ~{budget_steps} steps in "
          f"{args.max_minutes:.0f} min ({epochs} epochs of {steps_per_epoch})")

    def log_fn(step, total, loss, lr):
        el = time.time() - train_t0
        print(f"  step {step}/{total} loss={loss:.3f} lr={lr:.2e} "
              f"ppl~{math.exp(min(20, loss)):.1f} [{el:.0f}s]", flush=True)

    tcfg = TrainConfig(
        epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        warmup_steps=min(100, budget_steps // 10 + 1),
        log_every=25, eval_every=200,
        checkpoint_dir=args.run_dir, save_every=200,
        max_seconds=args.max_minutes * 60, log_fn=log_fn,
        resume_from=(os.path.join(args.run_dir, "final.pt")
                     if args.resume and os.path.exists(os.path.join(args.run_dir, "final.pt"))
                     else None),
    )

    print("\n[train] starting...")
    train_t0 = time.time()
    # rebuild a fresh model+optimizer inside train() (it makes its own optimizer);
    # our calibration optimizer above already nudged weights slightly -- fine.
    result = train(model, train_ds, val_ds, tcfg, config=cfg,
                   tokenizer_meta={"kind": "bpe", "vocab": len(tok), "path": tok_path})
    wall = time.time() - train_t0

    # 4. final metrics
    final = evaluate(model, DataLoader(val_ds, batch_size=args.batch_size), device, max_batches=32)
    print(f"\n[result] steps={result.total_steps} wall={wall:.0f}s "
          f"tok/s={result.tokens_per_sec:.0f}")
    print(f"[result] final val loss={final['loss']:.3f} "
          f"perplexity={final['perplexity']:.1f}  (chance={uniform_ppl:.0f})")
    improvement = uniform_ppl / max(1e-9, final["perplexity"])
    print(f"[result] {improvement:.1f}x better than chance")

    # 5. "after" generations
    print("\n[after training] samples:")
    after = sample_generations(model, tok, device)
    for p, c in after:
        print(f"  «{p}» -> {c[:100]!r}")

    # persist a machine-readable summary for the results doc
    summary = {
        "params": n_params,
        "config": cfg.to_dict(),
        "vocab": len(tok),
        "train_tokens": int(len(train_ds.data)),
        "steps": result.total_steps,
        "wall_seconds": round(wall, 1),
        "tokens_per_sec": round(result.tokens_per_sec, 1),
        "final_val_loss": round(final["loss"], 4),
        "final_val_ppl": round(final["perplexity"], 2),
        "uniform_ppl": uniform_ppl,
        "x_better_than_chance": round(improvement, 2),
        "val_loss_curve": [round(v, 4) for v in result.val_losses],
        "samples_after": [{"prompt": p, "text": c} for p, c in after],
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[done] summary -> {os.path.join(args.run_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
