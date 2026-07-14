"""
Command-line interface for Harmonic-LLM.

Usage:
    harmonic-llm info   --config configs/small.yaml
    harmonic-llm build  --config configs/small.yaml
    harmonic-llm demo
"""
from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def cmd_info(args):
    """Print the parameter breakdown for a config without a full forward."""
    import harmonic_llm as hl
    from harmonic_llm.builder import parameter_summary

    cfg = hl.ModelConfig.from_yaml(args.config) if args.config else hl.ModelConfig.small()
    model = hl.build_model(cfg)
    summary = parameter_summary(model)
    print(f"\nHarmonic-LLM v{hl.__version__}")
    print(f"  dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} vocab={cfg.vocab_size}")
    print(f"  total params    : {summary['_total']:,} ({summary['_total']/1e6:.1f}M)")
    print(f"  trainable params: {summary['_trainable']:,}")
    print("  top modules:")
    for name, n in sorted(summary.items(), key=lambda kv: -kv[1]):
        if not name.startswith("_"):
            print(f"    {name:28s} {n:>14,}")


def cmd_build(args):
    """Build a model and run a single forward pass to validate the config."""
    import torch
    import harmonic_llm as hl

    cfg = hl.ModelConfig.from_yaml(args.config) if args.config else hl.ModelConfig.tiny()
    model = hl.build_model(cfg, device=args.device)
    ids = torch.randint(0, cfg.vocab_size, (1, min(16, cfg.max_seq_len)))
    if args.device:
        ids = ids.to(args.device)
    out = model(ids, start_pos=0)
    logits = out[0] if isinstance(out, tuple) else out
    ok = not torch.isnan(logits).any()
    print(f"forward OK: shape={tuple(logits.shape)} finite={ok}")
    return 0 if ok else 1


def cmd_demo(args):
    """Build the tiny model and show a forward + one training step."""
    import torch
    import harmonic_llm as hl

    print("Building tiny demo model...")
    model = hl.build_model(hl.ModelConfig.tiny())
    ids = torch.randint(0, 256, (1, 16))
    out = model(ids, start_pos=0)
    logits = out[0] if isinstance(out, tuple) else out
    print(f"  forward: {tuple(logits.shape)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = logits.float().pow(2).mean()
    loss.backward()
    opt.step()
    print(f"  one training step OK, loss={loss.item():.4f}")


def cmd_generate(args):
    """Build a model and generate text from a prompt."""
    import harmonic_llm as hl
    from harmonic_llm.training import ByteTokenizer, generate_text

    cfg = hl.ModelConfig.from_yaml(args.config) if args.config else hl.ModelConfig.tiny()
    model = hl.build_model(cfg, device=args.device)
    tokenizer = ByteTokenizer()
    text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device or "cpu",
    )
    print(f"{args.prompt}{text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harmonic-llm", description="Harmonic-LLM CLI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="show parameter breakdown for a config")
    pi.add_argument("--config", help="path to a YAML config")
    pi.set_defaults(func=cmd_info)

    pb = sub.add_parser("build", help="build a model and validate with a forward pass")
    pb.add_argument("--config", help="path to a YAML config")
    pb.add_argument("--device", help="torch device (cpu/cuda)")
    pb.set_defaults(func=cmd_build)

    pd = sub.add_parser("demo", help="run the tiny end-to-end demo")
    pd.set_defaults(func=cmd_demo)

    pg = sub.add_parser("generate", help="generate text from a prompt")
    pg.add_argument("--config", help="path to a YAML config")
    pg.add_argument("--device", help="torch device (cpu/cuda)")
    pg.add_argument("--prompt", default="Hello", help="text prompt to continue")
    pg.add_argument("--max-new-tokens", type=int, default=64, dest="max_new_tokens")
    pg.add_argument("--temperature", type=float, default=0.8)
    pg.add_argument("--top-k", type=int, default=0, dest="top_k")
    pg.add_argument("--top-p", type=float, default=0.0, dest="top_p")
    pg.set_defaults(func=cmd_generate)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    result = args.func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
