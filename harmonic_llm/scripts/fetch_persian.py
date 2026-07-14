#!/usr/bin/env python3
"""
Fetch a bounded Persian (fa.wikipedia) text corpus for training.

Source: the ``wikimedia/wikipedia`` `20231101.fa` dump hosted on HuggingFace as
Parquet shards. A single shard downloads fast over the agent proxy (~20 MB/s),
whereas the live MediaWiki API is latency-bound (~1 KB/s here) -- so we pull one
shard and stream its ``text`` column row-group by row-group until the target
size, rather than making thousands of API calls.

Requires ``pyarrow`` (fetch-only; not a package dependency):  pip install pyarrow

Two environment specifics are handled: a descriptive ``User-Agent`` and the agent
proxy's CA bundle (``/root/.ccr/ca-bundle.crt``) when present; ``requests`` honours
HTTP(S)_PROXY from the environment automatically.

Usage:
    python scripts/fetch_persian.py --target-mb 15 --out data/fa_corpus.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

UA = "DiGiT-DIM-research/0.1 (https://github.com/; payamahadpoor08@gmail.com)"
_CA = "/root/.ccr/ca-bundle.crt"

# Smallest shard of the fa dump (~142 MB) -- enough for many MB of clean text.
SHARD_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "20231101.fa/train-00002-of-00004.parquet"
)


def _kw():
    return {"verify": _CA} if os.path.exists(_CA) else {}


def download_shard(dst: str) -> str:
    """Download the parquet shard to ``dst`` (skips if already present)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"shard already present: {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    print(f"downloading fa.wikipedia shard -> {dst} ...")
    t0 = time.time()
    n = 0
    with requests.get(SHARD_URL, headers={"User-Agent": UA}, stream=True,
                      timeout=600, **_kw()) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                n += len(chunk)
    print(f"  {n/1e6:.0f} MB in {time.time()-t0:.0f}s")
    return dst


def build_corpus(shard_path: str, out_path: str, target_mb: float,
                 min_chars: int = 200) -> dict:
    import pyarrow.parquet as pq

    target = int(target_mb * 1_000_000)
    pf = pq.ParquetFile(shard_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total, n_articles = 0, 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for rg in range(pf.num_row_groups):
            if total >= target:
                break
            texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            for t in texts:
                if not t:
                    continue
                # One article per line; collapse internal whitespace/newlines so
                # the LM dataset sees continuous prose.
                line = " ".join(t.split())
                if len(line) < min_chars:
                    continue
                f.write(line + "\n")
                total += len(line) + 1
                n_articles += 1
                if total >= target:
                    break
    return {
        "articles": n_articles,
        "chars": total,
        "mb": round(total / 1e6, 3),
        "seconds": round(time.time() - t0, 1),
        "path": out_path,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a Persian fa.wikipedia corpus")
    ap.add_argument("--target-mb", type=float, default=15.0)
    ap.add_argument("--out", default="data/fa_corpus.txt")
    ap.add_argument("--shard", default="data/fa_wiki_shard.parquet",
                    help="local path for the downloaded parquet shard (cached)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--keep-shard", action="store_true",
                    help="keep the parquet shard after building the corpus")
    args = ap.parse_args(argv)

    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"corpus already exists: {args.out} "
              f"({os.path.getsize(args.out)/1e6:.2f} MB) -- reusing. "
              f"Delete it to re-fetch.")
        return 0

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("ERROR: this fetcher needs pyarrow. Install it with:\n"
              "    pip install pyarrow", file=sys.stderr)
        return 2

    download_shard(args.shard)
    print(f"building corpus -> {args.out} (target {args.target_mb} MB)...")
    meta = build_corpus(args.shard, args.out, args.target_mb, min_chars=args.min_chars)
    print(f"done: {meta['articles']} articles, {meta['mb']} MB in "
          f"{meta['seconds']}s -> {meta['path']}")

    if not args.keep_shard and os.path.exists(args.shard):
        os.remove(args.shard)
        print(f"removed shard {args.shard} (pass --keep-shard to keep it)")
    return 0 if meta["articles"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
