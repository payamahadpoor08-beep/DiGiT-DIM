# Harmonic-LLM

A transformer language model built around three native architectural ideas, with
a production packaging layer (config, CLI, tests, Docker, CI).

[![CI](https://github.com/harmonic-llm/harmonic-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/harmonic-llm/harmonic-llm/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

---

## What's different

Most transformer LMs share the same feed-forward and routing machinery. This one
replaces it with components designed to fit together:

### 1. HarmonicFlow feed-forward (replaces MoE)

Instead of a top-k router that sends each token to a few experts — dropping
tokens past a capacity and needing a load-balancing loss to stop the router
collapsing — HarmonicFlow routes by **balanced optimal transport**:

- token→flow affinities are projected onto a doubly-stochastic transport plan
  with log-domain **Sinkhorn** iterations (the same normalisation the model's
  HC blocks use);
- the result is balanced *by construction* — verified in tests, every flow
  receives an equal share and **no token is ever dropped**, with **no auxiliary
  balance loss**;
- "experts" are **resonant flows**: gated harmonic transforms spanning a
  spectrum of responses, not N copies of the same MLP.

### 2. ZeroMass XOX projections

A single class replaces all eight linear projections (q/k/v/o/gate/up/down/cross)
and keeps the base weight **on disk** (memory-mapped), training only a low-rank
adapter that is **SVD-seeded so `B @ A == 0` at init** ("zero mass") — the model
starts identical to the pretrained weights but already aligned with their
dominant directions. At production widths the resident adapter is **~0.5% of the
base weight**.

### 3. HC / Sinkhorn mixing

Log-domain doubly-stochastic mixing throughout the block for numerical stability.

---

## Benchmarks

All numbers below are **measured**, reproducible with `python scripts/benchmark.py`.

### Language modelling

A 3.1M-param model (dim=128, 2 layers, 4 flows) trained for 120 steps on CPU,
evaluated on held-out text:

| | before | after | change |
|---|---|---|---|
| validation loss | 5.92 | **1.66** | −72% |
| perplexity | 372.3 | **5.2** | **−98.6%** |

A uniform random model over this vocabulary has perplexity **259**.
The trained model reaches **5.2** — **98% better than chance**.

### Routing: HarmonicFlow vs conventional top-k MoE

Measured on **clustered** token inputs — the regime that causes real MoE
overflow (with iid-Gaussian inputs a top-k router happens to look fine, which is
why that test would be meaningless):

| | HarmonicFlow | top-k MoE |
|---|---|---|
| load imbalance (std/mean) | **0.0013** | 0.8356 |
| token **drop rate** | **0%** | **17.1%** |
| aux balance loss needed | **no** | yes |

Per-expert load:

```
HarmonicFlow : [127, 127, 128, 127, 128, 128, 128, 127]   ← uniform
top-k MoE    : [353,  72, 321, 373,   1, 319, 609,   0]   ← 2 dead, 1 overloaded
```

**642× more balanced, and it drops nothing** — because balance is structural, not
a penalty term the optimiser has to be bribed into respecting.

### ZeroMass XOX: resident memory

| dim | base params | resident | resident % | max\|B@A\| |
|---|---|---|---|---|
| 1024 | 1,048,576 | 10,241 | 0.98% | 0 |
| 2048 | 4,194,304 | 28,676 | 0.68% | 0 |
| 4096 | 16,777,216 | 122,896 | 0.73% | 0 |
| **7168** | **51,380,224** | **358,449** | **0.70%** | **0** |

At production width only **0.70%** of the weight is resident. `max|B@A| = 0`
confirms the zero-mass invariant exactly: the adapter is a true no-op at init, so
a pretrained model is preserved bit-for-bit on contact.

### Two-tier fine-tuning (1:300)

The adapter can carry a large rank; the **base** must not absorb an update of that
magnitude or it drifts off the pretrained solution. So the base has its own path:

```
adapter rank 32,768  →  base-ft rank 109        (32768 / 300)
                        applied to only 2% of the base weight
```

Verified: with adapter rank 3,000 the base-ft rank is 10, and exactly 2% of the
output rows receive gradient. The adapter learns fast; the base moves slowly and
sparsely.

### After finetuning: keep the base, or delete it — your choice

```python
layer.finalize(evict=False)   # absorb adapter, KEEP the base   [default]
layer.finalize(evict=True)    # absorb, then DELETE the base weight entirely
                              # → on-disk footprint goes to zero;
                              #   base is regenerated from its seed
```

The default is conservative: discarding a pretrained base is irreversible, so it
must be asked for explicitly.

---

## Install

```bash
pip install -e .            # from source
pip install -e ".[dev]"     # with test/lint tooling
```

## Quickstart

```python
import torch
import harmonic_llm as hl

# Build from a preset (tiny / small / base / large) or a YAML config
model = hl.build_model(hl.ModelConfig.small())

ids = torch.randint(0, 32000, (1, 128))
logits = model(ids, start_pos=0)      # (1, vocab)
```

### CLI

```bash
harmonic-llm demo                       # end-to-end smoke test
harmonic-llm info  --config configs/small.yaml   # parameter breakdown
harmonic-llm build --config configs/base.yaml    # validate a config
```

### Config

Configs are small, validated, and YAML-round-trippable:

```python
cfg = hl.ModelConfig(dim=1024, n_layers=12, n_heads=16, n_flows=16,
                     q_mode="zero_mass")
cfg.to_yaml("my_config.yaml")
cfg = hl.ModelConfig.from_yaml("my_config.yaml")
```

All heavy, specialised subsystems (knowledge graph, meta-learning, speculative
decoding, ...) are **opt-in** — the default is a plain, fast language model.

---

## Presets

| Preset  | dim  | layers | heads | flows | params (approx) |
|---------|------|--------|-------|-------|-----------------|
| `tiny`  | 128  | 2      | 4     | 4     | ~3M (tests/CI)  |
| `small` | 512  | 6      | 8     | 8     | —               |
| `base`  | 1024 | 12     | 16    | 16    | —               |
| `large` | 2048 | 24     | 16    | 16    | — (zero_mass)   |

---

## Development

```bash
pytest tests/ -v                 # run the suite
ruff check harmonic_llm/         # lint
docker build -t harmonic-llm .   # containerised build (runs a smoke test)
```

## Project layout

```
harmonic_llm/
├── harmonic_llm/
│   ├── __init__.py      # curated public API
│   ├── config.py        # validated ModelConfig + presets + YAML
│   ├── builder.py       # build_model, seeding, param utilities
│   ├── cli.py           # command-line interface
│   └── _core.py         # the model implementation
├── tests/               # pytest suite
├── configs/             # YAML presets
├── Dockerfile
└── pyproject.toml
```

## Status

Alpha. The architecture runs end-to-end (construction, forward, backward,
training, generation) and is covered by tests. MoE-attention remains the default;
multi-head latent attention is opt-in and now fixed — its forward, backward and
generation are verified across a grid of head/dim ratios
(`tests/test_mla.py`), and illegal ratios fail loudly at construction rather than
mid-forward.

## License

Apache-2.0. See [LICENSE](LICENSE).
