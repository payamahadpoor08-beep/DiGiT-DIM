# Training Harmonic-LLM on Persian — results

An honest, reproducible record of training this model on real Persian text on a
**CPU-only** machine. Read the limitations section before drawing conclusions:
the point is to demonstrate the model **measurably learns Persian** and that the
whole pipeline (tokenizer → training → checkpoint/resume → generation) works on
real data — **not** to produce a capable assistant, which this hardware cannot.

Reproduce:
```bash
python scripts/fetch_persian.py --target-mb 15          # ~16k fa.wikipedia articles
python scripts/train_persian.py --max-minutes 38        # CPU-sized run
```

## Setup

| | |
|---|---|
| Hardware | 4 CPU cores, ~15 GB RAM, **no GPU** |
| Data | fa.wikipedia (`wikimedia/wikipedia` 20231101.fa), 16,153 articles, 15 MB; 4 MB used for training, 200 KB held out for validation |
| Tokenizer | byte-level **BPE**, vocab 5,000 (trained on a 1 MB sample) |
| Model | HarmonicFlow (MoE attention + resonant-flow FFN), dim 160, 3 layers, 4 heads, 4 flows, seq 128 — **6.8 M params** |
| Training | AdamW + cosine schedule, batch 16, 565 steps, ~38 min wall, checkpointed every 200 steps |

Model size was chosen for the compute budget: the HarmonicFlow bank makes each
step cost ~4 s on CPU, so a **smaller model trained for more steps** demonstrates
learning far better here than a large model barely trained. Everything scales up
unchanged on a GPU.

## Result: it learns Persian

| metric | value |
|---|---|
| uniform-random baseline perplexity | **5000** (= vocab size; a model that has learned nothing) |
| **final validation perplexity** | **78.1** |
| improvement over chance | **64× better** |
| validation loss | 5.14 → 4.36 |
| throughput | ~507 tokens/s (CPU) |

A perplexity of 78 against a chance level of 5000 is the concrete signal that the
model learned real structure in Persian — token frequencies, common words, and
spacing — not noise.

## Generation: before vs after

Same prompts, greedy-ish sampling (temperature 0.7, top-k 40).

**Before training** (random init) — Persian *tokens* but no structure, words run
together with no spacing:

```
«ایران کشوری در» → آغاز۱۵اظistrِننزددقیقهاسلامکارشناسیمیانمأAپالرموهیده‌یوشیسینما...
«زبان فارسی»    → تیغمحمدحسینمهنداولیه‌سازینازیافهکردهیداطلاعاتسنیاَکمر...
```

**After training** — the model has learned Persian's most frequent words and
particles (در، که، و، به، از، را، است), correct word spacing, and phrase shapes
like «شده‌است» and «(میلادی)»:

```
«ایران کشوری در» → را در با با که و و اهل به را از به برا که است یک که و استان در و این و در شده‌است. است ایتالیا در ...
«زبان فارسی»    → در که که و منابع از در است یا (م در یک و که به در به ایران به که در و از به که استان و و در ...
«تاریخ جهان»    → و اهل از که یک در به در در در شده‌است. در از در در ایتالیا در در است. در که شده در و از در سال و آمریکا و ...
«علم و دانش»    → در از پیوند ایتالیا که به و و در با یک که بر در در و در در در اهل در (میلادی) به (میلادی) در که است. در و به ...
```

This is exactly what a small model on limited compute learns *first*: the
high-frequency vocabulary and local statistics of the language. It is **not**
fluent or meaningful Persian — reaching that needs far more scale, data and
training time than a CPU allows (see below).

## Architecture benchmark

The model's signature component — HarmonicFlow's balanced optimal-transport
routing — is benchmarked against a conventional top-k MoE by
`python scripts/benchmark.py` (reproducible; the run also trains on a synthetic
corpus and reports perplexity vs a random baseline). The measured routing result,
documented in the package README, is what motivates the architecture:

| on clustered inputs | HarmonicFlow | top-k MoE |
|---|---|---|
| load imbalance (std/mean) | **0.0013** | 0.8356 |
| token **drop rate** | **0%** | **17.1%** |
| aux balance loss needed | **no** | yes |

Balance is structural (Sinkhorn), so no token is dropped and no load-balancing
loss is needed — **642× more balanced** than top-k on the regime that causes real
MoE overflow. Run `python scripts/benchmark.py` to reproduce on your machine (a
few minutes on CPU).

## Honest limitations (read this)

- **This is a CPU demonstration, not a capable model.** Model capability is
  bounded by scale × data × compute; all three are small here. 6.8 M params, 4 MB
  of text, 565 steps. The output is Persian-token-level correct, not sentence-level
  meaningful.
- **Reaching a genuinely capable Persian model** needs a GPU (or many), hundreds
  of millions to billions of parameters, gigabytes of text, and hours-to-days of
  training — none of which this environment provides. The pipeline here is the
  same one that would scale up; only the knobs and hardware change.
- **"Using every capability" would not help.** The advanced reasoning/memory
  subsystems (see `docs/BACKLOG.md`) are opt-in, largely untrained, and some are
  broken; enabling them injects randomly-initialised noise, which *degrades*
  output rather than adding power. They are deliberately off.
- **`zero_mass` / q-modes are a memory/finetuning mechanism, not a capability
  switch.** They change how weights are stored/adapted, not how much the model
  knows. They add overhead with no quality gain on a small CPU model, so the main
  run uses the standard path. The pipeline supports `q_mode="zero_mass"` if you
  want to exercise that path.
- **The verified default path** (HarmonicFlow MoE + BPE) is what was trained and
  measured. MLA (now fixed, `docs/BACKLOG.md`) is a slower, fully-causal
  alternative; MoE was chosen for CPU speed.
