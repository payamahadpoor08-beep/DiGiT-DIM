# Architecture status

This document is the honest record of what exists, what changed, and what's
deliberately deferred. It supersedes any external directive's quantitative
targets (line-count quotas, fixed "N new classes" counts) as the source of
truth for this repo's actual state.

## What was here before this session

A single file, `model.py` — 87,318 lines, 320 top-level classes, no tests,
no package structure, no dependency manifest. It mixes a genuine
transformer/MoE ML core (`Transformer`, `Block`, `Attention`,
`MultiHeadLatentAttention`, `DeepSeekMoE`, memory and reasoning modules) with
entirely unrelated product-feature domains bolted into the same file:
blockchain wallet management, drone/robot/IoT control, algorithmic trading,
DevOps tooling, security scanning, and a large media-generation subsystem
(anime/manhwa/video/audio generation). Class names collide (`Block` vs
`_Block`, `AdaGN` vs `_AdaGN`). Nothing in the file had ever been exercised
by a test.

## What this session did

**1. A real Block / Transflow / ModelArgument pipeline** — `digit_dim/pipeline/`:

- `model_argument.py` — `ModelArgument`: the payload/metadata/context/trace
  carrier. Single-writer per hop, `clone()` for fan-out isolation.
- `block.py` — `Block`: abstract single-stage contract (`process()`), with
  injectable logger/metrics hooks and lifecycle callbacks, wrapping failures
  in `BlockExecutionError`.
- `transflow.py` — `Transflow`: orchestrates Blocks — `run_sequential`,
  `run_conditional`, `run_parallel` (fan-out/fan-in with pluggable merge),
  and `stream` (bounded-concurrency backpressure via a sliding window, with
  blocking or fail-fast `BackpressureError` modes).
- `exceptions.py` — `PipelineError` hierarchy.

This is new, independent infrastructure with no torch dependency, so it's
fully unit-tested in this environment (`tests/test_pipeline.py`, 18 tests,
all passing) — including a caught-and-fixed deadlock in an early draft of
`stream()` and a caught-and-fixed trace-duplication bug in `run_parallel`'s
fan-in.

`model.py`'s existing ML core was **not** migrated onto this Trinity in this
session — see Roadmap.

**2. Three confirmed, tested bugfixes in `model.py`'s `ModelArgs`**
(now `@dataclass`-decorated; search `# BUGFIX:` for each):

- Missing `@dataclass` decorator meant `field(default_factory=...)`
  attributes were raw `dataclasses.Field` sentinels, `__post_init__` never
  ran, and `derive_inference_args()`'s `ModelArgs(**self.__dict__)` had
  nothing to unpack.
- `_generate_dynamic_compress_ratios()` hardcoded `n = 400` while
  `n_layers` defaults to `480`, violating `__post_init__`'s own
  `len(compress_ratios) == n_layers` check — masked until the decorator fix
  made `__post_init__` actually run.
- `o_lora_rank` defaulted to `36864`, exceeding `group_dim` (6144, derived
  from `n_heads // o_groups * head_dim`) — also only surfaced once
  `__post_init__` started running. Set to a conservative `3072` pending
  real model-design input on the intended value.

Verified in `tests/test_model_args_bugfix.py` by extracting and exec'ing the
actual `ModelArgs` source out of `model.py` (torch isn't installed in this
sandbox, but `ModelArgs` has no torch dependency, so this exercises real
current file content rather than a hand-copied reimplementation).

**3. `requirements.txt`** — `numpy`, `PyYAML`, `torch`, matching the actual
top-level imports in `model.py` (no redis/pynvml/cryptography — those are
referenced only by name in some docstrings/strings, not imported).

## What was deliberately not done, and why

The originating directive for this refactor asked for several things this
session intentionally did not do, per explicit direction:

- **Force every class (blockchain, drone, IoT, trading, security, media
  generation, etc.) to inherit from or depend on Block/Transflow/
  ModelArgument.** Declined — the Trinity is scoped to the actual ML
  pipeline. Forcing unrelated domains into that dependency tree would be
  fake coupling with no engineering value.
- **Hit fixed quantitative targets** (100,000–220,000+ total lines, exactly
  30 new classes, ≥70%-similarity mechanical merging). Declined — these
  are anti-patterns that reward padding over correctness. Code here grows
  only as real functionality demands.
- **Claim "zero bugs" across the whole file.** Scoped down — verified
  fixes are limited to what's actually testable in this environment
  (no GPU, no torch installed, no real drone/blockchain/SMTP endpoints to
  integration-test against). Everything outside that scope is unverified,
  not "confirmed bug-free."

## Roadmap for future sessions

Rough first-pass categorization of the 320 existing classes in `model.py`,
for whoever picks this up next:

- **Core ML pipeline (candidate for migration onto Block/Transflow/
  ModelArgument):** transformer/attention (`Transformer`, `Block`,
  `DynamicBlock`, `Attention`, `MultiHeadLatentAttention`, `GatedDeltaNet`,
  `RMSNorm`, embeddings/linear variants), MoE (`DeepSeekMoE`, `MoE`,
  `Expert`, `Gate`, `AdaptiveMoERouter`, `ExpertChoiceRouter`), memory
  (`InfiniMemory`, `NarrativeMemory`, `ExternalMemory`, `HierarchicalMemory`,
  `RecurrentMemoryBank`), reasoning (`CausalReasoner`, `BayesianReasoner`,
  `ChainOfThoughtEngine`, `NeuralSymbolicEngine`, `MonteCarloTreeSearch`),
  meta-learning (`MAMLAdapter`, `MetaLearner`, `ElasticWeightConsolidation`,
  `RecursiveSelfImprovement`), compression/quantization (`UnifiedSMMQP`,
  `Compressor`, `AdaptiveKVCompressor`, `TritonGEMMAutotuner`).
- **Out-of-pipeline product domains (candidates to relocate into their own
  top-level modules, not touched this session):** blockchain
  (`BlockchainIntegrator`), robotics/IoT/drone (`RobotController`,
  `IoTDeviceController`, `DronePilot`), trading (`TechnicalAnalyzer`,
  `TradeExecutor`, `PortfolioOptimizer`, `Backtester`), DevOps/security
  (`OSShellExecutor`, `CybersecurityScanner`, `OAuth2Manager`,
  `AuditLogger`), media generation (`AnimeGenerator`, `ManhwaGenerator`,
  `VideoDiffusionRenderer`, `VoiceSynthesizer`), domain assistants
  (`LegalAssistant`, `MedicalDiagnostician`, `HealthAdvisor`).
- **Known-unresolved naming collisions:** `Block` (line ~7484, a transformer
  decoder layer) vs `_Block`/`MTPBlock` (line ~52660) — these will need
  disambiguation before `Block` can safely become the pipeline base class
  name; a rename is likely required rather than reusing the name for two
  concepts.
- **Further bug-hunting:** the `ModelArgs.__post_init__` fix above only
  checked what its own asserts cover. Since that method (and by extension
  most of this file) has never actually executed before, treat any code
  path that hasn't been covered by a real test as unverified rather than
  correct — this file likely has more of the same class of bug (dead
  validation, inconsistent defaults) waiting to be found by actually
  running things.
- **Real deduplication, not mechanical merging:** a genuine similarity
  pass (e.g. `Neural3DHead`/`Neural3DHeadFull`, the several near-duplicate
  `*Generator`/`*Renderer` media classes) is worth doing, but each
  candidate merge needs a human/architectural judgment call on whether the
  overlap is coincidental or structural — not an automatic threshold.

## The `harmonic_llm/` package (added this session)

A self-contained, packaged transformer LM was added under `harmonic_llm/`. It
shares architectural lineage with `model.py` (same
`Transformer`/`MultiHeadLatentAttention`/MoE core, plus a HarmonicFlow /
ZeroMass / Sinkhorn rework) but ships a real productization layer around it:
`config.py` (validated `ModelConfig` + presets + YAML), `builder.py`
(`build_model`), `cli.py`, `training.py` (tokenizer + dataset + train loop),
and a passing `pytest` suite. Unlike the top-level `model.py`, this package is
importable and runnable in this environment (CPU torch), so it is the surface
where the active model work happens.

Five known gaps are being closed here, in phases (see the model-issues plan):

1. **MLA (`MultiHeadLatentAttention`) breaks at some head/dim ratios** — opt-in;
   MoE-attention is the verified default.
2. **Byte-level tokenizer only** — a real BPE tokenizer is being added.
3. **No checkpoint/resume** — being added to `training.py`.
4. **No generation loop** — `Transformer` returns logits only; a real
   incremental `generate()` + text/CLI glue is being added.
5. **Many unstrengthened classes in `_core.py`** — a bounded, default-path
   subset is being hardened and tested; the rest are catalogued as a backlog
   (consistent with this repo's rejection of quota-driven "N classes" targets).

## Metrics (honest, as of this session)

| | |
|---|---|
| `model.py` | 87,337 lines, 320 classes (unchanged in count; 3 lines fixed) |
| `digit_dim/pipeline/` | 4 modules, ~460 lines, 4 public classes, all new |
| Tests | 22 tests, all passing, 0 skipped |
| Bugs fixed (verified) | 3 |
| Bugs claimed fixed elsewhere in `model.py` | 0 — not yet investigated |
