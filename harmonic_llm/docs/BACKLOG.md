# Harmonic-LLM backlog

An honest catalogue of what is verified, what is known-broken/unverified, and
what remains untested. `_core.py` defines ~337 classes; only a small subset sit
on the default forward/train/generate path. This document tracks the rest so they
are visible, not silently assumed correct.

Consistent with `ARCHITECTURE.md`, this project rejects quota-driven "strengthen
N classes" targets: classes are hardened when they are actually reached and can
be verified, not to hit a count.

## Verified this session (default path + the five fixes)

Covered by the `pytest` suite and exercised end-to-end:

- **Core blocks** — `RMSNorm`, `ParallelEmbedding`, `ParallelHead`,
  `MoEAttention` (default attention), `ResonantFlow`, `SinkhornFlowRouter`
  (balance + no-drop), `HarmonicFlowFFN`/`HarmonicFlowMoE`, `Transformer`
  forward/backward.
- **Generation** — `Transformer.generate` + sampling + `generate_text` + CLI.
- **Checkpointing** — `save_checkpoint`/`load_checkpoint`/resume.
- **Tokenizers** — `ByteTokenizer`, `BPETokenizer`.
- **MLA** — `MultiHeadLatentAttention` forward/backward/generation across a
  head/dim ratio grid, with construction-time invariants.

## Known issues found but deliberately deferred

These were observed while fixing the five headline issues and are scoped out of
this round; each needs its own change with tests:

1. **MoEAttention routing is not strictly causal.** `_route` uses `x.mean(1)` —
   the mean over the whole sequence — to pick experts, so during a full-sequence
   training forward the expert choice for early positions can see later tokens.
   The per-expert attention (`_MHAExpert`) *is* causal, and single-token
   generation is unaffected (each step only feeds past tokens), but training-time
   routing leaks future information. Fix: causal/prefix-mean routing.
2. **MLA compression path unverified.** The `compress_ratios != 0` branch
   (`Compressor`/`Indexer`/`MultiScaleCompressor`, multi-scale KV cache) is not
   exercised by any config here (`ModelConfig` sets `compress_ratios = 0`). The
   plain MLA path is fixed and tested; the compressed path is untested.
3. **MLA heavy sub-modules default ON in `ModelArgs`.** `use_atr`,
   `use_fast_weights`, `use_flash_sparse`, and the MLA-only `CausalReasoner`
   (gated on `use_causal`) default to `True` in `ModelArgs` and are disabled by
   `ModelConfig.to_model_args` for the plain path. Several of them apply modules
   built for `(B,S,dim)` to the 4-D `(B,S,H,D)` attention output and would need
   per-module fixes before they can be enabled.
4. **`act_quant_triton` FP8 path** runs a quantise/dequantise on CPU (no real
   FP8, no Triton). It is out of the gradient path now; a proper FP8 story needs
   hardware and is untouched.

## Untested / opt-in subsystems (the long tail)

The remaining classes are heavy, specialised, opt-in subsystems, off by default
and not exercised here. Broad categories:

- **Reasoning** — `UltraAdvancedThinking`, `GraphReasoner`, `CausalReasoner`,
  `BayesianUncertainty`, `NeuralSymbolicEngine`, `MonteCarloTreeSearch`,
  `MultiAgentDebate`, `SpatialReasoner`, `TemporalReasoner`, ...
- **Memory** — `InfiniMemory`, `NarrativeMemory`, `DifferentiableExternalMemory`,
  `MetaMemoryBank`, `FastWeightUpdater`, `SelfOrganizingMap`, ...
- **Quantisation / compression** — `UnifiedSMMQP` (partly exercised via q_mode),
  `ZeroMassXOX` (tested), `Compressor`, `AdaptiveKVCompressor`,
  `TritonGEMMAutotuner`, `MultiScaleCompressor`.
- **Speculative decoding** — `MedusaTreeSpeculator`, `InternetAugmentedGenerator`
  (note: these carry their own `generate` methods requiring `self.model`; they
  are *not* the base decode loop, which now lives on `Transformer`).
- **Out-of-pipeline product domains** — media generation, robotics/IoT/drone,
  blockchain, trading, DevOps/security. These are unrelated to the LM core and
  are candidates to relocate out of `_core.py` entirely.

### How to strengthen one

When a subsystem is enabled and reached: construct it via `ModelConfig`/`ModelArgs`,
drive a real forward on the shapes it actually receives, add tests for
shape/finiteness/gradient-flow and its specific contract, then fix what breaks —
exactly the loop used for MLA and generation this session.
