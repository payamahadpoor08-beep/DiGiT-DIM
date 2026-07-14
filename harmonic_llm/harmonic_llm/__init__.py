"""
Harmonic-LLM
============

A transformer language model built around three native ideas:

* **HarmonicFlow feed-forward** -- balanced optimal-transport (Sinkhorn) routing
  over a bank of resonant flows, replacing top-k MoE. No dropped tokens, no
  load-balancing loss; balance is structural.
* **ZeroMass XOX projections** -- disk-resident base weights with a low-rank,
  SVD-seeded ("zero mass") adapter, so a large model can be finetuned with a
  fraction of it resident in memory.
* **HC / Sinkhorn mixing** -- log-domain doubly-stochastic mixing throughout the
  block for numerical stability.

Public API
----------
    from harmonic_llm import build_model, ModelConfig, Transformer

    cfg = ModelConfig.small()
    model = build_model(cfg)
    logits = model(input_ids)

The heavy implementation lives in ``harmonic_llm._core``; this module curates a
stable surface over it so downstream code does not depend on internal layout.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# The core implementation is a single large module. We import it lazily and
# re-export a curated set of names, so (a) importing the package is cheap and
# (b) the public API is decoupled from the internal file's contents.
# ---------------------------------------------------------------------------
_core = importlib.import_module("harmonic_llm._core")

# Core architecture
Transformer = _core.Transformer
Block = _core.Block
ModelArgs = _core.ModelArgs

# Native feed-forward
HarmonicFlowFFN = _core.HarmonicFlowFFN
HarmonicFlowMoE = _core.HarmonicFlowMoE
ResonantFlow = _core.ResonantFlow
SinkhornFlowRouter = _core.SinkhornFlowRouter

# ZeroMass
ZeroMassXOX = _core.ZeroMassXOX
ZeroMassRankAllocator = _core.ZeroMassRankAllocator

# Building blocks commonly needed downstream
RMSNorm = _core.RMSNorm
Linear = _core.Linear

from harmonic_llm.config import ModelConfig      # noqa: E402
from harmonic_llm.builder import build_model      # noqa: E402

__all__ = [
    "__version__",
    "Transformer", "Block", "ModelArgs",
    "HarmonicFlowFFN", "HarmonicFlowMoE", "ResonantFlow", "SinkhornFlowRouter",
    "ZeroMassXOX", "ZeroMassRankAllocator",
    "RMSNorm", "Linear",
    "ModelConfig", "build_model",
]
