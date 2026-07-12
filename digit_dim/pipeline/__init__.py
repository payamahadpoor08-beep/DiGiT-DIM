"""Block / Transflow / ModelArgument: the pipeline backbone.

This package is the sole dependency root for pipeline-participating
code in digit_dim: a Block processes a ModelArgument, a Transflow routes
ModelArgument instances between Blocks. See block.py, transflow.py, and
model_argument.py for the individual class docs, and
/ARCHITECTURE.md at the repo root for how this fits into the wider
model.py codebase and what is intentionally out of scope for now.
"""

from .block import Block, MetricsHook
from .exceptions import (
    BackpressureError,
    BlockExecutionError,
    PipelineError,
    StageTimeoutError,
    TransflowError,
)
from .model_argument import ModelArgument
from .transflow import FailurePolicy, MergeFn, Transflow

__all__ = [
    "Block",
    "MetricsHook",
    "ModelArgument",
    "Transflow",
    "FailurePolicy",
    "MergeFn",
    "PipelineError",
    "BlockExecutionError",
    "TransflowError",
    "BackpressureError",
    "StageTimeoutError",
]
