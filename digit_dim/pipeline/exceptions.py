"""Exception hierarchy for the Block / Transflow / ModelArgument pipeline.

Position in pipeline: cross-cutting. Every other module in this package
raises subclasses of PipelineError so callers can catch pipeline failures
without needing to know which stage produced them.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by the pipeline runtime."""


class BlockExecutionError(PipelineError):
    """Raised when a Block's process() call fails.

    Wraps the original exception so the stage name and the causal chain
    are both preserved for diagnostics.
    """

    def __init__(self, block_name: str, original: BaseException):
        self.block_name = block_name
        self.original = original
        super().__init__(f"Block '{block_name}' failed: {original!r}")


class TransflowError(PipelineError):
    """Raised for orchestration-level failures inside a Transflow.

    Distinct from BlockExecutionError: this covers failures in the flow's
    own control logic (e.g. an unmergeable fan-in), not a single stage.
    """


class BackpressureError(TransflowError):
    """Raised when a bounded Transflow.stream() cannot admit new work.

    Signals that producers are outrunning consumers; callers decide
    whether to retry, drop, or block upstream.
    """


class StageTimeoutError(BlockExecutionError):
    """Raised when a Block exceeds its configured execution deadline."""
