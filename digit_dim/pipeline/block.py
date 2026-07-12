"""Block: a single processing stage in the pipeline.

Position in pipeline: the unit of work a Transflow schedules. Every stage
in the system -- core-model layers, validation steps, ETL transforms,
etc. -- that wants to participate in the Block/Transflow/ModelArgument
architecture does so by subclassing Block and implementing process().

Design choice: Block is deliberately a thin contract (one abstract
method) plus optional lifecycle hooks, rather than a large base class
that subclasses must fight against. Cross-cutting concerns (logging,
metrics) are injected rather than hardcoded, per dependency inversion --
a Block never imports a concrete logger or metrics backend, it calls
whatever was handed to it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .exceptions import BlockExecutionError
from .model_argument import ModelArgument

MetricsHook = Callable[[str, str, float], None]
"""Signature: metrics_hook(block_name, event, duration_seconds)."""


class Block(ABC):
    """Abstract base for a single pipeline processing stage.

    Thread-safety: Block itself holds no mutable state beyond the
    injected logger/metrics hook, both of which are treated as
    read-only after construction, so a stateless Block instance is safe
    to share across threads within a parallel Transflow fan-out.
    Stateful subclasses (e.g. ones with an internal cache) are
    responsible for their own synchronization -- Block does not
    serialize calls to process() for them.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        metrics_hook: Optional[MetricsHook] = None,
    ):
        self.name = name or type(self).__name__
        self._logger = logger or logging.getLogger(f"digit_dim.pipeline.block.{self.name}")
        self._metrics_hook = metrics_hook

    @abstractmethod
    def process(self, arg: ModelArgument) -> ModelArgument:
        """Transform the given ModelArgument and return the result.

        Implementations should not mutate arg.payload in ways that break
        the single-writer rule described in ModelArgument's docstring:
        the caller relinquishes ownership of `arg` for the duration of
        this call and takes ownership of the returned value.
        """
        raise NotImplementedError

    def on_start(self, arg: ModelArgument) -> None:
        """Hook invoked immediately before process(). Override for setup/telemetry."""

    def on_success(self, arg: ModelArgument, result: ModelArgument, duration: float) -> None:
        """Hook invoked after a successful process() call."""

    def on_error(self, arg: ModelArgument, error: BaseException, duration: float) -> None:
        """Hook invoked after process() raises. Does not suppress the error."""

    def run(self, arg: ModelArgument) -> ModelArgument:
        """Execute this Block with lifecycle hooks, logging, and error wrapping.

        This is the entry point Transflow calls -- never process()
        directly -- so hooks and metrics fire consistently regardless of
        which orchestration mode (sequential/parallel/conditional) is in
        use.

        Raises:
            BlockExecutionError: if process() raises any exception. The
                original exception is available as .original.
        """
        import time

        start = time.monotonic()
        self.on_start(arg)
        try:
            result = self.process(arg)
        except Exception as exc:
            duration = time.monotonic() - start
            self._logger.exception("Block '%s' failed after %.4fs", self.name, duration)
            self.on_error(arg, exc, duration)
            if self._metrics_hook:
                self._metrics_hook(self.name, "error", duration)
            raise BlockExecutionError(self.name, exc) from exc

        duration = time.monotonic() - start
        result.record(self.name)
        self.on_success(arg, result, duration)
        if self._metrics_hook:
            self._metrics_hook(self.name, "success", duration)
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
