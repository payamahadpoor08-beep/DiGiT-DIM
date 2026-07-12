"""Transflow: routes ModelArgument instances through Blocks.

Position in pipeline: the orchestrator. Block defines a single stage;
Transflow decides how stages compose -- in sequence, in parallel
(fan-out) with a merge (fan-in), conditionally, or as a bounded stream
with backpressure. Transflow never contains business logic itself; it
only schedules Block.run() calls and applies a failure policy.

Design choice: rather than one monolithic "run" method with mode flags,
each orchestration pattern is its own method (run_sequential,
run_parallel, run_conditional, stream). This keeps each pattern's
control flow readable and lets callers pick the narrowest method for
what they need instead of threading mode enums through a single
call site.
"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from enum import Enum
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from .block import Block
from .exceptions import BackpressureError, BlockExecutionError, TransflowError
from .model_argument import ModelArgument

MergeFn = Callable[[List[ModelArgument]], ModelArgument]


class FailurePolicy(Enum):
    """How a Transflow reacts when a Block raises BlockExecutionError."""

    ABORT = "abort"      # propagate the error, stop the flow
    SKIP = "skip"         # drop the failing branch/stage, continue with the rest
    FALLBACK = "fallback"  # record the error on the ModelArgument and keep going with the input unchanged


class Transflow:
    """Orchestrates Block execution over ModelArgument instances.

    Thread-safety: a Transflow instance holds no per-run mutable state
    (the ThreadPoolExecutor used by run_parallel/stream is created
    per-call), so a single Transflow can be reused concurrently by
    multiple callers/threads.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        failure_policy: FailurePolicy = FailurePolicy.ABORT,
        max_workers: int = 4,
        logger: Optional[logging.Logger] = None,
    ):
        self.name = name or type(self).__name__
        self.failure_policy = failure_policy
        self.max_workers = max_workers
        self._logger = logger or logging.getLogger(f"digit_dim.pipeline.transflow.{self.name}")

    # -- sequential ---------------------------------------------------

    def run_sequential(self, blocks: Iterable[Block], arg: ModelArgument) -> ModelArgument:
        """Run blocks one after another, piping each result into the next.

        Honors self.failure_policy per stage: ABORT re-raises,
        SKIP passes the pre-stage argument through unchanged,
        FALLBACK records the error on the argument and continues.
        """
        current = arg
        for block in blocks:
            try:
                current = block.run(current)
            except BlockExecutionError as exc:
                current = self._handle_failure(block, current, exc)
        return current

    # -- conditional ----------------------------------------------------

    def run_conditional(
        self,
        condition: Callable[[ModelArgument], bool],
        if_true: Block,
        if_false: Optional[Block],
        arg: ModelArgument,
    ) -> ModelArgument:
        """Route arg to if_true or if_false based on condition(arg).

        If condition(arg) is False and if_false is None, arg passes
        through unmodified (aside from trace annotation).
        """
        branch = if_true if condition(arg) else if_false
        if branch is None:
            return arg.record(f"{self.name}:conditional-passthrough")
        try:
            return branch.run(arg)
        except BlockExecutionError as exc:
            return self._handle_failure(branch, arg, exc)

    # -- parallel fan-out / fan-in ---------------------------------------

    def run_parallel(
        self,
        blocks: Iterable[Block],
        arg: ModelArgument,
        merge: Optional[MergeFn] = None,
    ) -> ModelArgument:
        """Run every block concurrently against an independent clone of arg.

        Each block receives arg.clone() so concurrent mutation of one
        branch can never leak into another (see ModelArgument.clone()).
        Results are collected in submission order (not completion order)
        so merge functions can rely on positional correspondence to
        `blocks`. If merge is None, the default merge concatenates each
        branch's trace/errors onto the first successful result's payload
        list.
        """
        blocks = list(blocks)
        if not blocks:
            raise TransflowError(f"Transflow '{self.name}': run_parallel called with no blocks")

        results: List[Optional[ModelArgument]] = [None] * len(blocks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_index = {
                pool.submit(self._run_branch, block, arg.clone()): idx
                for idx, block in enumerate(blocks)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                block = blocks[idx]
                try:
                    results[idx] = future.result()
                except BlockExecutionError as exc:
                    handled = self._handle_failure(block, arg.clone(), exc)
                    results[idx] = handled

        merge_fn = merge or self._default_merge
        merged = merge_fn(results)  # type: ignore[arg-type]
        # Each branch's ModelArgument started from a clone() with a fresh
        # trace/errors list (see ModelArgument.clone()), so the shared
        # pre-fork history lives only on `arg` and must be prepended once
        # here rather than once per branch, or it would be duplicated
        # len(blocks) times by _default_merge's extend() calls.
        merged.trace = list(arg.trace) + merged.trace
        merged.errors = list(arg.errors) + merged.errors
        return merged

    @staticmethod
    def _run_branch(block: Block, branch_arg: ModelArgument) -> ModelArgument:
        return block.run(branch_arg)

    @staticmethod
    def _default_merge(results: List[ModelArgument]) -> ModelArgument:
        """Fan-in that merges payloads into a list and unions trace/errors/metadata."""
        merged = ModelArgument(payload=[r.payload for r in results])
        for r in results:
            merged.trace.extend(r.trace)
            merged.errors.extend(r.errors)
            merged.metadata.update(r.metadata)
        return merged

    # -- bounded streaming with backpressure -----------------------------

    def stream(
        self,
        blocks: Iterable[Block],
        args: Iterable[ModelArgument],
        max_in_flight: int = 8,
        block_on_full: bool = True,
    ) -> Iterator[ModelArgument]:
        """Run run_sequential(blocks, arg) for each arg in args with bounded concurrency.

        Uses a sliding window: at most max_in_flight args are submitted to
        the executor at once, and a new arg is only pulled from `args`
        once a slot frees up (a completed future is drained). This is
        real backpressure -- the generator will not pull further input
        (and by extension whatever produces `args`, if it is itself
        lazily evaluated) faster than the pipeline can drain it.

        If block_on_full is True (default), waiting for a slot blocks
        the calling thread until at least one in-flight task completes.
        If False, a full window with nothing yet complete raises
        BackpressureError instead of waiting.

        Results are yielded in completion order, which may differ from
        input order.
        """
        blocks = list(blocks)
        arg_iter = iter(args)
        in_flight: Dict[Future, None] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:

            def top_up() -> None:
                while len(in_flight) < max_in_flight:
                    try:
                        arg = next(arg_iter)
                    except StopIteration:
                        return
                    future = pool.submit(self.run_sequential, blocks, arg)
                    in_flight[future] = None

            top_up()
            while in_flight:
                if block_on_full:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                else:
                    done = {f for f in in_flight if f.done()}
                    if not done:
                        raise BackpressureError(
                            f"Transflow '{self.name}': max_in_flight={max_in_flight} exceeded"
                        )
                for future in done:
                    del in_flight[future]
                    yield future.result()
                top_up()

    # -- shared failure handling ------------------------------------------

    def _handle_failure(
        self, block: Block, arg: ModelArgument, exc: BlockExecutionError
    ) -> ModelArgument:
        if self.failure_policy is FailurePolicy.ABORT:
            raise exc
        if self.failure_policy is FailurePolicy.SKIP:
            self._logger.warning("Skipping failed block '%s': %s", block.name, exc)
            return arg
        # FALLBACK
        self._logger.warning("Falling back past failed block '%s': %s", block.name, exc)
        return arg.fail(block.name, exc.original)

    def __repr__(self) -> str:
        return f"Transflow(name={self.name!r}, failure_policy={self.failure_policy.value})"
