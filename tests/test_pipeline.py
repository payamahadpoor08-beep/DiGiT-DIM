"""Runtime tests for digit_dim.pipeline (Block / Transflow / ModelArgument).

Pure stdlib (unittest), no torch/numpy dependency, so these run in any
environment that can run Python -- including ones without the ML stack
installed, which is the actual constraint in this sandbox.
"""

import threading
import time
import unittest

from digit_dim.pipeline import (
    BackpressureError,
    Block,
    BlockExecutionError,
    FailurePolicy,
    ModelArgument,
    Transflow,
)


class UppercaseBlock(Block):
    def process(self, arg: ModelArgument) -> ModelArgument:
        return arg.with_payload(str(arg.payload).upper())


class AppendSuffixBlock(Block):
    def __init__(self, suffix: str, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix

    def process(self, arg: ModelArgument) -> ModelArgument:
        return arg.with_payload(f"{arg.payload}{self.suffix}")


class AlwaysFailsBlock(Block):
    def process(self, arg: ModelArgument) -> ModelArgument:
        raise ValueError("intentional failure")


class SlowBlock(Block):
    """Sleeps for `delay` seconds, tracking peak concurrency via a shared counter."""

    def __init__(self, delay: float, counter: "ConcurrencyCounter", **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.counter = counter

    def process(self, arg: ModelArgument) -> ModelArgument:
        self.counter.enter()
        try:
            time.sleep(self.delay)
        finally:
            self.counter.exit()
        return arg


class ConcurrencyCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = 0
        self.peak = 0

    def enter(self):
        with self._lock:
            self._current += 1
            self.peak = max(self.peak, self._current)

    def exit(self):
        with self._lock:
            self._current -= 1


class BlockLifecycleTests(unittest.TestCase):
    def test_run_records_trace_and_returns_result(self):
        block = UppercaseBlock()
        result = block.run(ModelArgument(payload="hello"))
        self.assertEqual(result.payload, "HELLO")
        self.assertEqual(result.trace, ["UppercaseBlock"])

    def test_run_wraps_exceptions_in_block_execution_error(self):
        block = AlwaysFailsBlock()
        with self.assertRaises(BlockExecutionError) as ctx:
            block.run(ModelArgument(payload="x"))
        self.assertEqual(ctx.exception.block_name, "AlwaysFailsBlock")
        self.assertIsInstance(ctx.exception.original, ValueError)


class TransflowSequentialTests(unittest.TestCase):
    def test_sequential_pipes_result_through_each_block(self):
        flow = Transflow()
        blocks = [UppercaseBlock(), AppendSuffixBlock("!")]
        result = flow.run_sequential(blocks, ModelArgument(payload="hi"))
        self.assertEqual(result.payload, "HI!")
        self.assertEqual(result.trace, ["UppercaseBlock", "AppendSuffixBlock"])

    def test_abort_policy_propagates_error(self):
        flow = Transflow(failure_policy=FailurePolicy.ABORT)
        blocks = [UppercaseBlock(), AlwaysFailsBlock(), AppendSuffixBlock("!")]
        with self.assertRaises(BlockExecutionError):
            flow.run_sequential(blocks, ModelArgument(payload="hi"))

    def test_skip_policy_passes_through_unchanged(self):
        flow = Transflow(failure_policy=FailurePolicy.SKIP)
        blocks = [UppercaseBlock(), AlwaysFailsBlock(), AppendSuffixBlock("!")]
        result = flow.run_sequential(blocks, ModelArgument(payload="hi"))
        # AlwaysFailsBlock is skipped, so AppendSuffixBlock still runs on HI
        self.assertEqual(result.payload, "HI!")

    def test_fallback_policy_records_error_and_continues(self):
        flow = Transflow(failure_policy=FailurePolicy.FALLBACK)
        blocks = [UppercaseBlock(), AlwaysFailsBlock()]
        result = flow.run_sequential(blocks, ModelArgument(payload="hi"))
        self.assertEqual(result.payload, "HI")
        self.assertTrue(result.has_errors())
        self.assertIn("AlwaysFailsBlock", result.errors[0])


class TransflowConditionalTests(unittest.TestCase):
    def test_routes_to_true_branch(self):
        flow = Transflow()
        result = flow.run_conditional(
            condition=lambda a: len(str(a.payload)) > 2,
            if_true=UppercaseBlock(),
            if_false=AppendSuffixBlock("?"),
            arg=ModelArgument(payload="hello"),
        )
        self.assertEqual(result.payload, "HELLO")

    def test_routes_to_false_branch(self):
        flow = Transflow()
        result = flow.run_conditional(
            condition=lambda a: len(str(a.payload)) > 2,
            if_true=UppercaseBlock(),
            if_false=AppendSuffixBlock("?"),
            arg=ModelArgument(payload="hi"),
        )
        self.assertEqual(result.payload, "hi?")

    def test_none_false_branch_passes_through(self):
        flow = Transflow()
        result = flow.run_conditional(
            condition=lambda a: False,
            if_true=UppercaseBlock(),
            if_false=None,
            arg=ModelArgument(payload="hi"),
        )
        self.assertEqual(result.payload, "hi")


class TransflowParallelTests(unittest.TestCase):
    def test_fan_out_fan_in_default_merge(self):
        flow = Transflow(max_workers=4)
        blocks = [UppercaseBlock(), AppendSuffixBlock("!")]
        result = flow.run_parallel(blocks, ModelArgument(payload="hi"))
        self.assertEqual(result.payload, ["HI", "hi!"])

    def test_fan_out_branches_do_not_leak_mutations(self):
        # Regression guard for ModelArgument.clone(): each branch mutates
        # its own payload; branches must not observe each other's writes.
        flow = Transflow(max_workers=4)
        blocks = [UppercaseBlock(), UppercaseBlock(), UppercaseBlock()]
        result = flow.run_parallel(blocks, ModelArgument(payload="same"))
        self.assertEqual(result.payload, ["SAME", "SAME", "SAME"])

    def test_custom_merge_function(self):
        flow = Transflow(max_workers=4)
        blocks = [UppercaseBlock(), AppendSuffixBlock("!")]
        result = flow.run_parallel(
            blocks,
            ModelArgument(payload="hi"),
            merge=lambda results: ModelArgument(payload="|".join(str(r.payload) for r in results)),
        )
        self.assertEqual(result.payload, "HI|hi!")

    def test_run_parallel_rejects_empty_block_list(self):
        flow = Transflow()
        with self.assertRaises(Exception):
            flow.run_parallel([], ModelArgument(payload="x"))


class TransflowStreamBackpressureTests(unittest.TestCase):
    def test_max_in_flight_bounds_concurrency(self):
        counter = ConcurrencyCounter()
        flow = Transflow(max_workers=8)
        blocks = [SlowBlock(delay=0.05, counter=counter)]
        args = [ModelArgument(payload=i) for i in range(6)]

        results = list(flow.stream(blocks, args, max_in_flight=2, block_on_full=True))

        self.assertEqual(len(results), 6)
        self.assertLessEqual(counter.peak, 2)

    def test_non_blocking_raises_backpressure_when_window_full(self):
        counter = ConcurrencyCounter()
        flow = Transflow(max_workers=4)
        blocks = [SlowBlock(delay=0.3, counter=counter)]
        args = [ModelArgument(payload=i) for i in range(4)]

        with self.assertRaises(BackpressureError):
            list(flow.stream(blocks, args, max_in_flight=2, block_on_full=False))


class ModelArgumentTests(unittest.TestCase):
    def test_clone_deep_copies_mutable_payload(self):
        original = ModelArgument(payload={"key": [1, 2, 3]})
        clone = original.clone()
        clone.payload["key"].append(4)
        self.assertEqual(original.payload["key"], [1, 2, 3])
        self.assertEqual(clone.payload["key"], [1, 2, 3, 4])

    def test_clone_resets_trace_and_errors(self):
        original = ModelArgument(payload="x")
        original.record("stage-a")
        original.fail("stage-b", ValueError("boom"))
        clone = original.clone()
        self.assertEqual(clone.trace, [])
        self.assertEqual(clone.errors, [])
        clone.record("stage-c")
        self.assertNotIn("stage-c", original.trace)

    def test_parallel_merge_does_not_duplicate_pre_fork_trace(self):
        flow = Transflow(max_workers=4)
        arg = ModelArgument(payload="hi")
        arg.record("pre-fork-stage")
        result = flow.run_parallel([UppercaseBlock(), UppercaseBlock()], arg)
        self.assertEqual(result.trace.count("pre-fork-stage"), 1)


if __name__ == "__main__":
    unittest.main()
