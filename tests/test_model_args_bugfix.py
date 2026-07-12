"""Regression test for the ModelArgs @dataclass bugfix in model.py.

model.py as a whole cannot be imported in this environment (it requires
torch, which is not installed here), but ModelArgs itself has no torch
dependency. This test extracts the exact ModelArgs source from model.py
and execs it in an isolated namespace, so it exercises the real,
current file content rather than a hand-copied reimplementation that
could drift out of sync with it.

Before the fix (no @dataclass decorator), this class was broken three
ways: field(default_factory=...) attributes stayed raw
dataclasses.Field sentinel objects, __post_init__ was never invoked, and
derive_inference_args()'s ModelArgs(**self.__dict__) had nothing to
unpack because self.__dict__ was empty (no generated __init__).
"""

import dataclasses
import pathlib
import unittest
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

MODEL_PY = pathlib.Path(__file__).resolve().parent.parent / "model.py"


def _load_model_args_class():
    source_lines = MODEL_PY.read_text().splitlines()
    start = next(i for i, line in enumerate(source_lines) if line.startswith("class ModelArgs:") or line.startswith("class ModelArgs("))
    # ModelArgs is immediately followed by "class Transformer(nn.Module):" at top level.
    end = next(i for i, line in enumerate(source_lines) if i > start and line.startswith("class Transformer("))
    # Include the @dataclass decorator line(s) directly above the class.
    decorator_start = start
    while decorator_start > 0 and source_lines[decorator_start - 1].strip().startswith(("@", "#")):
        decorator_start -= 1
    snippet = "\n".join(source_lines[decorator_start:end])

    namespace: Dict[str, Any] = {
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "Any": Any,
        "Callable": Callable,
        "Dict": Dict,
        "List": List,
        "Literal": Literal,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
    }
    exec(compile(snippet, str(MODEL_PY), "exec"), namespace)
    return namespace["ModelArgs"]


class ModelArgsBugfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ModelArgs = _load_model_args_class()

    def test_is_a_real_dataclass(self):
        self.assertTrue(dataclasses.is_dataclass(self.ModelArgs))

    def test_default_factory_fields_are_resolved_not_field_sentinels(self):
        args = self.ModelArgs()
        self.assertIsInstance(args.q_skip_layer_names, list)
        self.assertNotIsInstance(args.q_skip_layer_names, dataclasses.Field)
        self.assertEqual(args.q_skip_layer_names, ["embed", "head"])
        self.assertIsInstance(args.compress_ratios, tuple)
        self.assertIsInstance(args.manhwa_supported_langs, list)

    def test_post_init_validation_actually_runs(self):
        # __post_init__ raises ValueError when n_heads*head_dim != dim;
        # this only fires at all if @dataclass wired up the generated
        # __init__ to call it.
        with self.assertRaises(ValueError):
            self.ModelArgs(n_heads=7, head_dim=128, dim=999999)

    def test_derive_inference_args_round_trips_via_dict(self):
        args = self.ModelArgs()
        inference_args = args.derive_inference_args()
        self.assertEqual(inference_args.training_phase, "inference")
        self.assertEqual(inference_args.q_mode, "zero_mass")
        self.assertFalse(inference_args.save_base_weights)
        # Confirms self.__dict__ was actually populated by the generated __init__.
        self.assertEqual(inference_args.q_seed, args.q_seed)


if __name__ == "__main__":
    unittest.main()
