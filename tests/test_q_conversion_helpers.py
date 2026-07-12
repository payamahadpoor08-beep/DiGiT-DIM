"""Regression test for _resolve_q_init, part of this session's
TaskEmbedding/convert_to_* q-conversion subsystem added to model.py.

The convert_to_* functions themselves are nn.Module-based (need torch,
not installed in this sandbox) and are correctness-reviewed by reading
UnifiedSMMQP's real constructor rather than executed here -- see
ARCHITECTURE.md. _resolve_q_init is the one piece of that subsystem with
no torch dependency, so it's extracted and exec'd the same way the
ModelArgs bugfix test does, and is genuinely runnable.
"""

import pathlib
import unittest

MODEL_PY = pathlib.Path(__file__).resolve().parent.parent / "model.py"


def _load_resolve_q_init():
    source_lines = MODEL_PY.read_text().splitlines()
    start = next(i for i, line in enumerate(source_lines) if line.startswith("def _resolve_q_init("))
    end = next(i for i, line in enumerate(source_lines) if i > start and line.startswith("_Q_MODE_TO_SMMQP_MODE"))
    snippet = "\n".join(source_lines[start:end])
    namespace = {}
    exec(compile(snippet, str(MODEL_PY), "exec"), namespace)
    return namespace["_resolve_q_init"]


class ResolveQInitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve_q_init = staticmethod(_load_resolve_q_init())

    def test_parses_family_and_scale(self):
        family, scale = self.resolve_q_init("normal_1.0")
        self.assertEqual(family, "normal")
        self.assertEqual(scale, 1.0)

    def test_parses_multi_underscore_family(self):
        family, scale = self.resolve_q_init("truncated_normal_0.02")
        self.assertEqual(family, "truncated_normal")
        self.assertEqual(scale, 0.02)

    def test_falls_back_when_no_underscore(self):
        family, scale = self.resolve_q_init("normal")
        self.assertEqual(family, "normal")
        self.assertEqual(scale, 1.0)

    def test_falls_back_when_suffix_not_numeric(self):
        family, scale = self.resolve_q_init("normal_wide")
        self.assertEqual(family, "normal_wide")
        self.assertEqual(scale, 1.0)


if __name__ == "__main__":
    unittest.main()
