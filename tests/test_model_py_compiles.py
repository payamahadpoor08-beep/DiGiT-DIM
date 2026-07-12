"""Regression guard: model.py must compile.

This sounds trivial but wasn't: the file used to contain a second
`from __future__ import annotations` deep in the middle (a copy-paste
artifact from merging two source files), which is a hard SyntaxError
anywhere but the very first statement of a file. That meant the entire
87k-line file failed to even compile, let alone import -- worse than any
individual NameError, since it blocked all of them. This test would have
caught it immediately.
"""

import pathlib
import unittest

MODEL_PY = pathlib.Path(__file__).resolve().parent.parent / "model.py"


class ModelPyCompilesTests(unittest.TestCase):
    def test_compiles_without_syntax_error(self):
        source = MODEL_PY.read_text()
        compile(source, str(MODEL_PY), "exec")

    def test_future_import_is_the_only_one_and_is_first(self):
        lines = MODEL_PY.read_text().splitlines()
        future_import_lines = [i for i, line in enumerate(lines) if line.startswith("from __future__ import")]
        self.assertEqual(
            len(future_import_lines), 1,
            f"expected exactly one 'from __future__ import' statement, found at lines {future_import_lines}",
        )


if __name__ == "__main__":
    unittest.main()
