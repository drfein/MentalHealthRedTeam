from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from src import run


class RunnerIntegrationTests(unittest.TestCase):
    def test_all_subcommands_support_dry_run(self) -> None:
        commands = [
            ["harm", "seed-preferences"],
            ["harm", "generate-conversations"],
            ["harm", "compute-drift"],
            ["harm", "plot"],
            ["harm", "analyze-jumps"],
            ["harm", "plot-crossover"],
            ["dpo", "train-hh-token"],
            ["fairmt", "generate"],
            ["fairmt", "judge-heuristic"],
            ["fairmt", "judge-pairwise-llm"],
            ["token", "expound-hh-token-delta"],
        ]
        for command in commands:
            with self.subTest(command=command):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = run.main([*command, "--dry-run"])
                self.assertEqual(code, 0)
                self.assertIn("[dry-run]", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
