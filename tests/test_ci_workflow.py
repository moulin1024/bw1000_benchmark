from __future__ import annotations

import unittest
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "numerical-ci.yml"
)


class NumericalCIWorkflowTests(unittest.TestCase):
    def test_push_and_pull_requests_run_covering_matrix(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("push:", workflow)
        self.assertIn("pull_request:", workflow)
        self.assertIn("spectral-fd-regression --quick", workflow)
        self.assertIn("numerical-smoke.json", workflow)

    def test_full_matrix_is_manual_and_four_way_sharded(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("full_matrix:", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertIn("shard: [0, 1, 2, 3]", workflow)
        self.assertIn("--shard-count 4", workflow)
        self.assertIn("actions/upload-artifact@v7", workflow)


if __name__ == "__main__":
    unittest.main()
