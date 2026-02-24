#!/usr/bin/env python3
"""Smoke test for mission benchmark scoring."""

from __future__ import annotations

import socket
import unittest
from pathlib import Path

from tests.swarm_sim.mission_benchmark import MissionBenchmarkRunner, default_mission_cases


class MissionBenchmarkSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_single_case_benchmark_scores(self) -> None:
        if not self._can_bind_local_port():
            self.skipTest("local port binding is unavailable in current sandbox")

        cases = default_mission_cases()
        case = next(item for item in cases if item.case_id == "code-quality-parallel")
        runner = MissionBenchmarkRunner(
            repo_root=self.repo_root,
            workers=2,
            poll_interval=1.0,
            mock_task_delay=0.15,
            task_timeout=90,
            heartbeat_timeout=90,
            prefer_codex_cli=False,
        )
        benchmark_result = runner.run_case(case)
        self.assertEqual(benchmark_result.status, "ok")
        self.assertGreater(benchmark_result.total_score, 0.0)
        self.assertIsNotNone(benchmark_result.process)
        self.assertIsNotNone(benchmark_result.result)
        self.assertIsNotNone(benchmark_result.codex_judgement)
        assert benchmark_result.process is not None
        assert benchmark_result.result is not None
        self.assertEqual(benchmark_result.process.task_done, len(case.tasks))
        self.assertGreaterEqual(benchmark_result.result.completeness_ratio, 0.99)
        assert benchmark_result.codex_judgement is not None
        self.assertGreater(benchmark_result.codex_judgement.overall.score, 0.0)
        self.assertIsInstance(benchmark_result.bug_findings, list)

    @staticmethod
    def _can_bind_local_port() -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return True
        except OSError:
            return False
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
