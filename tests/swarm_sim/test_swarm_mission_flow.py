#!/usr/bin/env python3
"""Integration-style tests for simulated `/swarm <mission>` flow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.swarm_sim.harness import SwarmMissionHarness


class SwarmMissionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

    def _sample_tasks(self) -> list[dict]:
        return [
            {
                "id": "task-001",
                "type": "standard",
                "title": "分析插件入口日志注入",
                "description": "阅读 extension/plugins/opencode-agent-team/index.js，输出关键执行链路说明。",
                "dependencies": [],
                "status": "PENDING",
                "assignees": [],
            },
            {
                "id": "task-002",
                "type": "standard",
                "title": "汇总 `/swarm` 调度行为",
                "description": "结合 orchestrator 行为生成总结。",
                "dependencies": ["task-001"],
                "status": "PENDING",
                "assignees": [],
            },
        ]

    def test_swarm_mission_runs_to_done_with_mock_opencode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="swarm-sim-project-") as tmpdir:
            project_dir = Path(tmpdir)
            with SwarmMissionHarness(
                repo_root=self.repo_root,
                project_dir=project_dir,
                workers=2,
                poll_interval=1.0,
                mock_task_delay=0.15,
            ) as harness:
                result = harness.run_swarm_mission(
                    command_text="/swarm 模拟 opencode 中的 swarm mission 执行",
                    tasks=self._sample_tasks(),
                    timeout=60.0,
                )

                status_by_task = {task["id"]: task.get("status") for task in result.tasks}
                self.assertEqual(status_by_task.get("task-001"), "DONE")
                self.assertEqual(status_by_task.get("task-002"), "DONE")

                session_resources = project_dir / ".blackboard" / "sessions" / result.session_id / "resources"
                self.assertTrue((session_resources / "task-001.md").exists())
                self.assertTrue((session_resources / "task-002.md").exists())

                self.assertIn("swarm.command.received", result.events)
                self.assertIn("panel.on.mock", result.events)
                self.assertIn("plan.seed.write", result.events)
                self.assertTrue(any(item.startswith("orchestrator.exit:0") for item in result.events))

    def test_swarm_mission_requires_mission_argument(self) -> None:
        with tempfile.TemporaryDirectory(prefix="swarm-sim-project-") as tmpdir:
            project_dir = Path(tmpdir)
            with SwarmMissionHarness(repo_root=self.repo_root, project_dir=project_dir) as harness:
                with self.assertRaises(ValueError):
                    harness.run_swarm_mission(command_text="/swarm", tasks=self._sample_tasks(), timeout=10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

