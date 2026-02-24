# Swarm Simulation Test Framework

这个目录提供一个可复用的测试框架，用于模拟 opencode 中 `/swarm <mission>` 的 mission 执行流程。

## 目标

- 覆盖 `/swarm <mission>` 的主链路：
  1. 启动 orchestrator
  2. 默认 `panel on`（在测试中 mock）
  3. 生成并写入 seeded central plan
  4. worker 调度执行直到任务 DONE
- `opencode` 侧能力通过 mock 实现（不依赖真实 opencode 服务）

## 主要文件

- `mock_opencode_server.py`
  - 模拟 `opencode serve`，实现 orchestrator 依赖的 HTTP API。
- `harness.py`
  - `SwarmMissionHarness`：提供 `/swarm <mission>` mission 分支模拟执行器。
- `test_swarm_mission_flow.py`
  - `unittest` 示例，验证完整流程与产物。
- `mission_benchmark.py`
  - 多 mission 基准评测：过程评分（调度/恢复/效率）+ 结果评分（完成度/结构/语义相关性）+ Codex 日志审查（多维评语 + bug 发现）。
- `test_mission_benchmark.py`
  - benchmark smoke test。

## 运行

在仓库根目录执行：

```bash
python3 -m unittest tests.swarm_sim.test_swarm_mission_flow -v
```

## 多 Mission 优劣评测

运行默认 mission 集并输出评分报告：

```bash
python3 -m tests.swarm_sim.mission_benchmark
```

默认会尝试调用本地 `codex exec` 做日志评审（过程/结果/bug 多维打分）。

Codex 评测会自动写入 session 日志目录，便于回溯：

- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_prompt_*.md`
- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_schema_*.json`
- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_stdout_*.log`
- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_stderr_*.log`
- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_meta_*.json`
- `.blackboard/sessions/<session_id>/logs/judges/codex_judge_payload_*.json`（成功时）

内部调用形态（简化）：

```bash
codex exec \
  --skip-git-repo-check \
  --sandbox read-only \
  --ephemeral \
  -C <project_dir> \
  --add-dir <session_dir> \
  --output-schema <schema.json> \
  --output-last-message <judge_output.json> \
  -
```

导出 JSON + Markdown：

```bash
python3 -m tests.swarm_sim.mission_benchmark \
  --json-out /tmp/swarm-benchmark.json \
  --md-out /tmp/swarm-benchmark.md
```

只保留规则评分，关闭 Codex 日志审查：

```bash
python3 -m tests.swarm_sim.mission_benchmark --no-codex-judge
```

启用 Codex 审查但不调用外部 CLI（仅内部规则判审）：

```bash
python3 -m tests.swarm_sim.mission_benchmark --no-codex-cli
```
