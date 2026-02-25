#!/usr/bin/env python3
"""Mission benchmark for swarm simulation.

This module evaluates mission runs from two perspectives:
1. Process quality: dispatch stability, recovery/incident behavior, efficiency.
2. Result quality: completion ratio, report completeness, report structure, semantic relevance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.swarm_sim.harness import SwarmMissionHarness, SwarmRunResult


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _keyword_candidates(text: str, limit: int = 10) -> list[str]:
    import re

    if not text:
        return []
    tokens = re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", text)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _load_json_payload(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty JSON payload.")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    blocks = re.findall(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", raw, flags=re.MULTILINE)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except Exception:
            continue
        if isinstance(data, dict):
            return data

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except Exception as exc:
            raise ValueError(f"Cannot parse JSON payload: {exc}") from exc
        if isinstance(data, dict):
            return data

    raise ValueError("Cannot find JSON object in codex response.")


def _critical_path_len(tasks: list[dict[str, Any]]) -> int:
    nodes: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        deps = task.get("dependencies", [])
        dep_list = [str(dep).strip() for dep in deps] if isinstance(deps, list) else []
        nodes[task_id] = [dep for dep in dep_list if dep]

    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        if task_id in visiting:
            return 1
        visiting.add(task_id)
        deps = nodes.get(task_id, [])
        if not deps:
            result = 1
        else:
            result = 1 + max(depth(dep) if dep in nodes else 1 for dep in deps)
        visiting.remove(task_id)
        memo[task_id] = result
        return result

    if not nodes:
        return 0
    return max(depth(task_id) for task_id in nodes)


@dataclass
class MissionCase:
    case_id: str
    mission: str
    tasks: list[dict[str, Any]]
    expected_keywords: list[str] = field(default_factory=list)
    timeout: float = 30.0


@dataclass
class ProcessMetrics:
    duration_seconds: float
    task_total: int
    task_done: int
    critical_path_len: int
    expected_floor_seconds: float
    incidents: int
    recoveries: int
    task_releases: int
    dispatch_errors: int
    timeouts: int
    empty_plan_events: int
    scheduler_claims: int
    scheduler_dispatches: int
    completion_ratio: float
    efficiency_ratio: float
    score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ResultMetrics:
    report_files: int
    expected_reports: int
    avg_report_chars: float
    avg_report_lines: float
    completeness_ratio: float
    structure_ratio: float
    keyword_hit_ratio: float
    richness_ratio: float
    score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class BugFinding:
    bug_id: str
    severity: str
    category: str
    evidence: str
    impact: str
    suggestion: str
    count: int = 1


@dataclass
class DimensionJudgement:
    score: float
    evidence: list[str] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    suggestion: list[str] = field(default_factory=list)


@dataclass
class CodexJudgement:
    execution_stability: DimensionJudgement
    scheduling_quality: DimensionJudgement
    progress_efficiency: DimensionJudgement
    result_completeness: DimensionJudgement
    result_quality: DimensionJudgement
    collaboration_effectiveness: DimensionJudgement
    error_recovery_and_reflection: DimensionJudgement
    overall: DimensionJudgement


@dataclass
class MissionBenchmarkResult:
    case_id: str
    mission: str
    status: str
    verdict: str
    total_score: float
    process_score: float
    result_score: float
    duration_seconds: float
    process: ProcessMetrics | None
    result: ResultMetrics | None
    events: list[str]
    error: str = ""
    codex_judgement: CodexJudgement | None = None
    bug_findings: list[BugFinding] = field(default_factory=list)
    judge_source: str = "rules"
    codex_cli_invoked: bool = False
    codex_cli_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def default_mission_cases() -> list[MissionCase]:
    return [
        MissionCase(
            case_id="code-quality-parallel",
            mission="分析代码质量并给出改进建议",
            expected_keywords=["代码质量", "复杂度", "建议"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "复杂度扫描",
                    "description": "分析 orchestrator.py 的复杂度和关键风险点。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "日志质量检查",
                    "description": "检查日志结构一致性并提出改进建议。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "测试覆盖评估",
                    "description": "评估当前测试覆盖盲区并给出补充策略。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        MissionCase(
            case_id="auth-refactor-chain",
            mission="重构认证模块并确保回归稳定",
            expected_keywords=["认证", "重构", "回归", "稳定"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "认证现状梳理",
                    "description": "梳理现有认证流程和风险边界。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "重构方案设计",
                    "description": "设计认证重构步骤与回滚策略。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "回归验证矩阵",
                    "description": "定义认证相关回归测试矩阵。",
                    "dependencies": ["task-002"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        MissionCase(
            case_id="api-test-diamond",
            mission="提升 API 测试覆盖并聚焦关键路径",
            expected_keywords=["API", "测试", "覆盖", "关键路径"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "接口清单盘点",
                    "description": "盘点核心 API 及风险等级。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "单元测试策略",
                    "description": "制定 API 单元测试覆盖策略。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "集成测试策略",
                    "description": "制定 API 集成测试方案。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-004",
                    "type": "standard",
                    "title": "关键路径优先级",
                    "description": "综合单测与集成测试，输出关键路径优先级。",
                    "dependencies": ["task-002", "task-003"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        MissionCase(
            case_id="recovery-from-hallucination",
            mission="实现一个基于 Redis 的缓存模块，但项目中根本没有安装 Redis 依赖",
            expected_keywords=["Redis", "缓存", "失败", "恢复"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "尝试引入 Redis 客户端库",
                    "description": "编写一段 Python 代码尝试 import redis，并运行它以验证当前环境是否支持。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "缓存模块实现与异常处理",
                    "description": "基于 task-001 的结论，由于缺乏依赖，请不要强行成功，而是向黑板写入一个明确的失败原因或替代方案。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
            timeout=45.0,
        ),
        # Case 5: Wide fan-out star — 1 root → 4 parallel → 1 aggregator (6 tasks, 3 workers → forces queuing)
        MissionCase(
            case_id="perf-wide-fanout",
            mission="全面分析系统性能瓶颈并产出综合优化报告",
            expected_keywords=["性能", "瓶颈", "优化", "综合"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "性能基线采集",
                    "description": "采集系统各维度（CPU、内存、磁盘、网络）的基线性能数据，为后续并行分析提供输入。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "数据库查询瓶颈分析",
                    "description": "分析慢查询日志，找出 Top-5 高延迟 SQL 并给出索引优化建议。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "缓存命中率分析",
                    "description": "评估 Redis 缓存命中率，识别缓存穿透/击穿风险点。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-004",
                    "type": "standard",
                    "title": "网络 I/O 瓶颈分析",
                    "description": "分析服务间调用链路，识别高延迟 RPC 和带宽瓶颈。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-005",
                    "type": "standard",
                    "title": "计算密集型任务分析",
                    "description": "分析 CPU 高占用函数，找出热点算法并评估并发优化空间。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-006",
                    "type": "standard",
                    "title": "综合优化报告输出",
                    "description": "汇总 DB、缓存、网络、计算四个维度的分析结论，按优先级排列优化建议，产出综合性能优化报告。",
                    "dependencies": ["task-002", "task-003", "task-004", "task-005"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        # Case 6: Deep linear chain — 5-level strict sequence, tests cascading predecessor context
        MissionCase(
            case_id="security-deep-chain",
            mission="对系统进行全面安全审计并产出修复方案",
            expected_keywords=["安全", "漏洞", "审计", "修复"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "资产与攻击面盘点",
                    "description": "梳理系统所有对外暴露的接口、端口和服务，绘制完整攻击面地图。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "漏洞扫描与收集",
                    "description": "基于 task-001 的攻击面地图，执行漏洞扫描，收集所有已知 CVE 和配置错误。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "风险分级与优先级排序",
                    "description": "基于 task-002 的漏洞列表，按 CVSS 评分和业务影响进行风险分级，输出 P0/P1/P2 优先级矩阵。",
                    "dependencies": ["task-002"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-004",
                    "type": "standard",
                    "title": "修复方案设计",
                    "description": "基于 task-003 的优先级矩阵，为 P0 和 P1 漏洞设计具体修复步骤和回滚预案。",
                    "dependencies": ["task-003"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-005",
                    "type": "standard",
                    "title": "修复验证与关闭报告",
                    "description": "基于 task-004 的修复方案，定义验收测试矩阵，产出安全审计关闭报告和残余风险声明。",
                    "dependencies": ["task-004"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        # Case 7: Mid-chain failure — parallel prep → validation fails → recovery plan
        MissionCase(
            case_id="schema-migration-mid-failure",
            mission="将用户数据从旧格式迁移至新格式，但源数据存在严重的格式错误",
            expected_keywords=["迁移", "格式", "失败", "回退"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "源数据 Schema 分析",
                    "description": "读取源数据库 schema 定义，梳理字段类型、约束和历史变更记录。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "目标 Schema 定义",
                    "description": "定义迁移后的目标 schema，包括字段映射规则和数据转换逻辑。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "数据格式兼容性验证",
                    "description": "基于 task-001 和 task-002 的 schema，验证源数据是否满足目标格式要求。由于发现源数据存在严重格式错误，请不要强行成功，而是直接记录格式不一致的失败原因。",
                    "dependencies": ["task-001", "task-002"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-004",
                    "type": "standard",
                    "title": "迁移失败回退方案",
                    "description": "基于 task-003 的格式验证失败报告，制定完整的数据迁移回退策略和格式修复预案，确保生产数据安全。",
                    "dependencies": ["task-003"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
        # Case 8: Complex incident response — alert → 3 parallel investigations → root cause → fix plan (6 tasks)
        MissionCase(
            case_id="incident-response-complex",
            mission="线上告警：订单服务 P99 延迟从 200ms 飙升至 5s，请排查根因并给出修复方案",
            expected_keywords=["告警", "延迟", "根因", "修复"],
            tasks=[
                {
                    "id": "task-001",
                    "type": "standard",
                    "title": "告警上下文梳理",
                    "description": "收集告警触发时间窗口内的关键指标（QPS、错误率、Pod 重启次数），明确故障范围和影响面。",
                    "dependencies": [],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-002",
                    "type": "standard",
                    "title": "数据库层排查",
                    "description": "基于 task-001 的时间窗口，检查数据库慢查询日志、连接池状态和锁等待，判断 DB 是否为瓶颈。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-003",
                    "type": "standard",
                    "title": "服务依赖链路排查",
                    "description": "基于 task-001 的时间窗口，通过 trace 数据分析订单服务的上下游依赖，定位超时调用链路。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-004",
                    "type": "standard",
                    "title": "基础设施层排查",
                    "description": "基于 task-001 的时间窗口，检查节点 CPU/内存/网络压力、GC 暂停时间和容器资源限制。",
                    "dependencies": ["task-001"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-005",
                    "type": "standard",
                    "title": "根因确认与归因报告",
                    "description": "综合 task-002、task-003、task-004 的排查结论，确认主根因和贡献因子，产出结构化归因报告。",
                    "dependencies": ["task-002", "task-003", "task-004"],
                    "status": "PENDING",
                    "assignees": [],
                },
                {
                    "id": "task-006",
                    "type": "standard",
                    "title": "应急修复与预防措施",
                    "description": "基于 task-005 的根因报告，制定立即执行的应急修复步骤和中长期预防措施，包含回滚预案。",
                    "dependencies": ["task-005"],
                    "status": "PENDING",
                    "assignees": [],
                },
            ],
        ),
    ]


class MissionBenchmarkRunner:
    def __init__(
        self,
        repo_root: Path,
        *,
        workers: int = 3,
        poll_interval: float = 1.0,
        mock_task_delay: float = 0.15,
        task_timeout: int = 20,
        heartbeat_timeout: int = 20,
        work_root: Path | None = None,
        judge_by_codex: bool = True,
        prefer_codex_cli: bool = True,
        codex_cli_path: str = "codex",
        codex_cli_timeout: int = 60,
        codex_cli_model: str = "claude-sonnet-4-6",
        use_real_opencode: bool = False,
        real_case_timeout: float = 600.0,
        worker_model: str = "",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.workers = workers
        self.poll_interval = poll_interval
        self.mock_task_delay = mock_task_delay
        self.task_timeout = task_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.work_root = Path(work_root).resolve() if work_root else None
        self.judge_by_codex = judge_by_codex
        self.prefer_codex_cli = prefer_codex_cli
        self.codex_cli_path = codex_cli_path
        self.codex_cli_timeout = max(30, int(codex_cli_timeout))
        self.codex_cli_model = codex_cli_model.strip() or "claude-sonnet-4-6"
        self.use_real_opencode = use_real_opencode
        self.real_case_timeout = float(real_case_timeout)
        self.worker_model = worker_model.strip()
        self._run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def run_case(self, case: MissionCase) -> MissionBenchmarkResult:
        project_root = self._prepare_project_root(case.case_id)
        self._log(f"  project_root: {project_root}")
        try:
            with SwarmMissionHarness(
                repo_root=self.repo_root,
                project_dir=project_root,
                workers=self.workers,
                poll_interval=self.poll_interval,
                mock_task_delay=self.mock_task_delay,
                task_timeout=self.task_timeout,
                heartbeat_timeout=self.heartbeat_timeout,
                use_real_opencode=self.use_real_opencode,
                worker_model=self.worker_model,
            ) as harness:
                mode_label = "real opencode" if self.use_real_opencode else "mock servers"
                self._log(f"  Starting orchestrator & {mode_label}...")
                effective_timeout = self.real_case_timeout if self.use_real_opencode else case.timeout
                run = harness.run_swarm_mission(
                    command_text=f"/swarm {case.mission}",
                    tasks=case.tasks,
                    timeout=effective_timeout,
                )
                self._log(f"  Mission completed in {run.duration_seconds:.1f}s, session={run.session_id}")
                self._log("  Scoring process metrics...")
                events = harness.read_orchestrator_events(run.session_id)
                process_metrics = self._score_process(case, run, events)
                self._log(f"  Process score: {process_metrics.score:.1f}")
                self._log("  Scoring result metrics...")
                result_metrics = self._score_result(case, run, harness)
                self._log(f"  Result score: {result_metrics.score:.1f}")
                self._log("  Detecting bugs...")
                bug_findings = self._detect_bug_findings(
                    case=case,
                    run=run,
                    events=events,
                    process_metrics=process_metrics,
                    result_metrics=result_metrics,
                )
                self._log(f"  Rule-based bugs found: {len(bug_findings)}")
                codex_judgement: CodexJudgement | None = None
                judge_source = "rules"
                codex_cli_invoked = False
                codex_cli_error = ""

                if self.judge_by_codex and self.prefer_codex_cli:
                    codex_cli_invoked = True
                    self._log(f"  Invoking LLM judge via: {self.codex_cli_path} (timeout={self.codex_cli_timeout}s)...")
                    try:
                        codex_judgement, cli_findings = self._judge_with_codex_cli(
                            case=case,
                            run=run,
                            events=events,
                            process_metrics=process_metrics,
                            result_metrics=result_metrics,
                            baseline_findings=bug_findings,
                        )
                        judge_source = "codex_cli"
                        bug_findings = self._merge_bug_findings(cli_findings, bug_findings)
                        self._log(f"  LLM judge done: overall={codex_judgement.overall.score:.1f}, bugs={len(cli_findings)}")
                    except Exception as cli_exc:
                        codex_cli_error = str(cli_exc)
                        self._log(f"  LLM judge failed: {codex_cli_error.splitlines()[0][:120]}")
                        bug_findings.append(
                            BugFinding(
                                bug_id="BUG-CODEX-CLI-JUDGE-FAILED",
                                severity="low",
                                category="judge",
                                evidence=codex_cli_error.splitlines()[0][:280],
                                impact="Fell back to rule-based judgement; LLM qualitative review unavailable for this run.",
                                suggestion="Check codex CLI login/network/permissions and re-run benchmark.",
                                count=1,
                            )
                        )

                if codex_judgement is None and self.judge_by_codex:
                    codex_judgement = self._build_codex_judgement(
                        case=case,
                        run=run,
                        events=events,
                        process_metrics=process_metrics,
                        result_metrics=result_metrics,
                        bug_findings=bug_findings,
                    )

                total_score = _clamp(process_metrics.score * 0.55 + result_metrics.score * 0.45)
                process_score = round(process_metrics.score, 2)
                result_score = round(result_metrics.score, 2)
                if codex_judgement is not None:
                    process_score = round(
                        (
                            codex_judgement.execution_stability.score
                            + codex_judgement.scheduling_quality.score
                            + codex_judgement.progress_efficiency.score
                            + codex_judgement.error_recovery_and_reflection.score
                        )
                        / 4.0,
                        2,
                    )
                    result_score = round(
                        (
                            codex_judgement.result_completeness.score
                            + codex_judgement.result_quality.score
                            + codex_judgement.collaboration_effectiveness.score
                        )
                        / 3.0,
                        2,
                    )
                    total_score = _clamp(codex_judgement.overall.score)

                verdict = self._verdict(total_score)
                return MissionBenchmarkResult(
                    case_id=case.case_id,
                    mission=case.mission,
                    status="ok",
                    verdict=verdict,
                    total_score=round(total_score, 2),
                    process_score=process_score,
                    result_score=result_score,
                    duration_seconds=round(run.duration_seconds, 3),
                    process=process_metrics,
                    result=result_metrics,
                    events=run.events,
                    codex_judgement=codex_judgement,
                    bug_findings=bug_findings,
                    judge_source=judge_source,
                    codex_cli_invoked=codex_cli_invoked,
                    codex_cli_error=codex_cli_error,
                )
        except Exception as exc:
            err_text = str(exc)
            startup_bug = BugFinding(
                bug_id="BUG-STARTUP-PORT-UNAVAILABLE",
                severity="critical",
                category="startup",
                evidence=err_text.splitlines()[0][:300],
                impact="Orchestrator cannot allocate worker ports, mission execution aborts at startup.",
                suggestion="Run benchmark with local port binding enabled and verify no stale processes occupy port range.",
                count=1,
            )
            failure_judgement = (
                CodexJudgement(
                    execution_stability=DimensionJudgement(
                        score=0.0,
                        evidence=["startup failed before scheduler loop"],
                        risk=[startup_bug.bug_id],
                        suggestion=[startup_bug.suggestion],
                    ),
                    scheduling_quality=DimensionJudgement(
                        score=0.0,
                        evidence=["no scheduler tick available"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Fix startup failure before evaluating scheduling behavior."],
                    ),
                    progress_efficiency=DimensionJudgement(
                        score=0.0,
                        evidence=["duration unavailable due startup abort"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Re-run after startup fix to measure efficiency."],
                    ),
                    result_completeness=DimensionJudgement(
                        score=0.0,
                        evidence=["no tasks executed"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Ensure orchestrator can start workers and process tasks."],
                    ),
                    result_quality=DimensionJudgement(
                        score=0.0,
                        evidence=["no output artifacts generated"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Restore execution path to generate report artifacts."],
                    ),
                    collaboration_effectiveness=DimensionJudgement(
                        score=0.0,
                        evidence=["no tasks executed"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Ensure orchestrator can start workers minimum."],
                    ),
                    error_recovery_and_reflection=DimensionJudgement(
                        score=0.0,
                        evidence=["startup failed, no recovery logic triggered"],
                        risk=[startup_bug.bug_id],
                        suggestion=["Fix framework startup issue before benchmarking."],
                    ),
                    overall=DimensionJudgement(
                        score=0.0,
                        evidence=["benchmark case failed at startup"],
                        risk=[startup_bug.bug_id],
                        suggestion=[startup_bug.suggestion],
                    ),
                )
                if self.judge_by_codex
                else None
            )
            return MissionBenchmarkResult(
                case_id=case.case_id,
                mission=case.mission,
                status="error",
                verdict="失败",
                total_score=0.0,
                process_score=0.0,
                result_score=0.0,
                duration_seconds=0.0,
                process=None,
                result=None,
                events=[],
                error=err_text,
                codex_judgement=failure_judgement,
                bug_findings=[startup_bug],
                judge_source="startup_failure",
                codex_cli_invoked=False,
                codex_cli_error="",
            )

    def run_cases(self, cases: list[MissionCase]) -> list[MissionBenchmarkResult]:
        total = len(cases)
        self._log(f"Benchmark started: {total} case(s), run_stamp={self._run_stamp}")
        results: list[MissionBenchmarkResult] = []
        for idx, case in enumerate(cases, 1):
            self._log(f"[{idx}/{total}] Running case: {case.case_id}")
            result = self.run_case(case)
            results.append(result)
            self._log(
                f"[{idx}/{total}] Done: {case.case_id} | "
                f"status={result.status} score={result.total_score:.1f} "
                f"verdict={result.verdict} duration={result.duration_seconds:.1f}s"
            )
        self._log(f"Benchmark finished: {total} case(s)")
        return results

    @staticmethod
    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[bench {ts}] {msg}", file=sys.stderr, flush=True)

    def _prepare_project_root(self, case_id: str) -> Path:
        if self.work_root:
            base = self.work_root
        else:
            base = self.repo_root / ".benchmark_runs" / self._run_stamp
        target = base / case_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _score_process(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        events: list[dict[str, Any]],
    ) -> ProcessMetrics:
        event_names = [str(item.get("event", "")) for item in events]

        incidents = event_names.count("incident.generated")
        recoveries = event_names.count("worker.recover.start")
        releases = event_names.count("task.released")
        dispatch_errors = event_names.count("task.dispatched.error")
        timeouts = event_names.count("task.timeout")
        empty_plan = event_names.count("plan.empty")
        scheduler_claims = event_names.count("scheduler.assign.claimed")
        scheduler_dispatches = event_names.count("scheduler.assign.dispatched")

        task_total = len(case.tasks)
        task_done = sum(1 for task in run.tasks if str(task.get("status")) == "DONE")
        completion_ratio = _ratio(task_done, task_total)

        critical_path_len = _critical_path_len(case.tasks)
        expected_floor = (
            5.0
            + max(1, critical_path_len) * max(0.1, self.mock_task_delay)
            + max(0, critical_path_len - 1) * max(0.5, self.poll_interval) * 0.5
        )
        efficiency_ratio = min(1.0, _ratio(expected_floor, max(run.duration_seconds, 1e-6)))

        penalty = 0.0
        notes: list[str] = []

        if incidents:
            delta = incidents * 22.0
            penalty += delta
            notes.append(f"incident.generated={incidents} (-{delta:.1f})")
        if dispatch_errors:
            delta = dispatch_errors * 12.0
            penalty += delta
            notes.append(f"task.dispatched.error={dispatch_errors} (-{delta:.1f})")
        if timeouts:
            delta = timeouts * 10.0
            penalty += delta
            notes.append(f"task.timeout={timeouts} (-{delta:.1f})")
        if recoveries:
            delta = recoveries * 7.0
            penalty += delta
            notes.append(f"worker.recover.start={recoveries} (-{delta:.1f})")
        if releases:
            delta = releases * 3.0
            penalty += delta
            notes.append(f"task.released={releases} (-{delta:.1f})")
        if empty_plan:
            delta = empty_plan * 5.0
            penalty += delta
            notes.append(f"plan.empty={empty_plan} (-{delta:.1f})")

        delay_over = max(0.0, run.duration_seconds - expected_floor)
        if delay_over > 0:
            delta = min(30.0, delay_over * 2.0)
            penalty += delta
            notes.append(f"duration over floor +{delay_over:.2f}s (-{delta:.1f})")

        if completion_ratio < 1.0:
            delta = (1.0 - completion_ratio) * 45.0
            penalty += delta
            notes.append(f"completion_ratio={completion_ratio:.2f} (-{delta:.1f})")

        score = _clamp(100.0 - penalty)
        if not notes:
            notes.append("No stability penalties observed.")

        return ProcessMetrics(
            duration_seconds=round(run.duration_seconds, 3),
            task_total=task_total,
            task_done=task_done,
            critical_path_len=critical_path_len,
            expected_floor_seconds=round(expected_floor, 3),
            incidents=incidents,
            recoveries=recoveries,
            task_releases=releases,
            dispatch_errors=dispatch_errors,
            timeouts=timeouts,
            empty_plan_events=empty_plan,
            scheduler_claims=scheduler_claims,
            scheduler_dispatches=scheduler_dispatches,
            completion_ratio=round(completion_ratio, 4),
            efficiency_ratio=round(efficiency_ratio, 4),
            score=round(score, 2),
            notes=notes,
        )

    def _score_result(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        harness: SwarmMissionHarness,
    ) -> ResultMetrics:
        reports = harness.list_resource_files(run.session_id, "*.md")
        report_texts = [path.read_text(encoding="utf-8", errors="replace") for path in reports]
        expected_reports = len(case.tasks)

        expected_by_task: dict[str, Path] = {}
        resource_root = harness.resource_dir(run.session_id)
        for task in run.tasks:
            task_id = str(task.get("id") or "").strip()
            if not task_id:
                continue
            artifact_link = str(task.get("artifact_link") or "").strip()
            if artifact_link:
                candidate = Path(artifact_link)
                if not candidate.is_absolute():
                    candidate = (resource_root / candidate).resolve()
                expected_by_task[task_id] = candidate
                continue
            fallback_name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in task_id).strip("-") or task_id
            expected_by_task[task_id] = (resource_root / f"{fallback_name}.md").resolve()

        complete_hits = 0
        for task in run.tasks:
            task_id = str(task.get("id") or "").strip()
            status_done = str(task.get("status") or "") == "DONE"
            path = expected_by_task.get(task_id)
            exists = bool(path and path.exists() and path.stat().st_size > 0)
            if status_done and exists:
                complete_hits += 1
        completeness_ratio = _ratio(complete_hits, max(1, expected_reports))

        total_chars = sum(len(text) for text in report_texts)
        avg_chars = _ratio(total_chars, max(1, len(report_texts)))
        avg_lines = _ratio(
            sum(len([line for line in text.splitlines() if line.strip()]) for text in report_texts),
            max(1, len(report_texts)),
        )
        richness_ratio = min(1.0, avg_chars / 550.0)

        sections = ["## Task Context", "## Execution", "## Verification", "## Artifacts", "## Summary"]
        structure_scores: list[float] = []
        for text in report_texts:
            hits = sum(1 for section in sections if section in text)
            structure_scores.append(_ratio(hits, len(sections)))
        structure_ratio = _ratio(sum(structure_scores), max(1, len(structure_scores)))

        keywords = case.expected_keywords or _keyword_candidates(
            " ".join([case.mission] + [str(task.get("title") or "") for task in case.tasks]),
            limit=10,
        )
        combined_text = "\n".join(report_texts).lower()
        keyword_hits = sum(1 for keyword in keywords if keyword.lower() in combined_text)
        keyword_hit_ratio = _ratio(keyword_hits, max(1, len(keywords)))

        score = _clamp(
            completeness_ratio * 45.0
            + structure_ratio * 20.0
            + keyword_hit_ratio * 20.0
            + richness_ratio * 15.0
        )
        notes = [
            f"completeness={completeness_ratio:.2f}",
            f"structure={structure_ratio:.2f}",
            f"keyword_hit={keyword_hit_ratio:.2f}",
            f"richness={richness_ratio:.2f}",
        ]

        return ResultMetrics(
            report_files=len(reports),
            expected_reports=expected_reports,
            avg_report_chars=round(avg_chars, 2),
            avg_report_lines=round(avg_lines, 2),
            completeness_ratio=round(completeness_ratio, 4),
            structure_ratio=round(structure_ratio, 4),
            keyword_hit_ratio=round(keyword_hit_ratio, 4),
            richness_ratio=round(richness_ratio, 4),
            score=round(score, 2),
            notes=notes,
        )

    def _detect_bug_findings(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        events: list[dict[str, Any]],
        process_metrics: ProcessMetrics,
        result_metrics: ResultMetrics,
    ) -> list[BugFinding]:
        event_counter = Counter(str(item.get("event", "")) for item in events)
        findings: list[BugFinding] = []

        def add(
            bug_id: str,
            severity: str,
            category: str,
            evidence: str,
            impact: str,
            suggestion: str,
            count: int = 1,
        ) -> None:
            findings.append(
                BugFinding(
                    bug_id=bug_id,
                    severity=severity,
                    category=category,
                    evidence=evidence,
                    impact=impact,
                    suggestion=suggestion,
                    count=count,
                )
            )

        if event_counter["swarm.run.fatal"] > 0:
            count = event_counter["swarm.run.fatal"]
            add(
                bug_id="BUG-RUN-FATAL",
                severity="critical",
                category="runtime",
                evidence=f"swarm.run.fatal={count}",
                impact="Swarm run terminated unexpectedly and mission cannot complete deterministically.",
                suggestion="Inspect fatal traceback and add startup/runtime guards before scheduler loop.",
                count=count,
            )

        if event_counter["swarm.start.failed"] > 0:
            count = event_counter["swarm.start.failed"]
            add(
                bug_id="BUG-STARTUP-FAIL",
                severity="high",
                category="startup",
                evidence=f"swarm.start.failed={count}",
                impact="Workers/assigner failed to initialize; mission startup is unstable.",
                suggestion="Validate plugin/mcp runtime assets and worker process readiness checks.",
                count=count,
            )

        if process_metrics.timeouts > 0:
            add(
                bug_id="BUG-TASK-TIMEOUT",
                severity="high",
                category="execution",
                evidence=f"task.timeout={process_metrics.timeouts}",
                impact="Task execution exceeds timeout and causes retries/recovery churn.",
                suggestion="Investigate long-running prompts or tune task_timeout per mission profile.",
                count=process_metrics.timeouts,
            )

        if process_metrics.dispatch_errors > 0:
            add(
                bug_id="BUG-DISPATCH-ERROR",
                severity="high",
                category="dispatch",
                evidence=f"task.dispatched.error={process_metrics.dispatch_errors}",
                impact="Task dispatch to worker fails and mission progress can stall.",
                suggestion="Audit worker health endpoint and request payload compatibility.",
                count=process_metrics.dispatch_errors,
            )

        if process_metrics.recoveries > 0:
            add(
                bug_id="BUG-WORKER-RECOVERY",
                severity="medium",
                category="reliability",
                evidence=f"worker.recover.start={process_metrics.recoveries}",
                impact="Worker restarts indicate transient crashes or heartbeat instability.",
                suggestion="Correlate worker stderr logs with recoveries and fix recurrent crash roots.",
                count=process_metrics.recoveries,
            )

        if process_metrics.incidents > 0:
            add(
                bug_id="BUG-INCIDENT-GENERATED",
                severity="high",
                category="observability",
                evidence=f"incident.generated={process_metrics.incidents}",
                impact="Incident snapshots indicate orchestrator detected severe anomalies.",
                suggestion="Use incident bundle to reproduce and eliminate the triggering condition.",
                count=process_metrics.incidents,
            )

        if process_metrics.empty_plan_events > 0:
            add(
                bug_id="BUG-EMPTY-PLAN-FLOW",
                severity="medium",
                category="planning",
                evidence=f"plan.empty={process_metrics.empty_plan_events}",
                impact="Scheduler entered empty-plan path and spent cycles without actionable work.",
                suggestion="Strengthen mission decomposition guard to ensure non-empty task graph before run.",
                count=process_metrics.empty_plan_events,
            )

        if process_metrics.completion_ratio < 1.0:
            unfinished = max(0, process_metrics.task_total - process_metrics.task_done)
            add(
                bug_id="BUG-INCOMPLETE-MISSION",
                severity="critical",
                category="correctness",
                evidence=f"completion_ratio={process_metrics.completion_ratio:.2f}, unfinished={unfinished}",
                impact="Not all tasks completed, mission output is incomplete.",
                suggestion="Trace blocked/failed tasks and enforce dependency + retry closure before exit.",
                count=unfinished or 1,
            )

        if result_metrics.completeness_ratio < 1.0:
            missing = max(0, result_metrics.expected_reports - result_metrics.report_files)
            add(
                bug_id="BUG-MISSING-ARTIFACT",
                severity="medium",
                category="artifact",
                evidence=f"report_files={result_metrics.report_files}, expected={result_metrics.expected_reports}",
                impact="Some DONE tasks are missing required report artifacts.",
                suggestion="Enforce artifact path write-before-DONE in worker completion contract.",
                count=missing or 1,
            )

        if result_metrics.structure_ratio < 0.8:
            add(
                bug_id="BUG-REPORT-STRUCTURE",
                severity="medium",
                category="quality",
                evidence=f"structure_ratio={result_metrics.structure_ratio:.2f}",
                impact="Reports are hard to audit due to missing sections.",
                suggestion="Use a strict report template and validate sections before marking DONE.",
            )

        if result_metrics.keyword_hit_ratio < 0.6:
            expected = case.expected_keywords or _keyword_candidates(case.mission, limit=6)
            add(
                bug_id="BUG-LOW-MISSION-RELEVANCE",
                severity="medium",
                category="semantic",
                evidence=f"keyword_hit_ratio={result_metrics.keyword_hit_ratio:.2f}, expected={expected}",
                impact="Output may not be aligned with mission intent.",
                suggestion="Improve task prompts with explicit mission constraints and acceptance criteria.",
            )

        # Dependency correctness sanity check.
        status_by_task = {str(task.get("id") or ""): str(task.get("status") or "") for task in run.tasks}
        dependency_violations = 0
        for task in run.tasks:
            deps = task.get("dependencies", [])
            if not isinstance(deps, list):
                continue
            if str(task.get("status") or "") != "DONE":
                continue
            for dep in deps:
                dep_id = str(dep or "")
                if status_by_task.get(dep_id) != "DONE":
                    dependency_violations += 1
        if dependency_violations > 0:
            add(
                bug_id="BUG-DEPENDENCY-VIOLATION",
                severity="high",
                category="scheduler",
                evidence=f"dependency_violations={dependency_violations}",
                impact="Tasks completed without all dependencies DONE; schedule correctness is broken.",
                suggestion="Add dependency gate assertion before DONE transition.",
                count=dependency_violations,
            )

        return findings

    def _build_codex_judgement(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        events: list[dict[str, Any]],
        process_metrics: ProcessMetrics,
        result_metrics: ResultMetrics,
        bug_findings: list[BugFinding],
    ) -> CodexJudgement:
        severity_penalty = {"critical": 18.0, "high": 10.0, "medium": 5.0, "low": 2.0}
        aggregated_bug_penalty = 0.0
        for finding in bug_findings:
            aggregated_bug_penalty += severity_penalty.get(finding.severity, 3.0) * max(1, finding.count)
        aggregated_bug_penalty = min(40.0, aggregated_bug_penalty)

        event_counter = Counter(str(item.get("event", "")) for item in events)

        stability_score = _clamp(
            100.0
            - event_counter["incident.generated"] * 18.0
            - event_counter["task.timeout"] * 12.0
            - event_counter["task.dispatched.error"] * 12.0
            - event_counter["worker.recover.start"] * 6.0
            - event_counter["task.released"] * 3.0
            - event_counter["swarm.run.fatal"] * 25.0
        )
        stability = DimensionJudgement(
            score=round(stability_score, 2),
            evidence=[
                f"incident.generated={event_counter['incident.generated']}",
                f"task.timeout={event_counter['task.timeout']}",
                f"task.dispatched.error={event_counter['task.dispatched.error']}",
                f"worker.recover.start={event_counter['worker.recover.start']}",
            ],
            risk=[
                finding.evidence
                for finding in bug_findings
                if finding.category in {"runtime", "execution", "dispatch", "reliability", "observability"}
            ][:3],
            suggestion=[
                "Stabilize worker runtime and reduce timeout/recovery churn before scaling mission complexity."
            ],
        )

        claims_gap = abs(process_metrics.scheduler_claims - process_metrics.task_total)
        dispatch_gap = max(0, process_metrics.scheduler_claims - process_metrics.scheduler_dispatches)
        scheduling_score = _clamp(
            100.0
            - claims_gap * 8.0
            - dispatch_gap * 6.0
            - process_metrics.empty_plan_events * 6.0
            - (1.0 - process_metrics.completion_ratio) * 35.0
        )
        scheduling = DimensionJudgement(
            score=round(scheduling_score, 2),
            evidence=[
                f"scheduler_claimed={process_metrics.scheduler_claims}",
                f"scheduler_dispatched={process_metrics.scheduler_dispatches}",
                f"task_total={process_metrics.task_total}",
                f"completion_ratio={process_metrics.completion_ratio:.2f}",
            ],
            risk=[
                finding.evidence
                for finding in bug_findings
                if finding.category in {"scheduler", "planning", "correctness"}
            ][:3],
            suggestion=[
                "Tighten dependency and claim/dispatched consistency checks in scheduler trace review."
            ],
        )

        efficiency_score = _clamp(
            process_metrics.efficiency_ratio * 100.0
            - max(0.0, run.duration_seconds - process_metrics.expected_floor_seconds) * 1.8
        )
        efficiency = DimensionJudgement(
            score=round(efficiency_score, 2),
            evidence=[
                f"duration={run.duration_seconds:.2f}s",
                f"expected_floor={process_metrics.expected_floor_seconds:.2f}s",
                f"efficiency_ratio={process_metrics.efficiency_ratio:.2f}",
            ],
            risk=[
                "Execution latency significantly above critical-path floor."
            ]
            if run.duration_seconds > process_metrics.expected_floor_seconds * 1.8
            else [],
            suggestion=[
                "Reduce idle polling latency and optimize prompt/worker turnaround for chained tasks."
            ],
        )

        completeness_score = _clamp(
            result_metrics.completeness_ratio * 100.0
            - max(0.0, 1.0 - process_metrics.completion_ratio) * 20.0
        )
        completeness = DimensionJudgement(
            score=round(completeness_score, 2),
            evidence=[
                f"done_tasks={process_metrics.task_done}/{process_metrics.task_total}",
                f"report_files={result_metrics.report_files}/{result_metrics.expected_reports}",
                f"completeness_ratio={result_metrics.completeness_ratio:.2f}",
            ],
            risk=[
                finding.evidence for finding in bug_findings if finding.category in {"artifact", "correctness"}
            ][:3],
            suggestion=[
                "Require artifact existence and task DONE convergence checks before mission completion."
            ],
        )

        quality_score = _clamp(
            result_metrics.structure_ratio * 45.0
            + result_metrics.keyword_hit_ratio * 35.0
            + result_metrics.richness_ratio * 20.0
        )
        quality = DimensionJudgement(
            score=round(quality_score, 2),
            evidence=[
                f"structure_ratio={result_metrics.structure_ratio:.2f}",
                f"keyword_hit_ratio={result_metrics.keyword_hit_ratio:.2f}",
                f"richness_ratio={result_metrics.richness_ratio:.2f}",
            ],
            risk=[finding.evidence for finding in bug_findings if finding.category in {"quality", "semantic"}][:3],
            suggestion=[
                "Keep report template strict and bind each mission keyword to explicit acceptance output."
            ],
        )

        overall_score = _clamp(
            stability.score * 0.28
            + scheduling.score * 0.18
            + efficiency.score * 0.14
            + completeness.score * 0.20
            + quality.score * 0.20
            - aggregated_bug_penalty
        )
        top_bug_ids = [finding.bug_id for finding in bug_findings[:3]]
        overall = DimensionJudgement(
            score=round(overall_score, 2),
            evidence=[
                f"weighted_subscores={stability.score:.1f}/{scheduling.score:.1f}/{efficiency.score:.1f}/"
                f"{completeness.score:.1f}/{quality.score:.1f}",
                f"bug_count={len(bug_findings)}",
            ],
            risk=top_bug_ids,
            suggestion=[
                "Prioritize high/critical findings first, then optimize efficiency and output quality."
            ],
        )

        # --- error_recovery_and_reflection heuristic ---
        has_incidents = event_counter["incident.generated"] > 0
        has_timeouts = event_counter["task.timeout"] > 0
        has_recoveries = event_counter["worker.recover.start"] > 0
        if not has_incidents and not has_timeouts:
            err_recovery_score = 85.0  # no error events to judge
            err_evidence = ["no incident/timeout events observed; neutral score"]
        else:
            err_recovery_score = 85.0
            err_recovery_score -= event_counter["incident.generated"] * 15.0
            err_recovery_score -= event_counter["task.timeout"] * 10.0
            if has_recoveries:
                err_recovery_score += min(20.0, event_counter["worker.recover.start"] * 8.0)
            err_recovery_score = _clamp(err_recovery_score)
            err_evidence = [
                f"incident.generated={event_counter['incident.generated']}",
                f"task.timeout={event_counter['task.timeout']}",
                f"worker.recover.start={event_counter['worker.recover.start']}",
            ]
        error_recovery = DimensionJudgement(
            score=round(err_recovery_score, 2),
            evidence=err_evidence,
            risk=[f.evidence for f in bug_findings if f.category in {"reliability", "execution"}][:3],
            suggestion=["Use LLM judger for deeper error recovery analysis."],
        )

        # --- collaboration_effectiveness heuristic ---
        dep_total = 0
        dep_done = 0
        status_by_id = {str(t.get("id") or ""): str(t.get("status") or "") for t in run.tasks}
        for task in run.tasks:
            deps = task.get("dependencies", [])
            if not isinstance(deps, list):
                continue
            for dep in deps:
                dep_id = str(dep or "").strip()
                if dep_id:
                    dep_total += 1
                    if status_by_id.get(dep_id) == "DONE":
                        dep_done += 1
        if dep_total > 0:
            collab_score = _ratio(dep_done, dep_total) * 90.0 + 10.0
            collab_evidence = [f"dependency_edges_done={dep_done}/{dep_total}"]
        else:
            collab_score = 80.0  # no dependencies to judge
            collab_evidence = ["no dependency edges in task graph; neutral score"]
        collaboration = DimensionJudgement(
            score=round(_clamp(collab_score), 2),
            evidence=collab_evidence,
            risk=[f.evidence for f in bug_findings if f.category == "scheduler"][:3],
            suggestion=["Use LLM judger for deeper collaboration analysis."],
        )

        return CodexJudgement(
            execution_stability=stability,
            scheduling_quality=scheduling,
            progress_efficiency=efficiency,
            result_completeness=completeness,
            result_quality=quality,
            collaboration_effectiveness=collaboration,
            error_recovery_and_reflection=error_recovery,
            overall=overall,
        )

    @staticmethod
    def _merge_bug_findings(primary: list[BugFinding], fallback: list[BugFinding]) -> list[BugFinding]:
        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        merged: dict[tuple[str, str], BugFinding] = {}

        def put(item: BugFinding) -> None:
            key = (item.bug_id, item.category)
            if key not in merged:
                merged[key] = item
                return
            current = merged[key]
            current.count = max(1, int(current.count)) + max(0, int(item.count) - 1)
            if severity_rank.get(item.severity, 0) > severity_rank.get(current.severity, 0):
                current.severity = item.severity
            if len(item.evidence) > len(current.evidence):
                current.evidence = item.evidence
            if len(item.impact) > len(current.impact):
                current.impact = item.impact
            if len(item.suggestion) > len(current.suggestion):
                current.suggestion = item.suggestion

        for finding in fallback:
            put(finding)
        for finding in primary:
            put(finding)
        return list(merged.values())

    def _judge_with_codex_cli(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        events: list[dict[str, Any]],
        process_metrics: ProcessMetrics,
        result_metrics: ResultMetrics,
        baseline_findings: list[BugFinding],
    ) -> tuple[CodexJudgement, list[BugFinding]]:
        project_dir = run.session_dir.parents[2]
        prompt = self._build_codex_cli_prompt(
            case=case,
            run=run,
            events=events,
            process_metrics=process_metrics,
            result_metrics=result_metrics,
            baseline_findings=baseline_findings,
        )
        schema = self._codex_cli_output_schema()
        judge_log_dir = run.session_dir / "logs" / "judges"
        judge_log_dir.mkdir(parents=True, exist_ok=True)
        judge_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prompt_log_path = judge_log_dir / f"codex_judge_prompt_{judge_stamp}.md"
        schema_log_path = judge_log_dir / f"codex_judge_schema_{judge_stamp}.json"
        stdout_log_path = judge_log_dir / f"codex_judge_stdout_{judge_stamp}.log"
        stderr_log_path = judge_log_dir / f"codex_judge_stderr_{judge_stamp}.log"
        raw_log_path = judge_log_dir / f"codex_judge_raw_{judge_stamp}.txt"
        payload_log_path = judge_log_dir / f"codex_judge_payload_{judge_stamp}.json"
        meta_log_path = judge_log_dir / f"codex_judge_meta_{judge_stamp}.json"

        prompt_log_path.write_text(prompt, encoding="utf-8")
        schema_log_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="swarm-codex-judge-") as tmpdir:
            tmp_root = Path(tmpdir)
            schema_path = tmp_root / "judge_schema.json"
            output_path = tmp_root / "judge_output.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
            codex_home = tmp_root / "codex-home"
            codex_home.mkdir(parents=True, exist_ok=True)
            run_env = os.environ.copy()
            run_env["CODEX_HOME"] = str(codex_home)
            run_env.pop("CLAUDECODE", None)

            if "claude" in self.codex_cli_path:
                cmd = [
                    self.codex_cli_path,
                    "-p",
                    prompt,
                    "--model", self.codex_cli_model,
                    "--output-format", "json",
                    "--allowedTools", "Read,Glob,Grep,Bash(read-only)",
                    "--max-turns", "30",
                    "--verbose",
                ]
            else:
                cmd = [
                    self.codex_cli_path,
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--ephemeral",
                    "--color",
                    "never",
                    "-C",
                    str(project_dir),
                    "--add-dir",
                    str(run.session_dir),
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]

            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.codex_cli_timeout,
                check=False,
                env=run_env,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            stdout_log_path.write_text(stdout_text, encoding="utf-8", errors="replace")
            stderr_log_path.write_text(stderr_text, encoding="utf-8", errors="replace")
            meta_log_path.write_text(
                json.dumps(
                    {
                        "command": cmd,
                        "returncode": completed.returncode,
                        "timeout_seconds": self.codex_cli_timeout,
                        "project_dir": str(project_dir),
                        "session_dir": str(run.session_dir),
                        "prompt_path": str(prompt_log_path),
                        "schema_path": str(schema_log_path),
                        "stdout_path": str(stdout_log_path),
                        "stderr_path": str(stderr_log_path),
                        "raw_path": str(raw_log_path),
                        "payload_path": str(payload_log_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if completed.returncode != 0:
                message = self._summarize_cli_failure(
                    stderr_text=stderr_text,
                    stdout_text=stdout_text,
                    returncode=completed.returncode,
                )
                raise RuntimeError(f"codex exec failed ({meta_log_path}): {message[:800]}")

            raw = ""
            if output_path.exists():
                raw = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                raw = stdout_text.strip()
            raw_log_path.write_text(raw, encoding="utf-8", errors="replace")
            # Claude Code --output-format json emits a JSON array of messages.
            # Extract the last assistant text content which contains the actual
            # structured judgement payload.
            raw = self._extract_claude_code_text(raw) or raw
            payload = _load_json_payload(raw)
            payload_log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        judgement = self._parse_codex_judgement(payload)
        findings = self._parse_codex_bug_findings(payload.get("bug_findings"))
        return judgement, findings

    @staticmethod
    def _extract_claude_code_text(raw: str) -> str:
        """Extract the last assistant text from Claude Code JSON array output."""
        try:
            data = json.loads(raw)
        except Exception:
            return ""
        if not isinstance(data, list):
            return ""
        for obj in reversed(data):
            if not isinstance(obj, dict) or obj.get("type") != "assistant":
                continue
            msg = obj.get("message", {})
            if not isinstance(msg, dict):
                continue
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    return c.get("text", "")
        return ""

    @staticmethod
    def _summarize_cli_failure(stderr_text: str, stdout_text: str, returncode: int) -> str:
        text = stderr_text.strip() or stdout_text.strip()
        if not text:
            return f"returncode={returncode}"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return f"returncode={returncode}"
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("error:") or "timed out" in lowered or "failed" in lowered:
                return line
        return lines[-1]

    @staticmethod
    def _normalize_text_list(value: Any, *, max_items: int = 8, max_len: int = 220) -> list[str]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = [str(item) for item in value]
        else:
            items = []
        result: list[str] = []
        for item in items:
            text = " ".join(str(item).split()).strip()
            if not text:
                continue
            if len(text) > max_len:
                text = text[: max_len - 3] + "..."
            result.append(text)
            if len(result) >= max_items:
                break
        return result

    @staticmethod
    def _parse_dimension(payload: Any) -> DimensionJudgement:
        data = payload if isinstance(payload, dict) else {}
        try:
            score = float(data.get("score", 0.0))
        except Exception:
            score = 0.0
        return DimensionJudgement(
            score=round(_clamp(score), 2),
            evidence=MissionBenchmarkRunner._normalize_text_list(data.get("evidence", []), max_items=12),
            risk=MissionBenchmarkRunner._normalize_text_list(data.get("risk", []), max_items=8),
            suggestion=MissionBenchmarkRunner._normalize_text_list(data.get("suggestion", []), max_items=8),
        )

    @staticmethod
    def _parse_codex_judgement(payload: dict[str, Any]) -> CodexJudgement:
        return CodexJudgement(
            execution_stability=MissionBenchmarkRunner._parse_dimension(payload.get("execution_stability")),
            scheduling_quality=MissionBenchmarkRunner._parse_dimension(payload.get("scheduling_quality")),
            progress_efficiency=MissionBenchmarkRunner._parse_dimension(payload.get("progress_efficiency")),
            result_completeness=MissionBenchmarkRunner._parse_dimension(payload.get("result_completeness")),
            result_quality=MissionBenchmarkRunner._parse_dimension(payload.get("result_quality")),
            collaboration_effectiveness=MissionBenchmarkRunner._parse_dimension(payload.get("collaboration_effectiveness")),
            error_recovery_and_reflection=MissionBenchmarkRunner._parse_dimension(payload.get("error_recovery_and_reflection")),
            overall=MissionBenchmarkRunner._parse_dimension(payload.get("overall")),
        )

    @staticmethod
    def _parse_codex_bug_findings(payload: Any) -> list[BugFinding]:
        rows = payload if isinstance(payload, list) else []
        findings: list[BugFinding] = []
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            bug_id = str(item.get("bug_id") or f"BUG-LLM-{idx+1:03d}").strip() or f"BUG-LLM-{idx+1:03d}"
            severity = str(item.get("severity") or "medium").strip().lower()
            if severity not in {"critical", "high", "medium", "low"}:
                severity = "medium"
            category = str(item.get("category") or "unknown").strip() or "unknown"
            evidence = str(item.get("evidence") or "").strip() or "No evidence provided."
            impact = str(item.get("impact") or "").strip() or "Impact not specified."
            suggestion = str(item.get("suggestion") or "").strip() or "Suggestion not specified."
            try:
                count = int(item.get("count", 1))
            except Exception:
                count = 1
            findings.append(
                BugFinding(
                    bug_id=bug_id,
                    severity=severity,
                    category=category,
                    evidence=evidence[:320],
                    impact=impact[:320],
                    suggestion=suggestion[:320],
                    count=max(1, count),
                )
            )
        return findings

    @staticmethod
    def _codex_cli_output_schema() -> dict[str, Any]:
        dim = {
            "type": "object",
            "required": ["score", "evidence", "risk", "suggestion"],
            "additionalProperties": False,
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "risk": {"type": "array", "items": {"type": "string"}},
                "suggestion": {"type": "array", "items": {"type": "string"}},
            },
        }
        bug = {
            "type": "object",
            "required": ["bug_id", "severity", "category", "evidence", "impact", "suggestion", "count"],
            "additionalProperties": False,
            "properties": {
                "bug_id": {"type": "string"},
                "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                "category": {"type": "string"},
                "evidence": {"type": "string"},
                "impact": {"type": "string"},
                "suggestion": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "execution_stability",
                "scheduling_quality",
                "progress_efficiency",
                "result_completeness",
                "result_quality",
                "collaboration_effectiveness",
                "error_recovery_and_reflection",
                "overall",
                "bug_findings",
            ],
            "properties": {
                "execution_stability": dim,
                "scheduling_quality": dim,
                "progress_efficiency": dim,
                "result_completeness": dim,
                "result_quality": dim,
                "collaboration_effectiveness": dim,
                "error_recovery_and_reflection": dim,
                "overall": dim,
                "bug_findings": {"type": "array", "items": bug},
            },
        }

    def _build_codex_cli_prompt(
        self,
        case: MissionCase,
        run: SwarmRunResult,
        events: list[dict[str, Any]],
        process_metrics: ProcessMetrics,
        result_metrics: ResultMetrics,
        baseline_findings: list[BugFinding],
    ) -> str:
        session_dir = run.session_dir
        project_dir = session_dir.parents[2]
        orchestrator_log = session_dir / "logs" / "orchestrator" / "orchestrator.jsonl"
        scheduler_trace = session_dir / "logs" / "orchestrator" / "scheduler_trace.jsonl"
        registry_path = session_dir / "global_indices" / "registry.json"
        state_path = session_dir / "global_indices" / "orchestrator_state.json"
        completion_report = session_dir / "global_indices" / "swarm_completion_report.md"
        resources_dir = session_dir / "resources"
        worker_logs_dir = session_dir / "logs" / "workers"

        task_snapshot = [
            {
                "id": task.get("id"),
                "status": task.get("status"),
                "dependencies": task.get("dependencies", []),
                "assigned_worker": task.get("assigned_worker"),
                "result_summary": task.get("result_summary"),
                "artifact_link": task.get("artifact_link"),
            }
            for task in run.tasks
        ]

        baseline_bug_lines = [
            f"- [{item.severity}] {item.bug_id}: {item.evidence} (impact={item.impact})"
            for item in baseline_findings[:12]
        ]
        if not baseline_bug_lines:
            baseline_bug_lines = ["- (none)"]

        schema = self._codex_cli_output_schema()

        worker_mode_desc = (
            "真实 opencode serve + Claude worker（产物是真实 LLM 输出，质量评审应着重内容深度和可操作性）"
            if self.use_real_opencode
            else "opencode serve mock（产物为模板输出，质量评审应聚焦调度过程和结构完整性）"
        )
        background = [
            "你是 Swarm Mission 质量评审器。你需要审计一次 `/swarm <mission>` 的执行过程和结果。",
            "",
            "# 系统背景",
            "",
            f"/swarm mission 会启动 orchestrator，调度多个 worker（{worker_mode_desc}）执行 central plan。",
            "所有状态、日志、产物都在 session_dir 下，你需要自行读取相关文件来获取证据。",
            "正确执行应体现：任务依赖被遵守、任务最终 DONE、产物与任务语义一致。",
            "",
            "# 本次评审的 Mission 信息",
            "",
            f"- case_id: {case.case_id}",
            f"- mission: {case.mission}",
            f"- expected_keywords: {case.expected_keywords}",
            f"- task_graph: {json.dumps(case.tasks, ensure_ascii=False)}",
            f"- duration_seconds: {run.duration_seconds:.3f}",
            f"- task_snapshot: {json.dumps(task_snapshot, ensure_ascii=False)}",
            "",
            "# 关键文件路径（请按需读取）",
            "",
            "以下是本次执行产生的所有关键文件的绝对路径，请根据评审需要自行读取：",
            "",
            f"- session_dir: {session_dir}",
            f"- central_plan: {run.plan_path}",
            f"- orchestrator_log (JSONL): {orchestrator_log}",
            f"- scheduler_trace (JSONL): {scheduler_trace}",
            f"- registry: {registry_path}",
            f"- orchestrator_state: {state_path}",
            f"- completion_report: {completion_report}",
            f"- resources_dir (产物目录): {resources_dir}",
            f"- worker_logs_dir: {worker_logs_dir}",
            "",
            "建议的审查顺序：",
            "1. 先读 orchestrator_state 和 registry 了解最终状态",
            "2. 读 orchestrator_log 和 scheduler_trace 分析过程",
            "3. 读 resources_dir 下的产物文件评估结果质量",
            "4. 读 worker_logs_dir 下的日志排查异常",
            "5. 读 completion_report 查看系统自评",
            "",
            "# 规则评分基线（仅供参考，不是最终结论）",
            "",
            f"- process_metrics: {json.dumps(asdict(process_metrics), ensure_ascii=False)}",
            f"- result_metrics: {json.dumps(asdict(result_metrics), ensure_ascii=False)}",
            "",
            "基线 bug（规则检测）：",
            *baseline_bug_lines,
            "",
            "# 评审规则（硬约束）",
            "",
            "- 只读分析，不修改任何文件。",
            "- 必须引用证据（文件路径 + 具体内容）；不允许猜测。",
            '- 若证据不足，降低分数并在 evidence/risk 明确写出"证据不足"的原因。',
            "- 若发现 bug，必须给出 bug_id/severity/category/evidence/impact/suggestion/count。",
            "",
            "评分刻度（每项 0-100）：",
            "- 90-100: 证据充分且稳定/完整，未见中高风险问题。",
            "- 75-89: 总体良好，存在可恢复问题或轻微缺陷。",
            "- 60-74: 有明显缺陷，影响效率或结果可信度。",
            "- 40-59: 存在中高风险问题，过程或结果显著不可靠。",
            "- 0-39: 核心链路失败或基本不可用。",
            "",
            "评审维度（必须全部覆盖）：",
            "1) execution_stability: 崩溃/超时/重试/恢复闭环",
            "2) scheduling_quality: claim-dispatch 一致性、依赖约束、状态机合法性",
            "3) progress_efficiency: 耗时、空转、阻塞、关键路径执行效率",
            "4) result_completeness: DONE 比例、产物齐备度、artifact 与任务一致性",
            "5) result_quality: 报告结构、语义相关性、信息密度、可操作性",
            "6) collaboration_effectiveness: Worker 间上下文传递，前置任务产物是否被后续采纳",
            "7) error_recovery_and_reflection: 失败时是否反思环境并提出规避/恢复方案",
            "8) overall: 综合结论，不是简单平均，需反映主要风险",
            "",
            "Bug 发现标准：",
            "- category: startup/scheduling/recovery/timeout/state/artifact/semantic/quality/judge/other",
            "- severity: critical(阻断主链路), high(显著错误), medium(功能受损), low(轻微问题)",
            "- 同类问题按 bug_id 聚合，count 表示出现次数",
            "- evidence 需要带文件路径和具体内容，确保可复查",
            "",
            "# Known Limitations",
            "",
            "以下现象是已知的框架行为，不应标记为 bug：",
            "",
            "1. **`worker.idle` 事件中 `previous_task=null`**：这是 orchestrator `check_idle()` 的已知时序问题。",
            "   当 worker 首次进入 idle 状态或在任务完成与状态更新之间存在竞态时，`previous_task` 字段可能为 null。",
            "   这不影响任务调度的正确性，不应作为 bug 报告。",
            "",
            *(
                [
                    "2. **`scheduler_trace.jsonl` 为空或内容很少**：在 mock 运行环境中，scheduler trace 文件可能为空",
                    "   或仅包含少量条目，因为 mock worker 完成任务非常快。这是测试框架的预期行为，不代表调度器异常。",
                    "",
                ]
                if not self.use_real_opencode
                else [
                    "2. **真实 LLM worker 执行耗时差异大**：真实 Claude worker 的执行时间取决于任务复杂度，",
                    "   单 task 耗时 30-300s 均属正常范围，不应作为效率 bug 报告（除非有明确超时或阻塞）。",
                    "",
                ]
            ),
            "# 输出要求",
            "",
            "只输出严格 JSON，符合以下 schema：",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "注意：",
            "- 每个维度的 evidence/risk/suggestion 给出具体条目，避免空泛措辞。",
            "- overall 需要明确说明关键风险来源。",
            "- bug_findings 只保留有证据的问题；无问题时返回空数组。",
        ]
        return "\n".join(background)

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 85:
            return "优秀"
        if score >= 70:
            return "良好"
        if score >= 55:
            return "一般"
        return "较差"


def render_markdown_report(results: list[MissionBenchmarkResult]) -> str:
    lines: list[str] = []
    lines.append("# Swarm Mission Benchmark Report")
    lines.append("")
    lines.append(f"- generated_at: {_utc_now_iso()}")
    lines.append(f"- total_cases: {len(results)}")
    lines.append("")
    lines.append("| Case | Status | Process | Result | Total | Bugs | Verdict | Duration(s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for item in results:
        lines.append(
            f"| `{item.case_id}` | {item.status} | {item.process_score:.2f} | {item.result_score:.2f} | "
            f"{item.total_score:.2f} | {len(item.bug_findings)} | {item.verdict} | {item.duration_seconds:.2f} |"
        )

    for item in results:
        lines.append("")
        lines.append(f"## {item.case_id}")
        lines.append("")
        lines.append(f"- mission: {item.mission}")
        lines.append(f"- status: {item.status}")
        lines.append(f"- verdict: {item.verdict}")
        lines.append(f"- total_score: {item.total_score:.2f}")
        lines.append(f"- judge_source: {item.judge_source}")
        lines.append(f"- codex_cli_invoked: {item.codex_cli_invoked}")
        if item.codex_cli_error:
            lines.append(f"- codex_cli_error: {item.codex_cli_error}")
        if item.error:
            lines.append(f"- error: {item.error}")
            continue
        if item.process:
            lines.append(f"- process_notes: {'; '.join(item.process.notes)}")
        if item.result:
            lines.append(f"- result_notes: {'; '.join(item.result.notes)}")
        if item.codex_judgement:
            cj = item.codex_judgement
            lines.append("- codex_judgement:")
            lines.append(
                "  "
                + ", ".join(
                    [
                        f"execution_stability={cj.execution_stability.score:.1f}",
                        f"scheduling_quality={cj.scheduling_quality.score:.1f}",
                        f"progress_efficiency={cj.progress_efficiency.score:.1f}",
                        f"result_completeness={cj.result_completeness.score:.1f}",
                        f"result_quality={cj.result_quality.score:.1f}",
                        f"collaboration_effectiveness={cj.collaboration_effectiveness.score:.1f}",
                        f"error_recovery_and_reflection={cj.error_recovery_and_reflection.score:.1f}",
                        f"overall={cj.overall.score:.1f}",
                    ]
                )
            )
            if cj.overall.risk:
                lines.append(f"  risks: {', '.join(cj.overall.risk)}")
        if item.bug_findings:
            lines.append("- bug_findings:")
            for finding in item.bug_findings:
                lines.append(
                    "  "
                    + f"[{finding.severity}] {finding.bug_id} x{finding.count}: {finding.evidence} | "
                    + f"impact={finding.impact} | suggestion={finding.suggestion}"
                )
        else:
            lines.append("- bug_findings: none")
    lines.append("")
    return "\n".join(lines)


def _select_cases(all_cases: list[MissionCase], selected_ids: list[str]) -> list[MissionCase]:
    if not selected_ids:
        return all_cases
    wanted = {item.strip() for item in selected_ids if item.strip()}
    return [case for case in all_cases if case.case_id in wanted]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run swarm mission benchmark on simulated opencode runtime.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--mock-task-delay", type=float, default=0.15)
    parser.add_argument("--task-timeout", type=int, default=20)
    parser.add_argument("--heartbeat-timeout", type=int, default=20)
    parser.add_argument("--case", action="append", default=[], help="Case id to run (repeatable).")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--work-root", default="")
    parser.add_argument("--codex-cli-path", default="codex")
    parser.add_argument("--codex-cli-timeout", type=int, default=60)
    parser.add_argument("--codex-cli-model", default="claude-sonnet-4-6", help="Model ID for Claude Code judge.")
    parser.add_argument(
        "--judge-by-codex",
        dest="judge_by_codex",
        action="store_true",
        help="Enable Codex multi-dimension judgement and bug finding from logs.",
    )
    parser.add_argument(
        "--no-codex-judge",
        dest="judge_by_codex",
        action="store_false",
        help="Disable Codex judgement section; keep rule-based scores only.",
    )
    parser.add_argument(
        "--no-codex-cli",
        dest="prefer_codex_cli",
        action="store_false",
        help="Do not call codex CLI; use internal rule-based Codex judgement only.",
    )
    parser.add_argument(
        "--use-codex-cli",
        dest="prefer_codex_cli",
        action="store_true",
        help="Call codex CLI for judgement (default when Codex judgement enabled).",
    )
    parser.add_argument(
        "--real-opencode",
        dest="use_real_opencode",
        action="store_true",
        default=False,
        help="Use genuine opencode serve + Claude workers instead of the mock HTTP server.",
    )
    parser.add_argument(
        "--real-case-timeout",
        type=float,
        default=600.0,
        help="Wall-clock timeout (seconds) per case when running in real-opencode mode (default: 600).",
    )
    parser.add_argument(
        "--worker-model",
        dest="worker_model",
        default="",
        help=(
            "Model string (provider/model-id) injected into worker opencode config. "
            "Defaults to 'opencode/qwen3-coder' when --real-opencode is set. "
            "Example: 'opencode/qwen3-coder', 'anthropic/claude-sonnet-4-5'."
        ),
    )
    parser.set_defaults(judge_by_codex=True, prefer_codex_cli=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    all_cases = default_mission_cases()
    cases = _select_cases(all_cases, args.case)
    if not cases:
        raise SystemExit("No benchmark cases selected.")

    case_ids = [c.case_id for c in cases]
    print(
        f"[bench] cases={case_ids} workers={args.workers} "
        f"use_real_opencode={args.use_real_opencode} "
        f"judge_by_codex={args.judge_by_codex} prefer_codex_cli={args.prefer_codex_cli} "
        f"codex_cli_path={args.codex_cli_path} codex_cli_model={args.codex_cli_model}",
        file=sys.stderr,
        flush=True,
    )

    runner = MissionBenchmarkRunner(
        repo_root=repo_root,
        workers=args.workers,
        poll_interval=args.poll_interval,
        mock_task_delay=args.mock_task_delay,
        task_timeout=args.task_timeout,
        heartbeat_timeout=args.heartbeat_timeout,
        work_root=Path(args.work_root).resolve() if args.work_root else None,
        judge_by_codex=bool(args.judge_by_codex),
        prefer_codex_cli=bool(args.prefer_codex_cli),
        codex_cli_path=str(args.codex_cli_path),
        codex_cli_timeout=int(args.codex_cli_timeout),
        codex_cli_model=str(args.codex_cli_model),
        use_real_opencode=bool(args.use_real_opencode),
        real_case_timeout=float(args.real_case_timeout),
        worker_model=str(args.worker_model),
    )
    results = runner.run_cases(cases)

    md = render_markdown_report(results)
    print(md)

    if args.md_out:
        md_path = Path(args.md_out).resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")

    if args.json_out:
        payload = {
            "generated_at": _utc_now_iso(),
            "cases": [item.to_dict() for item in results],
        }
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
