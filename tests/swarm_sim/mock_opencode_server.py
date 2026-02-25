#!/usr/bin/env python3
"""Mock implementation of `opencode serve` for swarm integration tests.

This executable provides the minimal HTTP surface that orchestrator workers
expect from opencode:
- GET /global/health
- POST /session
- POST /session/<id>/message
- POST /session/<id>/prompt_async
- GET /session/status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import fcntl


TASK_ID_RE = re.compile(r"Task ID:\s*([^\n\r]+)")
TITLE_RE = re.compile(r"Title:\s*([^\n\r]+)")
DESCRIPTION_RE = re.compile(
    r"Description:\s*\n([\s\S]*?)\n\n(?:Predecessor Results|Detailed report path)",
    flags=re.IGNORECASE,
)
MISSION_RE = re.compile(r"Mission:\s*([^\n\r.]+)")
DETAIL_REPORT_RE = re.compile(
    r"Detailed report path \(absolute, required on DONE\):\s*\n([^\n\r]+)",
    flags=re.IGNORECASE,
)
WORKER_NAME_RE = re.compile(r"\bworker-\d+\b")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_fenced_json(raw: str) -> dict[str, Any]:
    matches = re.findall(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", raw, flags=re.MULTILINE)
    payload = matches[-1] if matches else raw
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def _to_fenced_json(data: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```\n"


def _extract_runtime_from_config(raw: str) -> dict[str, str]:
    runtime = {
        "project_dir": "",
        "swarm_session_id": "",
        "agent_name": "",
    }
    text = (raw or "").strip()
    if not text:
        return runtime
    try:
        config = json.loads(text)
    except Exception:
        return runtime

    mcp = config.get("mcp", {})
    if not isinstance(mcp, dict):
        return runtime
    bb = mcp.get("agent_team_blackboard", {})
    if not isinstance(bb, dict):
        return runtime
    command = bb.get("command", [])
    if isinstance(command, list):
        for idx, token in enumerate(command):
            if token == "--project-dir" and idx + 1 < len(command):
                runtime["project_dir"] = str(Path(str(command[idx + 1])).resolve())
                break

    env_map = bb.get("environment", {})
    if isinstance(env_map, dict):
        session_value = env_map.get("SWARM_SESSION_ID")
        if isinstance(session_value, str):
            runtime["swarm_session_id"] = session_value.strip()
        agent_value = env_map.get("AGENT_NAME")
        if isinstance(agent_value, str):
            runtime["agent_name"] = agent_value.strip()

    return runtime


@dataclass
class MockContext:
    project_dir: Path
    swarm_session_id: str
    agent_name: str
    task_delay_seconds: float

    @property
    def central_plan_path(self) -> Path:
        return (
            self.project_dir
            / ".blackboard"
            / "sessions"
            / self.swarm_session_id
            / "global_indices"
            / "central_plan.md"
        )

    @property
    def resources_dir(self) -> Path:
        return self.central_plan_path.parent.parent / "resources"


class MockRuntime:
    def __init__(self, context: MockContext) -> None:
        self.context = context
        self._lock = threading.Lock()
        self._session_states: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "MockRuntime":
        runtime_from_cfg = _extract_runtime_from_config(os.environ.get("OPENCODE_CONFIG_CONTENT", ""))
        project_dir = runtime_from_cfg.get("project_dir", "").strip() or str(Path.cwd())
        session_id = (
            runtime_from_cfg.get("swarm_session_id", "").strip()
            or (os.environ.get("SWARM_SESSION_ID", "") or "").strip()
            or "default"
        )
        agent_name = (
            runtime_from_cfg.get("agent_name", "").strip()
            or (os.environ.get("AGENT_NAME", "") or "").strip()
            or "worker-mock"
        )
        delay_raw = (os.environ.get("SWARM_MOCK_TASK_DELAY", "") or "").strip()
        try:
            delay = float(delay_raw) if delay_raw else 0.25
        except ValueError:
            delay = 0.25
        return cls(
            MockContext(
                project_dir=Path(project_dir).resolve(),
                swarm_session_id=session_id,
                agent_name=agent_name,
                task_delay_seconds=max(0.0, delay),
            )
        )

    def create_session(self) -> str:
        sid = "mock-" + uuid.uuid4().hex[:10]
        with self._lock:
            self._session_states[sid] = "idle"
        return sid

    def status_snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {sid: {"type": state} for sid, state in self._session_states.items()}

    def schedule_task_completion(self, sid: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._session_states[sid] = "busy"
        t = threading.Thread(target=self._complete_task, args=(sid, payload), daemon=True)
        t.start()

    def _complete_task(self, sid: str, payload: dict[str, Any]) -> None:
        try:
            time.sleep(self.context.task_delay_seconds)
            text = self._extract_first_text(payload)
            task_id = self._extract_task_id(text)
            title = self._extract_title(text, fallback=task_id)
            description = self._extract_description(text)
            mission = self._extract_mission(text)
            report_path = self._extract_report_path(text, task_id)
            self._mark_task_done(
                task_id=task_id,
                report_path=report_path,
                title=title,
                description=description,
                mission=mission,
            )
        finally:
            with self._lock:
                if sid in self._session_states:
                    self._session_states[sid] = "idle"

    @staticmethod
    def _extract_first_text(payload: dict[str, Any]) -> str:
        parts = payload.get("parts", [])
        if not isinstance(parts, list):
            return ""
        for part in parts:
            if isinstance(part, dict) and part.get("type") in {"text", "reasoning"}:
                text = part.get("text")
                if isinstance(text, str):
                    return text
        return ""

    @staticmethod
    def _extract_task_id(prompt_text: str) -> str:
        if not prompt_text:
            return "task-unknown"
        match = TASK_ID_RE.search(prompt_text)
        if match:
            return match.group(1).strip()
        return "task-unknown"

    @staticmethod
    def _extract_title(prompt_text: str, fallback: str = "task-unknown") -> str:
        if not prompt_text:
            return fallback
        match = TITLE_RE.search(prompt_text)
        if match:
            text = match.group(1).strip()
            if text:
                return text
        return fallback

    @staticmethod
    def _extract_description(prompt_text: str) -> str:
        if not prompt_text:
            return ""
        match = DESCRIPTION_RE.search(prompt_text)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1).strip())

    @staticmethod
    def _extract_mission(prompt_text: str) -> str:
        if not prompt_text:
            return ""
        match = MISSION_RE.search(prompt_text)
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_report_path(self, prompt_text: str, task_id: str) -> Path:
        if prompt_text:
            match = DETAIL_REPORT_RE.search(prompt_text)
            if match:
                token = match.group(1).strip()
                if token:
                    raw = Path(token)
                    return raw if raw.is_absolute() else (self.context.project_dir / raw).resolve()
        fallback_name = re.sub(r"[^A-Za-z0-9._-]+", "-", task_id).strip("-") or "task-unknown"
        return (self.context.resources_dir / f"{fallback_name}.md").resolve()

    @staticmethod
    def _keywords_from_text(text: str, limit: int = 8) -> list[str]:
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", text)
        seen: set[str] = set()
        result: list[str] = []
        for token in tokens:
            low = token.lower()
            if low in seen:
                continue
            seen.add(low)
            result.append(token)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _should_simulate_failure(description: str, mission: str = "") -> bool:
        """Detect if a task should simulate failure based on description and mission context."""
        if not description:
            return False

        # Direct failure signal words in the task description.
        failure_patterns = [
            r"不要强行成功",
            r"写入.*失败原因",
            r"替代方案",
            r"缺乏依赖",
            r"should\s+fail",
            r"report\s+failure",
            r"无法完成",
            r"不应成功",
        ]
        for pattern in failure_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return True

        # Verification task in a mission that implies expected dependency failure.
        # e.g. "尝试 import redis" in a mission stating Redis is not installed.
        mission_failure_signals = [
            r"没有安装",
            r"未安装",
            r"缺少依赖",
            r"not installed",
            r"missing dependency",
            r"无法使用",
            r"根本没有",
        ]
        task_verify_signals = [
            r"尝试.{0,10}(import|安装|引入)",
            r"验证.{0,15}(是否支持|是否可用|能否)",
            r"检测.{0,15}(是否|能否)",
            r"try.{0,10}import",
            r"verify.{0,15}support",
        ]
        mission_implies_failure = any(
            re.search(p, mission, re.IGNORECASE) for p in mission_failure_signals
        )
        task_is_verification = any(
            re.search(p, description, re.IGNORECASE) for p in task_verify_signals
        )
        if mission_implies_failure and task_is_verification:
            return True

        return False

    def _mark_task_done(
        self,
        task_id: str,
        report_path: Path,
        title: str,
        description: str,
        mission: str,
    ) -> None:
        plan_path = self.context.central_plan_path
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if not plan_path.exists():
            return

        with plan_path.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                raw = f.read()
                try:
                    plan = _parse_fenced_json(raw)
                except Exception:
                    return
                tasks = plan.get("tasks", [])
                if not isinstance(tasks, list):
                    tasks = []

                target: dict[str, Any] | None = None
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    if str(task.get("id") or "") == task_id:
                        target = task
                        break
                if target is None:
                    for task in tasks:
                        if not isinstance(task, dict):
                            continue
                        if task.get("status") == "IN_PROGRESS" and task.get("assigned_worker") == self.context.agent_name:
                            target = task
                            break
                if target is None:
                    return

                report_path.parent.mkdir(parents=True, exist_ok=True)
                completed_at = _utc_now_iso()
                safe_description = description or "No explicit task description."
                mission_text = mission or "unknown-mission"
                is_failure = self._should_simulate_failure(safe_description, mission=mission_text)

                # Issue 4: Build predecessor context from dependencies
                predecessor_lines: list[str] = []
                target_deps = target.get("dependencies", [])
                if isinstance(target_deps, list) and target_deps:
                    tasks_by_id = {
                        str(t.get("id") or ""): t for t in tasks if isinstance(t, dict)
                    }
                    for dep_id in target_deps:
                        dep_id_str = str(dep_id or "").strip()
                        dep_task = tasks_by_id.get(dep_id_str)
                        if dep_task:
                            dep_summary = str(dep_task.get("result_summary") or "N/A")
                            dep_artifact = str(dep_task.get("artifact_link") or "N/A")
                            predecessor_lines.append(f"- {dep_id_str}: summary={dep_summary}")
                            predecessor_lines.append(f"  artifact: {dep_artifact}")

                # Build report sections
                report_sections: list[str] = [
                    f"# Report for {task_id}",
                    "",
                    "## Task Context",
                    f"- mission: {mission_text}",
                    f"- worker: {self.context.agent_name}",
                    f"- title: {title}",
                    f"- description: {safe_description}",
                ]

                # Issue 4: Add predecessor context section
                if predecessor_lines:
                    report_sections.extend([
                        "",
                        "## Predecessor Context",
                        *predecessor_lines,
                    ])

                # Issue 3: Task-specific execution section
                desc_summary = safe_description[:120].rstrip(".")
                if is_failure:
                    # Issue 1: Failure report
                    report_sections.extend([
                        "",
                        "## Execution",
                        f"1. Received task `{title}` for mission `{mission_text}`.",
                        f"2. Analyzed task requirements: {desc_summary}.",
                        "3. Determined that task cannot be completed as specified.",
                        "",
                        "## Failure Analysis",
                        f"Task `{task_id}` cannot succeed because the required preconditions are not met.",
                        f"Description indicates: {safe_description}",
                        "The environment lacks necessary dependencies or capabilities to fulfill this task.",
                        "",
                        "## Alternative Proposal",
                        "- Investigate whether the missing dependency can be installed or substituted.",
                        "- Consider a fallback implementation that does not require the missing component.",
                        "- Document the gap and escalate to the mission owner for resolution.",
                    ])
                else:
                    report_sections.extend([
                        "",
                        "## Execution",
                        f"1. Received task `{title}` for mission `{mission_text}`.",
                        f"2. Analyzed requirements: {desc_summary}.",
                        "3. Executed task logic in mock opencode runtime.",
                        "4. Persisted DONE status and artifact metadata to central_plan.",
                    ])

                report_sections.extend([
                    "",
                    "## Verification",
                    f"- completed_at: {completed_at}",
                    "- status_written: DONE",
                    "- scheduler_visible: yes",
                    "",
                    "## Artifacts",
                    f"- report_path: {report_path}",
                    "",
                    "## Summary",
                ])

                if is_failure:
                    result_summary_text = (
                        f"{task_id} failed: preconditions not met; "
                        f"see failure analysis in report: {report_path}"
                    )
                    report_sections.append(
                        f"Task `{task_id}` could not be completed for mission `{mission_text}`. "
                        "See Failure Analysis above."
                    )
                else:
                    result_summary_text = (
                        f"{task_id} completed by {self.context.agent_name}; report: {report_path}"
                    )
                    report_sections.append(
                        f"Task `{task_id}` completed successfully for mission `{mission_text}`."
                    )

                report_path.write_text(
                    "\n".join(report_sections) + "\n",
                    encoding="utf-8",
                )

                target["status"] = "DONE"
                target["result_summary"] = result_summary_text
                target["result"] = target["result_summary"]
                target["artifact_link"] = str(report_path)
                target["end_time"] = completed_at
                plan["updated_at"] = completed_at

                f.seek(0)
                f.write(_to_fenced_json(plan))
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def choose_worker_from_message(payload: dict[str, Any]) -> str:
        prompt = ""
        parts = payload.get("parts", [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    prompt = part["text"]
                    break
        candidates = WORKER_NAME_RE.findall(prompt or "")
        seen: set[str] = set()
        ordered = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)
        return ordered[0] if ordered else "worker-0"


class MockServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], runtime: MockRuntime):
        super().__init__(server_address, MockHandler)
        self.runtime = runtime


class MockHandler(BaseHTTPRequestHandler):
    server: MockServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Keep tests clean from access logs.
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/global/health":
            self._write_json(200, {"status": "ok", "mock": True})
            return
        if self.path == "/session/status":
            self._write_json(200, self.server.runtime.status_snapshot())
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if self.path == "/session":
            sid = self.server.runtime.create_session()
            self._write_json(200, {"id": sid})
            return

        match_message = re.fullmatch(r"/session/([^/]+)/message", self.path)
        if match_message:
            selected = self.server.runtime.choose_worker_from_message(payload)
            self._write_json(200, {"parts": [{"type": "text", "text": selected}]})
            return

        match_async = re.fullmatch(r"/session/([^/]+)/prompt_async", self.path)
        if match_async:
            sid = match_async.group(1)
            self.server.runtime.schedule_task_completion(sid, payload)
            self._write_json(200, {"ok": True})
            return

        self._write_json(404, {"error": "not_found"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opencode")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run mock opencode HTTP server")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--hostname", default="127.0.0.1")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command != "serve":
        raise RuntimeError(f"Unsupported command: {args.command}")

    runtime = MockRuntime.from_env()
    server = MockServer((args.hostname, args.port), runtime)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
