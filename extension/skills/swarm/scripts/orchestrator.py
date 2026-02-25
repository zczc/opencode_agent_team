#!/usr/bin/env python3
"""Swarm orchestrator for opencode agent team plugin.

This module manages worker `opencode serve` processes, dispatches tasks from
`.blackboard/sessions/<session_id>/global_indices/central_plan.md`, and emits
structured debug logs for startup/runtime failures.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This orchestrator currently supports POSIX systems only.") from exc

import httpx

from logging_utils import JsonlLogger, read_logging_config, stable_hash, utc_now_iso


def _safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _now() -> float:
    return time.time()


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_parent_pid_arg(value: Any) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")

    raw = str(value or "").strip()
    if not raw:
        return 0

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip()

    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[+-]?\d+\.0+", raw):
        return int(float(raw))

    raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")


IDLE_STATUS_GRACE_SECONDS = max(1.0, _env_float("SWARM_IDLE_STATUS_GRACE_SECONDS", 8.0))
ARTIFACT_PATH_RE = re.compile(
    r"(?:/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.[A-Za-z0-9._-]{1,16}|(?:\./|\.\./)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\.[A-Za-z0-9._-]{1,16})"
)


def _extract_text_parts(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts = payload.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        text = part.get("text")
        if not isinstance(text, str):
            continue
        token = text.strip()
        if not token:
            continue
        if part_type in {"text", "reasoning"}:
            chunks.append(token)
    return "\n".join(chunks).strip()


@dataclass
class WorkerRoleSpec:
    name: str
    system_prompt: str
    keywords: list[str] = field(default_factory=list)


def _normalize_keyword_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        items = re.split(r"[,;\n]+", raw)
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        token = item.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return result


def _default_worker_roles(mission: str) -> list[WorkerRoleSpec]:
    mission_hint = mission.strip() or "swarm mission"
    return [
        WorkerRoleSpec(
            name="Scout Analyst",
            system_prompt=(
                "You are the scout analyst in this swarm. "
                "Focus on project understanding, dependency/context discovery, and risk identification. "
                f"Mission: {mission_hint}."
            ),
            keywords=[
                "collect",
                "survey",
                "inventory",
                "analyze",
                "dependency",
                "architecture",
                "risk",
                "readme",
                "收集",
                "调研",
                "依赖",
                "结构",
                "风险",
            ],
        ),
        WorkerRoleSpec(
            name="Implementation Engineer",
            system_prompt=(
                "You are the implementation engineer in this swarm. "
                "Focus on direct code changes, bug fixes, and feature delivery. "
                f"Mission: {mission_hint}."
            ),
            keywords=[
                "implement",
                "feature",
                "fix",
                "refactor",
                "code",
                "api",
                "实现",
                "修复",
                "开发",
                "编码",
            ],
        ),
        WorkerRoleSpec(
            name="Verification Engineer",
            system_prompt=(
                "You are the verification engineer in this swarm. "
                "Focus on tests, validation, and regression safety. "
                f"Mission: {mission_hint}."
            ),
            keywords=[
                "test",
                "verify",
                "validation",
                "regression",
                "qa",
                "lint",
                "coverage",
                "测试",
                "验证",
                "回归",
                "质量",
            ],
        ),
        WorkerRoleSpec(
            name="Documentation Synthesizer",
            system_prompt=(
                "You are the documentation synthesizer in this swarm. "
                "Focus on summaries, reports, and clear technical documentation. "
                f"Mission: {mission_hint}."
            ),
            keywords=[
                "document",
                "summary",
                "report",
                "guide",
                "readme",
                "文档",
                "总结",
                "报告",
                "说明",
                "结论",
            ],
        ),
    ]


def _auto_generate_worker_roles(worker_count: int, mission: str) -> list[WorkerRoleSpec]:
    defaults = _default_worker_roles(mission)
    roles: list[WorkerRoleSpec] = []
    for idx in range(worker_count):
        if idx < len(defaults):
            roles.append(defaults[idx])
            continue
        lane = idx + 1
        roles.append(
            WorkerRoleSpec(
                name=f"Generalist Worker {lane}",
                system_prompt=(
                    "You are a generalist worker in this swarm. "
                    "Handle tasks that do not strongly match a specialist role. "
                    f"Mission: {mission.strip() or 'swarm mission'}."
                ),
                keywords=["general", "support", "misc", "通用", "辅助", "支撑"],
            )
        )
    return roles


def _parse_worker_roles_json(raw: str) -> tuple[list[WorkerRoleSpec], str | None]:
    text = (raw or "").strip()
    if not text:
        return [], None
    try:
        payload = json.loads(text)
    except Exception as exc:
        return [], f"Invalid worker_roles_json: {exc}"
    if not isinstance(payload, list):
        return [], "worker_roles_json must be a JSON array."

    roles: list[WorkerRoleSpec] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"worker-role-{idx}")
        prompt = str(item.get("system_prompt") or item.get("prompt") or "").strip()
        if not prompt:
            prompt = f"You are {name}. Focus on tasks aligned with your specialization."
        keywords = _normalize_keyword_list(item.get("keywords") or item.get("tags") or [])
        roles.append(WorkerRoleSpec(name=name, system_prompt=prompt, keywords=keywords))
    return roles, None


def _as_plan_markdown(data: dict[str, Any]) -> str:
    return "```json\n" + _safe_json_dumps(data) + "\n```\n"


def _load_json_candidates(payload: str) -> Any:
    text = payload.strip()
    if not text:
        return {}
    candidates = [text]
    no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", text)
    if no_trailing_commas != text:
        candidates.append(no_trailing_commas)
    errors: list[str] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{exc.msg} at line {exc.lineno} col {exc.colno}")
    raise ValueError("; ".join(errors[-2:]))


def _parse_plan_text(raw: str) -> Any:
    if not raw.strip():
        return {}
    errors: list[str] = []

    blocks = re.findall(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", raw, flags=re.MULTILINE)
    for reverse_idx, block in enumerate(reversed(blocks)):
        idx = len(blocks) - 1 - reverse_idx
        try:
            return _load_json_candidates(block)
        except Exception as exc:
            errors.append(f"fenced[{idx}] {exc}")

    try:
        return _load_json_candidates(raw)
    except Exception as exc:
        errors.append(f"full_raw {exc}")

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _load_json_candidates(raw[start : end + 1])
        except Exception as exc:
            errors.append(f"brace_extract {exc}")

    raise ValueError("Cannot parse plan text: " + " | ".join(errors[-3:]))


def _tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return "<missing>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return "<unreadable>"


def _extract_human_error(raw: str, max_len: int = 280) -> str:
    if not raw:
        return ""

    direct_patterns = [
        r"Cannot find module[^\n\r<]+",
        r"EPERM:[^\n\r<]+",
        r"EACCES:[^\n\r<]+",
        r"ENOENT:[^\n\r<]+",
        r"Error:\s*[^\n\r<]+",
    ]
    for pattern in direct_patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()[:max_len]

    bun_payload = re.search(
        r'<script id="__bunfallback"[^>]*>\s*([A-Za-z0-9+/=\s]+)',
        raw,
        flags=re.IGNORECASE,
    )
    if bun_payload:
        b64 = re.sub(r"\s+", "", bun_payload.group(1))
        try:
            decoded = base64.b64decode(b64 + "===", validate=False).decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        if decoded:
            cleaned = re.sub(r"[^\x20-\x7E\n\r\t]+", " ", decoded)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            for pattern in [
                r"Cannot find module[^\n\r]+",
                r"POST\s+-\s+http[^\n\r]+failed",
                r"Error:\s*[^\n\r]+",
                r"EPERM:[^\n\r]+",
                r"EACCES:[^\n\r]+",
            ]:
                m = re.search(pattern, cleaned, flags=re.IGNORECASE)
                if m:
                    return m.group(0).strip()[:max_len]
            if cleaned:
                return cleaned[:max_len]

    fallback = re.sub(r"\s+", " ", raw).strip()
    return fallback[:max_len]


def _generate_server_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "swarm-" + "".join(random.choice(alphabet) for _ in range(length))


def _port_is_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _sanitize_session_id(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip()).strip("-")
    return value[:96] if value else "default"


def _default_session_id() -> str:
    return _sanitize_session_id(f"swarm-{int(_now())}-{random.randint(1000, 9999)}")


def _current_session_path(bb_root: Path) -> Path:
    return bb_root / "current_session"


def _read_current_session(bb_root: Path) -> str | None:
    selector = _current_session_path(bb_root)
    if not selector.exists():
        return None
    try:
        value = selector.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return _sanitize_session_id(value) if value else None


def _write_current_session(bb_root: Path, session_id: str) -> None:
    bb_root.mkdir(parents=True, exist_ok=True)
    selector = _current_session_path(bb_root)
    tmp = selector.with_suffix(".tmp")
    tmp.write_text(session_id + "\n", encoding="utf-8")
    tmp.replace(selector)


def _resolve_state_path(project_dir: Path, session_id: str | None) -> tuple[Path | None, str | None]:
    bb_root = project_dir / ".blackboard"
    resolved = _sanitize_session_id(session_id) if session_id else _read_current_session(bb_root)
    if resolved:
        path = bb_root / "sessions" / resolved / "global_indices" / "orchestrator_state.json"
        return (path if path.exists() else None), resolved

    candidates = sorted(
        (bb_root / "sessions").glob("*/global_indices/orchestrator_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if (bb_root / "sessions").exists() else []
    if not candidates:
        return None, None
    latest = candidates[0]
    return latest, latest.parents[1].name


@dataclass
class Worker:
    name: str
    port: int
    role_name: str
    role_system_prompt: str
    role_keywords: list[str]
    project_dir: Path
    mcp_script_path: Path
    plugin_path: Path
    server_password: str
    log_dir: Path
    run_id: str
    mission_id: str
    log_config_level: str
    swarm_session_id: str

    proc: subprocess.Popen | None = None
    session_id: str | None = None
    status: str = "starting"  # starting | idle | busy | dead
    current_task: str | None = None
    claimed_at: float | None = None
    last_heartbeat: float = field(default_factory=_now)
    restart_count: int = 0
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    stdout_stream: Any | None = None
    stderr_stream: Any | None = None

    def __post_init__(self) -> None:
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=30.0)
        self.dispatch_logger = JsonlLogger(
            self.log_dir / "workers" / f"{self.name}.dispatch.jsonl",
            component="worker",
            run_id=self.run_id,
            mission_id=self.mission_id,
            config=read_logging_config(self.log_config_level),
        )

    def _auth_headers(self) -> dict[str, str]:
        token = base64.b64encode(f"opencode:{self.server_password}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def _worker_config(self) -> dict[str, Any]:
        plugin_url = self.plugin_path.resolve().as_uri()
        config: dict[str, Any] = {
            "plugin": [plugin_url],
            "mcp": {
                "agent_team_blackboard": {
                    "type": "local",
                    "command": [
                        "python3",
                        str(self.mcp_script_path.resolve()),
                        "--project-dir",
                        str(self.project_dir.resolve()),
                    ],
                    "environment": {
                        "AGENT_NAME": self.name,
                        "SWARM_SESSION_ID": self.swarm_session_id,
                        "SWARM_WORKER_ROLE": self.role_name,
                        "SWARM_WORKER_ROLE_KEYWORDS": ",".join(self.role_keywords),
                    },
                    "enabled": True,
                }
            },
        }
        # Inject model for headless operation (no UI to select model interactively).
        worker_model = os.environ.get("SWARM_WORKER_MODEL", "").strip()
        if worker_model:
            config["model"] = worker_model
        return config

    async def start(self) -> None:
        ensure = self.log_dir / "workers"
        ensure.mkdir(parents=True, exist_ok=True)
        self.stdout_path = ensure / f"{self.name}.stdout.log"
        self.stderr_path = ensure / f"{self.name}.stderr.log"

        # Recovery path may call start() multiple times; close stale handles first.
        for attr in ("stdout_stream", "stderr_stream"):
            stream = getattr(self, attr, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
            setattr(self, attr, None)

        self.stdout_stream = self.stdout_path.open("ab")
        self.stderr_stream = self.stderr_path.open("ab")

        env = {
            **os.environ,
            "OPENCODE_SERVER_PASSWORD": self.server_password,
            "OPENCODE_SERVER_USERNAME": "opencode",
            "OPENCODE_CONFIG_CONTENT": json.dumps(self._worker_config(), ensure_ascii=False),
            "SWARM_DEBUG_LOG_LEVEL": self.log_config_level,
        }

        self.dispatch_logger.event(
            "info",
            "worker.start.begin",
            worker=self.name,
            status="starting",
            extra={
                "port": self.port,
                "project_dir": str(self.project_dir),
                "plugin_path": str(self.plugin_path),
                "mcp_script": str(self.mcp_script_path),
                "swarm_session_id": self.swarm_session_id,
            },
        )

        self.proc = subprocess.Popen(
            ["opencode", "serve", "--port", str(self.port), "--hostname", "127.0.0.1"],
            cwd=str(self.project_dir),
            env=env,
            stdout=self.stdout_stream,
            stderr=self.stderr_stream,
            start_new_session=True,
        )

        health_error = ""
        for _ in range(60):
            if self.proc.poll() is not None:
                stderr_tail = _tail_text(self.stderr_path, 80) if self.stderr_path else "<missing>"
                raise RuntimeError(
                    f"{self.name} exited early with code {self.proc.returncode}; stderr_tail={stderr_tail}"
                )
            try:
                resp = await self.client.get("/global/health", headers=self._auth_headers())
                if resp.status_code == 200:
                    break
            except Exception as exc:
                health_error = str(exc)
            await asyncio.sleep(1)
        else:
            stderr_tail = _tail_text(self.stderr_path, 120) if self.stderr_path else "<missing>"
            raise RuntimeError(
                f"{self.name} failed health check; last_error={health_error or 'none'}; stderr_tail={stderr_tail}"
            )

        resp = await self.client.post(
            "/session",
            headers=self._auth_headers(),
            json={"title": self.name},
        )
        if resp.status_code >= 400:
            body_raw = resp.text
            body = body_raw.strip().replace("\n", " ")
            decoded_error = _extract_human_error(body_raw)
            stderr_tail = _tail_text(self.stderr_path, 120) if self.stderr_path else "<missing>"
            merged_error = body[:600]
            if decoded_error and decoded_error not in merged_error:
                merged_error = f"{merged_error} | decoded={decoded_error}"
            self.dispatch_logger.event(
                "error",
                "worker.session.create.error",
                worker=self.name,
                status="failed",
                error_code=f"HTTP_{resp.status_code}",
                error=merged_error,
                extra={
                    "stderr_tail": stderr_tail,
                    "decoded_error": decoded_error,
                },
            )
            raise RuntimeError(
                f"{self.name} /session failed status={resp.status_code}; body={body[:300]}; "
                f"decoded={decoded_error[:180] if decoded_error else '<none>'}; stderr_tail={stderr_tail}"
            )

        try:
            self.session_id = resp.json()["id"]
        except Exception as exc:
            body = resp.text.strip().replace("\n", " ")
            raise RuntimeError(f"{self.name} /session parse failed: {exc}; body={body[:400]}") from exc
        self.status = "idle"
        self.current_task = None
        self.claimed_at = None
        self.last_heartbeat = _now()
        self.dispatch_logger.event(
            "info",
            "worker.start.ready",
            worker=self.name,
            session_id=self.session_id,
            status="ok",
        )

    async def send_task(self, task_id: str, prompt: str) -> None:
        if not self.session_id:
            raise RuntimeError(f"{self.name} has no session")
        t0 = _now()
        self.dispatch_logger.event("info", "task.dispatched", task_id=task_id, worker=self.name)
        resp = await self.client.post(
            f"/session/{self.session_id}/prompt_async",
            headers=self._auth_headers(),
            json={"parts": [{"type": "text", "text": prompt}]},
        )
        resp.raise_for_status()
        self.dispatch_logger.event(
            "info",
            "task.dispatched.ok",
            task_id=task_id,
            worker=self.name,
            duration_ms=int((_now() - t0) * 1000),
            status="ok",
        )
        self.status = "busy"
        self.current_task = task_id
        self.claimed_at = _now()

    async def prompt_sync(self, prompt: str, system_prompt: str = "") -> str:
        if not self.session_id:
            raise RuntimeError(f"{self.name} has no session")
        payload: dict[str, Any] = {"parts": [{"type": "text", "text": prompt}]}
        if system_prompt.strip():
            payload["system"] = system_prompt.strip()
        resp = await self.client.post(
            f"/session/{self.session_id}/message",
            headers=self._auth_headers(),
            json=payload,
        )
        resp.raise_for_status()
        return _extract_text_parts(resp.json())

    async def check_idle(self) -> bool:
        if not self.session_id:
            return True
        resp = await self.client.get("/session/status", headers=self._auth_headers())
        resp.raise_for_status()
        statuses: dict[str, Any] = resp.json()
        state = statuses.get(self.session_id)
        # opencode may briefly omit active sessions right after prompt_async.
        if state is None:
            if self.current_task and self.claimed_at and (_now() - self.claimed_at) < IDLE_STATUS_GRACE_SECONDS:
                return False
            self.status = "idle"
            self.current_task = None
            self.claimed_at = None
            return True
        if isinstance(state, dict) and state.get("type") == "idle":
            self.status = "idle"
            self.current_task = None
            self.claimed_at = None
            return True
        return False

    async def stop(self, force: bool = False) -> None:
        if self.proc and self.proc.poll() is None:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                self.proc.send_signal(sig)
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.status = "dead"

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass

        for attr in ("stdout_stream", "stderr_stream"):
            stream = getattr(self, attr, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
            setattr(self, attr, None)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_dir = Path(args.project_dir).resolve()
        self.bb_root = self.project_dir / ".blackboard"
        requested_session = args.session_id or os.environ.get("SWARM_SESSION_ID") or _default_session_id()
        self.session_id = _sanitize_session_id(requested_session)
        self.bb = self.bb_root / "sessions" / self.session_id
        self.plan_path = self.bb / "global_indices" / "central_plan.md"
        self.registry_path = self.bb / "global_indices" / "registry.json"
        self.state_path = self.bb / "global_indices" / "orchestrator_state.json"
        self.completion_report_path = self.bb / "global_indices" / "swarm_completion_report.md"

        self.run_id = f"run-{int(_now())}-{random.randint(1000, 9999)}"
        self.mission = args.mission or "swarm-mission"
        self.mission_id = stable_hash(self.mission)

        self.plugin_source_path = Path(args.plugin_path).resolve()
        self.mcp_script_path = Path(args.mcp_script).resolve()
        provided_password = (args.server_password or "").strip()
        self.server_password = provided_password if provided_password and provided_password != "swarm-internal" else _generate_server_password()
        self.runtime_plugin_path = self.bb / "runtime" / "plugins" / "opencode-agent-team.js"
        self._shutdown_started = False

        self.log_cfg = read_logging_config(args.debug_log_level)
        self.logger = JsonlLogger(
            self.bb / "logs" / "orchestrator" / "orchestrator.jsonl",
            component="orchestrator",
            run_id=self.run_id,
            mission_id=self.mission_id,
            config=self.log_cfg,
        )
        self.trace_logger = JsonlLogger(
            self.bb / "logs" / "orchestrator" / "scheduler_trace.jsonl",
            component="orchestrator",
            run_id=self.run_id,
            mission_id=self.mission_id,
            config=self.log_cfg,
        )
        self.worker_roles = self._build_worker_roles()

        self.running = True
        self.workers: list[Worker] = []
        self.assigner: Worker | None = None
        self.no_progress_rounds = 0
        self.last_signature = ""
        self.empty_plan_rounds = 0
        self.empty_plan_threshold = max(3, min(10, int(getattr(args, "no_progress_rounds", 30) or 30) // 3))
        self._last_tick_empty_plan = False
        self.parent_pid = int(getattr(args, "parent_pid", 0) or 0)
        self.parent_grace_seconds = max(3, int(getattr(args, "parent_grace_seconds", 20) or 20))
        self._parent_lost_at: float | None = None

    def _default_plan(self) -> dict[str, Any]:
        now_iso = utc_now_iso()
        return {
            "schema_version": "1.1",
            "mission_goal": self.mission,
            "status": "IN_PROGRESS",
            "summary": None,
            "session_id": self.session_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "tasks": [],
        }

    def _build_worker_roles(self) -> list[WorkerRoleSpec]:
        defaults = _auto_generate_worker_roles(self.args.workers, self.mission)
        raw = (getattr(self.args, "worker_roles_json", "") or "").strip()
        if not raw:
            self.logger.event(
                "info",
                "worker.roles.ready",
                status="ok",
                extra={"source": "auto", "workers": len(defaults)},
            )
            return defaults

        parsed, err = _parse_worker_roles_json(raw)
        if err:
            self.logger.event(
                "warn",
                "worker.roles.parse.error",
                status="failed",
                error=err,
                extra={"source": "worker_roles_json"},
            )
            return defaults
        if not parsed:
            return defaults

        merged: list[WorkerRoleSpec] = []
        for idx in range(self.args.workers):
            if idx < len(parsed):
                merged.append(parsed[idx])
            else:
                merged.append(defaults[idx])
        self.logger.event(
            "info",
            "worker.roles.ready",
            status="ok",
            extra={"source": "json", "configured": len(parsed), "workers": len(merged)},
        )
        return merged

    @staticmethod
    def _normalize_assignee_name(name: str) -> str:
        """Normalize worker names: worker-001 -> worker-0, worker-01 -> worker-0, etc."""
        m = re.match(r"^worker-0*(\d+)$", name)
        if m:
            return f"worker-{int(m.group(1))}"
        return name

    def _normalize_plan(self, raw_plan: Any) -> dict[str, Any]:
        if isinstance(raw_plan, list):
            plan = self._default_plan()
            plan["tasks"] = raw_plan
            plan["updated_at"] = utc_now_iso()
            return plan
        if not isinstance(raw_plan, dict):
            return self._default_plan()

        plan = dict(raw_plan)
        if not isinstance(plan.get("tasks"), list):
            plan["tasks"] = []
        plan.setdefault("schema_version", "1.1")
        plan.setdefault("mission_goal", self.mission)
        plan.setdefault("status", "IN_PROGRESS")
        plan.setdefault("summary", None)
        plan.setdefault("session_id", self.session_id)
        plan.setdefault("created_at", utc_now_iso())
        if plan.get("status") not in {"IN_PROGRESS", "DONE"}:
            plan["status"] = "IN_PROGRESS"
        plan["updated_at"] = utc_now_iso()

        # Normalize assignee names in all tasks (e.g. worker-001 -> worker-0)
        for task in plan.get("tasks", []):
            if not isinstance(task, dict):
                continue
            aw = task.get("assigned_worker")
            if isinstance(aw, str) and aw:
                task["assigned_worker"] = self._normalize_assignee_name(aw)
            assignees = task.get("assignees")
            if isinstance(assignees, list):
                task["assignees"] = [self._normalize_assignee_name(a) for a in assignees if isinstance(a, str)]

        return plan

    def _ensure_plan_template(self) -> None:
        if self.plan_path.exists():
            return
        template = self._default_plan()
        self.plan_path.write_text(_as_plan_markdown(template), encoding="utf-8")
        self.logger.event(
            "info",
            "plan.template.created",
            status="ok",
            session_id=self.session_id,
            extra={"path": str(self.plan_path), "schema_version": template["schema_version"]},
        )

    def _init_layout(self) -> None:
        (self.bb_root / "sessions").mkdir(parents=True, exist_ok=True)
        _write_current_session(self.bb_root, self.session_id)
        for p in [
            self.bb / "global_indices",
            self.bb / "resources",
            self.bb / "inboxes",
            self.bb / "heartbeats",
            self.bb / "logs" / "orchestrator",
            self.bb / "logs" / "workers",
            self.bb / "logs" / "mcp",
            self.bb / "logs" / "incidents",
        ]:
            p.mkdir(parents=True, exist_ok=True)
        # Pre-create scheduler_trace.jsonl so audit tools always find the file.
        (self.bb / "logs" / "orchestrator" / "scheduler_trace.jsonl").touch(exist_ok=True)

    def _migrate_legacy_plan(self) -> None:
        legacy = self.project_dir / "central_plan.md"
        if self.plan_path.exists() or not legacy.exists():
            return
        legacy_store = self.bb_root / "legacy"
        legacy_store.mkdir(parents=True, exist_ok=True)
        archived = legacy_store / f"central_plan.{int(_now())}.md"
        try:
            shutil.move(str(legacy), str(archived))
            raw = archived.read_text(encoding="utf-8", errors="replace")
            normalized = self._normalize_plan(_parse_plan_text(raw))
            self.plan_path.write_text(_as_plan_markdown(normalized), encoding="utf-8")
            self.logger.event(
                "warn",
                "plan.legacy_migrated",
                status="ok",
                session_id=self.session_id,
                extra={
                    "from": str(legacy),
                    "archived": str(archived),
                    "to": str(self.plan_path),
                },
            )
        except Exception as exc:
            self.logger.event(
                "warn",
                "plan.legacy_migrate.error",
                status="failed",
                session_id=self.session_id,
                error=str(exc),
            )

    def _validate_runtime_inputs(self) -> None:
        missing: list[str] = []
        if not self.plugin_source_path.exists():
            missing.append(f"plugin_path missing: {self.plugin_source_path}")
        if not self.mcp_script_path.exists():
            missing.append(f"mcp_script missing: {self.mcp_script_path}")
        if missing:
            message = "; ".join(missing)
            self.logger.event(
                "error",
                "swarm.start.input_error",
                session_id=self.session_id,
                status="failed",
                error_code="STARTUP_INPUT_MISSING",
                error=message,
            )
            raise FileNotFoundError(message)

    def _prepare_runtime_assets(self) -> None:
        self.runtime_plugin_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(self.plugin_source_path, self.runtime_plugin_path)
        except Exception as exc:
            self.logger.event(
                "error",
                "runtime.plugin.prepare.error",
                status="failed",
                session_id=self.session_id,
                error_code="RUNTIME_PLUGIN_PREPARE_FAILED",
                error=str(exc),
                extra={
                    "source": str(self.plugin_source_path),
                    "target": str(self.runtime_plugin_path),
                },
            )
            raise

        self.logger.event(
            "info",
            "runtime.plugin.prepared",
            status="ok",
            session_id=self.session_id,
            extra={
                "source": str(self.plugin_source_path),
                "target": str(self.runtime_plugin_path),
            },
        )

    def _write_state(self, status: str = "running") -> None:
        data = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "mission": self.mission,
            "mission_id": self.mission_id,
            "status": status,
            "updated_at": utc_now_iso(),
            "orchestrator_pid": os.getpid(),
            "workers": [
                {
                    "name": w.name,
                    "role": w.role_name,
                    "pid": w.proc.pid if w.proc else None,
                    "port": w.port,
                    "status": w.status,
                    "session_id": w.session_id,
                    "current_task": w.current_task,
                    "restart_count": w.restart_count,
                }
                for w in self.workers
            ],
            "assigner": (
                {
                    "name": self.assigner.name,
                    "role": self.assigner.role_name,
                    "pid": self.assigner.proc.pid if self.assigner.proc else None,
                    "port": self.assigner.port,
                    "status": self.assigner.status,
                    "session_id": self.assigner.session_id,
                    "current_task": self.assigner.current_task,
                    "restart_count": self.assigner.restart_count,
                }
                if self.assigner is not None
                else None
            ),
            "config": {
                "workers": self.args.workers,
                "port_start": self.args.port_start,
                "heartbeat_timeout": self.args.heartbeat_timeout,
                "task_timeout": self.args.task_timeout,
                "max_retries": self.args.max_retries,
                "poll_interval": self.args.poll_interval,
                "parent_pid": self.parent_pid,
                "parent_grace_seconds": self.parent_grace_seconds,
                "plugin_path": str(self.runtime_plugin_path.resolve()),
                "plugin_source_path": str(self.plugin_source_path),
                "mcp_script": str(self.mcp_script_path),
                "server_password_hash": stable_hash(self.server_password),
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(_safe_json_dumps(data), encoding="utf-8")

    def _should_shutdown_for_parent_loss(self) -> bool:
        if self.parent_pid <= 1:
            return False
        if _pid_alive(self.parent_pid):
            if self._parent_lost_at is not None:
                self.logger.event(
                    "info",
                    "parent.watchdog.recovered",
                    status="ok",
                    session_id=self.session_id,
                    extra={"parent_pid": self.parent_pid},
                )
            self._parent_lost_at = None
            return False

        now = _now()
        if self._parent_lost_at is None:
            self._parent_lost_at = now
            self.logger.event(
                "warn",
                "parent.watchdog.lost",
                status="degraded",
                session_id=self.session_id,
                extra={"parent_pid": self.parent_pid, "grace_seconds": self.parent_grace_seconds},
            )
            return False

        elapsed = now - self._parent_lost_at
        if elapsed < self.parent_grace_seconds:
            return False

        self.logger.event(
            "error",
            "parent.watchdog.shutdown",
            status="failed",
            session_id=self.session_id,
            error_code="PARENT_PROCESS_EXITED",
            error=f"Parent pid {self.parent_pid} exited; lost_for={int(elapsed)}s",
            extra={"parent_pid": self.parent_pid, "grace_seconds": self.parent_grace_seconds},
        )
        self._emit_incident(
            "parent_process_lost",
            {
                "parent_pid": self.parent_pid,
                "lost_for_seconds": int(elapsed),
                "grace_seconds": self.parent_grace_seconds,
            },
        )
        return True

    def _write_registry(self, status: str = "running") -> None:
        registry = {
            "status": status,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "mission_id": self.mission_id,
            "updated_at": utc_now_iso(),
            "workers": [
                {
                    "name": w.name,
                    "role": w.role_name,
                    "port": w.port,
                    "status": w.status,
                    "current_task": w.current_task,
                    "session_id": w.session_id,
                }
                for w in self.workers
            ],
            "assigner": (
                {
                    "name": self.assigner.name,
                    "role": self.assigner.role_name,
                    "port": self.assigner.port,
                    "status": self.assigner.status,
                    "current_task": self.assigner.current_task,
                    "session_id": self.assigner.session_id,
                }
                if self.assigner is not None
                else None
            ),
        }
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(_safe_json_dumps(registry), encoding="utf-8")

    def _load_plan(self) -> dict[str, Any] | None:
        if not self.plan_path.exists():
            return None
        text = self.plan_path.read_text(encoding="utf-8", errors="replace")
        try:
            return self._normalize_plan(_parse_plan_text(text))
        except Exception as exc:
            self.logger.event(
                "warn",
                "plan.parse.error",
                status="failed",
                session_id=self.session_id,
                error=str(exc),
            )
            return None

    def _write_plan(self, data: dict[str, Any]) -> None:
        content = _as_plan_markdown(self._normalize_plan(data))
        if not self.plan_path.exists():
            self.plan_path.write_text(_as_plan_markdown(self._default_plan()), encoding="utf-8")
        with self.plan_path.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                f.write(content)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _with_locked_plan(self, updater):
        if not self.plan_path.exists():
            self.plan_path.write_text(_as_plan_markdown(self._default_plan()), encoding="utf-8")
        with self.plan_path.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read()
                try:
                    parsed = _parse_plan_text(raw)
                except Exception as exc:
                    self.logger.event(
                        "warn",
                        "plan.parse.recovered",
                        status="degraded",
                        session_id=self.session_id,
                        error=str(exc),
                    )
                    parsed = {}
                data = self._normalize_plan(parsed)
                changed, result = updater(data)
                if changed:
                    f.seek(0)
                    f.write(_as_plan_markdown(self._normalize_plan(data)))
                    f.truncate()
                return data, result
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _dependencies_met(self, task: dict[str, Any], tasks: list[dict[str, Any]]) -> bool:
        deps = task.get("dependencies") or task.get("depends_on") or []
        done = {t.get("id") for t in tasks if t.get("status") == "DONE"}
        return all(dep in done for dep in deps)

    def _unblock_tasks(self, plan: dict[str, Any]) -> bool:
        changed = False
        tasks = plan.get("tasks", [])
        for task in tasks:
            if task.get("status") == "BLOCKED" and self._dependencies_met(task, tasks):
                task["status"] = "PENDING"
                changed = True
                self.logger.event("info", "task.unblocked", task_id=task.get("id"), status="ok")
        return changed

    def _retry_failed_tasks(self, plan: dict[str, Any]) -> bool:
        changed = False
        tasks = plan.get("tasks", [])
        for task in tasks:
            if task.get("status") != "FAILED":
                continue
            if not self._dependencies_met(task, tasks):
                continue

            retries = int(task.get("retry_count", 0) or 0)
            if retries >= self.args.max_retries:
                continue

            reason = str(task.get("result_summary") or task.get("result") or task.get("last_error") or "task_failed")
            task["retry_count"] = retries + 1
            task["last_error"] = reason
            task["status"] = "PENDING"
            task["assigned_worker"] = None
            task["assignees"] = []
            task["updated_at"] = _now()
            changed = True
            self.logger.event(
                "warn",
                "task.retry.scheduled",
                task_id=task.get("id"),
                status="retry",
                error=reason,
                extra={
                    "retry_count": task.get("retry_count"),
                    "max_retries": self.args.max_retries,
                },
            )
        return changed

    def _claim_task(self, task_id: str, worker_name: str) -> dict[str, Any] | None:
        def updater(plan: dict[str, Any]):
            tasks = plan.get("tasks", [])
            for task in tasks:
                if task.get("id") != task_id:
                    continue
                if task.get("status") != "PENDING":
                    return False, None
                task["status"] = "IN_PROGRESS"
                task["assigned_worker"] = worker_name
                assignees = task.get("assignees")
                if not isinstance(assignees, list):
                    assignees = []
                # Clean up non-standard assignee names from Architect
                assignees = [self._normalize_assignee_name(a) for a in assignees if isinstance(a, str)]
                if worker_name not in assignees:
                    assignees.append(worker_name)
                task["assignees"] = assignees
                task["claimed_at"] = _now()
                task["start_time"] = task.get("start_time") or utc_now_iso()
                return True, task
            return False, None

        _, task = self._with_locked_plan(updater)
        if task:
            self.logger.event(
                "info",
                "task.claimed",
                task_id=task_id,
                worker=worker_name,
                status="ok",
                extra={
                    "description": task.get("description", "")[:200],
                    "dependencies": task.get("dependencies", []),
                    "claimed_at": task.get("claimed_at"),
                },
            )
        else:
            self.logger.event(
                "warn",
                "task.claim_failed",
                task_id=task_id,
                worker=worker_name,
                status="failed",
                extra={"reason": "task_not_pending_or_not_found"},
            )
        return task

    def _task_status(self, task_id: str) -> str | None:
        plan = self._load_plan()
        if not plan:
            return None
        for t in plan.get("tasks", []):
            if t.get("id") == task_id:
                return t.get("status")
        return None

    def _find_in_progress_task_for_worker(self, worker_name: str) -> str | None:
        plan = self._load_plan()
        if not plan:
            return None
        for task in plan.get("tasks", []):
            if not isinstance(task, dict):
                continue
            if task.get("status") != "IN_PROGRESS":
                continue
            if task.get("assigned_worker") != worker_name:
                continue
            task_id = task.get("id")
            if isinstance(task_id, str) and task_id:
                return task_id
        return None

    def _release_task(self, task_id: str, reason: str, fail: bool = False) -> None:
        def updater(plan: dict[str, Any]):
            tasks = plan.get("tasks", [])
            for task in tasks:
                if task.get("id") != task_id:
                    continue
                retries = int(task.get("retry_count", 0))
                if fail and retries >= self.args.max_retries:
                    task["status"] = "FAILED"
                    task["last_error"] = reason
                    task["result_summary"] = reason
                    task["end_time"] = utc_now_iso()
                    return True, "failed"
                if fail:
                    task["retry_count"] = retries + 1
                    task["last_error"] = reason
                    task["status"] = "PENDING"
                    task["assigned_worker"] = None
                    task["assignees"] = []
                    return True, "retry"
                task["status"] = "PENDING"
                task["assigned_worker"] = None
                task["assignees"] = []
                task["last_error"] = reason
                return True, "released"
            return False, "missing"

        _, result = self._with_locked_plan(updater)
        self.logger.event(
            "warn" if fail else "info",
            "task.released",
            task_id=task_id,
            status=result,
            error=reason,
            extra={
                "fail": fail,
                "action": result,
            },
        )

    def _format_task_prompt(
        self,
        task: dict[str, Any],
        worker: Worker,
        all_tasks: list[dict[str, Any]] | None = None,
    ) -> str:
        task_id = str(task.get("id") or "task-unknown")
        task_file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", task_id).strip("-") or "task-unknown"
        detail_report_path = str((self.bb / "resources" / f"{task_file_id}.md").resolve())
        title = task.get("title") or task.get("description") or ""
        role_prompt = (worker.role_system_prompt or "").strip()
        if not role_prompt:
            role_prompt = f"You are {worker.role_name}. Prioritize tasks aligned to this role."

        # Build predecessor context for tasks that have completed dependencies.
        predecessor_block = ""
        deps = task.get("dependencies") or task.get("depends_on") or []
        if deps and isinstance(deps, list) and all_tasks:
            done_by_id = {
                str(t.get("id") or ""): t
                for t in all_tasks
                if isinstance(t, dict) and str(t.get("status") or "") == "DONE"
            }
            lines: list[str] = []
            for dep_id in deps:
                dep_id_str = str(dep_id or "").strip()
                dep_task = done_by_id.get(dep_id_str)
                if not dep_task:
                    continue
                summary = str(dep_task.get("result_summary") or dep_task.get("result") or "").strip()
                artifact = str(dep_task.get("artifact_link") or "").strip()
                dep_title = str(dep_task.get("title") or dep_id_str)
                line = f"- {dep_id_str} ({dep_title})"
                if summary:
                    line += f"\n  summary: {summary[:300]}"
                if artifact:
                    line += f"\n  artifact: {artifact}"
                lines.append(line)
            if lines:
                predecessor_block = (
                    "Predecessor Results (outputs from completed dependencies — build on these):\n"
                    + "\n".join(lines)
                    + "\n\n"
                )

        return (
            "You are a worker agent in a multi-agent swarm.\n\n"
            f"System role for {worker.name}: {worker.role_name}\n"
            f"Role instructions: {role_prompt}\n\n"
            f"Task ID: {task_id}\n"
            f"Title: {title}\n\n"
            f"Description:\n{task.get('description', '')}\n\n"
            f"{predecessor_block}"
            f"Detailed report path (absolute, required on DONE):\n{detail_report_path}\n\n"
            "Execution requirements:\n"
            "1. Complete the task by editing project files as needed.\n"
            "2. Before marking DONE, write a detailed execution report to the detailed report path.\n"
            "3. The detailed report must include: what changed, why, verification steps, and artifact paths (absolute paths if available).\n"
            "4. When done, call blackboard_update_task(task_id, new_status='DONE', result='concise summary + detailed report path').\n"
            "5. Keep central_plan result concise; put full details only in the report file.\n"
            "6. If blocked/failed, call blackboard_update_task(..., new_status='FAILED', result='reason + relevant paths').\n"
            "7. Check read_inbox() at task start for additional instructions.\n"
        )

    def _task_text(self, task: dict[str, Any]) -> str:
        title = str(task.get("title") or "")
        desc = str(task.get("description") or "")
        return f"{title}\n{desc}".lower()

    def _heuristic_score(self, task: dict[str, Any], worker: Worker) -> int:
        score = 0
        text = self._task_text(task)

        assigned_worker = str(task.get("assigned_worker") or "")
        if assigned_worker == worker.name:
            score += 200

        assignees = task.get("assignees")
        if isinstance(assignees, list):
            if worker.name in assignees:
                score += 150
            elif assignees:
                score -= 30

        role_name_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", worker.role_name.lower())
        for tok in role_name_tokens:
            if tok in text:
                score += 10
        for kw in worker.role_keywords:
            keyword = kw.strip().lower()
            if keyword and keyword in text:
                score += 25
        return score

    def _heuristic_pick_worker_for_task(self, task: dict[str, Any], candidates: list[Worker]) -> tuple[Worker | None, str]:
        if not candidates:
            return None, "no_idle_candidates"
        best_worker: Worker | None = None
        best_score: int | None = None
        for worker in candidates:
            score = self._heuristic_score(task, worker)
            if best_score is None or score > best_score:
                best_score = score
                best_worker = worker
        return best_worker, f"heuristic:{best_score if best_score is not None else 0}"

    async def _start_assigner(self) -> Worker | None:
        used_ports = {w.port for w in self.workers}
        if self.assigner and self.assigner.port:
            used_ports.add(self.assigner.port)
        preferred_port = self.args.port_start + self.args.workers
        allocated_port = self._allocate_worker_port(preferred_port, used_ports)
        assigner = Worker(
            name="orchestrator-assigner",
            port=allocated_port,
            role_name="Orchestrator Assigner",
            role_system_prompt=(
                "You are the orchestrator assigner. "
                "Given task information and candidate workers, pick one best worker name only."
            ),
            role_keywords=["assign", "dispatch", "schedule"],
            project_dir=self.project_dir,
            mcp_script_path=self.mcp_script_path,
            plugin_path=self.runtime_plugin_path,
            server_password=self.server_password,
            log_dir=self.bb / "logs",
            run_id=self.run_id,
            mission_id=self.mission_id,
            log_config_level=self.log_cfg.level,
            swarm_session_id=self.session_id,
        )
        self.logger.event(
            "info",
            "assigner.start.begin",
            status="starting",
            extra={"port": assigner.port},
        )
        try:
            await assigner.start()
        except Exception as exc:
            self.logger.event(
                "warn",
                "assigner.start.error",
                status="failed",
                error_code="ASSIGNER_START_FAILED",
                error=str(exc),
                extra={"traceback": traceback.format_exc(limit=8), "port": assigner.port},
            )
            try:
                await assigner.stop(force=True)
            except Exception:
                pass
            try:
                await assigner.close()
            except Exception:
                pass
            return None
        self.assigner = assigner
        self.logger.event(
            "info",
            "assigner.start.ready",
            status="ok",
            session_id=assigner.session_id,
            extra={"port": assigner.port},
        )
        return assigner

    async def _ensure_assigner_ready(self) -> Worker | None:
        if self.assigner and self.assigner.alive() and self.assigner.session_id:
            return self.assigner
        if self.assigner is not None:
            try:
                await self.assigner.stop(force=True)
            except Exception:
                pass
            try:
                await self.assigner.close()
            except Exception:
                pass
            self.assigner = None
        return await self._start_assigner()

    async def _llm_pick_worker_for_task(self, task: dict[str, Any], candidates: list[Worker]) -> tuple[Worker | None, str, str]:
        assigner = await self._ensure_assigner_ready()
        if assigner is None:
            return None, "llm_disabled_no_assigner", ""

        worker_lines = []
        for worker in candidates:
            worker_lines.append(
                f"- {worker.name}: role={worker.role_name}; keywords={','.join(worker.role_keywords[:8]) or 'none'}; status=idle"
            )
        prompt = (
            "Select exactly one worker for this task.\n"
            "Return only the worker name, no explanation.\n\n"
            f"Mission: {self.mission}\n"
            f"Task title: {task.get('title') or ''}\n"
            f"Task description: {task.get('description') or ''}\n\n"
            "Idle candidates:\n"
            + "\n".join(worker_lines)
        )
        try:
            text = await assigner.prompt_sync(
                prompt=prompt,
                system_prompt=(
                    "You are a strict scheduling assistant. "
                    "Pick one worker from the provided idle candidates and return only that worker name."
                ),
            )
        except Exception as exc:
            self.logger.event(
                "warn",
                "assigner.prompt.error",
                status="failed",
                error_code="ASSIGNER_PROMPT_FAILED",
                error=str(exc),
                extra={"task_id": task.get("id")},
            )
            try:
                await assigner.stop(force=True)
            except Exception:
                pass
            try:
                await assigner.close()
            except Exception:
                pass
            if self.assigner is assigner:
                self.assigner = None
            return None, "llm_call_error", str(exc)

        for worker in candidates:
            if re.search(rf"\b{re.escape(worker.name)}\b", text):
                return worker, "llm_selected", text
        return None, "llm_no_valid_worker", text

    async def _pick_worker_for_task(self, task: dict[str, Any], candidates: list[Worker]) -> tuple[Worker | None, str]:
        if not candidates:
            return None, "no_idle_candidates"
        llm_worker, llm_status, llm_detail = await self._llm_pick_worker_for_task(task, candidates)
        if llm_worker is not None:
            self.trace_logger.event(
                "debug",
                "scheduler.assign.llm",
                status="ok",
                worker=llm_worker.name,
                task_id=task.get("id"),
                extra={"reason": llm_status, "detail": llm_detail[:200]},
            )
            return llm_worker, llm_status

        heuristic_worker, heuristic_reason = self._heuristic_pick_worker_for_task(task, candidates)
        self.trace_logger.event(
            "debug",
            "scheduler.assign.fallback",
            status="ok",
            worker=heuristic_worker.name if heuristic_worker else None,
            task_id=task.get("id"),
            extra={
                "llm_status": llm_status,
                "llm_detail": llm_detail[:200],
                "fallback_reason": heuristic_reason,
            },
        )
        return heuristic_worker, heuristic_reason

    def _allocate_worker_port(self, desired_port: int, used: set[int]) -> int:
        port = max(1024, desired_port)
        while port <= 65535:
            if port in used:
                port += 1
                continue
            if _port_is_available(port):
                used.add(port)
                if port != desired_port:
                    self.logger.event(
                        "warn",
                        "worker.port.reassigned",
                        status="ok",
                        session_id=self.session_id,
                        extra={"requested": desired_port, "assigned": port},
                    )
                return port
            self.logger.event(
                "warn",
                "worker.port.in_use",
                status="in_use",
                session_id=self.session_id,
                extra={"port": port},
            )
            port += 1
        raise RuntimeError(f"No available worker port from {desired_port} to 65535")

    def _cleanup_stale_workers(self) -> None:
        """Kill leftover opencode serve processes from previous runs."""
        for port in range(self.args.port_start, self.args.port_start + self.args.workers + 5):
            if _port_is_available(port):
                continue
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=5,
                )
                for pid_str in result.stdout.strip().split("\n"):
                    if not pid_str.strip():
                        continue
                    pid = int(pid_str.strip())
                    if pid != os.getpid() and pid != self.parent_pid:
                        os.kill(pid, signal.SIGTERM)
                        self.logger.event(
                            "info",
                            "stale.process.killed",
                            session_id=self.session_id,
                            extra={"port": port, "pid": pid},
                        )
            except Exception:
                pass

    async def _start_workers(self) -> None:
        used_ports: set[int] = set()

        # Phase 1: Build all Worker objects (synchronous) and log spawn.start for each.
        pending: list[Worker] = []
        for i in range(self.args.workers):
            desired_port = self.args.port_start + i
            allocated_port = self._allocate_worker_port(desired_port, used_ports)
            role_spec = self.worker_roles[i] if i < len(self.worker_roles) else WorkerRoleSpec(
                name=f"Generalist Worker {i + 1}",
                system_prompt=f"You are Generalist Worker {i + 1}.",
                keywords=[],
            )
            worker = Worker(
                name=f"worker-{i}",
                port=allocated_port,
                role_name=role_spec.name,
                role_system_prompt=role_spec.system_prompt,
                role_keywords=role_spec.keywords,
                project_dir=self.project_dir,
                mcp_script_path=self.mcp_script_path,
                plugin_path=self.runtime_plugin_path,
                server_password=self.server_password,
                log_dir=self.bb / "logs",
                run_id=self.run_id,
                mission_id=self.mission_id,
                log_config_level=self.log_cfg.level,
                swarm_session_id=self.session_id,
            )
            self.logger.event(
                "info",
                "worker.spawn.start",
                worker=worker.name,
                session_id=self.session_id,
                extra={
                    "port": worker.port,
                    "requested_port": desired_port,
                    "role": worker.role_name,
                    "plugin_path": str(worker.plugin_path),
                },
            )
            pending.append(worker)

        # Phase 2: Start all workers concurrently.
        results = await asyncio.gather(*(w.start() for w in pending), return_exceptions=True)

        # Phase 3: Process results — register successes, collect first failure.
        first_exc: BaseException | None = None
        for worker, result in zip(pending, results):
            if isinstance(result, BaseException):
                if first_exc is None:
                    first_exc = result
                self.logger.event(
                    "error",
                    "worker.spawn.error",
                    worker=worker.name,
                    session_id=self.session_id,
                    status="failed",
                    error_code="WORKER_START_FAILED",
                    error=str(result),
                    extra={
                        "plugin_path": str(worker.plugin_path),
                        "mcp_script": str(worker.mcp_script_path),
                        "port": worker.port,
                    },
                )
                self._emit_incident(
                    "worker_spawn_failed",
                    {
                        "worker": worker.name,
                        "session_id": self.session_id,
                        "error": str(result),
                        "plugin_path": str(worker.plugin_path),
                        "mcp_script": str(worker.mcp_script_path),
                        "port": worker.port,
                    },
                )
            else:
                self.logger.event(
                    "info",
                    "worker.spawn.ready",
                    worker=worker.name,
                    session_id=worker.session_id,
                    status="ok",
                    extra={"port": worker.port},
                )
                self.workers.append(worker)
                inbox = self.bb / "inboxes" / f"{worker.name}.json"
                if not inbox.exists():
                    inbox.write_text("[]", encoding="utf-8")

        if first_exc is not None:
            # Ensure no orphan worker process survives partial startup failures.
            for w in pending:
                try:
                    await w.stop(force=True)
                except Exception:
                    pass
                try:
                    await w.close()
                except Exception:
                    pass
            self.workers.clear()
            raise first_exc

    async def _recover_worker(self, worker: Worker, reason: str) -> None:
        self.logger.event("warn", "worker.recover.start", worker=worker.name, error=reason)
        task_to_release = worker.current_task or self._find_in_progress_task_for_worker(worker.name)
        if task_to_release:
            if worker.current_task is None:
                self.logger.event(
                    "warn",
                    "worker.recover.reconciled_task",
                    worker=worker.name,
                    task_id=task_to_release,
                    status="ok",
                )
            self._release_task(task_to_release, reason=f"worker_recover:{reason}", fail=True)

        await worker.stop(force=True)
        worker.restart_count += 1
        if worker.restart_count > 3:
            self._emit_incident("worker_recovery_exhausted", {"worker": worker.name, "reason": reason})

        try:
            await worker.start()
        except Exception as exc:
            try:
                await worker.stop(force=True)
            except Exception:
                pass
            worker.status = "dead"
            worker.current_task = None
            worker.claimed_at = None
            self.logger.event(
                "error",
                "worker.recover.failed",
                worker=worker.name,
                status="failed",
                error_code="WORKER_RECOVERY_FAILED",
                error=str(exc),
                extra={"traceback": traceback.format_exc(limit=8), "reason": reason},
            )
            self._emit_incident(
                "worker_recovery_failed",
                {"worker": worker.name, "reason": reason, "error": str(exc)},
            )
            return

        self.logger.event(
            "info",
            "worker.recover.done",
            worker=worker.name,
            session_id=worker.session_id,
            status="ok",
            attempt=worker.restart_count,
        )

    def _heartbeat_timeout(self, worker: Worker) -> bool:
        hb_file = self.bb / "heartbeats" / f"{worker.name}.json"
        if not hb_file.exists():
            return False
        try:
            hb = json.loads(hb_file.read_text(encoding="utf-8"))
            ts = float(hb.get("timestamp", 0))
            hb_pid = int(hb.get("pid", 0) or 0)
        except Exception:
            return False
        proc_pid = int(worker.proc.pid) if worker.proc else 0
        # Ignore stale heartbeat files left by older worker processes.
        if hb_pid and proc_pid and hb_pid != proc_pid:
            return False
        if ts <= 0:
            return False
        return (_now() - ts) > self.args.heartbeat_timeout

    def _emit_incident(self, reason: str, details: dict[str, Any] | None = None) -> None:
        ts = int(_now())
        incident = self.bb / "logs" / "incidents" / f"incident-{self.run_id}-{ts}.md"
        details = dict(details or {})

        raw_error = str(details.get("error") or "")
        decoded_error = _extract_human_error(raw_error)
        if decoded_error and decoded_error not in raw_error:
            details["decoded_error"] = decoded_error

        text = [
            f"# Incident {self.run_id}",
            "",
            f"- session_id: {self.session_id}",
            f"- blackboard_session_dir: {self.bb}",
            f"- reason: {reason}",
            f"- at: {utc_now_iso()}",
            f"- details: `{json.dumps(details, ensure_ascii=False)}`",
            "",
            "## orchestrator tail",
            "```jsonl",
            _tail_text(self.bb / "logs" / "orchestrator" / "orchestrator.jsonl", 200),
            "```",
            "",
            "## current plan",
            "```json",
            _safe_json_dumps(self._load_plan() or {}),
            "```",
            "",
            "## registry",
            "```json",
            _tail_text(self.registry_path, 400),
            "```",
            "",
            "## state",
            "```json",
            _tail_text(self.state_path, 400),
            "```",
        ]

        for worker in self.workers:
            err_log = self.bb / "logs" / "workers" / f"{worker.name}.stderr.log"
            out_log = self.bb / "logs" / "workers" / f"{worker.name}.stdout.log"
            dispatch_log = self.bb / "logs" / "workers" / f"{worker.name}.dispatch.jsonl"
            text.extend([
                "",
                f"## {worker.name} dispatch tail",
                "```jsonl",
                _tail_text(dispatch_log, 120),
                "```",
                "",
                f"## {worker.name} stderr tail",
                "```text",
                _tail_text(err_log, 120),
                "```",
                "",
                f"## {worker.name} stdout tail",
                "```text",
                _tail_text(out_log, 120),
                "```",
            ])

        if self.assigner is not None:
            err_log = self.bb / "logs" / "workers" / f"{self.assigner.name}.stderr.log"
            out_log = self.bb / "logs" / "workers" / f"{self.assigner.name}.stdout.log"
            dispatch_log = self.bb / "logs" / "workers" / f"{self.assigner.name}.dispatch.jsonl"
            text.extend([
                "",
                f"## {self.assigner.name} dispatch tail",
                "```jsonl",
                _tail_text(dispatch_log, 120),
                "```",
                "",
                f"## {self.assigner.name} stderr tail",
                "```text",
                _tail_text(err_log, 120),
                "```",
                "",
                f"## {self.assigner.name} stdout tail",
                "```text",
                _tail_text(out_log, 120),
                "```",
            ])

        incident.write_text("\n".join(text), encoding="utf-8")
        self.logger.event("error", "incident.generated", status="ok", error=reason, extra={"path": str(incident)})

    def _summarize_task_result(self, task: dict[str, Any], max_len: int = 220) -> str:
        summary = (
            str(task.get("result_summary") or "").strip()
            or str(task.get("result") or "").strip()
            or str(task.get("description") or "").strip()
            or "No result provided."
        )
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) <= max_len:
            return summary
        return summary[: max(3, max_len - 3)] + "..."

    def _extract_artifact_candidates(self, text: str) -> list[str]:
        if not text:
            return []
        candidates: list[str] = []
        for match in ARTIFACT_PATH_RE.finditer(text):
            token = match.group(0).strip().strip("`'\"()[]{}<>.,;:")
            if not token or "://" in token:
                continue
            candidates.append(token)
        return candidates

    def _collect_artifacts(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for task in tasks:
            task_id = str(task.get("id") or "unknown")
            seeds: list[str] = []

            artifact_link = task.get("artifact_link")
            if isinstance(artifact_link, str) and artifact_link.strip():
                seeds.extend(self._extract_artifact_candidates(artifact_link))
                seeds.append(artifact_link.strip())

            for field in ("result", "result_summary"):
                value = task.get(field)
                if isinstance(value, str) and value.strip():
                    seeds.extend(self._extract_artifact_candidates(value))

            for raw in seeds:
                cleaned = str(raw).strip().strip("`'\"()[]{}<>.,;:")
                if not cleaned or "://" in cleaned:
                    continue
                candidate = Path(cleaned)
                resolved = candidate if candidate.is_absolute() else (self.project_dir / candidate).resolve()

                # If the claimed path doesn't exist, try to find it in common locations
                actual_path = resolved
                if not resolved.exists():
                    # Try searching in session directories
                    filename = resolved.name
                    search_dirs = [
                        self.bb / "resources",
                        self.bb / "global_indices",
                        self.bb / "artifacts",
                    ]
                    for search_dir in search_dirs:
                        candidate_path = search_dir / filename
                        if candidate_path.exists():
                            actual_path = candidate_path
                            break

                normalized = str(actual_path)
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                artifacts.append({
                    "task_id": task_id,
                    "path": normalized,
                    "exists": actual_path.exists(),
                })
        return artifacts

    def _build_mission_summary(self, tasks: list[dict[str, Any]]) -> str:
        highlights = [
            f"{str(task.get('id') or 'task')}: {self._summarize_task_result(task, max_len=160)}"
            for task in tasks
        ]
        if not highlights:
            return f"Mission completed: {self.mission}"
        if len(highlights) <= 2:
            return " | ".join(highlights)
        return " | ".join(highlights[:2]) + f" | +{len(highlights) - 2} more task(s)"

    def _append_inbox_message(self, to: str, content: str, msg_type: str = "message") -> None:
        inbox_path = self.bb / "inboxes" / f"{to}.json"
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        if not inbox_path.exists():
            inbox_path.write_text("[]", encoding="utf-8")

        with inbox_path.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                raw = f.read()
                try:
                    messages = json.loads(raw) if raw.strip() else []
                except Exception:
                    messages = []
                if not isinstance(messages, list):
                    messages = []
                messages.append(
                    {
                        "id": f"orchestrator-{int(_now() * 1000)}",
                        "from": "orchestrator",
                        "to": to,
                        "type": msg_type,
                        "content": content,
                        "timestamp": _now(),
                        "read": False,
                    }
                )
                f.seek(0)
                f.write(json.dumps(messages, ensure_ascii=False, indent=2))
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write_completion_report(self, plan: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
        tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
        done = sum(1 for task in tasks if task.get("status") == "DONE")
        failed = sum(1 for task in tasks if task.get("status") == "FAILED")
        in_progress = sum(1 for task in tasks if task.get("status") == "IN_PROGRESS")
        pending = sum(1 for task in tasks if task.get("status") in {"PENDING", "BLOCKED"})
        artifacts = self._collect_artifacts(tasks)

        lines = [
            "# Swarm Completion Report",
            "",
            f"- session_id: {self.session_id}",
            f"- run_id: {self.run_id}",
            f"- mission: {plan.get('mission_goal') or self.mission}",
            f"- completed_at: {utc_now_iso()}",
            f"- mission_status: {plan.get('status') or 'DONE'}",
            f"- tasks: total={len(tasks)}, done={done}, failed={failed}, in_progress={in_progress}, pending_or_blocked={pending}",
            "",
            "## Task Answers",
        ]

        if tasks:
            for task in tasks:
                task_id = str(task.get("id") or "task")
                status = str(task.get("status") or "UNKNOWN")
                summary = self._summarize_task_result(task)
                lines.append(f"- [{status}] {task_id}: {summary}")
        else:
            lines.append("- (no tasks)")

        lines.extend([
            "",
            "## Output Artifacts",
        ])

        if artifacts:
            for artifact in artifacts:
                state = "exists" if artifact.get("exists") else "missing"
                lines.append(f"- {artifact.get('path')} ({state}; from {artifact.get('task_id')})")
        else:
            lines.append("- (none detected)")

        lines.extend([
            "",
            "## Important Paths",
            f"- completion_report: {self.completion_report_path}",
            f"- central_plan: {self.plan_path}",
            "",
        ])

        self.completion_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.completion_report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.completion_report_path, artifacts

    def _finalize_completion(self) -> str | None:
        def updater(plan: dict[str, Any]):
            tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
            if not tasks:
                return False, False
            if any(task.get("status") != "DONE" for task in tasks):
                return False, False

            changed = False
            if plan.get("status") != "DONE":
                plan["status"] = "DONE"
                changed = True
            if not str(plan.get("summary") or "").strip():
                plan["summary"] = self._build_mission_summary(tasks)
                changed = True
            if changed:
                plan["updated_at"] = utc_now_iso()
            return changed, True

        try:
            plan, completed = self._with_locked_plan(updater)
            if not completed:
                return None

            report_path, artifacts = self._write_completion_report(plan)
            tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
            done = sum(1 for task in tasks if task.get("status") == "DONE")
            failed = sum(1 for task in tasks if task.get("status") == "FAILED")

            task_lines = [
                f"- {str(task.get('id') or 'task')}: {self._summarize_task_result(task, max_len=180)}"
                for task in tasks[:6]
            ]
            if len(tasks) > 6:
                task_lines.append(f"- ... and {len(tasks) - 6} more task(s)")

            artifact_lines = [f"- {artifact.get('path')}" for artifact in artifacts[:10]]
            if len(artifacts) > 10:
                artifact_lines.append(f"- ... and {len(artifacts) - 10} more artifact(s)")

            notice = "\n".join(
                [
                    "Swarm mission completed.",
                    f"Mission: {plan.get('mission_goal') or self.mission}",
                    f"Status: {plan.get('status') or 'DONE'}",
                    f"Tasks: total={len(tasks)}, done={done}, failed={failed}",
                    "Answer summary:",
                    *(task_lines or ["- (none)"]),
                    "Artifacts:",
                    *(artifact_lines or ["- (none detected)"]),
                    f"Completion report: {report_path}",
                ]
            )
            self._append_inbox_message("architect", notice, msg_type="mission_completed")

            # Print completion report to stdout for user visibility
            print("\n" + "=" * 80)
            print(notice)
            print("=" * 80 + "\n")

            self.logger.event(
                "info",
                "swarm.completion.report",
                status="ok",
                session_id=self.session_id,
                extra={"path": str(report_path), "artifacts": len(artifacts)},
            )
            return str(report_path)
        except Exception as exc:
            self.logger.event(
                "warn",
                "swarm.completion.report.error",
                status="failed",
                session_id=self.session_id,
                error=str(exc),
                extra={"traceback": traceback.format_exc(limit=8)},
            )
            return None

    def _schedule_signature(self, plan: dict[str, Any] | None) -> str:
        if not plan:
            return "no_plan"
        compact = [
            {
                "id": t.get("id"),
                "status": t.get("status"),
                "assigned": t.get("assigned_worker"),
                "retry": t.get("retry_count", 0),
            }
            for t in plan.get("tasks", [])
        ]
        return stable_hash(json.dumps(compact, sort_keys=True, ensure_ascii=False))

    async def _tick(self) -> bool:
        self._last_tick_empty_plan = False
        plan = self._load_plan()
        if not plan:
            self.trace_logger.event("debug", "scheduler.tick", status="no_plan")
            return False

        tasks = plan.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []
        if len(tasks) == 0:
            self._last_tick_empty_plan = True
            self.empty_plan_rounds += 1
            if self.empty_plan_rounds == 1:
                self.logger.event(
                    "error",
                    "plan.empty",
                    session_id=self.session_id,
                    status="failed",
                    error_code="EMPTY_TASK_GRAPH",
                    error="central_plan.md has zero tasks; scheduler has nothing to dispatch",
                    extra={
                        "rounds": self.empty_plan_rounds,
                        "threshold": self.empty_plan_threshold,
                        "mission_status": plan.get("status", "IN_PROGRESS"),
                    },
                )
            self.trace_logger.event(
                "warn",
                "scheduler.tick.empty_plan",
                status="empty_plan",
                extra={
                    "rounds": self.empty_plan_rounds,
                    "threshold": self.empty_plan_threshold,
                    "mission_status": plan.get("status", "IN_PROGRESS"),
                },
            )
            if self.empty_plan_rounds >= self.empty_plan_threshold:
                self._emit_incident(
                    "empty_plan_tasks",
                    {
                        "rounds": self.empty_plan_rounds,
                        "threshold": self.empty_plan_threshold,
                        "hint": "Architect must rewrite central_plan.md with non-empty tasks.",
                    },
                )
                self.empty_plan_rounds = 0
            return False

        if self.empty_plan_rounds > 0:
            self.logger.event(
                "info",
                "plan.empty.recovered",
                session_id=self.session_id,
                status="ok",
            )
        self.empty_plan_rounds = 0

        changed = self._unblock_tasks(plan)
        if self._retry_failed_tasks(plan):
            changed = True
        if changed:
            self._write_plan(plan)

        done_or_failed = [t for t in tasks if t.get("status") in {"DONE", "FAILED"}]
        in_progress = [t for t in tasks if t.get("status") == "IN_PROGRESS"]
        pending = [t for t in tasks if t.get("status") == "PENDING" and self._dependencies_met(t, tasks)]

        idle_workers = [w for w in self.workers if w.status == "idle"]
        self.trace_logger.event(
            "debug",
            "scheduler.tick",
            status="ok",
            extra={
                "assignable": len(pending),
                "idle_workers": len(idle_workers),
                "in_progress": len(in_progress),
                "done_or_failed": len(done_or_failed),
            },
        )

        # Dispatch: pick from current idle workers only.
        dispatch_workers = list(idle_workers)
        while pending and dispatch_workers:
            task = pending[0]
            picked_worker, picked_reason = await self._pick_worker_for_task(task, dispatch_workers)
            if picked_worker is None:
                break
            pending.pop(0)
            dispatch_workers = [w for w in dispatch_workers if w.name != picked_worker.name]
            claimed = self._claim_task(task.get("id"), picked_worker.name)
            if not claimed:
                self.logger.event(
                    "warn",
                    "scheduler.assign.claim_failed",
                    worker=picked_worker.name,
                    task_id=task.get("id"),
                    status="failed",
                )
                continue
            self.logger.event(
                "info",
                "scheduler.assign.claimed",
                worker=picked_worker.name,
                task_id=claimed.get("id"),
                status="ok",
                extra={
                    "task_description": claimed.get("description", "")[:200],
                    "dependencies": claimed.get("dependencies", []),
                    "reason": picked_reason,
                    "worker_role": picked_worker.role_name,
                },
            )
            self.trace_logger.event(
                "debug",
                "scheduler.assign.final",
                status="ok",
                worker=picked_worker.name,
                task_id=claimed.get("id"),
                extra={
                    "reason": picked_reason,
                    "worker_role": picked_worker.role_name,
                },
            )
            prompt = self._format_task_prompt(claimed, picked_worker, all_tasks=tasks)
            self.logger.event(
                "debug",
                "scheduler.assign.prompt",
                worker=picked_worker.name,
                task_id=claimed.get("id"),
                extra={
                    "prompt_len": len(prompt),
                    "prompt_preview": prompt[:500] if len(prompt) > 500 else prompt,
                },
            )
            try:
                await picked_worker.send_task(claimed.get("id"), prompt)
                self.logger.event(
                    "info",
                    "scheduler.assign.dispatched",
                    worker=picked_worker.name,
                    task_id=claimed.get("id"),
                    status="ok",
                )
            except Exception as exc:
                self.logger.event(
                    "error",
                    "task.dispatched.error",
                    worker=picked_worker.name,
                    task_id=claimed.get("id"),
                    error_code="DISPATCH_HTTP_ERROR",
                    error=str(exc),
                )
                self._release_task(claimed.get("id"), reason=str(exc), fail=True)
                picked_worker.status = "idle"
                picked_worker.current_task = None
                picked_worker.claimed_at = None

        # Health and completion checks
        for worker in self.workers:
            if not worker.alive():
                self.logger.event(
                    "error",
                    "worker.health.dead",
                    worker=worker.name,
                    status="dead",
                    extra={
                        "current_task": worker.current_task,
                        "pid": worker.proc.pid if worker.proc else None,
                        "port": worker.port,
                        "restart_count": worker.restart_count,
                    },
                )
                await self._recover_worker(worker, reason="process_dead")
                continue

            if self._heartbeat_timeout(worker):
                self.logger.event(
                    "warn",
                    "worker.health.timeout",
                    worker=worker.name,
                    error="heartbeat_timeout",
                    extra={
                        "current_task": worker.current_task,
                        "last_heartbeat_age_s": int(_now() - worker.last_heartbeat),
                        "heartbeat_timeout": self.args.heartbeat_timeout,
                        "port": worker.port,
                    },
                )
                await self._recover_worker(worker, reason="heartbeat_timeout")
                continue

            if worker.status == "busy" and worker.claimed_at and (_now() - worker.claimed_at) > self.args.task_timeout:
                self.logger.event(
                    "warn",
                    "task.timeout",
                    worker=worker.name,
                    task_id=worker.current_task,
                    status="timeout",
                    extra={
                        "elapsed_s": int(_now() - worker.claimed_at),
                        "task_timeout": self.args.task_timeout,
                        "port": worker.port,
                    },
                )
                if worker.current_task:
                    self._release_task(worker.current_task, reason="task_timeout", fail=True)
                await self._recover_worker(worker, reason="task_timeout")
                continue

            if worker.status == "busy":
                # Save before check_idle() clears current_task and claimed_at.
                _previous_task = worker.current_task
                _busy_since = worker.claimed_at
                try:
                    became_idle = await worker.check_idle()
                except Exception as exc:
                    self.logger.event(
                        "warn",
                        "worker.status.error",
                        worker=worker.name,
                        error=str(exc),
                        extra={
                            "current_task": worker.current_task,
                            "port": worker.port,
                        },
                    )
                    await self._recover_worker(worker, reason="status_api_error")
                    continue
                if became_idle:
                    self.logger.event(
                        "info",
                        "worker.idle",
                        worker=worker.name,
                        status="ok",
                        extra={
                            "previous_task": _previous_task,
                            "busy_duration_s": int(_now() - _busy_since) if _busy_since else None,
                        },
                    )

        return True

    def _all_completed(self) -> bool:
        plan = self._load_plan()
        if not plan:
            return False
        tasks = [task for task in plan.get("tasks", []) if isinstance(task, dict)]
        if not tasks:
            return False
        return all(task.get("status") in ("DONE", "FAILED") for task in tasks)

    async def run(self) -> None:
        self._init_layout()
        self.logger.event(
            "info",
            "swarm.start",
            status="ok",
            session_id=self.session_id,
            extra={
                "mission": self.mission,
                "blackboard_session_dir": str(self.bb),
                "parent_pid": self.parent_pid,
                "parent_grace_seconds": self.parent_grace_seconds,
            },
        )
        self._migrate_legacy_plan()
        self._ensure_plan_template()
        self._validate_runtime_inputs()
        self._prepare_runtime_assets()

        signal.signal(signal.SIGTERM, lambda *_: self._request_stop("sigterm"))
        signal.signal(signal.SIGINT, lambda *_: self._request_stop("sigint"))

        force_shutdown = False
        try:
            try:
                self._cleanup_stale_workers()
                await self._start_workers()
            except Exception as exc:
                force_shutdown = True
                self.logger.event(
                    "error",
                    "swarm.start.failed",
                    status="failed",
                    session_id=self.session_id,
                    error_code="SWARM_START_FAILED",
                    error=str(exc),
                    extra={"traceback": traceback.format_exc(limit=8)},
                )
                raise
            await self._ensure_assigner_ready()
            self._write_registry(status="running")
            self._write_state(status="running")

            while self.running:
                if self._should_shutdown_for_parent_loss():
                    force_shutdown = True
                    self.running = False
                    break
                try:
                    await self._tick()
                except Exception as exc:
                    self.logger.event(
                        "error",
                        "scheduler.tick.error",
                        error_code="SCHEDULER_LOOP_ERROR",
                        error=str(exc),
                        extra={"traceback": traceback.format_exc(limit=8)},
                    )

                plan = self._load_plan()
                if self._last_tick_empty_plan:
                    self.no_progress_rounds = 0
                    self.last_signature = self._schedule_signature(plan)
                else:
                    sig = self._schedule_signature(plan)
                    if sig == self.last_signature:
                        self.no_progress_rounds += 1
                    else:
                        self.no_progress_rounds = 0
                        self.last_signature = sig

                if (not self._last_tick_empty_plan) and self.no_progress_rounds >= self.args.no_progress_rounds:
                    self._emit_incident(
                        "no_progress_detected",
                        {
                            "rounds": self.no_progress_rounds,
                            "threshold": self.args.no_progress_rounds,
                        },
                    )
                    self.no_progress_rounds = 0

                    # Recovery: release IN_PROGRESS tasks whose worker is idle/dead/missing/timeout
                    recovery_plan = self._load_plan()
                    if recovery_plan:
                        for task in recovery_plan.get("tasks", []):
                            if task.get("status") != "IN_PROGRESS":
                                continue
                            assigned = task.get("assigned_worker")
                            tid = task.get("id")
                            if not assigned:
                                self._release_task(tid, reason="no_progress_no_assignee")
                                continue
                            worker = next((w for w in self.workers if w.name == assigned), None)
                            if worker is None:
                                self._release_task(tid, reason="no_progress_worker_missing")
                            elif worker.status in ("idle", "dead"):
                                reason = f"no_progress_worker_{worker.status}"
                                self._release_task(tid, reason=reason)
                            elif worker.status == "busy":
                                # Check if busy worker has timed out (no heartbeat)
                                time_since_heartbeat = _now() - worker.last_heartbeat
                                if time_since_heartbeat > self.args.heartbeat_timeout:
                                    self._release_task(tid, reason=f"no_progress_worker_busy_timeout_{int(time_since_heartbeat)}s")


                self._write_registry(status="running")
                self._write_state(status="running")

                if self._all_completed():
                    completion_report = self._finalize_completion()
                    if completion_report:
                        self.logger.event(
                            "info",
                            "swarm.completed",
                            status="ok",
                            session_id=self.session_id,
                            extra={"completion_report": completion_report},
                        )
                    else:
                        self.logger.event("info", "swarm.completed", status="ok", session_id=self.session_id)
                    break

                await asyncio.sleep(self.args.poll_interval)
        except Exception as exc:
            force_shutdown = True
            self.logger.event(
                "error",
                "swarm.run.fatal",
                status="failed",
                session_id=self.session_id,
                error_code="SWARM_RUN_FATAL",
                error=str(exc),
                extra={"traceback": traceback.format_exc(limit=8)},
            )
            raise
        finally:
            await self.shutdown(force=force_shutdown)

    async def shutdown(self, force: bool = False) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True

        self.running = False
        self.logger.event(
            "info",
            "swarm.stop.requested",
            status="ok",
            session_id=self.session_id,
            extra={"force": force},
        )

        survivors: list[Worker] = []
        for worker in list(self.workers):
            try:
                await worker.stop(force=force)
            except Exception as exc:
                self.logger.event(
                    "warn",
                    "worker.stop.error",
                    worker=worker.name,
                    session_id=self.session_id,
                    error=str(exc),
                )
            if worker.alive():
                survivors.append(worker)

        if self.assigner is not None:
            try:
                await self.assigner.stop(force=force)
            except Exception as exc:
                self.logger.event(
                    "warn",
                    "assigner.stop.error",
                    session_id=self.session_id,
                    error=str(exc),
                )
            if self.assigner.alive():
                survivors.append(self.assigner)

        if survivors and not force:
            self.logger.event(
                "warn",
                "swarm.stop.escalate",
                status="force_kill",
                session_id=self.session_id,
                extra={"workers": [w.name for w in survivors]},
            )
            for worker in survivors:
                try:
                    await worker.stop(force=True)
                except Exception as exc:
                    self.logger.event(
                        "warn",
                        "worker.stop.force.error",
                        worker=worker.name,
                        session_id=self.session_id,
                        error=str(exc),
                    )

        for worker in list(self.workers):
            try:
                await worker.close()
            except Exception as exc:
                self.logger.event(
                    "warn",
                    "worker.close.error",
                    worker=worker.name,
                    session_id=self.session_id,
                    error=str(exc),
                )
        self.workers.clear()
        if self.assigner is not None:
            try:
                await self.assigner.close()
            except Exception as exc:
                self.logger.event(
                    "warn",
                    "assigner.close.error",
                    session_id=self.session_id,
                    error=str(exc),
                )
            self.assigner = None

        self._write_registry(status="stopped")
        self._write_state(status="stopped")
        self.logger.event("info", "swarm.stopped", status="ok", session_id=self.session_id)

    def _request_stop(self, reason: str) -> None:
        self.logger.event(
            "info",
            "swarm.stop.signal",
            status="ok",
            session_id=self.session_id,
            extra={"reason": reason},
        )
        self.running = False


async def start_command(args: argparse.Namespace) -> None:
    orchestrator = Orchestrator(args)
    await orchestrator.run()


def stop_command(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    state_path, resolved_session = _resolve_state_path(project_dir, args.session_id)
    if not state_path:
        print("No orchestrator_state.json found; nothing to stop.")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))

    def terminate_pid(pid: int, label: str) -> None:
        if not pid or not _pid_alive(pid):
            print(f"{label} pid={pid} is not running")
            return

        initial_sig = signal.SIGKILL if args.force else signal.SIGTERM
        try:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, initial_sig)
            except Exception:
                os.kill(pid, initial_sig)
            print(f"Sent {initial_sig.name} to {label} pid={pid} session={resolved_session}")
        except ProcessLookupError:
            print(f"{label} pid={pid} already exited")
            return

        if initial_sig == signal.SIGKILL:
            return

        deadline = _now() + 8
        while _now() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.2)

        if _pid_alive(pid):
            try:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    os.kill(pid, signal.SIGKILL)
                print(f"Escalated SIGKILL to {label} pid={pid} session={resolved_session}")
            except ProcessLookupError:
                pass

    orchestrator_pid = state.get("orchestrator_pid")
    if orchestrator_pid:
        terminate_pid(int(orchestrator_pid), "orchestrator")

    for worker in state.get("workers", []):
        pid = worker.get("pid")
        if pid:
            terminate_pid(int(pid), worker.get("name") or "worker")
    assigner = state.get("assigner") if isinstance(state, dict) else None
    if isinstance(assigner, dict):
        pid = assigner.get("pid")
        if pid:
            terminate_pid(int(pid), assigner.get("name") or "orchestrator-assigner")


def status_command(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    state_path, _resolved_session = _resolve_state_path(project_dir, args.session_id)
    if not state_path:
        print("No orchestrator state found.")
        return
    print(state_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    extension_root = script_dir.parent.parent.parent

    parser = argparse.ArgumentParser(description="OpenCode Agent Team Orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start orchestrator")
    start.add_argument("--project-dir", default=".")
    start.add_argument("--mission", default="")
    start.add_argument("--session-id", default="")
    start.add_argument("--workers", type=int, default=3)
    start.add_argument("--worker-roles-json", default=os.environ.get("SWARM_WORKER_ROLES_JSON", ""))
    start.add_argument("--port-start", type=int, default=4401)
    start.add_argument("--heartbeat-timeout", type=int, default=60)
    start.add_argument("--task-timeout", type=int, default=900)
    start.add_argument("--max-retries", type=int, default=2)
    start.add_argument("--poll-interval", type=int, default=3)
    start.add_argument("--no-progress-rounds", type=int, default=30)
    start.add_argument("--parent-pid", type=_parse_parent_pid_arg, default=0)
    start.add_argument("--parent-grace-seconds", type=int, default=20)
    start.add_argument("--server-password", default="")
    start.add_argument(
        "--mcp-script",
        default=str((script_dir / "blackboard_mcp_server.py").resolve()),
    )
    start.add_argument(
        "--plugin-path",
        default=str((extension_root / "plugins" / "opencode-agent-team.js").resolve()),
    )
    start.add_argument("--debug-log-level", default=os.environ.get("SWARM_DEBUG_LOG_LEVEL", "info"))

    stop = sub.add_parser("stop", help="Stop orchestrator")
    stop.add_argument("--project-dir", default=".")
    stop.add_argument("--session-id", default="")
    stop.add_argument("--force", action="store_true")

    status = sub.add_parser("status", help="Print orchestrator state")
    status.add_argument("--project-dir", default=".")
    status.add_argument("--session-id", default="")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "start":
        asyncio.run(start_command(args))
    elif args.command == "stop":
        stop_command(args)
    elif args.command == "status":
        status_command(args)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
