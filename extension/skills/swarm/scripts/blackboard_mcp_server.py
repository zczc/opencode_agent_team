#!/usr/bin/env python3
"""Blackboard MCP server with session isolation and structured debug logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import fcntl

from logging_utils import JsonlLogger, read_logging_config

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "mcp>=1.0 is required. Install with: pip install 'mcp>=1.0.0'"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent team blackboard MCP server")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--session-id", default="")
    return parser.parse_args()


ARGS = parse_args()
AGENT_NAME = os.environ.get("AGENT_NAME", "architect")
PROJECT_DIR = Path(ARGS.project_dir).resolve()
BB_ROOT = PROJECT_DIR / ".blackboard"
HEARTBEAT_INTERVAL_SECONDS = max(1.0, float(os.environ.get("SWARM_HEARTBEAT_INTERVAL_SECONDS", "5")))
_HEARTBEAT_STOP = threading.Event()


def _sanitize_session_id(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip()).strip("-")
    return value[:96] if value else "default"


def _current_session_file() -> Path:
    return BB_ROOT / "current_session"


def _read_current_session() -> str | None:
    selector = _current_session_file()
    if not selector.exists():
        return None
    try:
        value = selector.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return _sanitize_session_id(value) if value else None


def _resolve_session_id() -> str:
    if ARGS.session_id:
        return _sanitize_session_id(ARGS.session_id)
    env = os.environ.get("SWARM_SESSION_ID", "").strip()
    if env:
        return _sanitize_session_id(env)
    selected = _read_current_session()
    if selected:
        return selected
    return "default"


def _session_root(session_id: str) -> Path:
    return BB_ROOT / "sessions" / session_id


def _session_paths(session_id: str) -> dict[str, Path]:
    bb = _session_root(session_id)
    return {
        "bb": bb,
        "global_indices": bb / "global_indices",
        "plan": bb / "global_indices" / "central_plan.md",
        "registry": bb / "global_indices" / "registry.json",
        "state": bb / "global_indices" / "orchestrator_state.json",
        "inboxes": bb / "inboxes",
        "heartbeats": bb / "heartbeats",
        "logs_mcp": bb / "logs" / "mcp",
    }


def _ensure_layout(session_id: str) -> None:
    BB_ROOT.mkdir(parents=True, exist_ok=True)
    paths = _session_paths(session_id)
    for p in [
        paths["global_indices"],
        paths["bb"] / "resources",
        paths["inboxes"],
        paths["heartbeats"],
        paths["logs_mcp"],
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _run_id(session_id: str) -> str:
    state_path = _session_paths(session_id)["state"]
    if not state_path.exists():
        return "run-unknown"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data.get("run_id") or "run-unknown"
    except Exception:
        return "run-unknown"


def _mission_id(session_id: str) -> str:
    state_path = _session_paths(session_id)["state"]
    if not state_path.exists():
        return "mission-unknown"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data.get("mission_id") or "mission-unknown"
    except Exception:
        return "mission-unknown"


_LOGGERS: dict[str, JsonlLogger] = {}


def _get_logger(session_id: str) -> JsonlLogger:
    logger = _LOGGERS.get(session_id)
    if logger is None:
        path = _session_paths(session_id)["logs_mcp"] / f"blackboard-{AGENT_NAME}.jsonl"
        logger = JsonlLogger(
            path,
            component="mcp",
            run_id=_run_id(session_id),
            mission_id=_mission_id(session_id),
            config=read_logging_config(),
        )
        _LOGGERS[session_id] = logger
    return logger


def _log(level: str, event: str, session_id: str | None = None, **kwargs: Any) -> None:
    sid = session_id or _resolve_session_id()
    logger = _get_logger(sid)
    logger.run_id = _run_id(sid)
    logger.mission_id = _mission_id(sid)
    logger.event(level, event, agent=AGENT_NAME, session_id=sid, **kwargs)


def _update_heartbeat(session_id: str) -> None:
    hb = _session_paths(session_id)["heartbeats"] / f"{AGENT_NAME}.json"
    hb.write_text(
        json.dumps(
            {
                "agent": AGENT_NAME,
                "pid": os.getpid(),
                "timestamp": time.time(),
                "session_id": session_id,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _heartbeat_pump(session_id: str) -> None:
    while not _HEARTBEAT_STOP.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            _update_heartbeat(session_id)
        except Exception:
            # Heartbeat failures should not crash the MCP server process.
            pass


def _sha(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _backup_corrupt_plan(session_id: str, raw: str, reason: str) -> str:
    incidents_dir = _session_paths(session_id)["bb"] / "logs" / "incidents"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    path = incidents_dir / f"plan-corrupt-{int(time.time())}.md"
    content = [
        "# Corrupt central_plan backup",
        "",
        f"- session_id: {session_id}",
        f"- reason: {reason}",
        f"- at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "```text",
        raw[-20000:],
        "```",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return str(path)


def _json_error_context(payload: str, line: int, col: int, radius: int = 80) -> str:
    lines = payload.splitlines()
    if line <= 0 or line > len(lines):
        compact = re.sub(r"\s+", " ", payload).strip()
        return compact[:radius]
    text = lines[line - 1]
    if col <= 0:
        return text[:radius]
    start = max(0, col - 1 - radius // 2)
    end = min(len(text), col - 1 + radius // 2)
    return text[start:end]


def _try_json_load_candidates(payload: str) -> tuple[Any | None, str | None, list[dict[str, Any]]]:
    normalized = payload.strip()
    if not normalized:
        return {}, "empty", []

    parse_errors: list[dict[str, Any]] = []
    candidates: list[tuple[str, str]] = [("raw", normalized)]
    no_trailing = re.sub(r",(\s*[}\]])", r"\1", normalized)
    if no_trailing != normalized:
        candidates.append(("drop_trailing_commas", no_trailing))

    for strategy, candidate in candidates:
        try:
            return json.loads(candidate), strategy, parse_errors
        except json.JSONDecodeError as exc:
            parse_errors.append(
                {
                    "strategy": strategy,
                    "message": exc.msg,
                    "line": exc.lineno,
                    "col": exc.colno,
                    "context": _json_error_context(candidate, exc.lineno, exc.colno),
                }
            )
    return None, None, parse_errors


def _parse_plan_with_diagnostics(raw: str) -> tuple[Any | None, dict[str, Any] | None]:
    if not raw.strip():
        return {}, None

    attempts: list[dict[str, Any]] = []

    fenced_blocks = re.findall(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", raw, flags=re.MULTILINE)
    for reverse_idx, block in enumerate(reversed(fenced_blocks)):
        idx = len(fenced_blocks) - 1 - reverse_idx
        value, strategy, errors = _try_json_load_candidates(block)
        if strategy is not None:
            return value, None
        for item in errors:
            attempts.append({"stage": f"fenced[{idx}]", **item})

    value, strategy, errors = _try_json_load_candidates(raw)
    if strategy is not None:
        return value, None
    for item in errors:
        attempts.append({"stage": "full_raw", **item})

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        brace_candidate = raw[start : end + 1]
        value, strategy, errors = _try_json_load_candidates(brace_candidate)
        if strategy is not None:
            return value, None
        for item in errors:
            attempts.append({"stage": "brace_extract", **item})

    primary = attempts[-1] if attempts else {"message": "unknown parse failure", "line": None, "col": None, "context": ""}
    summary = f"{primary.get('message')} at line {primary.get('line')} col {primary.get('col')}"
    return (
        None,
        {
            "summary": summary,
            "primary": primary,
            "attempts": attempts[-10:],
            "suggestions": [
                "Use exactly one JSON object (or one ```json fenced block).",
                "Remove trailing commas and keep key/value quoting valid JSON.",
                "Keep all required top-level fields and a valid tasks array.",
            ],
        },
    )


def _parse_loose_list(text: str) -> list[Any]:
    payload = text.strip()
    if not payload:
        return []
    value, strategy, _ = _try_json_load_candidates(payload)
    if strategy is not None and isinstance(value, list):
        return value
    if payload.startswith("[") and payload.endswith("]"):
        single_quote_payload = payload.replace("'", '"')
        value2, strategy2, _ = _try_json_load_candidates(single_quote_payload)
        if strategy2 is not None and isinstance(value2, list):
            return value2
        body = payload[1:-1]
        return [item.strip().strip("'\"") for item in body.split(",") if item.strip()]
    return []


def _parse_loose_scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    if value.lower() in {"none", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return _parse_loose_list(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except Exception:
            return value
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_kv_line(line: str) -> tuple[str, Any] | None:
    match = re.match(
        r"^(?:[-*]\s*)?(?:\*\*)?([A-Za-z_][A-Za-z0-9 _-]*)(?:\*\*)?\s*:\s*(.+?)\s*$",
        line.strip(),
    )
    if not match:
        return None
    key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
    value = _parse_loose_scalar(match.group(2))
    return key, value


def _normalize_task_id(raw_id: Any, fallback_index: int) -> str:
    text = str(raw_id or "").strip()
    if not text:
        return f"task-{fallback_index:03d}"
    if re.fullmatch(r"\d+", text):
        return f"task-{int(text):03d}"
    return text


def _parse_markdown_plan_fallback(raw: str, session_id: str) -> dict[str, Any] | None:
    lowered = raw.lower()
    if "## tasks" not in lowered and "# tasks" not in lowered and "id:" not in lowered:
        return None

    section = ""
    mission_goal = ""
    mission_status = "IN_PROGRESS"
    task_rows: list[dict[str, Any]] = []
    current_task: dict[str, Any] | None = None

    def flush_task() -> None:
        nonlocal current_task
        if current_task and current_task.get("id"):
            task_rows.append(current_task)
        current_task = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if re.match(r"^#+\s*mission\b", line, flags=re.IGNORECASE):
            section = "mission"
            continue
        if re.match(r"^#+\s*tasks\b", line, flags=re.IGNORECASE):
            section = "tasks"
            flush_task()
            continue

        parsed = _parse_kv_line(line)
        if parsed is None:
            continue
        key, value = parsed

        if section == "mission":
            if key in {"goal", "mission_goal", "objective"}:
                mission_goal = str(value).strip()
            elif key in {"status", "mission_status"}:
                mission_status = str(value).strip().upper().replace(" ", "_")
            continue

        if key == "id":
            flush_task()
            current_task = {"id": value}
            continue
        if current_task is None:
            continue
        current_task[key] = value

    flush_task()
    if not task_rows:
        return None

    id_map: dict[str, str] = {}
    for idx, row in enumerate(task_rows, start=1):
        normalized_id = _normalize_task_id(row.get("id"), idx)
        id_map[str(row.get("id"))] = normalized_id
        row["id"] = normalized_id

    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(task_rows, start=1):
        tid = str(row.get("id") or _normalize_task_id(None, idx)).strip()
        ttype = str(row.get("type") or "standard").strip().lower()
        if ttype not in {"standard", "standing"}:
            ttype = "standard"
        description = str(row.get("description") or row.get("title") or f"Task {idx}").strip()
        title = str(row.get("title") or description).strip()
        status = str(row.get("status") or "PENDING").strip().upper().replace(" ", "_")
        if status not in {"PENDING", "IN_PROGRESS", "BLOCKED", "DONE", "FAILED"}:
            status = "PENDING"

        deps_raw = row.get("dependencies", row.get("depends_on", []))
        if isinstance(deps_raw, list):
            deps = [str(x).strip() for x in deps_raw if str(x).strip()]
        elif deps_raw is None:
            deps = []
        else:
            dep_text = str(deps_raw).strip()
            deps = [dep_text] if dep_text else []
        dependencies = [id_map.get(dep, _normalize_task_id(dep, 0) if re.fullmatch(r"\d+", dep) else dep) for dep in deps]

        assignees_raw = row.get("assignees", [])
        if isinstance(assignees_raw, list):
            assignees = [str(x).strip() for x in assignees_raw if str(x).strip()]
        elif assignees_raw is None:
            assignees = []
        else:
            assignees = [str(assignees_raw).strip()] if str(assignees_raw).strip() else []

        assigned_worker = row.get("assigned_worker")
        assigned_worker_text = str(assigned_worker).strip() if assigned_worker is not None else None
        if not assigned_worker_text:
            assigned_worker_text = assignees[0] if assignees else None

        task = {
            "id": tid,
            "type": ttype,
            "title": title,
            "description": description,
            "status": status,
            "dependencies": dependencies,
            "assignees": assignees,
            "assigned_worker": assigned_worker_text,
            "start_time": row.get("start_time"),
            "end_time": row.get("end_time"),
            "artifact_link": row.get("artifact_link"),
            "result_summary": row.get("result_summary"),
            "result": row.get("result"),
            "retry_count": _safe_int(row.get("retry_count", 0) or 0, 0),
            "last_error": row.get("last_error"),
        }
        tasks.append(task)

    plan = _default_plan(session_id, mission_goal=mission_goal)
    if mission_goal:
        plan["mission_goal"] = mission_goal
    if mission_status in {"IN_PROGRESS", "DONE"}:
        plan["status"] = mission_status
    plan["tasks"] = tasks
    plan["updated_at"] = time.time()
    return plan


def _read_plan_from_text(raw: str) -> Any:
    value, parse_error = _parse_plan_with_diagnostics(raw)
    if parse_error is not None:
        raise ValueError(parse_error.get("summary") or "Cannot parse central_plan.md")
    return value


def _format_parse_error_message(parse_error: dict[str, Any] | None) -> str:
    if not parse_error:
        return "Cannot parse central_plan.md"
    primary = parse_error.get("primary") if isinstance(parse_error, dict) else None
    if not isinstance(primary, dict):
        return str(parse_error.get("summary") or "Cannot parse central_plan.md")
    message = str(primary.get("message") or parse_error.get("summary") or "Cannot parse central_plan.md")
    line = primary.get("line")
    col = primary.get("col")
    context = str(primary.get("context") or "")
    return f"{message} (line={line}, col={col}, context={context})"


def _write_plan_text(plan: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(plan, ensure_ascii=False, indent=2) + "\n```\n"


def _default_plan(session_id: str, mission_goal: str = "") -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "mission_goal": mission_goal or "swarm-mission",
        "status": "IN_PROGRESS",
        "summary": None,
        "session_id": session_id,
        "created_at": None,
        "updated_at": None,
        "tasks": [],
    }


def _normalize_plan(raw_plan: Any, session_id: str, mission_goal: str = "") -> dict[str, Any]:
    if isinstance(raw_plan, list):
        plan = _default_plan(session_id, mission_goal=mission_goal)
        plan["tasks"] = raw_plan
        return plan
    if not isinstance(raw_plan, dict):
        return _default_plan(session_id, mission_goal=mission_goal)

    plan = dict(raw_plan)
    if not isinstance(plan.get("tasks"), list):
        plan["tasks"] = []
    plan.setdefault("schema_version", "1.1")
    plan.setdefault("mission_goal", mission_goal or "swarm-mission")
    plan.setdefault("status", "IN_PROGRESS")
    plan.setdefault("summary", None)
    plan.setdefault("session_id", session_id)
    plan.setdefault("created_at", None)
    plan.setdefault("updated_at", None)
    return plan


def _task_deps(task: dict[str, Any]) -> list[str]:
    deps = task.get("dependencies", [])
    if not isinstance(deps, list):
        return []
    return [str(dep) for dep in deps]


def _normalize_dependency_statuses(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list):
        return []

    task_by_id: dict[str, dict[str, Any]] = {
        str(task.get("id")): task
        for task in tasks
        if isinstance(task, dict) and task.get("id") is not None
    }
    adjustments: list[dict[str, Any]] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "")
        if not task_id:
            continue

        status = task.get("status")
        deps = _task_deps(task)
        if not deps:
            continue

        unmet: list[str] = []
        for dep in deps:
            dep_task = task_by_id.get(str(dep))
            if dep_task is None or dep_task.get("status") != "DONE":
                unmet.append(str(dep))

        if status == "PENDING" and unmet:
            task["status"] = "BLOCKED"
            task["updated_at"] = time.time()
            adjustments.append(
                {
                    "task_id": task_id,
                    "from_status": "PENDING",
                    "to_status": "BLOCKED",
                    "reason": "dependencies_not_done",
                    "unmet_dependencies": unmet,
                }
            )
            continue

        if status == "BLOCKED" and not unmet:
            task["status"] = "PENDING"
            task["updated_at"] = time.time()
            adjustments.append(
                {
                    "task_id": task_id,
                    "from_status": "BLOCKED",
                    "to_status": "PENDING",
                    "reason": "dependencies_satisfied",
                    "unmet_dependencies": [],
                }
            )

    if adjustments:
        plan["updated_at"] = time.time()
    return adjustments


def _normalization_warnings(adjustments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "code": "TASK_STATUS_NORMALIZED",
            "task_id": item.get("task_id"),
            "from": item.get("from_status"),
            "to": item.get("to_status"),
            "reason": item.get("reason"),
            "unmet_dependencies": item.get("unmet_dependencies", []),
        }
        for item in adjustments
    ]


def _log_normalizations(session_id: str, tool: str, adjustments: list[dict[str, Any]]) -> None:
    if not adjustments:
        return
    _log(
        "info",
        "plan.status.normalized",
        session_id=session_id,
        tool=tool,
        extra={"count": len(adjustments), "adjustments": adjustments},
    )


def _validate_plan(plan: dict[str, Any]) -> str | None:
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list):
        return "'tasks' must be a list."
    mission_status = plan.get("status", "IN_PROGRESS")
    if mission_status not in {"IN_PROGRESS", "DONE"}:
        return "Mission 'status' must be IN_PROGRESS or DONE."
    if mission_status == "IN_PROGRESS" and len(tasks) == 0:
        return (
            "Mission is IN_PROGRESS but 'tasks' is empty. "
            "Provide at least one task with id/type/description/status/dependencies/assignees."
        )
    if mission_status == "DONE":
        not_done = [str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("status") != "DONE"]
        if not_done:
            return f"Mission is DONE but tasks are not DONE: {not_done}."

    task_ids = {str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("id") is not None}
    for task in tasks:
        if not isinstance(task, dict):
            return "Each task must be an object."
        tid = str(task.get("id"))
        if not tid:
            return "Each task must include non-empty 'id'."
        deps = task.get("dependencies", [])
        if not isinstance(deps, list):
            return f"Task {tid} 'dependencies' must be a list."
        for dep in deps:
            dep_id = str(dep)
            if dep_id == tid:
                return f"Task {tid} depends on itself."
            if dep_id not in task_ids:
                return f"Task {tid} depends on non-existent task {dep_id}."
        status = task.get("status")
        if status not in {"PENDING", "IN_PROGRESS", "BLOCKED", "DONE", "FAILED"}:
            return f"Task {tid} has invalid status: {status}."
        if status == "PENDING":
            for dep in deps:
                dep_task = next((t for t in tasks if isinstance(t, dict) and str(t.get("id")) == str(dep)), None)
                if dep_task and dep_task.get("status") != "DONE":
                    return f"Task {tid} is PENDING but dependency {dep} is not DONE."
        if status == "DONE" and not (task.get("result_summary") or task.get("result")):
            return f"Task {tid} is DONE but missing result_summary/result."
    return None


def _atomic_file(path: Path, fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    with path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return fn(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and ".." not in name


def _run_tool(tool: str, fn, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    session_id = _resolve_session_id()
    _ensure_layout(session_id)
    _update_heartbeat(session_id)
    _log("debug", "mcp.tool.start", session_id=session_id, tool=tool, extra={**(extra or {}), "agent": AGENT_NAME})

    try:
        result = fn(session_id)
    except Exception as exc:
        _log(
            "error",
            "mcp.tool.exception",
            session_id=session_id,
            tool=tool,
            status="failed",
            error_code="UNHANDLED_EXCEPTION",
            error=str(exc),
            extra={"traceback": traceback.format_exc(limit=8), "agent": AGENT_NAME},
        )
        return {
            "ok": False,
            "code": "INTERNAL_ERROR",
            "error": f"{tool} failed: {exc}",
        }

    duration_ms = int((time.perf_counter() - start) * 1000)
    if isinstance(result, dict) and result.get("ok") is False:
        _log(
            "warn",
            "mcp.tool.error",
            session_id=session_id,
            tool=tool,
            status="failed",
            error_code=result.get("code", "TOOL_FAILED"),
            error=result.get("error"),
            duration_ms=duration_ms,
            extra={"agent": AGENT_NAME},
        )
    else:
        _log(
            "info",
            "mcp.tool.ok",
            session_id=session_id,
            tool=tool,
            status="ok",
            duration_ms=duration_ms,
            extra={"agent": AGENT_NAME},
        )
    return result


VALID_TRANSITIONS = {
    "PENDING": {"IN_PROGRESS", "BLOCKED"},
    "IN_PROGRESS": {"DONE", "FAILED", "PENDING"},
    "FAILED": {"PENDING"},
    "BLOCKED": {"PENDING"},
}

MISSION_TRANSITIONS = {
    "IN_PROGRESS": {"DONE"},
    "DONE": {"IN_PROGRESS"},
}

mcp = FastMCP("")


@mcp.tool()
def blackboard_read(name: str) -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        if not _safe_name(name):
            return {"ok": False, "code": "INVALID_NAME", "error": f"Invalid file name: {name}"}
        path = _session_paths(session_id)["global_indices"] / name
        if not path.exists():
            return {"ok": False, "code": "FILE_NOT_FOUND", "error": f"File not found: {name}"}
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "name": name, "content": content, "checksum": _sha(content)}

    return _run_tool("blackboard_read", impl, extra={"name": name})


@mcp.tool()
def blackboard_write(name: str, content: str, checksum: str = "") -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        if not _safe_name(name):
            return {"ok": False, "code": "INVALID_NAME", "error": f"Invalid file name: {name}"}
        path = _session_paths(session_id)["global_indices"] / name
        payload_content = content
        warnings: list[dict[str, Any]] = []
        if name == "central_plan.md":
            parsed, parse_error = _parse_plan_with_diagnostics(content)
            if parse_error is not None:
                recovered = _parse_markdown_plan_fallback(content, session_id)
                if recovered is None:
                    backup_path = _backup_corrupt_plan(session_id, content, reason=f"blackboard_write: {_format_parse_error_message(parse_error)}")
                    return {
                        "ok": False,
                        "code": "PLAN_PARSE_ERROR",
                        "error": f"Cannot parse central_plan.md: {_format_parse_error_message(parse_error)}",
                        "parse_error": parse_error,
                        "backup_path": backup_path,
                    }
                parsed = recovered
                warnings.append(
                    {
                        "code": "PLAN_PARSE_RECOVERED",
                        "reason": "markdown_fallback",
                        "details": _format_parse_error_message(parse_error),
                    }
                )
                _log(
                    "warn",
                    "plan.parse.recovered",
                    session_id=session_id,
                    tool="blackboard_write",
                    extra={"strategy": "markdown_fallback", "summary": parse_error.get("summary", "")},
                )
            normalized = _normalize_plan(parsed, session_id)
            adjustments = _normalize_dependency_statuses(normalized)
            _log_normalizations(session_id, "blackboard_write", adjustments)
            warnings.extend(_normalization_warnings(adjustments))
            err = _validate_plan(normalized)
            if err:
                return {"ok": False, "code": "PLAN_VALIDATION_ERROR", "error": err}
            payload_content = _write_plan_text(normalized)

        def writer(f):
            existing = f.read()
            if existing and checksum and _sha(existing) != checksum:
                _log(
                    "warn",
                    "mcp.cas.conflict",
                    session_id=session_id,
                    tool="blackboard_write",
                    status="failed",
                    extra={"name": name, "expected": checksum, "actual": _sha(existing)},
                )
                return {
                    "ok": False,
                    "error": "Checksum mismatch. Re-read and retry.",
                    "code": "CAS_CONFLICT",
                }
            # Protect central_plan.md from full overwrite when tasks are active
            if name == "central_plan.md" and not checksum and existing:
                try:
                    existing_parsed, _ = _parse_plan_with_diagnostics(existing)
                    if isinstance(existing_parsed, dict):
                        active_tasks = [
                            t for t in existing_parsed.get("tasks", [])
                            if isinstance(t, dict) and t.get("status") in ("IN_PROGRESS", "DONE")
                        ]
                        if active_tasks:
                            _log(
                                "warn",
                                "mcp.plan.overwrite.blocked",
                                session_id=session_id,
                                tool="blackboard_write",
                                status="blocked",
                                extra={"active_count": len(active_tasks)},
                            )
                            return {
                                "ok": False,
                                "code": "PLAN_OVERWRITE_BLOCKED",
                                "error": (
                                    f"Cannot overwrite central_plan.md without checksum: "
                                    f"{len(active_tasks)} task(s) are IN_PROGRESS or DONE. "
                                    f"Use blackboard_update_task() to update individual tasks, "
                                    f"or provide checksum for CAS write."
                                ),
                            }
                except Exception:
                    pass
            f.seek(0)
            f.write(payload_content)
            f.truncate()
            result = {"ok": True, "checksum": _sha(payload_content)}
            if warnings:
                result["warnings"] = warnings
            return result

        return _atomic_file(path, writer)

    return _run_tool("blackboard_write", impl, extra={"name": name})


@mcp.tool()
def blackboard_list() -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        indices = _session_paths(session_id)["global_indices"]
        files = sorted([p.name for p in indices.iterdir() if p.is_file()]) if indices.exists() else []
        return {"ok": True, "files": files}

    return _run_tool("blackboard_list", impl)


@mcp.tool()
def blackboard_plan_template(mission_goal: str = "") -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        template_path = Path(__file__).resolve().parent.parent / "templates" / "central_plan.template.md"
        if template_path.exists():
            template_content = template_path.read_text(encoding="utf-8", errors="replace")
        else:
            template_content = _write_plan_text(_default_plan(session_id, mission_goal=mission_goal))
        if mission_goal:
            plan = _normalize_plan(_default_plan(session_id, mission_goal=mission_goal), session_id, mission_goal=mission_goal)
            seeded = _write_plan_text(plan)
        else:
            seeded = _write_plan_text(_default_plan(session_id, mission_goal=mission_goal))
        return {
            "ok": True,
            "session_id": session_id,
            "template_name": "central_plan.template.md",
            "template_content": template_content,
            "seeded_plan_content": seeded,
        }

    return _run_tool("blackboard_plan_template", impl, extra={"mission_goal": mission_goal})


@mcp.tool()
def blackboard_update_task(task_id: str, new_status: str, result: str = "") -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        plan_path = _session_paths(session_id)["plan"]

        def updater(f):
            raw = f.read()
            parsed, parse_error = _parse_plan_with_diagnostics(raw)
            if parse_error is not None:
                backup_path = _backup_corrupt_plan(
                    session_id,
                    raw,
                    reason=f"blackboard_update_task: {_format_parse_error_message(parse_error)}",
                )
                return {
                    "ok": False,
                    "code": "PLAN_PARSE_ERROR",
                    "error": f"Cannot parse central_plan.md: {_format_parse_error_message(parse_error)}",
                    "parse_error": parse_error,
                    "backup_path": backup_path,
                }
            plan = _normalize_plan(parsed, session_id)

            tasks = plan.get("tasks", [])
            target = next((t for t in tasks if str(t.get("id")) == str(task_id)), None)
            if not target:
                return {"ok": False, "code": "TASK_NOT_FOUND", "error": f"Task not found: {task_id}"}
            if plan.get("status") == "DONE":
                return {
                    "ok": False,
                    "code": "MISSION_ALREADY_DONE",
                    "error": "Mission already DONE. Reopen mission first with blackboard_update_mission(..., new_status='IN_PROGRESS').",
                }

            # Scheduler-owned status: prevent Architect from faking execution progress.
            if new_status == "IN_PROGRESS" and AGENT_NAME == "architect":
                return {
                    "ok": False,
                    "code": "STATUS_OWNERSHIP_VIOLATION",
                    "error": (
                        "Architect cannot set task to IN_PROGRESS. "
                        "This status is assigned by orchestrator scheduler when a worker claims the task."
                    ),
                }

            current = target.get("status")
            allowed = VALID_TRANSITIONS.get(current, set())
            if new_status not in allowed:
                _log(
                    "warn",
                    "mcp.transition.invalid",
                    session_id=session_id,
                    task_id=task_id,
                    status="failed",
                    extra={"current": current, "new": new_status, "allowed": sorted(list(allowed))},
                )
                return {
                    "ok": False,
                    "code": "INVALID_TRANSITION",
                    "error": f"Invalid transition {current} -> {new_status}",
                }

            if new_status == "DONE" and AGENT_NAME.startswith("worker"):
                assigned = target.get("assigned_worker")
                assignees = target.get("assignees") if isinstance(target.get("assignees"), list) else []
                if assigned and assigned != AGENT_NAME and AGENT_NAME not in assignees:
                    return {
                        "ok": False,
                        "code": "ASSIGNMENT_MISMATCH",
                        "error": f"Task {task_id} assigned to {assigned}, not {AGENT_NAME}",
                    }

            target["status"] = new_status
            target["updated_at"] = time.time()
            if result:
                target["result"] = result
                target["result_summary"] = result
            if new_status == "DONE":
                target["completed_at"] = time.time()
                target["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if new_status == "IN_PROGRESS" and not target.get("start_time"):
                target["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if AGENT_NAME.startswith("worker"):
                assignees = target.get("assignees") if isinstance(target.get("assignees"), list) else []
                if AGENT_NAME not in assignees:
                    assignees.append(AGENT_NAME)
                target["assignees"] = assignees

            # Log detailed task status transition
            _log(
                "info",
                "mcp.task.status_changed",
                session_id=session_id,
                task_id=task_id,
                status="ok",
                extra={
                    "agent": AGENT_NAME,
                    "previous_status": current,
                    "new_status": new_status,
                    "assigned_worker": target.get("assigned_worker"),
                    "result_preview": (result[:300] + "...") if len(result) > 300 else result if result else None,
                    "description": str(target.get("description") or "")[:200],
                },
            )

            plan["status"] = "IN_PROGRESS"
            adjustments = _normalize_dependency_statuses(plan)
            _log_normalizations(session_id, "blackboard_update_task", adjustments)
            err = _validate_plan(plan)
            if err:
                return {"ok": False, "code": "PLAN_VALIDATION_ERROR", "error": err}

            f.seek(0)
            f.write(_write_plan_text(plan))
            f.truncate()
            response = {
                "ok": True,
                "task_id": task_id,
                "requested_status": new_status,
                "status": target.get("status"),
            }
            if adjustments:
                response["warnings"] = _normalization_warnings(adjustments)
            return response

        return _atomic_file(plan_path, updater)

    return _run_tool(
        "blackboard_update_task",
        impl,
        extra={"task_id": task_id, "new_status": new_status},
    )


@mcp.tool()
def blackboard_update_mission(new_status: str, summary: str = "") -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        plan_path = _session_paths(session_id)["plan"]

        def updater(f):
            raw = f.read()
            parsed, parse_error = _parse_plan_with_diagnostics(raw)
            if parse_error is not None:
                backup_path = _backup_corrupt_plan(
                    session_id,
                    raw,
                    reason=f"blackboard_update_mission: {_format_parse_error_message(parse_error)}",
                )
                return {
                    "ok": False,
                    "code": "PLAN_PARSE_ERROR",
                    "error": f"Cannot parse central_plan.md: {_format_parse_error_message(parse_error)}",
                    "parse_error": parse_error,
                    "backup_path": backup_path,
                }
            plan = _normalize_plan(parsed, session_id)

            current = plan.get("status", "IN_PROGRESS")
            allowed = MISSION_TRANSITIONS.get(current, set())
            if new_status not in allowed:
                return {
                    "ok": False,
                    "code": "INVALID_MISSION_TRANSITION",
                    "error": f"Invalid mission transition {current} -> {new_status}",
                }

            if new_status == "DONE":
                tasks = plan.get("tasks", [])
                not_done = [str(t.get("id")) for t in tasks if isinstance(t, dict) and t.get("status") != "DONE"]
                if not_done:
                    return {
                        "ok": False,
                        "code": "MISSION_NOT_FINISHED",
                        "error": f"Cannot mark mission DONE. Tasks not DONE: {not_done}",
                    }
                if not summary.strip():
                    return {
                        "ok": False,
                        "code": "MISSION_SUMMARY_REQUIRED",
                        "error": "summary is required when mission transitions to DONE.",
                    }

            plan["status"] = new_status
            if summary.strip():
                plan["summary"] = summary.strip()

            adjustments = _normalize_dependency_statuses(plan)
            _log_normalizations(session_id, "blackboard_update_mission", adjustments)
            err = _validate_plan(plan)
            if err:
                return {"ok": False, "code": "PLAN_VALIDATION_ERROR", "error": err}

            f.seek(0)
            f.write(_write_plan_text(plan))
            f.truncate()
            response = {"ok": True, "status": new_status, "summary": plan.get("summary")}
            if adjustments:
                response["warnings"] = _normalization_warnings(adjustments)
            return response

        return _atomic_file(plan_path, updater)

    return _run_tool(
        "blackboard_update_mission",
        impl,
        extra={"new_status": new_status},
    )


@mcp.tool()
def swarm_status() -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        paths = _session_paths(session_id)
        plan = {}
        plan_parse_error = ""
        plan_parse_error_detail: dict[str, Any] | None = None
        if paths["plan"].exists():
            raw_plan = paths["plan"].read_text(encoding="utf-8", errors="replace")
            parsed, parse_error = _parse_plan_with_diagnostics(raw_plan)
            if parse_error is None:
                plan = parsed if isinstance(parsed, dict) else {}
            else:
                plan = {}
                plan_parse_error = _format_parse_error_message(parse_error)
                plan_parse_error_detail = parse_error

        tasks = plan.get("tasks", [])
        summary = {
            "total": len(tasks),
            "done": sum(1 for t in tasks if t.get("status") == "DONE"),
            "in_progress": sum(1 for t in tasks if t.get("status") == "IN_PROGRESS"),
            "pending": sum(1 for t in tasks if t.get("status") == "PENDING"),
            "blocked": sum(1 for t in tasks if t.get("status") == "BLOCKED"),
            "failed": sum(1 for t in tasks if t.get("status") == "FAILED"),
        }

        registry = {}
        if paths["registry"].exists():
            try:
                registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
            except Exception:
                registry = {}

        inbox = paths["inboxes"] / f"{AGENT_NAME}.json"
        unread = 0
        if inbox.exists():
            try:
                msgs = json.loads(inbox.read_text(encoding="utf-8") or "[]")
                unread = sum(1 for m in msgs if not m.get("read"))
            except Exception:
                unread = 0

        return {
            "ok": True,
            "session_id": session_id,
            "mission_goal": plan.get("mission_goal"),
            "mission_status": plan.get("status", "IN_PROGRESS"),
            "mission_summary": plan.get("summary"),
            "task_summary": summary,
            "tasks": tasks,
            "workers": registry.get("workers", []),
            "run_id": registry.get("run_id"),
            "unread_messages": unread,
            "plan_parse_error": plan_parse_error or None,
            "plan_parse_error_detail": plan_parse_error_detail,
            "commands": [
                "/swarm status",
                "/swarm panel on",
                "/swarm panel off",
                "/swarm panel status",
                "/swarm send worker-0 <instruction>",
                "/swarm stop",
                "/swarm stop --force",
            ],
        }

    return _run_tool("swarm_status", impl)


@mcp.tool()
def send_message(to: str, content: str, msg_type: str = "message") -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        path = _session_paths(session_id)["inboxes"] / f"{to}.json"

        def sender(f):
            raw = f.read()
            msgs = json.loads(raw) if raw.strip() else []
            item = {
                "id": f"{AGENT_NAME}-{int(time.time() * 1000)}",
                "from": AGENT_NAME,
                "to": to,
                "type": msg_type,
                "content": content,
                "timestamp": time.time(),
                "read": False,
            }
            msgs.append(item)
            f.seek(0)
            f.write(json.dumps(msgs, ensure_ascii=False, indent=2))
            f.truncate()
            return item

        item = _atomic_file(path, sender)
        return {"ok": True, "message_id": item["id"], "to": to}

    return _run_tool("send_message", impl, extra={"to": to, "type": msg_type})


@mcp.tool()
def read_inbox(mark_read: bool = True) -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        path = _session_paths(session_id)["inboxes"] / f"{AGENT_NAME}.json"

        def reader(f):
            raw = f.read()
            msgs = json.loads(raw) if raw.strip() else []
            unread = [m for m in msgs if not m.get("read")]
            if mark_read and unread:
                for m in msgs:
                    m["read"] = True
                f.seek(0)
                f.write(json.dumps(msgs, ensure_ascii=False, indent=2))
                f.truncate()
            return unread

        unread = _atomic_file(path, reader)
        return {"ok": True, "messages": unread}

    return _run_tool("read_inbox", impl)


@mcp.tool()
def broadcast_message(content: str) -> dict[str, Any]:
    def impl(session_id: str) -> dict[str, Any]:
        inboxes = _session_paths(session_id)["inboxes"]
        delivered: list[str] = []
        if not inboxes.exists():
            return {"ok": False, "error": "No inboxes directory", "code": "NO_INBOXES"}

        for file in inboxes.glob("*.json"):
            target = file.stem
            if target == AGENT_NAME:
                continue

            def sender(f):
                raw = f.read()
                msgs = json.loads(raw) if raw.strip() else []
                msgs.append(
                    {
                        "id": f"{AGENT_NAME}-broadcast-{int(time.time() * 1000)}",
                        "from": AGENT_NAME,
                        "to": target,
                        "type": "broadcast",
                        "content": content,
                        "timestamp": time.time(),
                        "read": False,
                    }
                )
                f.seek(0)
                f.write(json.dumps(msgs, ensure_ascii=False, indent=2))
                f.truncate()

            _atomic_file(file, sender)
            delivered.append(target)

        return {"ok": True, "delivered_to": delivered}

    return _run_tool("broadcast_message", impl)


def main() -> None:
    session_id = _resolve_session_id()
    _ensure_layout(session_id)
    _update_heartbeat(session_id)
    _HEARTBEAT_STOP.clear()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_pump,
        args=(session_id,),
        name=f"{AGENT_NAME}-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    _log(
        "info",
        "mcp.server.start",
        session_id=session_id,
        status="ok",
        extra={
            "agent": AGENT_NAME,
            "project_dir": str(PROJECT_DIR),
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        },
    )
    try:
        mcp.run()
    finally:
        _HEARTBEAT_STOP.set()
        try:
            _update_heartbeat(session_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
