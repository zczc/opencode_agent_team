#!/usr/bin/env python3
"""Swarm side panel for live status display (tmux or standalone window)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_session_id(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip()).strip("-")
    return value[:96] if value else "default"


def _read_text(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(_read_text(path, ""))
    except Exception:
        return fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_plan(path: Path) -> dict[str, Any]:
    raw = _read_text(path, "")
    if not raw:
        return {}
    blocks = re.findall(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", raw)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _current_session(bb_root: Path) -> str | None:
    selector = bb_root / "current_session"
    if selector.exists():
        value = _read_text(selector, "").strip()
        if value:
            return _sanitize_session_id(value)

    sessions_root = bb_root / "sessions"
    if not sessions_root.exists():
        return None
    candidates = sorted(
        sessions_root.glob("*/global_indices/orchestrator_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parents[1].name


def _tail_lines(path: Path, max_lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    text = _read_text(path, "")
    if not text:
        return []
    lines = text.splitlines()
    return lines[-max_lines:]


def _tail_jsonl(path: Path, max_lines: int = 6) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in _tail_lines(path, max_lines=max_lines * 3):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            result.append(data)
    return result[-max_lines:]


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: max(0, n - 3)] + "..."


def _panel_storage_dir(project_dir: Path) -> Path:
    preferred = project_dir / ".blackboard" / "panel"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        digest = hashlib.sha1(str(project_dir).encode("utf-8")).hexdigest()[:12]
        fallback = Path("/tmp") / "opencode-swarm-panel" / digest
        fallback.mkdir(parents=True, exist_ok=True)
        probe = fallback / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return fallback


def _fallback_panel_dir(project_dir: Path) -> Path:
    digest = hashlib.sha1(str(project_dir).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / "opencode-swarm-panel" / digest


def _panel_state_candidates(project_dir: Path) -> list[Path]:
    preferred = project_dir / ".blackboard" / "panel" / "state.json"
    fallback = _fallback_panel_dir(project_dir) / "state.json"
    return [preferred, fallback]


def _panel_state_path(project_dir: Path) -> Path:
    # Write to best-effort storage dir. Reads should check all candidates.
    return _panel_storage_dir(project_dir) / "state.json"


def _read_panel_state(project_dir: Path) -> dict[str, Any]:
    for path in _panel_state_candidates(project_dir):
        if not path.exists():
            continue
        state = _read_json(path, {})
        if isinstance(state, dict) and state:
            return state
    return {}


def _write_panel_state(project_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(_panel_state_path(project_dir), payload)


def _clear_panel_state(project_dir: Path) -> None:
    for path in _panel_state_candidates(project_dir):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _pid_command(pid: int) -> str:
    if pid <= 1:
        return ""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _is_standalone_panel_pid(pid: int, project_dir: Path) -> bool:
    if not _pid_alive(pid):
        return False
    cmd = _pid_command(pid)
    if not cmd:
        return False
    if "swarm_panel.py" not in cmd or " render " not in f" {cmd} ":
        return False
    # Ensure this process is for the same project.
    preferred_panel_dir = project_dir / ".blackboard" / "panel"
    return (
        str(project_dir) in cmd
        or str(preferred_panel_dir) in cmd
        or str(_fallback_panel_dir(project_dir)) in cmd
    )


def _cleanup_standalone_state(project_dir: Path) -> None:
    state = _read_panel_state(project_dir)
    if not isinstance(state, dict) or not state:
        _clear_panel_state(project_dir)
        return
    if state.get("mode") != "standalone":
        _clear_panel_state(project_dir)
        return
    pid = int(state.get("pid") or 0)
    if not _is_standalone_panel_pid(pid, project_dir):
        _clear_panel_state(project_dir)


def _collect_state(project_dir: Path) -> dict[str, Any]:
    bb_root = project_dir / ".blackboard"
    session_id = _current_session(bb_root)
    if not session_id:
        return {"has_session": False, "project_dir": str(project_dir)}

    session_dir = bb_root / "sessions" / session_id
    registry = _read_json(session_dir / "global_indices" / "registry.json", {})
    orchestrator_state = _read_json(session_dir / "global_indices" / "orchestrator_state.json", {})
    plan = _parse_plan(session_dir / "global_indices" / "central_plan.md")

    return {
        "has_session": True,
        "project_dir": str(project_dir),
        "session_id": session_id,
        "session_dir": str(session_dir),
        "registry": registry if isinstance(registry, dict) else {},
        "state": orchestrator_state if isinstance(orchestrator_state, dict) else {},
        "plan": plan if isinstance(plan, dict) else {},
        "orchestrator_tail": _tail_jsonl(session_dir / "logs" / "orchestrator" / "orchestrator.jsonl", max_lines=6),
        "execution_tail": _tail_jsonl(session_dir / "logs" / "execution" / "architect.jsonl", max_lines=6),
    }


LOG_MODE_KEYS = {
    "m": "merged",
    "o": "orchestrator",
    "e": "execution",
    "w": "workers",
    "s": "status",
}
LOG_MODE_LABELS = {
    "merged": "merged",
    "orchestrator": "orchestrator",
    "execution": "execution",
    "workers": "workers",
    "status": "status",
}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_GRAY = "\x1b[90m"
ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_BLUE = "\x1b[34m"
ANSI_CYAN = "\x1b[36m"


def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(sys.stdout.isatty())


def _colorize(text: str, *codes: str) -> str:
    if not text or not codes or not _colors_enabled():
        return text
    prefix = "".join(code for code in codes if code)
    if not prefix:
        return text
    return f"{prefix}{text}{ANSI_RESET}"


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _take_visible(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    out: list[str] = []
    visible = 0
    i = 0
    while i < len(text) and visible < limit:
        if text[i] == "\x1b":
            match = ANSI_RE.match(text, i)
            if match:
                out.append(match.group(0))
                i = match.end()
                continue
        out.append(text[i])
        visible += 1
        i += 1
    joined = "".join(out)
    if "\x1b[" in joined and not joined.endswith(ANSI_RESET):
        joined += ANSI_RESET
    return joined


def _fit_width(text: str, width: int) -> str:
    clean = str(text).replace("\n", " ").replace("\r", " ")
    if width <= 0:
        return ""
    visible_len = len(_strip_ansi(clean))
    if visible_len <= width:
        return clean
    if width <= 3:
        return _take_visible(clean, width)
    return _take_visible(clean, width - 3) + "..."


def _normalize_keywords(keywords: list[str] | None) -> list[str]:
    if not keywords:
        return []
    result: list[str] = []
    for keyword in keywords:
        token = str(keyword or "").strip().lower()
        if token:
            result.append(token)
    return result


def _collect_worker_options(data: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    registry = data.get("registry") if isinstance(data.get("registry"), dict) else {}
    workers = registry.get("workers") if isinstance(registry, dict) else []
    if isinstance(workers, list):
        for worker in workers:
            if isinstance(worker, dict):
                name = str(worker.get("name") or "").strip()
                if name:
                    names.add(name)

    plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    tasks = plan.get("tasks") if isinstance(plan, dict) else []
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            assigned = str(task.get("assigned_worker") or "").strip()
            if assigned:
                names.add(assigned)

    if data.get("has_session"):
        session_dir = Path(str(data.get("session_dir")))
        workers_dir = session_dir / "logs" / "workers"
        for dispatch in workers_dir.glob("worker-*.dispatch.jsonl"):
            names.add(dispatch.stem.replace(".dispatch", ""))

    return sorted(names)


def _collect_task_options(data: dict[str, Any]) -> list[str]:
    task_ids: list[str] = []
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    tasks = plan.get("tasks") if isinstance(plan, dict) else []
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "").strip()
            if task_id:
                task_ids.append(task_id)
    return sorted(set(task_ids))


def _entry_search_text(item: dict[str, Any], formatted: str) -> str:
    parts = [formatted]
    for key in ("worker", "task_id", "event", "status", "error_code", "error", "session_id", "run_id", "mission_id"):
        value = item.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}:{value}")
    payload = item.get("payload")
    if isinstance(payload, dict):
        try:
            parts.append(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass
    return " | ".join(parts).lower()


def _jsonl_to_entries(path: Path, label: str, max_lines: int = 500, keywords: list[str] | None = None) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    wanted = _normalize_keywords(keywords)
    if not path.exists():
        return entries

    raw_lines = _tail_lines(path, max_lines=max_lines * 3)
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue

        ts_full = str(item.get("ts") or "")
        ts = ts_full[-8:] if ts_full else "--:--:--"
        level = str(item.get("level") or "info").upper()
        event = str(item.get("event") or "-")

        hints: list[str] = []
        for key in ("worker", "task_id", "status", "error_code"):
            value = item.get(key)
            if value not in (None, "", []):
                hints.append(f"{key}={value}")

        payload = item.get("payload")
        if isinstance(payload, dict):
            if payload.get("tool"):
                hints.append(f"tool={payload.get('tool')}")
            if payload.get("part_id"):
                hints.append(f"part={payload.get('part_id')}")
            if payload.get("delta_len") is not None:
                hints.append(f"delta={payload.get('delta_len')}")

        suffix = f" | {' '.join(hints[:4])}" if hints else ""
        formatted = f"{ts} [{label}] {level:<5} {event}{suffix}"
        if wanted:
            searchable = _entry_search_text(item, formatted)
            if any(token not in searchable for token in wanted):
                continue
        entries.append((ts_full, formatted))

    return entries[-max_lines:]


def _collect_log_lines(
    data: dict[str, Any],
    mode: str,
    width: int,
    keywords: list[str] | None = None,
    worker_filter: str = "",
    task_filter: str = "",
) -> list[str]:
    if not data.get("has_session"):
        return ["No active session logs. Start with /swarm <mission>."]
    if mode == "status":
        return _collect_status_lines(data, width=width, worker_filter=worker_filter, task_filter=task_filter)

    session_dir = Path(str(data["session_dir"]))
    logs_dir = session_dir / "logs"
    orchestrator_dir = logs_dir / "orchestrator"
    workers_dir = logs_dir / "workers"
    execution_dir = logs_dir / "execution"

    entries: list[tuple[str, str]] = []
    if mode in {"merged", "orchestrator"}:
        entries.extend(_jsonl_to_entries(orchestrator_dir / "orchestrator.jsonl", "orch", max_lines=700, keywords=keywords))
        entries.extend(_jsonl_to_entries(orchestrator_dir / "scheduler_trace.jsonl", "sched", max_lines=600, keywords=keywords))
    if mode in {"merged", "execution"}:
        entries.extend(_jsonl_to_entries(execution_dir / "architect.jsonl", "exec", max_lines=700, keywords=keywords))
    if mode in {"merged", "workers"}:
        for dispatch in sorted(workers_dir.glob("worker-*.dispatch.jsonl")):
            label = dispatch.stem.replace(".dispatch", "")
            entries.extend(_jsonl_to_entries(dispatch, label, max_lines=350, keywords=keywords))

    if mode == "merged":
        entries.sort(key=lambda x: x[0] or "")

    lines = [_fit_width(line, width) for _, line in entries[-1600:]]
    if not lines:
        return [f"No logs for mode={mode} yet."]
    return lines


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(0, total)
    done = max(0, min(done, total if total else done))
    if total <= 0:
        return "[" + ("-" * width) + "] 0%"
    ratio = done / total
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {int(ratio * 100):>3d}%"


def _task_priority_key(task: dict[str, Any]) -> tuple[int, str]:
    status = str(task.get("status") or "")
    order = {
        "FAILED": 0,
        "IN_PROGRESS": 1,
        "BLOCKED": 2,
        "PENDING": 3,
        "DONE": 4,
    }
    return (order.get(status, 9), str(task.get("id") or ""))


def _task_status_style(status: str) -> str:
    value = status.upper().strip()
    mapping = {
        "FAILED": ANSI_RED,
        "DONE": ANSI_GREEN,
        "IN_PROGRESS": ANSI_CYAN,
        "BLOCKED": ANSI_YELLOW,
        "PENDING": ANSI_BLUE,
    }
    return mapping.get(value, ANSI_GRAY)


def _worker_status_style(status: str) -> str:
    value = status.lower().strip()
    if value in {"dead", "failed", "stopped"}:
        return ANSI_RED
    if value in {"busy", "running"}:
        return ANSI_CYAN
    if value in {"idle", "ready"}:
        return ANSI_GREEN
    return ANSI_GRAY


def _pending_deps(task: dict[str, Any], task_state: dict[str, str]) -> list[str]:
    deps = task.get("dependencies")
    if not isinstance(deps, list):
        return []
    pending: list[str] = []
    for dep in deps:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        if task_state.get(dep_id, "") != "DONE":
            pending.append(dep_id)
    return pending


def _task_live_workers_map(workers: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        name = str(worker.get("name") or "").strip()
        task_id = str(worker.get("current_task") or "").strip()
        if not name or not task_id or task_id == "-":
            continue
        mapping.setdefault(task_id, []).append(name)
    return mapping


def _task_worker_display(task: dict[str, Any], live_workers: dict[str, list[str]]) -> tuple[str, str]:
    task_id = str(task.get("id") or "").strip()
    assigned = str(task.get("assigned_worker") or "").strip()
    assignees_raw = task.get("assignees")
    assignees = []
    if isinstance(assignees_raw, list):
        assignees = [str(x).strip() for x in assignees_raw if str(x).strip()]
    live = list(live_workers.get(task_id, []))

    primary = "--"
    if live:
        primary = ",".join(live[:2]) + (f"+{len(live)-2}" if len(live) > 2 else "")
    elif assigned:
        primary = assigned
    elif assignees:
        primary = ",".join(assignees[:2]) + (f"+{len(assignees)-2}" if len(assignees) > 2 else "")

    note_parts: list[str] = []
    if assigned and live and assigned not in live:
        note_parts.append(f"plan={assigned} live={','.join(live[:2])}")
    if assignees and live and not set(live).issubset(set(assignees)):
        note_parts.append("assignees!=live")
    note = "; ".join(note_parts)
    return primary, note


def _collect_status_lines(data: dict[str, Any], width: int, worker_filter: str = "", task_filter: str = "") -> list[str]:
    if not data.get("has_session"):
        return ["No active session. Start with /swarm <mission>."]

    registry = data.get("registry") if isinstance(data.get("registry"), dict) else {}
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}

    tasks_raw = plan.get("tasks")
    tasks = [t for t in tasks_raw if isinstance(t, dict)] if isinstance(tasks_raw, list) else []
    workers_raw = registry.get("workers")
    workers = [w for w in workers_raw if isinstance(w, dict)] if isinstance(workers_raw, list) else []
    live_workers = _task_live_workers_map(workers)

    task_state = {str(t.get("id") or ""): str(t.get("status") or "") for t in tasks}
    run_id = str(registry.get("run_id") or state.get("run_id") or "unknown")
    mission = str(plan.get("mission_goal") or state.get("mission") or "unknown")

    lines: list[str] = []
    lines.append(_colorize("STATUS SNAPSHOT", ANSI_BOLD, ANSI_CYAN) + f"  run={run_id}")
    lines.append(_colorize(f"Mission: {mission}", ANSI_DIM, ANSI_GRAY))
    lines.append("")
    lines.append(_colorize("Tasks", ANSI_BOLD))
    if not tasks:
        lines.append(_colorize("  (no tasks)", ANSI_DIM, ANSI_GRAY))
    shown_tasks = 0
    hidden_tasks = 0
    for task in sorted(tasks, key=_task_priority_key):
        task_id = str(task.get("id") or "").strip() or "-"
        assigned_worker = str(task.get("assigned_worker") or "").strip()
        status_raw = str(task.get("status") or "-")
        title = str(task.get("title") or task.get("description") or "")
        if task_filter and task_id != task_filter:
            hidden_tasks += 1
            continue
        if worker_filter:
            possible_workers: set[str] = set()
            if assigned_worker:
                possible_workers.add(assigned_worker)
            for lw in live_workers.get(task_id, []):
                possible_workers.add(lw)
            assignees_raw = task.get("assignees")
            if isinstance(assignees_raw, list):
                for who in assignees_raw:
                    value = str(who).strip()
                    if value:
                        possible_workers.add(value)
            if worker_filter not in possible_workers:
                hidden_tasks += 1
                continue

        worker_display, mismatch_note = _task_worker_display(task, live_workers)
        if worker_filter and worker_filter not in worker_display:
            hidden_tasks += 1
            continue
        blocked_by = _pending_deps(task, task_state)
        status_colored = _colorize(status_raw, _task_status_style(status_raw), ANSI_BOLD)
        line = f"  {task_id:<12} {status_colored:<12} worker={worker_display}"
        if blocked_by:
            line += " " + _colorize(f"wait={','.join(blocked_by[:3])}", ANSI_YELLOW)
        if mismatch_note:
            line += " " + _colorize(f"| {mismatch_note}", ANSI_YELLOW)
        if title:
            line += " " + _colorize("|", ANSI_DIM, ANSI_GRAY) + " " + title
        lines.append(_fit_width(line, width))
        shown_tasks += 1
    if hidden_tasks > 0:
        lines.append(_colorize(f"  ... ({hidden_tasks} task(s) filtered)", ANSI_DIM, ANSI_GRAY))

    lines.append("")
    lines.append(_colorize("Workers", ANSI_BOLD))
    if not workers:
        lines.append(_colorize("  (no workers)", ANSI_DIM, ANSI_GRAY))
    shown_workers = 0
    hidden_workers = 0
    for worker in workers:
        name = str(worker.get("name") or "-")
        status_raw = str(worker.get("status") or "-")
        current_task = str(worker.get("current_task") or "-")
        if worker_filter and name != worker_filter:
            hidden_workers += 1
            continue
        if task_filter and current_task != task_filter:
            hidden_workers += 1
            continue
        status_colored = _colorize(status_raw, _worker_status_style(status_raw), ANSI_BOLD)
        session_id = str(worker.get("session_id") or "")
        sid_short = session_id[:14] + "..." if len(session_id) > 17 else session_id
        line = f"  {name:<12} {status_colored:<10} current={current_task}"
        if sid_short:
            line += " " + _colorize("|", ANSI_DIM, ANSI_GRAY) + " " + _colorize(f"sid={sid_short}", ANSI_DIM, ANSI_GRAY)
        lines.append(_fit_width(line, width))
        shown_workers += 1
    if hidden_workers > 0:
        lines.append(_colorize(f"  ... ({hidden_workers} worker(s) filtered)", ANSI_DIM, ANSI_GRAY))

    lines.append("")
    lines.append(_colorize("Hint:", ANSI_DIM, ANSI_GRAY) + " press " + _colorize("m/o/e/w", ANSI_BOLD) + " for log views")
    lines.append(_colorize("Hint:", ANSI_DIM, ANSI_GRAY) + " worker/task filter from top panel applies here")
    _ = shown_tasks, shown_workers
    return lines


def _build_top_lines(
    data: dict[str, Any],
    width: int,
    worker_filter: str,
    task_filter: str,
    worker_options: list[str],
    task_options: list[str],
) -> list[str]:
    now_text = _colorize(_now_str(), ANSI_DIM, ANSI_GRAY)
    header = _colorize("Swarm Panel", ANSI_BOLD, ANSI_CYAN)
    lines: list[str] = [f"{header}  {now_text}", _colorize("=" * max(10, width), ANSI_DIM, ANSI_GRAY)]

    if not data.get("has_session"):
        lines.extend([
            _colorize("No active swarm session", ANSI_BOLD, ANSI_YELLOW),
            _colorize(f"Project: {data.get('project_dir')}", ANSI_DIM, ANSI_GRAY),
            "Run /swarm <mission> first.",
            "",
            _colorize("Commands:", ANSI_DIM, ANSI_GRAY) + " /swarm status | /swarm panel off | /swarm panel status | /swarm stop",
            _colorize("Panel keys:", ANSI_DIM, ANSI_GRAY) + " j/k(up/down) u/d(page) g/G(top/bottom) s(status) m/o/e/w(log) v/V(worker) c/C(task) x(clear) q(quit)",
        ])
        return [_fit_width(x, width) for x in lines]

    session_id = str(data["session_id"])
    session_dir = str(data["session_dir"])
    registry = data["registry"] if isinstance(data.get("registry"), dict) else {}
    state = data["state"] if isinstance(data.get("state"), dict) else {}
    plan = data["plan"] if isinstance(data.get("plan"), dict) else {}

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    workers = registry.get("workers")
    if not isinstance(workers, list):
        workers = []
    live_workers = _task_live_workers_map([w for w in workers if isinstance(w, dict)])

    mission = str(plan.get("mission_goal") or state.get("mission") or "unknown")
    plan_status = str(plan.get("status") or "IN_PROGRESS")
    registry_status = str(registry.get("status") or state.get("status") or "unknown")
    done = len([t for t in tasks if isinstance(t, dict) and t.get("status") == "DONE"])
    in_progress = len([t for t in tasks if isinstance(t, dict) and t.get("status") == "IN_PROGRESS"])
    pending = len([t for t in tasks if isinstance(t, dict) and t.get("status") == "PENDING"])
    blocked = len([t for t in tasks if isinstance(t, dict) and t.get("status") == "BLOCKED"])
    failed = len([t for t in tasks if isinstance(t, dict) and t.get("status") == "FAILED"])
    total = len(tasks)

    worker_busy = 0
    worker_idle = 0
    worker_dead = 0
    worker_other = 0
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        status = str(worker.get("status") or "").lower()
        if status in {"busy", "running"}:
            worker_busy += 1
        elif status in {"idle", "ready"}:
            worker_idle += 1
        elif status in {"dead", "failed", "stopped"}:
            worker_dead += 1
        else:
            worker_other += 1

    run_id = str(registry.get("run_id") or state.get("run_id") or "unknown")
    lines.append(_colorize("SWARM DASHBOARD", ANSI_BOLD) + f" | session={session_id} | run={run_id}")
    lines.append(_colorize("Mission:", ANSI_DIM, ANSI_GRAY) + " " + _fit_width(mission, max(20, width - 9)))
    lines.append(
        _colorize("State:", ANSI_DIM, ANSI_GRAY)
        + " "
        + f"plan={_colorize(plan_status, _task_status_style(plan_status), ANSI_BOLD)} "
        + f"registry={_colorize(registry_status, _worker_status_style(registry_status), ANSI_BOLD)}"
    )
    progress_text = _progress_bar(done, total, width=min(36, max(18, width // 4)))
    lines.append(_colorize("Progress:", ANSI_DIM, ANSI_GRAY) + f" {progress_text}   ({done}/{total})")
    lines.append(
        _colorize("Task KPI:", ANSI_DIM, ANSI_GRAY)
        + f" total={total} "
        + _colorize(f"done={done}", ANSI_GREEN)
        + " "
        + _colorize(f"in_progress={in_progress}", ANSI_CYAN)
        + " "
        + _colorize(f"pending={pending}", ANSI_BLUE)
        + " "
        + _colorize(f"blocked={blocked}", ANSI_YELLOW)
        + " "
        + _colorize(f"failed={failed}", ANSI_RED, ANSI_BOLD)
    )
    lines.append(
        _colorize("Worker KPI:", ANSI_DIM, ANSI_GRAY)
        + f" total={len(workers)} "
        + _colorize(f"busy={worker_busy}", ANSI_CYAN)
        + " "
        + _colorize(f"idle={worker_idle}", ANSI_GREEN)
        + " "
        + _colorize(f"dead={worker_dead}", ANSI_RED, ANSI_BOLD)
        + " "
        + _colorize(f"other={worker_other}", ANSI_GRAY)
    )
    if failed > 0:
        lines.append(_colorize(f"Alert: {failed} task(s) FAILED. Prioritize recovery or retry.", ANSI_RED, ANSI_BOLD))
    elif blocked > 0:
        lines.append(_colorize(f"Alert: {blocked} task(s) BLOCKED. Check dependency chain / worker assignment.", ANSI_YELLOW, ANSI_BOLD))
    elif in_progress > 0:
        lines.append(_colorize(f"Focus: {in_progress} task(s) currently IN_PROGRESS.", ANSI_CYAN))
    elif plan_status == "DONE" and done > 0:
        summary = str(plan.get("summary") or "").strip()
        if summary:
            lines.append(_colorize(f"✓ Mission completed! {summary}", ANSI_GREEN, ANSI_BOLD))
        else:
            lines.append(_colorize(f"✓ Mission completed! All {done} task(s) finished successfully.", ANSI_GREEN, ANSI_BOLD))
    else:
        lines.append(_colorize("Focus: no active tasks. Check plan generation / worker dispatch.", ANSI_DIM, ANSI_GRAY))

    lines.append(
        _colorize("Filters:", ANSI_DIM, ANSI_GRAY)
        + f" worker={worker_filter or 'ALL'} task={task_filter or 'ALL'} "
        f"| worker_options={len(worker_options)} task_options={len(task_options)}"
    )
    if worker_options:
        lines.append(_colorize("Worker choices:", ANSI_DIM, ANSI_GRAY) + " " + " | ".join(worker_options[:8]) + (" | ..." if len(worker_options) > 8 else ""))
    if task_options:
        lines.append(_colorize("Task choices:", ANSI_DIM, ANSI_GRAY) + " " + " | ".join(task_options[:6]) + (" | ..." if len(task_options) > 6 else ""))
    lines.append(_colorize("-" * max(10, width), ANSI_DIM, ANSI_GRAY))
    lines.append(_colorize("Top Tasks", ANSI_BOLD) + " (priority: FAILED > IN_PROGRESS > BLOCKED > PENDING > DONE)")
    sorted_tasks = sorted([t for t in tasks if isinstance(t, dict)], key=_task_priority_key)
    for task in sorted_tasks[:8]:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "-")
        status = str(task.get("status") or "-")
        worker, _mismatch_note = _task_worker_display(task, live_workers)
        title = str(task.get("title") or task.get("description") or "")
        status_text = _colorize(status, _task_status_style(status), ANSI_BOLD)
        lines.append(f"{task_id:<12} {status_text:<12} {worker:<10} {_fit_width(title, max(20, width - 40))}")

        # Show result summary for DONE tasks
        if status == "DONE":
            result_summary = str(task.get("result_summary") or task.get("result") or "").strip()
            if result_summary:
                # Truncate and indent the result
                max_result_len = max(40, width - 20)
                if len(result_summary) > max_result_len:
                    result_summary = result_summary[:max_result_len - 3] + "..."
                lines.append(_colorize(f"  └─ {result_summary}", ANSI_DIM, ANSI_GRAY))
    if len(sorted_tasks) > 8:
        lines.append(_colorize(f"... ({len(sorted_tasks) - 8} more tasks)", ANSI_DIM, ANSI_GRAY))

    worker_text = []
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        name = str(worker.get("name") or "-")
        status = str(worker.get("status") or "-")
        current = str(worker.get("current_task") or "-")
        worker_text.append(f"{name}:{status}({current})")
    lines.append(_colorize("Workers:", ANSI_DIM, ANSI_GRAY) + " " + (_fit_width(" | ".join(worker_text), max(20, width - 9)) if worker_text else "none"))
    lines.append(_colorize(f"Session dir: {session_dir}", ANSI_DIM, ANSI_GRAY))
    lines.append(_colorize("Commands:", ANSI_DIM, ANSI_GRAY) + " /swarm status | /swarm panel off | /swarm panel status | /swarm send worker-0 <instruction>")
    lines.append(_colorize("Panel keys:", ANSI_DIM, ANSI_GRAY) + " j/k(up/down) u/d(page) g/G(top/bottom) s(status) m/o/e/w(log) v/V(worker) c/C(task) x(clear) q(quit)")
    return [_fit_width(x, width) for x in lines]


def _render_screen(project_dir: Path, ui_state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    data = _collect_state(project_dir)
    term = shutil.get_terminal_size((120, 42))
    width = max(60, term.columns)
    height = max(24, term.lines)

    top_height = max(12, min((height * 3) // 5, height - 9))
    bottom_body_height = max(5, height - top_height - 3)

    worker_options = _collect_worker_options(data)
    task_options = _collect_task_options(data)
    worker_filter = str(ui_state.get("worker_filter") or "")
    task_filter = str(ui_state.get("task_filter") or "")
    if worker_filter and worker_filter not in worker_options:
        worker_filter = ""
    if task_filter and task_filter not in task_options:
        task_filter = ""
    ui_state["worker_filter"] = worker_filter
    ui_state["task_filter"] = task_filter

    keywords = [x for x in [worker_filter, task_filter] if x]
    top_lines = _build_top_lines(
        data,
        width,
        worker_filter=worker_filter,
        task_filter=task_filter,
        worker_options=worker_options,
        task_options=task_options,
    )
    if len(top_lines) < top_height:
        top_lines.extend([""] * (top_height - len(top_lines)))
    else:
        top_lines = top_lines[:top_height]

    mode = str(ui_state.get("log_mode") or "merged")
    log_lines = _collect_log_lines(
        data,
        mode=mode,
        width=width,
        keywords=keywords,
        worker_filter=worker_filter,
        task_filter=task_filter,
    )
    max_scroll = max(0, len(log_lines) - bottom_body_height)
    scroll = int(ui_state.get("log_scroll") or 0)
    scroll = min(max(scroll, 0), max_scroll)
    ui_state["log_scroll"] = scroll

    start = max(0, len(log_lines) - bottom_body_height - scroll)
    visible = log_lines[start : start + bottom_body_height]
    if len(visible) < bottom_body_height:
        visible.extend([""] * (bottom_body_height - len(visible)))

    mode_label = LOG_MODE_LABELS.get(mode, mode)
    filter_label = f"worker={worker_filter or 'ALL'} task={task_filter or 'ALL'}"
    log_header = (
        _colorize("[View]", ANSI_DIM, ANSI_GRAY)
        + f" mode={_colorize(mode_label, ANSI_BOLD, ANSI_CYAN)} "
        + f"filters={filter_label} lines={len(log_lines)} scroll={scroll}/{max_scroll}"
    )
    footer = (
        _colorize("Keys:", ANSI_DIM, ANSI_GRAY)
        + " j/k arrows u/d page g/G home/end "
        + _colorize("s", ANSI_BOLD)
        + "(status) "
        + _colorize("m/o/e/w", ANSI_BOLD)
        + "(logs) v/V(worker) c/C(task) x clear q quit"
    )

    final_lines: list[str] = []
    final_lines.extend(_fit_width(line, width) for line in top_lines)
    final_lines.append("-" * width)
    final_lines.append(_fit_width(log_header, width))
    final_lines.extend(_fit_width(line, width) for line in visible)
    final_lines.append(_fit_width(footer, width))
    return (
        "\n".join(final_lines),
        {
            "max_scroll": max_scroll,
            "log_lines": len(log_lines),
            "window": bottom_body_height,
            "worker_options": worker_options,
            "task_options": task_options,
            "worker_filter": worker_filter,
            "task_filter": task_filter,
        },
    )


def _tmux(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _in_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def _current_tmux_pane() -> str:
    try:
        proc = _tmux(["display-message", "-p", "#{pane_id}"])
        return proc.stdout.strip()
    except Exception:
        return ""


def _get_panel_pane() -> str:
    try:
        proc = _tmux(["show-options", "-wqv", "@swarm_panel_pane"])
        return proc.stdout.strip()
    except Exception:
        return ""


def _pane_info(pane_id: str) -> dict[str, Any]:
    if not pane_id:
        return {"exists": False, "dead": False, "command": ""}
    try:
        lines = _tmux(["list-panes", "-a", "-F", "#{pane_id}\t#{pane_dead}\t#{pane_current_command}"]).stdout.splitlines()
    except Exception:
        return {"exists": False, "dead": False, "command": ""}
    for line in lines:
        parts = line.split("\t")
        if not parts:
            continue
        if parts[0] != pane_id:
            continue
        dead = len(parts) > 1 and parts[1] == "1"
        cmd = parts[2] if len(parts) > 2 else ""
        return {"exists": True, "dead": dead, "command": cmd}
    return {"exists": False, "dead": False, "command": ""}


def _pane_running(pane_id: str) -> bool:
    info = _pane_info(pane_id)
    return bool(info.get("exists")) and not bool(info.get("dead"))


def _cleanup_tmux_panel_reference() -> None:
    pane_id = _get_panel_pane()
    if not pane_id:
        return
    info = _pane_info(pane_id)
    if not info.get("exists"):
        _tmux(["set-option", "-wu", "@swarm_panel_pane"], check=False)
        return
    if info.get("dead"):
        _tmux(["kill-pane", "-t", pane_id], check=False)
        _tmux(["set-option", "-wu", "@swarm_panel_pane"], check=False)


def _build_render_cmd(project_dir: Path, interval: float, python_bin: str) -> str:
    script = Path(__file__).resolve()
    state_file = _panel_state_path(project_dir)
    return (
        f"{shlex.quote(python_bin)} {shlex.quote(str(script))} render "
        f"--project-dir {shlex.quote(str(project_dir))} "
        f"--interval {interval:.2f} "
        f"--mode standalone "
        f"--state-file {shlex.quote(str(state_file))}"
    )


def _build_launcher_script(project_dir: Path, render_cmd: str) -> Path:
    panel_dir = _panel_storage_dir(project_dir)
    launcher = panel_dir / "swarm-panel.command"
    crash_log = panel_dir / "standalone-last.log"
    content = (
        "#!/bin/bash\n"
        f"cd {shlex.quote(str(project_dir))}\n"
        f"{render_cmd} >> {shlex.quote(str(crash_log))} 2>&1\n"
        "status=$?\n"
        "if [ \"$status\" -ne 0 ]; then\n"
        "  echo \"[swarm-panel] exited with code $status\"\n"
        f"  echo \"log: {shlex.quote(str(crash_log))}\"\n"
        "  tail -n 80 "
        f"{shlex.quote(str(crash_log))}\n"
        "  echo \"Press Enter to close...\"\n"
        "  read -r _\n"
        "fi\n"
        "exit \"$status\"\n"
    )
    launcher.write_text(content, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _open_external_terminal(render_cmd: str, project_dir: Path) -> tuple[bool, str]:
    if sys.platform == "darwin":
        launcher = _build_launcher_script(project_dir, render_cmd)
        try:
            proc = subprocess.run(
                ["open", "-na", "/System/Applications/Utilities/Terminal.app", str(launcher)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _ = proc
            return True, "Opened standalone swarm panel in Terminal."
        except Exception as open_exc_path:
            try:
                proc = subprocess.run(
                    ["open", "-a", "Terminal", str(launcher)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                _ = proc
                return True, "Opened standalone swarm panel in Terminal."
            except Exception as open_exc:
                open_error = f"path_open={open_exc_path}; app_open={open_exc}"
                if shutil.which("osascript"):
                    shell_cmd = f"cd {shlex.quote(str(project_dir))}; {render_cmd}"
                    lines = [
                        'tell application "Terminal"',
                        "activate",
                        f"do script {json.dumps(shell_cmd)}",
                        "end tell",
                    ]
                    args: list[str] = []
                    for line in lines:
                        args.extend(["-e", line])
                    try:
                        subprocess.run(["osascript", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        return True, "Opened standalone swarm panel in Terminal (osascript fallback)."
                    except Exception as osa_exc:
                        return False, f"Failed to open Terminal (open/osascript): {open_error}; osascript={osa_exc}"
                return False, f"Failed to open Terminal: {open_error}"

    if shutil.which("x-terminal-emulator"):
        try:
            subprocess.Popen(["x-terminal-emulator", "-e", "bash", "-lc", render_cmd], cwd=str(project_dir))
            return True, "Opened standalone swarm panel in x-terminal-emulator."
        except Exception as exc:
            return False, f"Failed to open x-terminal-emulator: {exc}"

    if shutil.which("gnome-terminal"):
        try:
            subprocess.Popen(["gnome-terminal", "--", "bash", "-lc", render_cmd], cwd=str(project_dir))
            return True, "Opened standalone swarm panel in gnome-terminal."
        except Exception as exc:
            return False, f"Failed to open gnome-terminal: {exc}"

    return False, "No supported terminal launcher found."


def _open_tmux_panel(project_dir: Path, percent: int, interval: float, python_bin: str, focus: str) -> tuple[int, str]:
    _cleanup_tmux_panel_reference()
    pane_id = _get_panel_pane()
    if pane_id and _pane_running(pane_id):
        if focus == "panel":
            _tmux(["select-pane", "-t", pane_id], check=False)
        return 0, f"Swarm panel already open in tmux: {pane_id}"
    if pane_id and not _pane_running(pane_id):
        _tmux(["set-option", "-wu", "@swarm_panel_pane"], check=False)

    script = Path(__file__).resolve()
    cmd = (
        f"{shlex.quote(python_bin)} {shlex.quote(str(script))} render "
        f"--project-dir {shlex.quote(str(project_dir))} --interval {interval:.2f} --mode tmux"
    )
    current_pane = _current_tmux_pane()
    split_args = ["split-window", "-h", "-p", str(percent), "-P", "-F", "#{pane_id}", cmd]
    if focus != "panel":
        split_args.insert(4, "-d")
    proc = _tmux(split_args)
    new_pane = proc.stdout.strip()
    if not new_pane:
        return 1, f"Failed to create tmux panel: {proc.stderr.strip()}"
    _tmux(["set-option", "-wq", "@swarm_panel_pane", new_pane], check=False)
    if focus == "panel":
        _tmux(["select-pane", "-t", new_pane], check=False)
    elif current_pane:
        _tmux(["select-pane", "-t", current_pane], check=False)
    return 0, f"Swarm panel opened in tmux: {new_pane} (focus={focus})"


def _open_standalone_panel(project_dir: Path, interval: float, python_bin: str) -> tuple[int, str]:
    _cleanup_standalone_state(project_dir)
    state = _read_panel_state(project_dir)
    existing_pid = int(state.get("pid") or 0)
    if state.get("mode") == "standalone" and _is_standalone_panel_pid(existing_pid, project_dir):
        return 0, f"Standalone swarm panel already open: pid={existing_pid}"

    render_cmd = _build_render_cmd(project_dir, interval, python_bin)
    ok, msg = _open_external_terminal(render_cmd, project_dir)
    if ok:
        return 0, msg
    launcher = _build_launcher_script(project_dir, render_cmd)
    manual = f"{render_cmd}"
    return 1, f"{msg}\nRun manually in another terminal:\n{manual}\nOr run launcher file:\n{launcher}"


def open_panel(project_dir: Path, percent: int, interval: float, python_bin: str, focus: str = "main") -> int:
    focus = "panel" if str(focus).lower() == "panel" else "main"
    if shutil.which("tmux") and _in_tmux():
        code, msg = _open_tmux_panel(
            project_dir=project_dir,
            percent=percent,
            interval=interval,
            python_bin=python_bin,
            focus=focus,
        )
        print(msg)
        return code

    code, msg = _open_standalone_panel(project_dir=project_dir, interval=interval, python_bin=python_bin)
    print(msg)
    return code


def close_panel(project_dir: Path) -> int:
    closed_any = False

    if shutil.which("tmux") and _in_tmux():
        _cleanup_tmux_panel_reference()
        pane_id = _get_panel_pane()
        if pane_id and _pane_running(pane_id):
            _tmux(["kill-pane", "-t", pane_id], check=False)
            closed_any = True
        _tmux(["set-option", "-wu", "@swarm_panel_pane"], check=False)

    _cleanup_standalone_state(project_dir)
    state = _read_panel_state(project_dir)
    pid = int(state.get("pid") or 0)
    if state.get("mode") == "standalone" and _is_standalone_panel_pid(pid, project_dir):
        try:
            os.kill(pid, signal.SIGTERM)
            closed_any = True
        except OSError:
            pass
    _clear_panel_state(project_dir)

    if closed_any:
        print("Swarm panel closed.")
    else:
        print("Swarm panel is not open.")
    return 0


def panel_status(project_dir: Path) -> int:
    if shutil.which("tmux") and _in_tmux():
        _cleanup_tmux_panel_reference()
        pane_id = _get_panel_pane()
        if pane_id and _pane_running(pane_id):
            print(f"Swarm panel open (tmux): {pane_id}")
            return 0

    _cleanup_standalone_state(project_dir)
    state = _read_panel_state(project_dir)
    pid = int(state.get("pid") or 0)
    if state.get("mode") == "standalone" and _is_standalone_panel_pid(pid, project_dir):
        print(f"Swarm panel open (standalone): pid={pid}")
        return 0

    print("Swarm panel closed.")
    return 0


@contextlib.contextmanager
def _terminal_cbreak(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _decode_key_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(raw):
        if raw.startswith("\x1b[A", i):
            tokens.append("UP")
            i += 3
            continue
        if raw.startswith("\x1b[B", i):
            tokens.append("DOWN")
            i += 3
            continue
        if raw.startswith("\x1b[5~", i):
            tokens.append("PGUP")
            i += 4
            continue
        if raw.startswith("\x1b[6~", i):
            tokens.append("PGDN")
            i += 4
            continue
        ch = raw[i]
        if ch in ("k", "K"):
            tokens.append("UP")
        elif ch in ("j", "J"):
            tokens.append("DOWN")
        elif ch in ("u", "U"):
            tokens.append("PGUP")
        elif ch in ("d", "D"):
            tokens.append("PGDN")
        elif ch in ("g",):
            tokens.append("HOME")
        elif ch in ("G",):
            tokens.append("END")
        elif ch in ("q", "Q", "\x03"):
            tokens.append("QUIT")
        elif ch in ("v",):
            tokens.append("WORKER_NEXT")
        elif ch in ("V",):
            tokens.append("WORKER_PREV")
        elif ch in ("c",):
            tokens.append("TASK_NEXT")
        elif ch in ("C",):
            tokens.append("TASK_PREV")
        elif ch in ("x", "X"):
            tokens.append("FILTER_CLEAR")
        elif ch in ("s", "S"):
            tokens.append("MODE_status")
        elif ch in LOG_MODE_KEYS:
            tokens.append("MODE_" + LOG_MODE_KEYS[ch])
        i += 1
    return tokens


def _read_key_tokens(timeout: float) -> list[str]:
    if not sys.stdin.isatty():
        time.sleep(max(0.0, timeout))
        return []
    ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
    if not ready:
        return []
    try:
        raw = os.read(sys.stdin.fileno(), 64).decode("utf-8", errors="ignore")
    except Exception:
        return []
    return _decode_key_tokens(raw)


def _cycle_option(current: str, options: list[str], step: int) -> str:
    if not options:
        return ""
    pool = [""] + options
    try:
        idx = pool.index(current)
    except ValueError:
        idx = 0
    idx = (idx + step) % len(pool)
    return pool[idx]


def _apply_tokens(ui_state: dict[str, Any], tokens: list[str], metrics: dict[str, Any]) -> bool:
    max_scroll = int(metrics.get("max_scroll", 0))
    page = max(5, int(metrics.get("window", 8)) - 2)
    scroll = int(ui_state.get("log_scroll") or 0)
    mode = str(ui_state.get("log_mode") or "merged")
    worker_filter = str(ui_state.get("worker_filter") or "")
    task_filter = str(ui_state.get("task_filter") or "")
    worker_options = list(metrics.get("worker_options") or [])
    task_options = list(metrics.get("task_options") or [])
    running = True

    for token in tokens:
        if token == "UP":
            scroll = min(max_scroll, scroll + 1)
        elif token == "DOWN":
            scroll = max(0, scroll - 1)
        elif token == "PGUP":
            scroll = min(max_scroll, scroll + page)
        elif token == "PGDN":
            scroll = max(0, scroll - page)
        elif token == "HOME":
            scroll = max_scroll
        elif token == "END":
            scroll = 0
        elif token.startswith("MODE_"):
            next_mode = token[5:]
            if next_mode in LOG_MODE_LABELS:
                mode = next_mode
                scroll = 0
        elif token == "WORKER_NEXT":
            worker_filter = _cycle_option(worker_filter, worker_options, +1)
            scroll = 0
        elif token == "WORKER_PREV":
            worker_filter = _cycle_option(worker_filter, worker_options, -1)
            scroll = 0
        elif token == "TASK_NEXT":
            task_filter = _cycle_option(task_filter, task_options, +1)
            scroll = 0
        elif token == "TASK_PREV":
            task_filter = _cycle_option(task_filter, task_options, -1)
            scroll = 0
        elif token == "FILTER_CLEAR":
            worker_filter = ""
            task_filter = ""
            scroll = 0
        elif token == "QUIT":
            running = False

    ui_state["log_mode"] = mode
    ui_state["worker_filter"] = worker_filter
    ui_state["task_filter"] = task_filter
    ui_state["log_scroll"] = min(max(scroll, 0), max_scroll)
    return running


def render_loop(project_dir: Path, interval: float, mode: str, state_file: Path | None) -> int:
    interval = max(0.5, interval)
    ui_state: dict[str, Any] = {"log_mode": "status", "log_scroll": 0, "worker_filter": "", "task_filter": ""}
    interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())

    if mode == "standalone" and state_file:
        _write_json(
            state_file,
            {
                "mode": "standalone",
                "pid": os.getpid(),
                "project_dir": str(project_dir),
                "started_at": int(time.time()),
            },
        )

    try:
        with _terminal_cbreak(interactive):
            running = True
            while running:
                screen, metrics = _render_screen(project_dir, ui_state)
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(screen + "\n")
                sys.stdout.flush()

                if interactive:
                    tokens = _read_key_tokens(interval)
                    if tokens:
                        running = _apply_tokens(ui_state, tokens, metrics)
                else:
                    time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        if mode == "standalone" and state_file:
            cur = _read_json(state_file, {})
            if isinstance(cur, dict) and int(cur.get("pid") or 0) == os.getpid():
                try:
                    state_file.unlink(missing_ok=True)
                except Exception:
                    pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swarm live panel")
    sub = parser.add_subparsers(dest="command", required=True)

    open_parser = sub.add_parser("open")
    open_parser.add_argument("--project-dir", default=".")
    open_parser.add_argument("--interval", type=float, default=2.0)
    open_parser.add_argument("--percent", type=int, default=50)
    open_parser.add_argument("--python-bin", default="python3")
    open_parser.add_argument("--focus", choices=["panel", "main"], default="main")

    close_parser = sub.add_parser("close")
    close_parser.add_argument("--project-dir", default=".")

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--project-dir", default=".")

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--project-dir", default=".")
    render_parser.add_argument("--interval", type=float, default=2.0)
    render_parser.add_argument("--mode", choices=["tmux", "standalone"], default="tmux")
    render_parser.add_argument("--state-file", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "open":
        percent = min(80, max(25, int(args.percent)))
        raise SystemExit(
            open_panel(
                project_dir=Path(args.project_dir).resolve(),
                percent=percent,
                interval=max(0.5, float(args.interval)),
                python_bin=str(args.python_bin),
                focus=str(args.focus),
            )
        )
    if args.command == "close":
        raise SystemExit(close_panel(Path(args.project_dir).resolve()))
    if args.command == "status":
        raise SystemExit(panel_status(Path(args.project_dir).resolve()))
    if args.command == "render":
        state_file = Path(args.state_file).resolve() if str(args.state_file).strip() else None
        raise SystemExit(
            render_loop(
                Path(args.project_dir).resolve(),
                interval=max(0.5, float(args.interval)),
                mode=str(args.mode),
                state_file=state_file,
            )
        )
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
