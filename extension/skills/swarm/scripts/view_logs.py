#!/usr/bin/env python3
"""
Swarm Log Viewer - View detailed logs from swarm sessions

Usage:
    python view_logs.py <session_id>                    # View all logs
    python view_logs.py <session_id> --component orchestrator  # View orchestrator logs
    python view_logs.py <session_id> --component worker-0      # View worker-0 logs
    python view_logs.py <session_id> --component mcp           # View MCP server logs
    python view_logs.py <session_id> --component plugin        # View plugin logs
    python view_logs.py <session_id> --component execution     # View execution logs
    python view_logs.py <session_id> --event task.claimed      # Filter by event
    python view_logs.py <session_id> --level error             # Filter by level
    python view_logs.py <session_id> --tail 50                 # Show last 50 lines
    python view_logs.py <session_id> --follow                  # Follow logs (like tail -f)
"""

import argparse
import json
import sys
import time
from pathlib import Path


def find_session_dir(session_id: str) -> Path | None:
    """Find the session directory."""
    cwd = Path.cwd()
    bb_root = cwd / ".blackboard" / "sessions"

    if not bb_root.exists():
        return None

    # Exact match
    session_dir = bb_root / session_id
    if session_dir.exists():
        return session_dir

    # Partial match
    matches = [d for d in bb_root.iterdir() if d.is_dir() and session_id in d.name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple sessions match '{session_id}':", file=sys.stderr)
        for m in matches:
            print(f"  - {m.name}", file=sys.stderr)
        return None

    return None


def parse_jsonl_line(line: str) -> dict | None:
    """Parse a JSONL line."""
    try:
        return json.loads(line.strip())
    except:
        return None


def format_log_entry(entry: dict, show_full: bool = False) -> str:
    """Format a log entry for display."""
    ts = entry.get("ts", "")
    level = entry.get("level", "info").upper()
    component = entry.get("component", "")
    event = entry.get("event", "")
    agent = entry.get("agent", "")
    worker = entry.get("worker", "")
    task_id = entry.get("task_id", "")
    status = entry.get("status", "")
    error = entry.get("error", "")
    extra = entry.get("extra", {})

    # Color codes
    colors = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[36m",   # Cyan
        "WARN": "\033[33m",   # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m",
    }

    color = colors.get(level, colors["RESET"])
    reset = colors["RESET"]

    # Build the line
    parts = [f"{color}[{ts}]{reset}"]
    parts.append(f"{color}[{level}]{reset}")

    if component:
        parts.append(f"[{component}]")

    if agent:
        parts.append(f"agent={agent}")

    if worker:
        parts.append(f"worker={worker}")

    if task_id:
        parts.append(f"task={task_id}")

    parts.append(f"{event}")

    if status:
        parts.append(f"status={status}")

    if error:
        parts.append(f"error={error}")

    line = " ".join(parts)

    if show_full and extra:
        line += f"\n  extra: {json.dumps(extra, indent=2)}"

    return line


def view_logs(
    session_dir: Path,
    component: str | None = None,
    event: str | None = None,
    level: str | None = None,
    tail: int | None = None,
    follow: bool = False,
    show_full: bool = False,
):
    """View logs from a session."""
    logs_dir = session_dir / "logs"

    if not logs_dir.exists():
        print(f"No logs directory found in {session_dir}", file=sys.stderr)
        return

    # Collect log files
    log_files = []

    if component:
        # Specific component
        if component == "orchestrator":
            log_files.append(logs_dir / "orchestrator" / "orchestrator.jsonl")
        elif component.startswith("worker-"):
            log_files.append(logs_dir / "workers" / f"{component}.dispatch.jsonl")
        elif component == "mcp":
            for f in (logs_dir / "mcp").glob("*.jsonl"):
                log_files.append(f)
        elif component == "plugin":
            for f in (logs_dir / "plugin").glob("*.jsonl"):
                log_files.append(f)
        elif component == "execution":
            for f in (logs_dir / "execution").glob("*.jsonl"):
                log_files.append(f)
    else:
        # All logs
        for subdir in ["orchestrator", "workers", "mcp", "plugin", "execution"]:
            subdir_path = logs_dir / subdir
            if subdir_path.exists():
                for f in subdir_path.glob("*.jsonl"):
                    log_files.append(f)

    log_files = [f for f in log_files if f.exists()]

    if not log_files:
        print(f"No log files found", file=sys.stderr)
        return

    # Read and filter logs
    entries = []
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                for line in f:
                    entry = parse_jsonl_line(line)
                    if not entry:
                        continue

                    # Apply filters
                    if event and entry.get("event") != event:
                        continue

                    if level and entry.get("level", "").upper() != level.upper():
                        continue

                    entries.append(entry)
        except Exception as e:
            print(f"Error reading {log_file}: {e}", file=sys.stderr)

    # Sort by timestamp
    entries.sort(key=lambda e: e.get("ts", ""))

    # Apply tail
    if tail:
        entries = entries[-tail:]

    # Display
    for entry in entries:
        print(format_log_entry(entry, show_full))

    # Follow mode
    if follow:
        print("\n--- Following logs (Ctrl+C to stop) ---\n")
        file_positions = {f: f.stat().st_size for f in log_files}

        try:
            while True:
                time.sleep(0.5)

                for log_file in log_files:
                    if not log_file.exists():
                        continue

                    current_size = log_file.stat().st_size
                    last_pos = file_positions.get(log_file, 0)

                    if current_size > last_pos:
                        with open(log_file, "r") as f:
                            f.seek(last_pos)
                            for line in f:
                                entry = parse_jsonl_line(line)
                                if not entry:
                                    continue

                                # Apply filters
                                if event and entry.get("event") != event:
                                    continue

                                if level and entry.get("level", "").upper() != level.upper():
                                    continue

                                print(format_log_entry(entry, show_full))

                        file_positions[log_file] = current_size
        except KeyboardInterrupt:
            print("\nStopped following logs")


def main():
    parser = argparse.ArgumentParser(description="View swarm session logs")
    parser.add_argument("session_id", help="Session ID (full or partial)")
    parser.add_argument("--component", help="Filter by component (orchestrator, worker-0, mcp, plugin, execution)")
    parser.add_argument("--event", help="Filter by event name")
    parser.add_argument("--level", help="Filter by log level (debug, info, warn, error)")
    parser.add_argument("--tail", type=int, help="Show last N lines")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs (like tail -f)")
    parser.add_argument("--full", action="store_true", help="Show full log entries with extra fields")

    args = parser.parse_args()

    session_dir = find_session_dir(args.session_id)
    if not session_dir:
        print(f"Session not found: {args.session_id}", file=sys.stderr)
        sys.exit(1)

    print(f"Viewing logs from: {session_dir.name}\n")

    view_logs(
        session_dir,
        component=args.component,
        event=args.event,
        level=args.level,
        tail=args.tail,
        follow=args.follow,
        show_full=args.full,
    )


if __name__ == "__main__":
    main()
