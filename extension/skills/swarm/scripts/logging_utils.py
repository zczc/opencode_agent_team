#!/usr/bin/env python3
"""Structured JSONL logging helpers for the opencode agent team extension."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEVEL_ORDER = {
    "debug": 10,
    "info": 20,
    "warn": 30,
    "error": 40,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggerConfig:
    level: str = "info"
    rotate_mb: int = 20
    retention_files: int = 10

    @property
    def min_level(self) -> int:
        return LEVEL_ORDER.get(self.level.lower(), LEVEL_ORDER["info"])

    @property
    def rotate_bytes(self) -> int:
        return max(self.rotate_mb, 1) * 1024 * 1024

    @property
    def retention(self) -> int:
        return max(self.retention_files, 1)


class JsonlLogger:
    """Lightweight JSONL logger with size-based rotation."""

    def __init__(
        self,
        file_path: Path,
        component: str,
        run_id: str,
        mission_id: str,
        config: LoggerConfig | None = None,
    ) -> None:
        self.file_path = file_path
        self.component = component
        self.run_id = run_id
        self.mission_id = mission_id
        self.config = config or LoggerConfig()
        ensure_dir(file_path.parent)

    def _should_log(self, level: str) -> bool:
        return LEVEL_ORDER.get(level.lower(), LEVEL_ORDER["info"]) >= self.config.min_level

    def _rotate(self) -> None:
        if not self.file_path.exists():
            return
        if self.file_path.stat().st_size < self.config.rotate_bytes:
            return
        ts = int(time.time())
        rotated = self.file_path.with_suffix(self.file_path.suffix + f".{ts}")
        self.file_path.rename(rotated)

        pattern = f"{self.file_path.name}.*"
        backups = sorted(self.file_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[self.config.retention :]:
            try:
                old.unlink()
            except OSError:
                pass

    def event(self, level: str, event: str, **kwargs: Any) -> None:
        if not self._should_log(level):
            return
        self._rotate()
        record = {
            "ts": utc_now_iso(),
            "level": level.lower(),
            "component": self.component,
            "event": event,
            "run_id": self.run_id,
            "mission_id": self.mission_id,
            **kwargs,
        }
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Logging failures must not crash orchestrator/MCP.
            pass


def read_logging_config(default_level: str = "info") -> LoggerConfig:
    level = os.environ.get("SWARM_DEBUG_LOG_LEVEL", default_level)
    rotate = _parse_int(os.environ.get("SWARM_DEBUG_LOG_ROTATION_MB"), 20)
    retention = _parse_int(os.environ.get("SWARM_DEBUG_LOG_RETENTION_FILES"), 10)
    return LoggerConfig(level=level, rotate_mb=rotate, retention_files=retention)


def _parse_int(raw: str | None, fallback: int) -> int:
    if not raw:
        return fallback
    try:
        value = int(raw)
        return value if value > 0 else fallback
    except ValueError:
        return fallback
