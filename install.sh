#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/extension"

OPENCODE_DIR="$HOME/.opencode"
PYTHON_BIN="python3"
PROJECT_DIR="."

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opencode-dir)
      OPENCODE_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: ./install.sh [options]

Options:
  --opencode-dir <dir>   Install target (default: ~/.opencode)
  --python <bin>         Python executable for MCP (default: python3)
  --project-dir <path>   Default project dir arg for MCP config (default: .)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OPENCODE_DIR/skills" "$OPENCODE_DIR/plugins"
rm -rf "$OPENCODE_DIR/skills/swarm"
mkdir -p "$OPENCODE_DIR/skills/swarm"
cp -R "$SRC_DIR/skills/swarm/." "$OPENCODE_DIR/skills/swarm/"
mkdir -p "$OPENCODE_DIR/plugins/opencode-agent-team"
cp "$SRC_DIR/plugins/opencode-agent-team/index.js" "$OPENCODE_DIR/plugins/opencode-agent-team.js"
cp "$SRC_DIR/plugins/opencode-agent-team/index.js" "$OPENCODE_DIR/plugins/opencode-agent-team/index.js"
chmod +x \
  "$OPENCODE_DIR/skills/swarm/scripts/orchestrator.py" \
  "$OPENCODE_DIR/skills/swarm/scripts/blackboard_mcp_server.py" \
  "$OPENCODE_DIR/skills/swarm/scripts/swarm_panel.py"

CONFIG_FILE="$OPENCODE_DIR/opencode.json"
PLUGIN_URI="file://$OPENCODE_DIR/plugins/opencode-agent-team.js"
MCP_SCRIPT="$OPENCODE_DIR/skills/swarm/scripts/blackboard_mcp_server.py"

CONFIG_FILE_ENV="$CONFIG_FILE" \
PLUGIN_URI_ENV="$PLUGIN_URI" \
OPENCODE_DIR_ENV="$OPENCODE_DIR" \
PYTHON_BIN_ENV="$PYTHON_BIN" \
MCP_SCRIPT_ENV="$MCP_SCRIPT" \
PROJECT_DIR_ENV="$PROJECT_DIR" \
python3 - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import unquote

config_file = Path(os.environ["CONFIG_FILE_ENV"])
plugin_uri = os.environ["PLUGIN_URI_ENV"]
opencode_dir = Path(os.environ["OPENCODE_DIR_ENV"]).expanduser().resolve()
python_bin = os.environ["PYTHON_BIN_ENV"]
mcp_script = os.environ["MCP_SCRIPT_ENV"]
project_dir = os.environ["PROJECT_DIR_ENV"]

if config_file.exists():
    data = json.loads(config_file.read_text(encoding="utf-8"))
else:
    data = {}

plugins_raw = data.get("plugin") or []
plugins: list[str] = []
for item in plugins_raw:
    if isinstance(item, str):
        plugins.append(item)

plugin_base = (opencode_dir / "plugins" / "opencode-agent-team").resolve()
plugin_base_index = (plugin_base / "index.js").resolve()
plugin_file = (opencode_dir / "plugins" / "opencode-agent-team.js").resolve()

def normalize_plugin_entry(entry: str) -> str:
    value = entry.strip()
    if value.startswith("file://"):
        value = unquote(value[7:])
    p = Path(value).expanduser()
    if p.is_absolute():
        try:
            return str(p.resolve(strict=False))
        except TypeError:
            return str(p.resolve())
    return value

legacy_candidates = {
    str(plugin_base),
    str(plugin_base_index),
    str(plugin_file),
}

cleaned_plugins: list[str] = []
seen_plugins: set[str] = set()
for item in plugins:
    normalized = normalize_plugin_entry(item)
    if normalized in legacy_candidates:
        continue
    if normalized.startswith(str(plugin_base) + os.sep):
        continue
    if item in seen_plugins:
        continue
    seen_plugins.add(item)
    cleaned_plugins.append(item)

if plugin_uri not in seen_plugins:
    cleaned_plugins.append(plugin_uri)
data["plugin"] = cleaned_plugins

mcp = data.get("mcp") or {}
mcp["agent_team_blackboard"] = {
    "type": "local",
    "command": [python_bin, mcp_script, "--project-dir", project_dir],
    "environment": {"AGENT_NAME": "architect"},
    "enabled": True,
}
data["mcp"] = mcp

config_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY

cat <<EOF
Installed opencode agent team extension.

Installed files:
  $OPENCODE_DIR/skills/swarm
  $OPENCODE_DIR/plugins/opencode-agent-team.js
  $OPENCODE_DIR/plugins/opencode-agent-team/index.js
  $CONFIG_FILE (updated)

Next steps:
  1) Restart opencode.
  2) In your project, run: /swarm <your mission>
EOF
