#!/usr/bin/env bash
set -euo pipefail

OPENCODE_DIR="$HOME/.opencode"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opencode-dir)
      OPENCODE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: ./uninstall.sh [options]

Options:
  --opencode-dir <dir>   Uninstall target (default: ~/.opencode)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

rm -rf "$OPENCODE_DIR/skills/swarm"
rm -f "$OPENCODE_DIR/plugins/opencode-agent-team.js"
rm -rf "$OPENCODE_DIR/plugins/opencode-agent-team"

CONFIG_FILE="$OPENCODE_DIR/opencode.json"
PLUGIN_URI="file://$OPENCODE_DIR/plugins/opencode-agent-team.js"
PLUGIN_URI_DIR="file://$OPENCODE_DIR/plugins/opencode-agent-team/index.js"
PLUGIN_URI_BASE="file://$OPENCODE_DIR/plugins/opencode-agent-team"

if [[ -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE_ENV="$CONFIG_FILE" \
  PLUGIN_URI_ENV="$PLUGIN_URI" \
  PLUGIN_URI_DIR_ENV="$PLUGIN_URI_DIR" \
  PLUGIN_URI_BASE_ENV="$PLUGIN_URI_BASE" \
  OPENCODE_DIR_ENV="$OPENCODE_DIR" \
  python3 - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import unquote

config_file = Path(os.environ["CONFIG_FILE_ENV"])
plugin_uri = os.environ["PLUGIN_URI_ENV"]
plugin_uri_dir = os.environ["PLUGIN_URI_DIR_ENV"]
plugin_uri_base = os.environ["PLUGIN_URI_BASE_ENV"]
opencode_dir = Path(os.environ["OPENCODE_DIR_ENV"]).expanduser().resolve()

try:
    data = json.loads(config_file.read_text(encoding="utf-8"))
except Exception:
    data = {}

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
    plugin_uri,
    plugin_uri_dir,
    plugin_uri_base,
    str(plugin_base),
    str(plugin_base_index),
    str(plugin_file),
}

plugins = data.get("plugin") or []
keep_plugins: list[str] = []
seen_plugins: set[str] = set()
for item in plugins:
    if not isinstance(item, str):
        continue
    normalized = normalize_plugin_entry(item)
    if item in legacy_candidates or normalized in legacy_candidates:
        continue
    if normalized.startswith(str(plugin_base) + os.sep):
        continue
    if item in seen_plugins:
        continue
    seen_plugins.add(item)
    keep_plugins.append(item)
plugins = keep_plugins
if plugins:
    data["plugin"] = plugins
elif "plugin" in data:
    del data["plugin"]

mcp = data.get("mcp") or {}
if "agent_team_blackboard" in mcp:
    del mcp["agent_team_blackboard"]
if mcp:
    data["mcp"] = mcp
elif "mcp" in data:
    del data["mcp"]

config_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
fi

cat <<EOF
Uninstalled opencode agent team extension from:
  $OPENCODE_DIR
EOF
