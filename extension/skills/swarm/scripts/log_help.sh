#!/bin/bash
# Swarm Log Viewer - Quick Reference

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SWARM LOG VIEWER - QUICK REFERENCE                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  view_logs.py <session-id>                    # View all logs
  view_logs.py <session-id> --tail 50          # Last 50 lines
  view_logs.py <session-id> --follow           # Follow logs (Ctrl+C to stop)

FILTER BY COMPONENT:
  --component orchestrator     # Orchestrator logs
  --component worker-0         # Worker-0 logs
  --component mcp              # MCP server logs
  --component plugin           # Plugin logs
  --component execution        # Execution logs

FILTER BY EVENT:
  --event task.claimed                # Task assignments
  --event mcp.task.status_changed     # Task status changes
  --event worker.idle                 # Worker idle events
  --event worker.health.timeout       # Worker timeouts
  --event task.released               # Task releases

FILTER BY LEVEL:
  --level error                # Only errors
  --level warn                 # Only warnings
  --level info                 # Only info
  --level debug                # Only debug

COMMON DEBUGGING COMMANDS:
  # Task stuck?
  view_logs.py <session-id> --event task.claimed --tail 10
  view_logs.py <session-id> --event worker.health.timeout

  # Worker crashed?
  view_logs.py <session-id> --event worker.health.dead
  view_logs.py <session-id> --component worker-0 --level error

  # Task failed?
  view_logs.py <session-id> --event task.released
  view_logs.py <session-id> --component mcp --level error

  # See what's happening now?
  view_logs.py <session-id> --follow

EXAMPLES:
  # View orchestrator errors
  view_logs.py swarm-123 --component orchestrator --level error

  # Follow task status changes
  view_logs.py swarm-123 --event mcp.task.status_changed --follow

  # Last 20 worker-0 logs
  view_logs.py swarm-123 --component worker-0 --tail 20

  # All errors across all components
  view_logs.py swarm-123 --level error

FULL DOCUMENTATION:
  ~/.opencode/skills/swarm/docs/LOGGING.md

EOF
