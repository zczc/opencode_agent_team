# opencode_agent_team

A zero-core-change Agent Team extension for [opencode](https://github.com/anomalyco/opencode).

English | [中文说明](./README_CN.md)

## Features

- **Multi-Agent Collaboration**: Architect plans, multiple Workers execute tasks in parallel
- **Real-time Status Panel**: Live task progress, worker status, and logs via tmux split-pane
- **Session Isolation**: Each mission runs in an isolated session directory
- **Smart Scheduling**: Automatic task assignment, dependency management, and retry logic
- **Structured Logging**: JSONL logs for debugging and auditing
- **Safe Cleanup**: Auto-cleanup of stale processes, graceful and forced shutdown support

## Requirements

- **Python 3.10+**
- **tmux**: Required for status panel display
- **opencode**: Installed and configured

### Installing tmux

**macOS:**
```bash
brew install tmux
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tmux
```

**Other systems:**
See [tmux installation guide](https://github.com/tmux/tmux/wiki/Installing)

## How To Use

### 1) Install

```bash
./install.sh
```

Optional arguments:

```bash
./install.sh --opencode-dir /path/to/.opencode --project-dir . --python python3
```

The installer will:
- Copy plugin, skill, and MCP configs to `~/.opencode/`
- Install Python dependencies (mcp, anthropic, etc.)
- Configure MCP server and blackboard tools

### 2) Restart opencode

After install, restart opencode so plugin/skill/MCP config is reloaded.

### 3) Start a mission

```text
/swarm <your mission>
```

Examples:

```text
/swarm Analyze this project's architecture and generate documentation
/swarm Refactor the authentication module for better maintainability
/swarm Add unit tests for all API endpoints
```

This will:
1. Start the orchestrator process
2. Open status panel in tmux (auto split-pane)
3. Architect generates task plan
4. Workers execute tasks in parallel
5. Display completion summary when done

### 4) Check status

**View current status:**
```text
/swarm status
```

**Open/close status panel:**
```text
/swarm panel on      # Open status panel in tmux
/swarm panel off     # Close status panel
/swarm panel status  # Check panel status
```

**Status panel keybindings:**
- `j/k` - Scroll up/down
- `u/d` - Page up/down
- `g/G` - Jump to top/bottom
- `s` - Switch to status view
- `m/o/e/w` - Switch log view (main/orchestrator/error/worker)
- `v/V` - Filter by worker
- `c/C` - Filter by task
- `x` - Clear filters
- `q` - Quit panel

### 5) Interact with Workers

Send instructions to specific workers:

```text
/swarm send worker-0 Please prioritize authentication-related tasks
/swarm send worker-1 Skip tests and commit directly
```

### 6) Stop mission

**Graceful stop (wait for current tasks to finish):**
```text
/swarm stop
```

**Force stop (terminate all processes immediately):**
```text
/swarm stop --force
```

### 7) Remove extension

```bash
./uninstall.sh
```

Optional:

```bash
./uninstall.sh --opencode-dir /path/to/.opencode
```

## Directory Structure

```
.blackboard/
├── sessions/
│   └── swarm-<timestamp>-<pid>/
│       ├── global_indices/
│       │   ├── central_plan.md          # Task plan (JSON format)
│       │   ├── orchestrator_state.json  # Orchestrator state
│       │   ├── registry.json            # Worker registry
│       │   └── swarm_completion_report.md # Completion report
│       ├── resources/                   # Worker output files
│       │   ├── task-001.md
│       │   └── task-002.md
│       ├── logs/                        # Structured logs
│       │   ├── orchestrator/
│       │   └── workers/
│       └── inboxes/                     # Agent message queues
│           ├── architect.json
│           └── worker-0.json
└── current_session                      # Current active session ID
```

## Workflow

1. **User starts mission** → `/swarm <mission>`
2. **Architect generates plan** → Creates `central_plan.md` with task list and dependencies
3. **Orchestrator schedules** → Assigns tasks to idle workers based on dependencies
4. **Workers execute** → Execute tasks in parallel, update status and results
5. **State sync** → Read/write shared state via blackboard MCP tools
6. **Completion summary** → Generate `swarm_completion_report.md`, output to terminal

## Design Approach

The design keeps opencode source unchanged and builds everything through extension points.

- **Skill-driven entry**: `/swarm` is the Architect command surface
- **Runtime orchestrator**: A Python orchestrator starts and monitors worker agents
- **Shared blackboard**: Session-scoped plan/state/message files under `.blackboard/sessions/<session_id>/`
- **MCP coordination**: Architect/Workers update shared state through MCP tools
- **Optional live panel**: A side panel (tmux) or standalone terminal renders status/log views
- **Safety-first execution**: Process cleanup, session isolation, and explicit status transitions

In short: **plugin + skill + MCP + blackboard + orchestrator**, without modifying opencode core.

## FAQ

**Q: Why is tmux required?**
A: The status panel needs to display real-time information in a separate terminal pane. tmux provides split-pane and session management capabilities, making it the best choice.

**Q: Can I run without the status panel?**
A: Yes. Use `/swarm panel off` to disable the panel and check status with `/swarm status`.

**Q: Can I adjust the number of workers?**
A: Yes. Modify the orchestrator startup parameter `--workers N`, or adjust the default value in the skill configuration.

**Q: What happens if a task fails?**
A: The orchestrator will automatically retry failed tasks (up to 2 times). If it still fails, it will be marked as FAILED and other tasks will continue.

**Q: How do I view detailed logs?**
A: Logs are located in `.blackboard/sessions/<session-id>/logs/`. Use the status panel's log view (keybindings `m/o/e/w`) or directly view the JSONL files.

**Q: Are session directories automatically cleaned up?**
A: No. Old session directories need to be manually cleaned. It's recommended to periodically delete old directories under `.blackboard/sessions/`.

## Contributing

Issues and Pull Requests are welcome!

## License

MIT