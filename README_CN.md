# opencode_agent_team

一个基于 [opencode](https://github.com/anomalyco/opencode) 的多 Agent 扩展方案，目标是：
**不改 opencode 核心源码，只通过扩展点实现 Agent Team 协作能力。**

[English](./README.md) | 中文

## 功能特性

- **多 Agent 协作**：Architect 负责规划，多个 Worker 并行执行任务
- **实时状态面板**：通过 tmux 分屏实时查看任务进度、Worker 状态和日志
- **会话隔离**：每次任务在独立的会话目录中运行，互不干扰
- **智能调度**：自动任务分配、依赖管理、失败重试
- **完整日志**：结构化 JSONL 日志，便于调试和审计
- **安全清理**：自动清理僵尸进程，支持优雅停止和强制终止

## 系统要求

- **Python 3.10+**
- **tmux**：用于状态面板显示（必需）
- **opencode**：已安装并配置

### 安装 tmux

**macOS:**
```bash
brew install tmux
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tmux
```

**其他系统:**
参考 [tmux 官方文档](https://github.com/tmux/tmux/wiki/Installing)

## 怎么用

### 1）安装

```bash
./install.sh
```

可选参数：

```bash
./install.sh --opencode-dir /path/to/.opencode --project-dir . --python python3
```

安装脚本会：
- 复制 plugin、skill、MCP 配置到 `~/.opencode/`
- 安装 Python 依赖（mcp、anthropic 等）
- 配置 MCP server 和 blackboard 工具

### 2）重启 opencode

安装后需要重启 opencode，确保插件、skill、MCP 配置生效。

### 3）发起任务

```text
/swarm <你的任务描述>
```

示例：

```text
/swarm 分析这个项目的架构并生成文档
/swarm 重构认证模块，提高代码可维护性
/swarm 为所有 API 端点添加单元测试
```

执行后会：
1. 启动 orchestrator 进程
2. 在 tmux 中打开状态面板（自动分屏）
3. Architect 生成任务计划
4. Worker 并行执行任务
5. 完成后显示结果摘要

### 4）查看状态

**查看当前状态：**
```text
/swarm status
```

**打开/关闭状态面板：**
```text
/swarm panel on    # 在 tmux 中打开状态面板
/swarm panel off   # 关闭状态面板
/swarm panel status # 查看面板状态
```

**状态面板快捷键：**
- `j/k` - 上下滚动
- `u/d` - 翻页
- `g/G` - 跳到顶部/底部
- `s` - 切换到状态视图
- `m/o/e/w` - 切换日志视图（main/orchestrator/error/worker）
- `v/V` - 过滤 worker
- `c/C` - 过滤任务
- `x` - 清除过滤器
- `q` - 退出面板

### 5）与 Worker 交互

向特定 Worker 发送指令：

```text
/swarm send worker-0 请优先处理认证相关的任务
/swarm send worker-1 跳过测试，直接提交代码
```

### 6）停止任务

**优雅停止（等待当前任务完成）：**
```text
/swarm stop
```

**强制停止（立即终止所有进程）：**
```text
/swarm stop --force
```

### 7）卸载

```bash
./uninstall.sh
```

可选：

```bash
./uninstall.sh --opencode-dir /path/to/.opencode
```

## 目录结构

```
.blackboard/
├── sessions/
│   └── swarm-<timestamp>-<pid>/
│       ├── global_indices/
│       │   ├── central_plan.md          # 任务计划（JSON 格式）
│       │   ├── orchestrator_state.json  # Orchestrator 状态
│       │   ├── registry.json            # Worker 注册表
│       │   └── swarm_completion_report.md # 完成报告
│       ├── resources/                   # Worker 输出文件
│       │   ├── task-001.md
│       │   └── task-002.md
│       ├── logs/                        # 结构化日志
│       │   ├── orchestrator/
│       │   └── workers/
│       └── inboxes/                     # Agent 消息队列
│           ├── architect.json
│           └── worker-0.json
└── current_session                      # 当前活跃会话 ID
```

## 工作流程

1. **用户发起任务** → `/swarm <mission>`
2. **Architect 生成计划** → 创建 `central_plan.md`，包含任务列表和依赖关系
3. **Orchestrator 调度** → 根据依赖关系分配任务给空闲 Worker
4. **Worker 执行** → 并行执行任务，更新状态和结果
5. **状态同步** → 通过 blackboard MCP 工具读写共享状态
6. **完成汇总** → 生成 `swarm_completion_report.md`，输出到终端

## 设计思路

整体思路是”扩展式编排”，用最小侵入方式把多 Agent 流程挂到 opencode 上。

- **Skill 作为入口**：`/swarm` 负责把用户任务交给 Architect
- **Orchestrator 负责运行**：Python orchestrator 启动并管理多个 worker
- **Blackboard 共享状态**：任务计划、状态、消息都在会话级目录中维护
- **MCP 作为协作通道**：Architect/Worker 通过 MCP 工具读写黑板
- **状态面板用于观察**：可在 tmux 分屏或独立终端里实时看执行状态
- **安全与清理优先**：会话隔离、状态约束、进程可回收

一句话：**plugin + skill + MCP + blackboard + orchestrator**，完成多 Agent 协作。

## 常见问题

**Q: 为什么需要 tmux？**
A: 状态面板需要在独立的终端窗格中显示实时信息。tmux 提供了分屏和会话管理能力，是最佳选择。

**Q: 可以不用状态面板吗？**
A: 可以。使用 `/swarm panel off` 关闭面板，通过 `/swarm status` 查看状态。

**Q: Worker 数量可以调整吗？**
A: 可以。修改 orchestrator 启动参数 `--workers N`，或在 skill 配置中调整默认值。

**Q: 任务失败了怎么办？**
A: Orchestrator 会自动重试失败的任务（最多 2 次）。如果仍然失败，会标记为 FAILED 并继续执行其他任务。

**Q: 如何查看详细日志？**
A: 日志位于 `.blackboard/sessions/<session-id>/logs/`，使用状态面板的日志视图（快捷键 `m/o/e/w`）或直接查看 JSONL 文件。

**Q: 会话目录会自动清理吗？**
A: 不会。旧会话目录需要手动清理。建议定期删除 `.blackboard/sessions/` 下的旧目录。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT
