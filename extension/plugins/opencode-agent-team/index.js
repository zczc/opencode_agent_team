import fs from "fs"
import path from "path"

const EXEC_MAX_TEXT = Number(process.env.SWARM_EXEC_LOG_MAX_TEXT || 4000)
const EXEC_MAX_ARGS = Number(process.env.SWARM_EXEC_LOG_MAX_ARGS || 3000)
const EXEC_MAX_META = Number(process.env.SWARM_EXEC_LOG_MAX_META || 2000)
const EXEC_TEXT_STEP = Number(process.env.SWARM_EXEC_LOG_TEXT_STEP || 240)

function parseJSONSafe(raw, fallback = {}) {
  try {
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

function readFileSafe(filePath, fallback = "") {
  try {
    return fs.readFileSync(filePath, "utf-8")
  } catch {
    return fallback
  }
}

function ensureDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true })
  } catch {}
}

function appendJsonl(filePath, payload) {
  try {
    ensureDir(path.dirname(filePath))
    fs.appendFileSync(filePath, JSON.stringify(payload) + "\n", "utf-8")
  } catch {}
}

function tailText(text, maxLen = 200) {
  if (!text) return ""
  return text.length <= maxLen ? text : text.slice(0, maxLen) + "..."
}

function simpleHash(text) {
  let hash = 2166136261
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(16)
}

function isSensitiveKey(key) {
  return /token|secret|password|authorization|api[-_]?key|cookie|credential/i.test(String(key || ""))
}

function sanitizeValue(value, depth = 0) {
  if (depth >= 4) return "[truncated-depth]"
  if (value === null || value === undefined) return value
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return value
  if (Array.isArray(value)) return value.slice(0, 40).map((item) => sanitizeValue(item, depth + 1))
  if (typeof value === "object") {
    const result = {}
    for (const [key, val] of Object.entries(value)) {
      if (isSensitiveKey(key)) {
        result[key] = "[redacted]"
      } else {
        result[key] = sanitizeValue(val, depth + 1)
      }
    }
    return result
  }
  return String(value)
}

function stringifyPreview(value, maxLen) {
  if (typeof value === "string") return tailText(value, maxLen)
  try {
    return tailText(JSON.stringify(sanitizeValue(value)), maxLen)
  } catch {
    return tailText(String(value), maxLen)
  }
}

function parsePlan(raw) {
  const matches = [...raw.matchAll(/```(?:json|JSON)?\s*\n([\s\S]*?)\n```/g)]
  for (let i = matches.length - 1; i >= 0; i -= 1) {
    const m = matches[i]
    const parsed = parseJSONSafe(m[1], null)
    if (parsed && typeof parsed === "object") return parsed
  }
  const full = parseJSONSafe(raw, null)
  if (full && typeof full === "object") return full
  const start = raw.indexOf("{")
  const end = raw.lastIndexOf("}")
  if (start >= 0 && end > start) {
    return parseJSONSafe(raw.slice(start, end + 1), {})
  }
  return {}
}

function statusIcon(status) {
  return (
    {
      DONE: "✅",
      IN_PROGRESS: "🔄",
      PENDING: "⏳",
      BLOCKED: "🔒",
      FAILED: "❌",
    }[status] || "•"
  )
}

export default async function opencodeAgentTeamPlugin(input) {
  const bbRoot = path.join(input.directory, ".blackboard")
  const agentName = process.env.AGENT_NAME || "architect"
  let lastSummaryHash = ""
  const lastPartTextLen = new Map()

  function resolveSessionId() {
    const envSession = (process.env.SWARM_SESSION_ID || "").trim()
    if (envSession) return envSession
    const fromFile = readFileSafe(path.join(bbRoot, "current_session"), "").trim()
    return fromFile || "default"
  }

  function resolveSessionIdFromEvent(evt) {
    if (!evt || !evt.properties) return ""
    const props = evt.properties
    if (props.sessionID) return String(props.sessionID)
    if (props.info && props.info.sessionID) return String(props.info.sessionID)
    if (props.part && props.part.sessionID) return String(props.part.sessionID)
    return ""
  }

  function sessionState(sessionId) {
    const bbDir = path.join(bbRoot, "sessions", sessionId)
    const stateRaw = readFileSafe(path.join(bbDir, "global_indices", "orchestrator_state.json"), "{}")
    const state = parseJSONSafe(stateRaw, {})
    return { bbDir, state }
  }

  function logEvent(level, event, extra = {}, sessionIdOverride = "") {
    const sessionId = sessionIdOverride || resolveSessionId()
    const { bbDir, state } = sessionState(sessionId)
    appendJsonl(path.join(bbDir, "logs", "plugin", `${agentName}.jsonl`), {
      ts: new Date().toISOString(),
      level,
      component: "plugin",
      event,
      run_id: state.run_id || "run-unknown",
      mission_id: state.mission_id || "mission-unknown",
      session_id: sessionId,
      agent: agentName,
      ...extra,
    })
  }

  function logExecution(level, event, payload = {}, sessionIdOverride = "") {
    const sessionId = sessionIdOverride || resolveSessionId()
    const { bbDir, state } = sessionState(sessionId)
    appendJsonl(path.join(bbDir, "logs", "execution", `${agentName}.jsonl`), {
      ts: new Date().toISOString(),
      level,
      component: "execution",
      event,
      run_id: state.run_id || "run-unknown",
      mission_id: state.mission_id || "mission-unknown",
      session_id: sessionId,
      agent: agentName,
      payload,
    })
  }

  function summarizePart(part) {
    if (!part || typeof part !== "object") return null
    const base = {
      part_id: part.id,
      message_id: part.messageID,
      part_type: part.type,
    }
    if (part.type === "text" || part.type === "reasoning") {
      const text = String(part.text || "")
      const key = String(part.id || "")
      const prev = lastPartTextLen.get(key) || 0
      const next = text.length
      const shouldLog = next === 0 || next - prev >= EXEC_TEXT_STEP
      if (!shouldLog) return null
      lastPartTextLen.set(key, next)
      return {
        ...base,
        text_len: next,
        text_preview: tailText(text, EXEC_MAX_TEXT),
      }
    }
    if (part.type === "tool") {
      const state = part.state || {}
      return {
        ...base,
        tool: part.tool || "",
        call_id: part.callID || "",
        status: state.status || "",
        input_preview: stringifyPreview(state.input || {}, EXEC_MAX_ARGS),
        output_preview: stringifyPreview(state.output || "", EXEC_MAX_TEXT),
        metadata_preview: stringifyPreview(state.metadata || {}, EXEC_MAX_META),
        error_preview: tailText(String(state.error || ""), 400),
      }
    }
    return base
  }

  return {
    event: async ({ event }) => {
      const sessionId = resolveSessionIdFromEvent(event) || resolveSessionId()
      if (!event || !event.type) return

      if (event.type === "message.updated") {
        const info = event.properties?.info || {}
        if (info.role !== "assistant") return
        logExecution(
          "info",
          "assistant.message.updated",
          {
            message_id: info.id || "",
            parent_id: info.parentID || "",
            model: info.model ? `${info.model.providerID}/${info.model.modelID}` : "",
            finish: info.finish || "",
            cost: info.cost ?? null,
            tokens: info.tokens ?? null,
            time: info.time ?? null,
          },
          sessionId,
        )
        return
      }

      if (event.type === "message.part.delta") {
        const props = event.properties || {}
        const delta = String(props.delta || "")
        if (!delta) return
        logExecution(
          "debug",
          "assistant.part.delta",
          {
            message_id: props.messageID || "",
            part_id: props.partID || "",
            field: props.field || "",
            delta_len: delta.length,
            delta_preview: tailText(delta, EXEC_MAX_TEXT),
          },
          sessionId,
        )
        return
      }

      if (event.type === "message.part.updated") {
        const summary = summarizePart(event.properties?.part)
        if (!summary) return
        logExecution("info", "assistant.part.updated", summary, sessionId)
      }
    },
    "tool.execute.after": async (toolInput, toolOutput) => {
      const sessionId = toolInput?.sessionID || resolveSessionId()
      logExecution(
        "info",
        "tool.execute.after",
        {
          tool: toolInput?.tool || "",
          call_id: toolInput?.callID || "",
          args_preview: stringifyPreview(toolInput?.args || {}, EXEC_MAX_ARGS),
          title: String(toolOutput?.title || ""),
          output_len: typeof toolOutput?.output === "string" ? toolOutput.output.length : null,
          output_preview: stringifyPreview(toolOutput?.output || "", EXEC_MAX_TEXT),
          metadata_preview: stringifyPreview(toolOutput?.metadata || {}, EXEC_MAX_META),
        },
        sessionId,
      )
    },
    "experimental.chat.system.transform": async (_hookInput, output) => {
      const sessionId = resolveSessionId()
      const bbDir = path.join(bbRoot, "sessions", sessionId)
      const registryPath = path.join(bbDir, "global_indices", "registry.json")
      if (!fs.existsSync(registryPath)) {
        logEvent("debug", "plugin.system_transform.skipped", { reason: "no_registry", session_id: sessionId })
        return
      }

      const registry = parseJSONSafe(readFileSafe(registryPath, "{}"), {})
      const workers = Array.isArray(registry.workers) ? registry.workers : []
      const running = registry.status === "running" || workers.length > 0
      if (!running) {
        logEvent("debug", "plugin.system_transform.skipped", { reason: "not_running", session_id: sessionId })
        return
      }

      const planPath = path.join(bbDir, "global_indices", "central_plan.md")
      const plan = fs.existsSync(planPath) ? parsePlan(readFileSafe(planPath, "")) : {}
      const tasks = Array.isArray(plan.tasks) ? plan.tasks : []
      const taskLines = tasks.map((task) => {
        const icon = statusIcon(task.status)
        const worker = task.assigned_worker ? ` (${task.assigned_worker})` : ""
        const title = task.title || task.description || ""
        return `  ${icon} ${task.id}: ${title}${worker}`
      })

      const inboxPath = path.join(bbDir, "inboxes", `${agentName}.json`)
      const inboxRaw = readFileSafe(inboxPath, "[]")
      const inbox = parseJSONSafe(inboxRaw, [])
      const unread = Array.isArray(inbox) ? inbox.filter((m) => !m.read) : []
      const unreadPreview = unread
        .slice(0, 3)
        .map((m) => `  [${m.from || "unknown"}] ${tailText(String(m.content || ""), 100)}`)
        .join("\n")

      const workerLines = workers.map((w) => {
        const suffix = w.current_task ? ` -> ${w.current_task}` : ""
        return `  ${w.name}: ${w.status || "unknown"}${suffix}`
      })

      const summary = [
        "[Swarm Status]",
        `Run: ${registry.run_id || "unknown"}`,
        `Mission: ${plan.mission_goal || "unknown"} (${plan.status || "IN_PROGRESS"})`,
        `Workers: ${workers.length}`,
        `Tasks: total=${tasks.length}, done=${tasks.filter((t) => t.status === "DONE").length}, in_progress=${tasks.filter((t) => t.status === "IN_PROGRESS").length}`,
        taskLines.length ? "Tasks:\n" + taskLines.join("\n") : "Tasks: (none)",
        workerLines.length ? "Workers:\n" + workerLines.join("\n") : "Workers: (none)",
        unread.length ? `Unread Messages (${unread.length}):\n${unreadPreview}` : "Unread Messages: 0",
        "Commands:\n"
          + "  /swarm status\n"
          + "  /swarm panel on\n"
          + "  /swarm panel off\n"
          + "  /swarm panel status\n"
          + "  /swarm send worker-0 <instruction>\n"
          + "  /swarm stop\n"
          + "  /swarm stop --force",
      ].join("\n\n")

      output.system.push(summary)

      const summaryHash = simpleHash(summary)
      if (summaryHash !== lastSummaryHash) {
        lastSummaryHash = summaryHash
        logEvent("info", "plugin.system_transform.applied", {
          session_id: sessionId,
          tasks: tasks.length,
          workers: workers.length,
          unread: unread.length,
          injected_chars: summary.length,
        })
      }
    },
  }
}
