---
name: Central Mission Plan
description: "JSON-structured Task Graph for the opencode agent team."
usage_policy:
  - "Architect manages task graph and dependencies."
  - "Workers only claim and update compatible tasks."
  - "Use CAS writes for concurrent updates."
schema: JSON Code Block
---

# Mission Plan (JSON)

```json
{
  "schema_version": "1.1",
  "mission_goal": "[High-level Goal]",
  "status": "IN_PROGRESS",
  "summary": null,
  "session_id": "[set by orchestrator]",
  "created_at": null,
  "updated_at": null,
  "tasks": [
    {
      "id": "task-001",
      "type": "standard",
      "title": "Initial Analysis",
      "description": "Analyze project status and create implementation checklist.",
      "status": "PENDING",
      "dependencies": [],
      "assignees": [],
      "assigned_worker": null,
      "start_time": null,
      "end_time": null,
      "artifact_link": null,
      "result_summary": null,
      "result": null,
      "retry_count": 0,
      "last_error": null
    },
    {
      "id": "task-002",
      "type": "standard",
      "title": "Implement Main Change",
      "description": "Implement the requested change based on task-001 output.",
      "status": "BLOCKED",
      "dependencies": ["task-001"],
      "assignees": [],
      "assigned_worker": null,
      "start_time": null,
      "end_time": null,
      "artifact_link": null,
      "result_summary": null,
      "result": null,
      "retry_count": 0,
      "last_error": null
    }
  ]
}
```
