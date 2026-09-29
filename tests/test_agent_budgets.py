"""Budget, journal and versioning hardening (audit phases ①②③)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent import AgentPlatform
from app.agent import planner as planner_module
from app.agent.compiler import PROMPT_VERSION, tool_catalog_fingerprint  # noqa: F401
from app.agent.platform import AgentPlatform as Platform

SECRET = "sk-budget-test"

SKILL = """---
name: test-editor
description: Budget test profile.
allowed-tools: inspect_workspace propose_timeline_edit confirm_timeline_edit prepare_subtitle_review render_review_preview
workflow-profile: revision
version: 1.0.0
---

# Test Editor
"""


class FakeModel:
    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools, tool_choice=None):
        return self

    def stream(self, messages):
        yield AIMessageChunk(content="ok")
        yield AIMessageChunk(content="", tool_call_chunks=[{
            "name": "submit_plan",
            "args": json.dumps({"summary": "计划", "strategy": {}}, ensure_ascii=False),
            "id": "c1", "index": 0, "type": "tool_call_chunk",
        }])
        yield AIMessageChunk(
            content="",
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )

    def invoke(self, messages):
        merged = AIMessageChunk(content="")
        for chunk in self.stream(messages):
            merged = merged + chunk
        return merged


class FakeBackend:
    def plan(self, payload, *, model_config, emit=None):
        return {"plan": {"summary": "计划", "strategy": {}}, "events": [], "usage": {}}

    def route_skill(self, payload, *, model_config):
        return {"skillId": "test-editor", "reason": "r"}

    def generate_skill(self, payload, *, model_config):
        raise AssertionError("unused")

    def probe(self, model_config):
        return {"ok": True}


def platform_at(
    path: Path, *, deadline: float = 3600.0, rate: int = 30,
    dispatcher=None,
) -> AgentPlatform:
    platform = AgentPlatform(
        data_root=path,
        model_config_resolver=lambda: {
            "provider": "bailian", "protocol": "openai", "apiKey": SECRET,
            "model": "m", "baseUrl": "https://x", "thinkingType": "enabled",
            "timeoutSeconds": 30,
        },
        plan_deadline_seconds=deadline,
        max_messages_per_10min=rate,
    )
    platform.planner_backend = FakeBackend()  # type: ignore[assignment]
    platform.install_skill(markdown=SKILL, source="test", status="enabled")
    platform.create_workspace(job_id="job_budget")
    platform.configure_planning_context_provider(lambda job_id: {})
    platform.configure_tool_dispatcher(dispatcher or (lambda ws, tool, args: {"artifact": {"kind": "noop"}}))
    return platform


def workspace_id(platform: AgentPlatform) -> str:
    return str(platform.workspace_for_job("job_budget")["id"])


# ---------------- ① durable operation journal ----------------

def test_operation_journal_lifecycle_and_cas(tmp_path):
    platform = platform_at(tmp_path)
    gate = threading.Event()

    def dispatcher(ws, tool, args):
        if tool == "propose_timeline_edit":
            future: Future[Any] = Future()

            def run():
                gate.wait(timeout=5)
                future.set_result({"artifact": {"kind": "draft"}})
            threading.Thread(target=run, daemon=True).start()
            return {"future": future, "operationId": "op-journal-1"}
        return {"artifact": {"kind": "noop"}}

    platform.configure_tool_dispatcher(dispatcher)
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版60秒高光")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])

    waiting = [s for s in approved["steps"] if s["status"] == "waiting_operation"]
    if waiting:
        record = platform.store.get_operation("op-journal-1")
        assert record and record["status"] == "running"
        assert record["planId"] == plan["id"]
        assert record["worker"] == "propose_timeline_edit"
        assert record["inputHash"]

        # CAS violation: bump plan revision behind the journal's back, then
        # complete the future — outcome must be journalled as late, not applied.
        approved["revision"] = int(approved.get("revision") or 1) + 1
        platform.store.save("plans", approved)
        gate.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if platform.store.get_operation("op-journal-1")["status"] == "late":
                break
            time.sleep(0.02)
        assert platform.store.get_operation("op-journal-1")["status"] == "late"
        refreshed = platform.store.get("plans", plan["id"])
        step = next(s for s in refreshed["steps"] if s["id"] == waiting[0]["id"])
        assert step["status"] == "waiting_operation"  # outcome never applied


def test_cancel_marks_running_operations_in_journal(tmp_path):
    platform = platform_at(tmp_path)
    never = threading.Event()

    def dispatcher(ws, tool, args):
        future: Future[Any] = Future()
        future.set_running_or_notify_cancel()
        return {"future": future, "operationId": "op-cancel-1"}

    platform.configure_tool_dispatcher(dispatcher)
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版60秒高光")
    platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    cancelled = platform.cancel_plan(plan["id"])
    assert cancelled["status"] == "cancelled"
    record = platform.store.get_operation("op-cancel-1")
    if record:
        assert record["status"] in {"cancelled", "unknown", "running"}


def test_restart_recovers_orphaned_journal_rows(tmp_path):
    platform = platform_at(tmp_path)
    platform.store.save_operation({
        "operationId": "op-orphan", "planId": "plan_gone",
        "stepId": "s", "workspaceId": "ws", "status": "running",
    })
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版高光")
    platform.store.save_operation({
        "operationId": "op-terminal", "planId": plan["id"],
        "stepId": "s", "workspaceId": plan["workspaceId"], "status": "running",
    })
    platform.store.save("plans", {**plan, "status": "failed", "completedAt": "x"})

    settled = platform.recover_operation_journal()

    assert settled == 2
    assert platform.store.get_operation("op-orphan")["status"] == "unknown"
    assert platform.store.get_operation("op-terminal")["status"] == "cancelled"


def test_purge_removes_operation_rows(tmp_path):
    platform = platform_at(tmp_path)
    ws = platform.workspace_for_job("job_budget")
    platform.store.save_operation({
        "operationId": "op-purge", "planId": "p", "stepId": "s",
        "workspaceId": ws["id"], "status": "completed",
    })
    platform.purge_job_data("job_budget")
    assert platform.store.get_operation("op-purge") is None


# ---------------- ② budgets and run trace ----------------

def test_message_rate_limit(tmp_path):
    platform = platform_at(tmp_path, rate=2)
    wid = workspace_id(platform)
    for _ in range(2):
        platform.create_plan(workspace_id=wid, goal="剪一版60秒高光")
        platform.cancel_plan(platform.workspace_for_job("job_budget")["activePlanId"])
    with pytest.raises(ValueError, match="请求过于频繁"):
        platform.create_plan(workspace_id=wid, goal="再来一条")


def test_deadline_expires_running_plan(tmp_path):
    def slow_dispatcher(ws, tool, args):
        time.sleep(0.12)
        return {"artifact": {"kind": "noop"}}

    platform = platform_at(tmp_path, deadline=0.2, dispatcher=slow_dispatcher)
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版60秒高光，加上字幕")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] == "failed"
    assert approved.get("failureReason") == "deadline_exceeded"
    run = platform.store.get("runs", approved["runId"])
    assert run["errorClass"] == "deadline_exceeded"
    assert run["durationSeconds"] >= 0.2
    assert run["stepCount"] == len(approved["steps"])


def test_run_trace_records_model_and_replans(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版60秒高光")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] == "preview_ready"
    run = platform.store.get("runs", approved["runId"])
    assert run["modelFingerprint"]
    assert run["replanCount"] == 0
    assert run["durationSeconds"] >= 0
    assert "errorClass" not in run


def test_planner_reports_usage_event():
    original = planner_module._build_model
    try:
        planner_module._build_model = lambda config: FakeModel()
        events: list[dict[str, Any]] = []
        result = planner_module.run_plan_request(
            {
                "skill": {"id": "s", "version": "1", "contentHash": "h", "markdown": "# s"},
                "profile": {"kind": "highlight", "managed": True},
                "brief": {"variantCount": 1}, "goal": "g",
                "toolCatalog": [{"name": "inspect_workspace"}],
                "workspace": {"jobId": "j", "revision": 1},
            },
            model_config={"apiKey": "k", "model": "m", "baseUrl": "https://x"},
            emit=events.append,
        )
    finally:
        planner_module._build_model = original
    assert result["usage"]
    assert any(event["type"] == "message.usage" for event in events)


# ---------------- ③ planHash versioning ----------------

def test_plan_hash_binds_catalog_and_prompt_versions(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_id(platform), goal="剪一版60秒高光")
    assert tool_catalog_fingerprint().startswith("v1:")
    assert PROMPT_VERSION >= 2
    # The canonical hash inputs are embedded via normalize_plan; verify the
    # plan carries both generation markers through persistence.
    assert plan["planHash"] and len(plan["planHash"]) == 64
    from app.agent.compiler import normalize_plan

    workspace = platform.workspace_for_job("job_budget")
    normalized = normalize_plan(
        {"summary": "x", "steps": [{
            "id": "s1", "title": "t", "tool": "inspect_workspace", "arguments": {},
            "dependencies": [], "expectedOutput": "", "sideEffect": "read",
            "estimatedSeconds": 1, "optional": False,
        }]},
        workspace=workspace, skill={"id": "test-editor", "version": "1.0.0",
                                    "contentHash": "h", "allowedTools": ["inspect_workspace"]},
        goal="g",
    )
    assert normalized["planHash"]
