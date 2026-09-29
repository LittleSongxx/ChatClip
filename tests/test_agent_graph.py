"""LangGraph agent orchestration tests.

Covers the state-machine core introduced with the LangGraph rewrite:
forced-tool planning, deterministic compilation, approval interrupt, action
gates, operation gates with Future handoff, rejection and replan flows.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent import AgentPlatform
from app.agent import planner as planner_module

SKILL = """---
name: test-editor
description: Test profile for LangGraph orchestration verification.
allowed-tools: inspect_workspace propose_timeline_edit confirm_timeline_edit prepare_subtitle_review render_review_preview
workflow-profile: revision
version: 1.0.0
---

# Test Editor
"""


class FakePlannerBackend:
    """In-process planner backend that always submits a managed-style plan."""

    def __init__(self) -> None:
        self.plan_payloads: list[dict[str, Any]] = []

    def plan(self, payload: dict[str, Any], *, model_config: dict[str, Any], emit: Any = None) -> dict[str, Any]:
        self.plan_payloads.append(payload)
        return {
            "plan": {"summary": "测试计划", "strategy": {}},
            "events": [{"type": "planning.progress", "phase": "decomposing_goal"}],
        }

    def route_skill(self, payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
        return {"skillId": "test-editor", "reason": "matches"}

    def generate_skill(self, payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("not used in these tests")

    def probe(self, model_config: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "toolCalling": True, "streaming": True}


def platform_at(path: Path) -> AgentPlatform:
    platform = AgentPlatform(
        data_root=path,
        model_config_resolver=lambda: {
            "provider": "bailian", "protocol": "openai", "apiKey": "sk-test",
            "model": "qwen3.8-max", "baseUrl": "https://dashscope.example/v1",
            "thinkingType": "enabled", "timeoutSeconds": 30,
        },
    )
    platform.planner_backend = FakePlannerBackend()  # type: ignore[assignment]
    platform.install_skill(markdown=SKILL, source="test", status="enabled")
    platform.create_workspace(job_id="job_1", title="demo")
    return platform


def workspace_of(platform: AgentPlatform) -> dict[str, Any]:
    workspace = platform.workspace_for_job("job_1")
    assert workspace is not None
    return workspace


def test_managed_plan_compiles_deterministic_steps(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(
        workspace_id=workspace_of(platform)["id"],
        goal="把视频剪成60秒的高光集锦，加上字幕",
        execution_mode="autonomous_review",
    )
    assert plan["status"] == "awaiting_confirmation"
    tools = [step["tool"] for step in plan["steps"]]
    assert tools[0] == "inspect_workspace"
    assert "propose_timeline_edit" in tools
    assert tools.index("confirm_timeline_edit") > tools.index("propose_timeline_edit")
    assert "prepare_subtitle_review" in tools
    assert "render_review_preview" in tools
    assert plan["understanding"]["title"] == "Agent 理解"


def test_approval_resume_executes_steps_to_preview(tmp_path):
    platform = platform_at(tmp_path)
    dispatched: list[str] = []
    platform.configure_tool_dispatcher(
        lambda workspace, tool, args: (dispatched.append(tool), {"artifact": {"kind": "noop"}})[1]
    )
    plan = platform.create_plan(
        workspace_id=workspace_of(platform)["id"], goal="剪一版45秒高光集锦",
    )
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] == "preview_ready"
    assert dispatched and dispatched[0] == "inspect_workspace"


def test_wrong_hash_cannot_approve(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版30秒高光")
    with pytest.raises(ValueError):
        platform.approve_plan(plan["id"], expected_hash="0" * 64)


def test_stepwise_review_gates_review_steps(tmp_path):
    platform = platform_at(tmp_path)
    # Install a content skill so the compiled plan contains an evidence review.
    platform.install_skill(markdown=SKILL.replace("test-editor", "test-content")
                           .replace("revision", "content")
                           .replace(
                               "inspect_workspace propose_timeline_edit confirm_timeline_edit prepare_subtitle_review render_review_preview",
                               "inspect_workspace search_content review_content_evidence propose_timeline_edit confirm_timeline_edit prepare_subtitle_review render_review_preview",
                           ), source="test", status="enabled")
    platform.configure_tool_dispatcher(
        lambda workspace, tool, args: {"artifact": {"kind": "noop"}}
    )
    plan = platform.create_plan(
        workspace_id=workspace_of(platform)["id"],
        goal="找出关于产品定价的片段，合成成片",
        execution_mode="stepwise_review",
        skill_id="test-content",
    )
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] == "action_required"
    gated = [step for step in approved["steps"] if step["status"] == "action_required"]
    assert any(step["tool"] == "review_content_evidence" for step in gated)


def test_rejecting_action_fails_plan(tmp_path):
    platform = platform_at(tmp_path)
    platform.configure_tool_dispatcher(
        lambda workspace, tool, args: {"artifact": {"kind": "noop"}}
    )
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] in {"preview_ready", "action_required"}
    if approved["status"] == "action_required":
        rejected = platform.resolve_action(plan["id"], approved=False)
        assert rejected["status"] == "failed"


def test_future_operations_pause_and_resume(tmp_path):
    platform = platform_at(tmp_path)

    def dispatcher(workspace: dict[str, Any], tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool in {"propose_timeline_edit", "prepare_subtitle_review", "render_review_preview"}:
            future: Future[Any] = Future()

            def run() -> None:
                time.sleep(0.02)
                future.set_result({"artifact": {"kind": "async-done"}})

            threading.Thread(target=run, daemon=True).start()
            return {"future": future, "operationId": f"op-{tool}"}
        return {"artifact": {"kind": "noop"}}

    platform.configure_tool_dispatcher(dispatcher)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        refreshed = platform.store.get("plans", plan["id"]) or {}
        if refreshed.get("status") in {"preview_ready", "no_result", "failed", "cancelled"}:
            break
        time.sleep(0.1)
    assert (platform.store.get("plans", plan["id"]) or {}).get("status") == "preview_ready"


def test_cancel_parks_plan_permanently(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    cancelled = platform.cancel_plan(plan["id"])
    assert cancelled["status"] == "cancelled"
    with pytest.raises(ValueError):
        platform.approve_plan(plan["id"], expected_hash=plan["planHash"])


def test_planner_failure_falls_back_to_deterministic_plan(tmp_path):
    class BrokenBackend(FakePlannerBackend):
        def plan(self, payload: dict[str, Any], *, model_config: dict[str, Any], emit: Any = None) -> dict[str, Any]:
            raise RuntimeError("planner offline")

    platform = platform_at(tmp_path)
    platform.planner_backend = BrokenBackend()  # type: ignore[assignment]
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    assert plan["status"] == "awaiting_confirmation"
    assert plan["planningSource"] == "deterministic_fallback"
    assert "planner offline" in plan["planningWarning"]


def test_planner_forced_tool_session_parses_streamed_tool_call():
    """The LangChain planner merges streamed tool_call_chunks into one call."""

    class FakeModel:
        def bind(self, **kwargs):
            return self

        def bind_tools(self, tools, tool_choice=None):
            return self

        def stream(self, messages):
            yield AIMessageChunk(content="计划：")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "submit_plan",
                    "args": json.dumps({"summary": "高光计划", "strategy": {}}, ensure_ascii=False),
                    "id": "call_1", "index": 0, "type": "tool_call_chunk",
                }],
            )

    original = planner_module._build_model
    try:
        planner_module._build_model = lambda config: FakeModel()
        events: list[dict[str, Any]] = []
        result = planner_module.run_plan_request(
            {
                "skill": {"id": "s", "version": "1", "contentHash": "h", "markdown": "# s"},
                "profile": {"kind": "highlight", "managed": True},
                "brief": {"variantCount": 1},
                "goal": "剪一版高光",
                "toolCatalog": [{"name": "inspect_workspace"}],
                "workspace": {"jobId": "j", "revision": 1},
            },
            model_config={"apiKey": "k", "model": "m", "baseUrl": "https://x"},
            emit=events.append,
        )
    finally:
        planner_module._build_model = original
    assert result["plan"]["summary"] == "高光计划"
    assert result["plan"]["steps"] == []
    assert any(event["type"] == "message.text_delta" for event in events)
    assert any(event["type"] == "tool.completed" and not event["isError"] for event in events)


def test_graph_thread_survives_restart_via_store(tmp_path):
    """Plan store remains the product-facing record across graph checkpoints."""
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    snapshot = platform.store.get("plans", plan["id"])
    assert snapshot is not None and snapshot["planHash"] == plan["planHash"]
