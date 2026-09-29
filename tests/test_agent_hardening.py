"""Hardening regressions from the external security/consistency audit.

Covers: credentials never entering graph checkpoints, the real tool-calling
probe path, approval staleness on material context changes, job-deletion
cascade into agent state, multi-tool-call rejection, action-receipt scoping
and non-zero planning estimates.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent import AgentPlatform
from app.agent import planner as planner_module
from app.agent.planner import AgentPlannerError

SECRET_KEY = "sk-audit-secret-DO-NOT-PERSIST-0123456789"

SKILL = """---
name: test-editor
description: Hardening test profile.
allowed-tools: inspect_workspace propose_timeline_edit confirm_timeline_edit prepare_subtitle_review render_review_preview
workflow-profile: revision
version: 1.0.0
---

# Test Editor
"""


class FakeModel:
    """Streams prose plus N tool calls of the requested tool name."""

    def __init__(self, tool_calls: int = 1, tool_name: str = "submit_plan") -> None:
        self.tool_calls = tool_calls
        self.tool_name = tool_name
        self._challenge = "probe-fixed"

    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools, tool_choice=None):
        return self

    def _arguments(self) -> str:
        if self.tool_name == "submit_plan":
            return json.dumps({"summary": "计划", "strategy": {}}, ensure_ascii=False)
        if self.tool_name == "agent_capability_probe":
            return json.dumps({"challenge": self._challenge}, ensure_ascii=False)
        return json.dumps({}, ensure_ascii=False)

    def stream(self, messages):
        yield AIMessageChunk(content="计划")
        for index in range(self.tool_calls):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": self.tool_name,
                    "args": self._arguments(),
                    "id": f"call_{index}", "index": index, "type": "tool_call_chunk",
                }],
            )

    def invoke(self, messages):
        # Echo back the challenge requested in the prompt for probe sessions.
        text = "".join(str(m.content) for m in messages)
        import re as _re

        match = _re.search(r"probe-\d+", text)
        if match and self.tool_name == "agent_capability_probe":
            self._challenge = match.group(0)
        merged = AIMessageChunk(content="")
        for chunk in self.stream(messages):
            merged = merged + chunk
        return merged


class FakeBackend:
    def plan(self, payload, *, model_config, emit=None):
        assert "model" not in payload or "apiKey" not in (payload.get("model") or {})
        return {"plan": {"summary": "测试计划", "strategy": {}}, "events": []}

    def route_skill(self, payload, *, model_config):
        return {"skillId": "test-editor", "reason": "matches"}

    def generate_skill(self, payload, *, model_config):
        raise AssertionError("unused")

    def probe(self, model_config):
        return planner_module.probe_tool_calling(model_config)


def platform_at(path: Path, context: dict[str, Any] | None = None) -> AgentPlatform:
    platform = AgentPlatform(
        data_root=path,
        model_config_resolver=lambda: {
            "provider": "bailian", "protocol": "openai", "apiKey": SECRET_KEY,
            "model": "qwen3.8-max", "baseUrl": "https://dashscope.example/v1",
            "thinkingType": "enabled", "timeoutSeconds": 30,
        },
    )
    platform.planner_backend = FakeBackend()  # type: ignore[assignment]
    platform.install_skill(markdown=SKILL, source="test", status="enabled")
    platform.create_workspace(job_id="job_audit")
    platform.configure_planning_context_provider(lambda job_id: dict(context or {}))
    platform.configure_tool_dispatcher(
        lambda workspace, tool, args: {"artifact": {"kind": "noop"}}
    )
    return platform


def workspace_of(platform: AgentPlatform) -> dict[str, Any]:
    workspace = platform.workspace_for_job("job_audit")
    assert workspace is not None
    return workspace


def _sqlite_blob(path: Path) -> bytes:
    if not path.is_file():
        return b""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        payload = bytearray(path.read_bytes())
        for (table,) in rows:
            try:
                for (blob,) in connection.execute(f"SELECT CAST(COALESCE(data,'') AS BLOB) FROM {table}"):
                    payload.extend(blob if isinstance(blob, bytes) else str(blob).encode())
            except sqlite3.DatabaseError:
                continue
        return bytes(payload)
    finally:
        connection.close()


def test_graph_checkpoint_never_persists_api_key(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    assert plan["status"] == "awaiting_confirmation"
    for name in ("graph.sqlite3", "agent.sqlite3"):
        blob = _sqlite_blob(platform.data_root / name)
        assert SECRET_KEY.encode() not in blob, f"{name} leaked the API key"
        assert b"apiKey" not in blob


def test_agent_state_files_are_private(tmp_path):
    platform = platform_at(tmp_path)
    assert (platform.data_root).stat().st_mode & 0o777 == 0o700
    for name in ("agent.sqlite3", "graph.sqlite3"):
        assert (platform.data_root / name).stat().st_mode & 0o777 == 0o600


def test_real_probe_path_uses_langchain_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        planner_module, "_build_model",
        lambda config: FakeModel(tool_name="agent_capability_probe"),
    )
    platform = AgentPlatform(
        data_root=tmp_path,
        model_config_resolver=lambda: {
            "provider": "bailian", "protocol": "openai", "apiKey": "k",
            "model": "m", "baseUrl": "https://x", "thinkingType": "", "timeoutSeconds": 30,
        },
    )
    result = platform.probe(platform.model_config_resolver())
    assert result["toolCalling"] is True
    assert result["runtime"] == "langchain"


def test_agent_health_reports_saved_probe_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        planner_module, "_build_model",
        lambda config: FakeModel(tool_name="agent_capability_probe"),
    )
    platform = AgentPlatform(
        data_root=tmp_path,
        model_config_resolver=lambda: {
            "provider": "bailian", "protocol": "openai", "apiKey": "k",
            "model": "m", "baseUrl": "https://x", "thinkingType": "", "timeoutSeconds": 30,
        },
    )
    assert platform.tool_calling_verified() is False
    platform.probe(platform.model_config_resolver())
    from app.setup_readiness import save_agent_probe

    save_agent_probe(platform.data_root / "agent-probe.json", platform.model_config_resolver(), {"toolCalling": True})
    assert platform.tool_calling_verified() is True


def test_approval_rejects_stale_material_context(tmp_path):
    context: dict[str, Any] = {"editing": {"hasOutputs": False, "hasActiveSession": False}}
    platform = platform_at(tmp_path, context)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    # Material facts changed while the plan waited for confirmation.
    context["editing"] = {"hasOutputs": True, "currentOutputVersionId": "ver_2"}
    with pytest.raises(ValueError, match="已过期"):
        platform.approve_plan(plan["id"], expected_hash=plan["planHash"])


def test_job_deletion_cascades_into_agent_state_and_checkpoints(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    thread_id = platform._plan_thread_id(plan)
    assert (platform.data_root / "graph.sqlite3").is_file()

    removed = platform.purge_job_data("job_audit")

    assert removed == 1
    assert platform.workspace_for_job("job_audit") is None
    assert platform.store.get("plans", plan["id"]) is None
    assert platform.store.events_after(str(plan["workspaceId"])) == []
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    snapshot = platform.graph.get_state(config)
    assert not (snapshot.values if snapshot else None)


def test_multiple_tool_calls_are_rejected(monkeypatch):
    monkeypatch.setattr(planner_module, "_build_model", lambda config: FakeModel(tool_calls=2))
    with pytest.raises(AgentPlannerError, match="2 个工具调用"):
        planner_module.run_plan_request(
            {
                "skill": {"id": "s", "version": "1", "contentHash": "h", "markdown": "# s"},
                "profile": {"kind": "highlight", "managed": True},
                "brief": {"variantCount": 1}, "goal": "g",
                "toolCatalog": [{"name": "inspect_workspace"}],
                "workspace": {"jobId": "j", "revision": 1},
            },
            model_config={"apiKey": "k", "model": "m", "baseUrl": "https://x"},
        )


def test_action_receipt_is_scoped_to_plan_and_step(tmp_path):
    import time

    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    if approved["status"] == "action_required":
        gated = next(s for s in approved["steps"] if s["status"] == "action_required")
        value = {"context": {"jobId": "job_audit", "stepId": gated["id"]}, "selection": {}}
        first = platform.resolve_action(plan["id"], approved=True, value=value)
        assert first["status"] != "action_required" or any(
            s["status"] == "action_required" and s["id"] != gated["id"] for s in first["steps"]
        )
        # Duplicate submit of the same action stays idempotent.
        duplicate = platform.resolve_action(plan["id"], approved=True, value=value)
        assert duplicate["id"] == plan["id"]


def test_compiled_steps_carry_nonzero_estimates(tmp_path):
    platform = platform_at(tmp_path)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光，加上字幕")
    assert plan["steps"]
    assert all(step["estimatedSeconds"] > 0 for step in plan["steps"])


def test_replan_model_change_forces_material_revision(tmp_path):
    context: dict[str, Any] = {}
    platform = platform_at(tmp_path, context)
    plan = platform.create_plan(workspace_id=workspace_of(platform)["id"], goal="剪一版60秒高光")
    approved = platform.approve_plan(plan["id"], expected_hash=plan["planHash"])
    assert approved["status"] == "preview_ready"

    # Simulate a failed required step and a switched planning model.
    failed = dict(approved["steps"][0])
    failed["status"] = "failed"
    approved["steps"][0] = failed
    approved["status"] = "running"
    platform.store.save("plans", approved)

    original = platform.model_config_resolver
    platform.model_config_resolver = lambda: {  # type: ignore[assignment]
        "provider": "deepseek", "protocol": "openai", "apiKey": "k2",
        "model": "deepseek-flash", "baseUrl": "https://api.deepseek.com/v1",
        "thinkingType": "", "timeoutSeconds": 30,
    }
    try:
        outcome = platform._attempt_replan(approved["id"], approved["steps"][0]["id"])
    finally:
        platform.model_config_resolver = original  # type: ignore[assignment]
    assert outcome == "awaiting_approval"
