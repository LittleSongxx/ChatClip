"""LangChain planner: forced single-tool-call sessions.

Replaces the retired Node/Pi agent service. Each entry point binds exactly
one custom tool with ``tool_choice`` forced, streams model output through an
``emit`` callback (same event names the frontend already consumes), and
returns the structured submission — the LangChain-native equivalent of the
old "one prompt, one forced tool call" Pi sessions.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from ..llm.factory import create_chat_model, normalize_protocol
from .prompts import (
    PROBE_TOOL,
    SELECT_SKILL_TOOL,
    SUBMIT_SKILL_TOOL,
    plan_system_prompt,
    probe_system_prompt,
    router_system_prompt,
    skill_system_prompt,
    submit_plan_tool_schema,
)

Emit = Callable[[dict[str, Any]], None]

PLANNER_RUNTIME = "langchain"
PLANNER_MAX_TOKENS = 16384


class AgentPlannerError(RuntimeError):
    """Planning/model failure; API maps this to a degraded response."""

    def __init__(self, message: str, *, unconfigured: bool = False) -> None:
        super().__init__(message)
        self.unconfigured = unconfigured


def _build_model(model_config: dict[str, Any]):
    if not all(str(model_config.get(key) or "").strip() for key in ("apiKey", "model", "baseUrl")):
        raise AgentPlannerError(
            "Agent 模型尚未配置完整：请在设置中配置 Agent 模型，或先保存可复用的文本模型连接",
            unconfigured=True,
        )
    return create_chat_model(model_config)


def _bind_forced(model: Any, tool: dict[str, Any], *, config: dict[str, Any]) -> Any:
    tool_choice = "any" if normalize_protocol(config) == "anthropic" else "required"
    return model.bind_tools([tool], tool_choice=tool_choice)


def _merge_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    merged = AIMessageChunk(content="")
    for chunk in chunks:
        merged = merged + chunk
    return merged


def _stream_tool_session(
    *,
    model_config: dict[str, Any],
    tool: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    emit: Emit | None,
    heartbeat_phase: str = "decomposing_goal",
    heartbeat_title: str = "正在等待规划结果",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one forced-tool session, streaming deltas through ``emit``.

    Returns the parsed tool-call arguments plus the normalized event list
    (mirroring the retired Pi wire format).
    """
    events: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        events.append(event)
        if emit is not None:
            emit(event)

    model = _build_model(model_config)
    bound = _bind_forced(model, tool, config=model_config).bind(max_tokens=PLANNER_MAX_TOKENS)
    messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt[:4000])]

    started = time.monotonic()
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(6.0):
            elapsed = max(1, int(time.monotonic() - started))
            record({
                "type": "planning.progress", "phase": heartbeat_phase,
                "title": heartbeat_title,
                "detail": f"已等待 {elapsed} 秒，计划返回后即可确认。",
            })

    pacer = threading.Thread(target=heartbeat, daemon=True)
    pacer.start()
    thinking_open = False
    chunks: list[AIMessageChunk] = []
    try:
        for chunk in bound.stream(messages):
            chunks.append(chunk)
            text = chunk.content if isinstance(chunk.content, str) else "".join(
                str(item.get("text", "")) for item in chunk.content if isinstance(item, dict)
            ) if isinstance(chunk.content, list) else ""
            if text:
                record({"type": "message.text_delta", "delta": text})
            reasoning = str((chunk.additional_kwargs or {}).get("reasoning_content") or "")
            if reasoning and not thinking_open:
                thinking_open = True
                record({"type": "message.thinking"})
    finally:
        stop_heartbeat.set()
        pacer.join(timeout=0.5)
    if thinking_open:
        record({"type": "message.thinking_end"})
    merged = _merge_chunks(chunks)
    tool_calls = list(getattr(merged, "tool_calls", None) or [])
    record({"type": "message.toolcall", "tool": tool_calls[0]["name"] if tool_calls else ""})
    record({"type": "tool.started", "tool": tool_calls[0]["name"] if tool_calls else ""})
    if not tool_calls:
        record({"type": "tool.completed", "tool": "", "isError": True})
        raise AgentPlannerError("Agent 未调用 %s；请确认模型支持 Tool Calling" % tool["function"]["name"])
    call = tool_calls[0]
    if call.get("name") != tool["function"]["name"]:
        record({"type": "tool.completed", "tool": str(call.get("name") or ""), "isError": True})
        raise AgentPlannerError(f"Agent 调用了意外的工具：{call.get('name')}")
    arguments = call.get("args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as error:
            record({"type": "tool.completed", "tool": call["name"], "isError": True})
            raise AgentPlannerError("Agent 工具参数不是合法 JSON") from error
    if not isinstance(arguments, dict):
        record({"type": "tool.completed", "tool": call["name"], "isError": True})
        raise AgentPlannerError("Agent 工具参数格式无效")
    record({"type": "tool.completed", "tool": call["name"], "isError": False})
    return arguments, events


def _invoke_tool_session(
    *,
    model_config: dict[str, Any],
    tool: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    model = _build_model(model_config)
    bound = _bind_forced(model, tool, config=model_config).bind(max_tokens=PLANNER_MAX_TOKENS)
    answer = bound.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt[:4000]),
    ])
    tool_calls = list(getattr(answer, "tool_calls", None) or [])
    if not tool_calls or tool_calls[0].get("name") != tool["function"]["name"]:
        raise AgentPlannerError(f"Agent 未调用 {tool['function']['name']}；请确认模型支持 Tool Calling")
    arguments = tool_calls[0].get("args")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as error:
            raise AgentPlannerError("Agent 工具参数不是合法 JSON") from error
    if not isinstance(arguments, dict):
        raise AgentPlannerError("Agent 工具参数格式无效")
    return arguments


def run_plan_request(
    payload: dict[str, Any], *, model_config: dict[str, Any], emit: Emit | None = None,
) -> dict[str, Any]:
    """Generate a plan via a forced ``submit_plan`` call (streamed)."""
    if not payload.get("skill", {}).get("markdown") or not isinstance(payload.get("toolCatalog"), list):
        raise AgentPlannerError("规划上下文不完整")
    managed = payload.get("profile", {}).get("managed") is not False and str(payload.get("profile", {}).get("kind") or "") != "custom"
    if emit is not None:
        emit({
            "type": "planning.progress", "phase": "skill_selected",
            "title": "已选择剪辑 Skill",
            "detail": f"使用 {str(payload.get('skill', {}).get('id') or '智能剪辑 Skill')}",
        })
        emit({
            "type": "planning.progress", "phase": "context_loading",
            "title": "正在核对素材范围与可用工具",
            "detail": "此阶段只准备计划上下文，不会分析视频或开始渲染。",
        })
    if emit is not None:
        emit({
            "type": "planning.progress", "phase": "decomposing_goal",
            "title": "正在构建可确认的剪辑方案",
            "detail": "正在把目标、素材范围和交付要求整理成可审核步骤。",
        })
    arguments, events = _stream_tool_session(
        model_config=model_config,
        tool=submit_plan_tool_schema(managed),
        system_prompt=plan_system_prompt(payload),
        user_prompt=f"为以下目标生成完整执行计划：\n{str(payload.get('goal') or '')}",
        emit=emit,
    )
    if emit is not None:
        step_count = (
            int((payload.get("brief") or {}).get("variantCount") or 1)
            if managed else len(arguments.get("steps") or [])
        )
        emit({
            "type": "planning.progress", "phase": "validating_plan",
            "title": "正在校验计划步骤与人工确认点",
            "detail": f"已拆解为 {step_count} 个待确认步骤。" if step_count else "正在校验待确认步骤。",
        })
    submitted_plan = {
        **arguments,
        "steps": [] if managed else arguments.get("steps"),
        "skillChain": [
            {
                "id": item.get("id"), "version": item.get("version"),
                "contentHash": item.get("contentHash"),
                "role": item.get("role") or ("primary" if index == 0 else "addon"),
                "workflowProfile": item.get("workflowProfile") or "",
            }
            for index, item in enumerate(payload.get("skills") or [payload["skill"]])
        ],
    }
    return {
        "plan": {
            **submitted_plan,
            "agent": {
                "provider": str(model_config.get("provider") or ""),
                "model": str(model_config.get("model") or ""),
                "runtime": PLANNER_RUNTIME,
                "version": PLANNER_RUNTIME,
            },
        },
        "events": events,
    }


def generate_skill_markdown(payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
    arguments = _invoke_tool_session(
        model_config=model_config,
        tool=SUBMIT_SKILL_TOOL,
        system_prompt=skill_system_prompt(payload),
        user_prompt=str(payload.get("request") or ""),
    )
    available = {item["name"] for item in payload.get("toolCatalog") or []}
    required = [str(name) for name in arguments.get("requiredTools") or []]
    missing = [name for name in required if name not in available]
    return {
        "skillMarkdown": str(arguments.get("skillMarkdown") or ""),
        "simulation": {
            "valid": not missing,
            "summary": str(arguments.get("simulationSummary") or ""),
            "requiredTools": required,
            "missingTools": missing,
        },
    }


def route_skill_llm(payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("skills"), list) or not payload["skills"]:
        raise AgentPlannerError("没有可路由的 Skill")
    arguments = _invoke_tool_session(
        model_config=model_config,
        tool=SELECT_SKILL_TOOL,
        system_prompt=router_system_prompt(payload),
        user_prompt=str(payload.get("goal") or ""),
    )
    allowed = {str(item["id"]) for item in payload["skills"]}
    skill_id = str(arguments.get("skillId") or "")
    if skill_id not in allowed:
        raise AgentPlannerError("Agent 没有选择有效的 Skill")
    return {"skillId": skill_id, "reason": str(arguments.get("reason") or "")}


def probe_tool_calling(model_config: dict[str, Any]) -> dict[str, Any]:
    challenge = f"probe-{int(time.time() * 1000)}"
    arguments = _invoke_tool_session(
        model_config=model_config,
        tool=PROBE_TOOL,
        system_prompt=probe_system_prompt(challenge),
        user_prompt=f"调用工具并回传 challenge：{challenge}",
    )
    if str(arguments.get("challenge") or "") != challenge:
        raise AgentPlannerError("模型没有完成结构化 Tool Calling 探测")
    return {
        "ok": True, "toolCalling": True, "streaming": True,
        "model": str(model_config.get("model") or ""),
        "runtime": PLANNER_RUNTIME,
    }
