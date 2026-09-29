"""LangGraph state machine for the ChatClip agent.

Topology (one thread per plan, checkpointed in ``data/agent/graph.db``)::

    START ──(op=plan)──► planner ─► compile ─► approval ─► advance ◄─────────┐
          ──(op=execute)─────────────────────────┘           │              │
                                                               ├─► dispatch ──┤
                                                               ├─► gate_action ─► advance
                                                               ├─► operation_gate ─► advance
                                                               ├─► replan ─► (advance | approval | finish)
                                                               └─► finish ─► END

Human gates are LangGraph ``interrupt()`` pauses: the plan approval, the
structured action confirmations and background-operation waits all park the
run in a checkpoint. API confirmations, action resolutions and media-future
callbacks resume the same thread with ``Command(resume=...)``. Step-level
product state (statuses, attempts, operation ids) lives in the AgentStore;
the checkpoint carries graph control state only.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .catalog import (
    ACTION_REQUIRED_EFFECTS,
    AUTONOMOUS_REVIEW,
    FORBIDDEN_AUTONOMOUS_EFFECTS,
    SAFE_AUTONOMOUS_EXPORT_TOOLS,
    STEPWISE_REVIEW,
)

TERMINAL_PLAN_STATUSES = frozenset({"preview_ready", "no_result", "failed", "cancelled"})


class AgentState(TypedDict, total=False):
    op: str                    # plan | execute
    workspace_id: str
    goal: str
    skill_id: str
    execution_mode: str
    workspace: dict[str, Any]
    skill: dict[str, Any]
    payload: dict[str, Any]
    phase: str                 # planning | compiling | awaiting_approval | executing | finished
    plan_id: str
    planning_warning: str
    raw_result: dict[str, Any]
    final_status: str
    error: str


def build_agent_graph(platform: Any):
    """Assemble the agent graph around an ``AgentPlatform`` instance."""

    def planner_node(state: AgentState) -> dict[str, Any]:
        payload = state["payload"]
        managed = bool(payload.get("profile", {}).get("managed"))
        emit = platform._make_stream_emit(state["workspace_id"])
        try:
            result = _run_planner(payload, emit)
            return {"phase": "compiling", "raw_result": result, "planning_warning": ""}
        except Exception as error:  # noqa: BLE001 - routed to deterministic fallback
            if not managed:
                raise
            # First-party managed Skills always have a deterministic plan.
            return {
                "phase": "compiling",
                "planning_warning": str(error)[:500],
                "raw_result": {
                    "plan": {
                        "summary": state["goal"],
                        "strategy": {},
                        "planningSource": "deterministic_fallback",
                        "planningWarning": str(error)[:500],
                    },
                    "events": [{
                        "phase": "plan_ready", "title": "已切换本地确定性规划",
                        "detail": "在线规划服务暂不可用，已按内置 Skill 工作流生成可确认计划。",
                    }],
                },
            }

    def _run_planner(payload: dict[str, Any], emit: Any) -> dict[str, Any]:
        return platform.planner_backend.plan(
            payload, model_config=platform.model_config_resolver(), emit=emit,
        )

    def compile_node(state: AgentState) -> dict[str, Any]:
        raw_result = state.get("raw_result") or {}
        if state.get("planning_warning"):
            platform.store.append_event(state["workspace_id"], "planning.fallback", {
                "reason": state["planning_warning"],
            })
            platform.record_planning_progress(state["workspace_id"], {
                "phase": "plan_ready", "title": "已切换本地确定性规划",
                "detail": "在线规划服务暂不可用，已按内置 Skill 工作流生成可确认计划。",
            })
        plan = platform.persist_plan_result(
            workspace=state["workspace"], skill=state["skill"],
            goal=state["goal"], result=raw_result,
            execution_mode=state.get("execution_mode") or AUTONOMOUS_REVIEW,
        )
        return {"phase": "awaiting_approval", "plan_id": plan["id"]}

    def approval_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        if plan and str(plan.get("status") or "") == "awaiting_confirmation":
            value = interrupt({
                "gate": "plan_approval", "planId": plan["id"],
                "planHash": plan["planHash"], "summary": plan.get("summary") or "",
            })
            if not isinstance(value, dict) or value.get("action") != "approve":
                return {"phase": "finished", "final_status": "cancelled"}
        return {"phase": "executing"}

    def advance_node(state: AgentState) -> dict[str, Any]:
        return {}

    def gate_action_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        step = _step_with_status(plan, "action_required")
        if step is None:
            # Fresh entry routed here by route_advance: only mark when the
            # next ready step genuinely requires a human gate. On resume
            # re-entry the gated step is already resolved, and a plain kick
            # must fall through to the dispatch loop untouched.
            candidate = _next_ready_step(plan)
            if candidate is not None and _requires_action_gate(plan, candidate):
                platform._mark_action_required(plan, candidate)
                plan = platform.store.get("plans", plan["id"]) or plan
                step = _step_with_status(plan, "action_required") or candidate
        if step is None:
            return {}
        interrupt({
            "gate": "action", "planId": plan["id"], "stepId": step["id"],
            "tool": step.get("tool") or "",
            "message": str((step.get("result") or {}).get("message") or ""),
        })
        return {}

    def dispatch_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        step = _next_ready_step(plan)
        if step is None or not plan or str(plan.get("status") or "") != "running":
            return {}
        outcome = platform._execute_tool_step(plan, step)
        return {"phase": "executing", **({"dispatch_outcome": outcome} if outcome else {})}

    def operation_gate_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        step = _step_with_status(plan, "waiting_operation")
        if step is None:
            return {}
        outcome = interrupt({
            "gate": "operation", "planId": plan["id"], "stepId": step["id"],
            "operationId": str(step.get("operationId") or ""),
        })
        # Only real Future completions carry the apply marker; resume kicks
        # from approve/resolve/retry must not settle a waiting operation.
        if isinstance(outcome, dict) and outcome.get("apply") is True:
            platform._apply_operation_outcome(
                str(plan["id"]), str(step["id"]), outcome,
            )
        return {}

    def replan_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        if not plan:
            return {"phase": "finished", "final_status": "failed"}
        failed_step = next(
            (item for item in plan.get("steps") or []
             if item.get("status") == "failed" and not item.get("optional")),
            None,
        )
        if failed_step is None:
            return {"phase": "executing"}
        outcome = platform._attempt_replan(str(plan["id"]), str(failed_step["id"]))
        return {"phase": outcome}

    def finish_node(state: AgentState) -> dict[str, Any]:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        if plan and str(plan.get("status") or "") not in TERMINAL_PLAN_STATUSES:
            failed_required = any(
                item.get("status") == "failed" and not item.get("optional")
                for item in plan.get("steps") or []
            )
            no_result = any(
                isinstance(item.get("result"), dict)
                and str(item["result"].get("terminalStatus") or "") == "no_result"
                for item in plan.get("steps") or []
            )
            final = "failed" if failed_required else "no_result" if no_result else "preview_ready"
            platform._finish_plan(plan, final)
        return {"phase": "finished"}

    # ------------------------------------------------------------- routing

    def _step_with_status(plan: dict[str, Any] | None, status: str) -> dict[str, Any] | None:
        if not plan:
            return None
        return next(
            (item for item in plan.get("steps") or [] if item.get("status") == status),
            None,
        )

    def _next_ready_step(plan: dict[str, Any] | None) -> dict[str, Any] | None:
        if not plan or str(plan.get("status") or "") != "running":
            return None
        completed = {
            item["id"] for item in plan.get("steps") or []
            if item.get("status") in {"completed", "skipped"}
        }
        return next((
            item for item in plan.get("steps") or []
            if item.get("status") == "pending"
            and set(item.get("dependencies") or []).issubset(completed)
        ), None)

    def _requires_action_gate(plan: dict[str, Any], step: dict[str, Any]) -> bool:
        """Only identity/review side effects and unauthorized exports gate."""
        execution_mode = str(plan.get("executionMode") or STEPWISE_REVIEW)
        return (
            (
                step["sideEffect"] in FORBIDDEN_AUTONOMOUS_EFFECTS
                and str(step.get("tool") or "") not in SAFE_AUTONOMOUS_EXPORT_TOOLS
            )
            or (
                step["sideEffect"] in ACTION_REQUIRED_EFFECTS
                and execution_mode != AUTONOMOUS_REVIEW
            )
        )

    def route_entry(state: AgentState) -> str:
        return "planner" if state.get("op") == "plan" else "advance"

    def route_approval(state: AgentState) -> str:
        return "finish" if state.get("final_status") else "advance"

    def route_dispatch(state: AgentState) -> str:
        outcome = str(state.get("dispatch_outcome") or "")
        return {"operation": "operation_gate", "action": "gate_action"}.get(outcome, "advance")

    def route_replan(state: AgentState) -> str:
        phase = str(state.get("phase") or "")
        if phase == "awaiting_approval":
            return "approval"
        if phase == "finished":
            return "finish"
        return "advance"

    def route_advance(state: AgentState) -> str:
        plan = platform.store.get("plans", state.get("plan_id") or "")
        if not plan or str(plan.get("status") or "") in TERMINAL_PLAN_STATUSES:
            return "finish"
        if str(plan.get("status") or "") == "awaiting_confirmation":
            return "approval"
        steps = plan.get("steps") or []
        if _step_with_status(plan, "waiting_operation") is not None:
            return "operation_gate"
        if _step_with_status(plan, "action_required") is not None:
            return "gate_action"
        if any(item.get("status") == "running" for item in steps):
            return "finish"
        failed_required = any(
            item.get("status") == "failed" and not item.get("optional") for item in steps
        )
        if failed_required:
            if int(plan.get("replanCount") or 0) < 2:
                return "replan"
            return "finish"
        next_step = _next_ready_step(plan)
        if next_step is None:
            return "finish"
        return "gate_action" if _requires_action_gate(plan, next_step) else "dispatch"

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("compile", compile_node)
    builder.add_node("approval", approval_node)
    builder.add_node("advance", advance_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("gate_action", gate_action_node)
    builder.add_node("operation_gate", operation_gate_node)
    builder.add_node("replan", replan_node)
    builder.add_node("finish", finish_node)

    builder.add_conditional_edges(START, route_entry, {"planner": "planner", "advance": "advance"})
    builder.add_edge("planner", "compile")
    builder.add_edge("compile", "approval")
    builder.add_conditional_edges("approval", route_approval, {"advance": "advance", "finish": "finish"})
    builder.add_conditional_edges(
        "advance", route_advance,
        {
            "dispatch": "dispatch", "gate_action": "gate_action",
            "operation_gate": "operation_gate", "replan": "replan",
            "approval": "approval", "finish": "finish",
        },
    )
    builder.add_conditional_edges(
        "dispatch", route_dispatch,
        {"advance": "advance", "operation_gate": "operation_gate", "gate_action": "gate_action"},
    )
    builder.add_edge("gate_action", "advance")
    builder.add_edge("operation_gate", "advance")
    builder.add_conditional_edges(
        "replan", route_replan,
        {"advance": "advance", "approval": "approval", "finish": "finish"},
    )
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=platform.checkpointer)
