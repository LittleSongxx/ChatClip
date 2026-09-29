"""AgentPlatform: LangGraph-orchestrated agent coordination.

Owns the AgentStore (workspaces/skills/plans/runs/events), the compiled
:class:`~app.agent.graph` state machine and its SQLite checkpointer. The
media tool implementations are injected through ``configure_tool_dispatcher``
(the seam into the FastAPI media kernel); everything else — planning,
approval gating, step execution, replans and recovery — lives here.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ..agent_store import AgentStore, content_hash, now_iso, parse_skill_markdown
from ..quality_repair import quality_repair_plan
from . import planner as planner_module
from .brief import (
    _cover_intro_duration,
    _cover_intro_requested,
    _retrieval_query,
    _social_delivery,
    editing_brief,
)
from .catalog import (
    ACTIVE_PLAN_STATUSES,
    AUTONOMOUS_REVIEW,
    STEPWISE_REVIEW,
    VALID_EXECUTION_MODES,
    action_required_message,
    tool_catalog,
)
from .compiler import (
    automatic_replan_block_reason,
    compile_profile_plan,
    normalize_plan,
    replan_force_replay_tools,
    step_execution_signatures,
)
from .graph import TERMINAL_PLAN_STATUSES, build_agent_graph
from .planner import AgentPlannerError
from .skills import (
    WORKFLOW_PROFILE_KINDS,
    compose_skills,
    profile_for_skill,
    skill_routing_eligibility,
)

# Backwards-compatible alias: API layers map this to 503 responses.
AgentServiceError = AgentPlannerError

ToolDispatcher = Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]]
ModelConfigResolver = Callable[[], dict[str, Any]]
WorkspaceStateListener = Callable[[dict[str, Any], dict[str, Any] | None], None]
ActionResolutionValidator = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]
PlanningContextProvider = Callable[[str], dict[str, Any]]

GRAPH_RECURSION_LIMIT = 400

LOGGER = logging.getLogger("chatclip.agent")


class LangChainPlannerBackend:
    """Default planner backend: the in-process LangChain forced-tool sessions."""

    def plan(self, payload: dict[str, Any], *, model_config: dict[str, Any], emit: Any = None) -> dict[str, Any]:
        return planner_module.run_plan_request(payload, model_config=model_config, emit=emit)

    def route_skill(self, payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
        return planner_module.route_skill_llm(payload, model_config=model_config)

    def generate_skill(self, payload: dict[str, Any], *, model_config: dict[str, Any]) -> dict[str, Any]:
        return planner_module.generate_skill_markdown(payload, model_config=model_config)

    def probe(self, model_config: dict[str, Any]) -> dict[str, Any]:
        return self.planner_backend.probe(model_config)


class AgentPlatform:
    """Plan-gated LangGraph orchestration around the media kernel."""

    def __init__(
        self, *, data_root: Path,
        model_config_resolver: ModelConfigResolver, timeout_seconds: float = 120.0,
    ) -> None:
        self.data_root = data_root / "agent"
        self.skills_root = self.data_root / "skills"
        self.skills_root.mkdir(parents=True, exist_ok=True)
        self.store = AgentStore(self.data_root / "agent.sqlite3")
        self.model_config_resolver = model_config_resolver
        self.timeout_seconds = timeout_seconds
        self._dispatch_tool: ToolDispatcher | None = None
        self._workspace_state_listener: WorkspaceStateListener | None = None
        self._action_resolution_validator: ActionResolutionValidator | None = None
        self._planning_context_provider: PlanningContextProvider | None = None
        self._execution_lock = threading.RLock()
        self._thread_locks: dict[str, threading.RLock] = {}
        self._operation_handles: dict[tuple[str, str], dict[str, Any]] = {}
        # Single-worker driver for Future-completion resumes: delivery always
        # happens OFF the graph thread, so it blocks on the per-plan thread
        # lock until the owning invoke parks at its operation-gate interrupt
        # instead of racing a still-running dispatch node.
        self._graph_driver = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chatclip-agent-graph",
        )
        self._checkpointer_context = SqliteSaver.from_conn_string(
            str(self.data_root / "graph.sqlite3"),
        )
        self.checkpointer = self._checkpointer_context.__enter__()
        self.planner_backend = LangChainPlannerBackend()
        self.graph = build_agent_graph(self)

    def close(self) -> None:
        # Wait for in-flight Future deliveries so callers observe quiescence
        # (tests clean up their data directories immediately afterwards).
        self._graph_driver.shutdown(wait=True)
        try:
            self._checkpointer_context.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass

    # ------------------------------------------------------------- wiring

    def configure_tool_dispatcher(self, dispatcher: ToolDispatcher) -> None:
        self._dispatch_tool = dispatcher

    def configure_workspace_state_listener(self, listener: WorkspaceStateListener) -> None:
        """Persist a compact Agent summary alongside the media job when available."""
        self._workspace_state_listener = listener

    def configure_action_resolution_validator(self, validator: ActionResolutionValidator) -> None:
        """Attach product-specific validation for review selections."""
        self._action_resolution_validator = validator

    def configure_planning_context_provider(self, provider: PlanningContextProvider) -> None:
        """Provide a compact, read-only material snapshot before planning."""
        self._planning_context_provider = provider

    def _notify_workspace_state(
        self, workspace: dict[str, Any], plan: dict[str, Any] | None = None,
    ) -> None:
        listener = self._workspace_state_listener
        if listener is None:
            return
        try:
            listener(copy.deepcopy(workspace), copy.deepcopy(plan) if plan else None)
        except Exception:
            # Agent orchestration must not lose a completed plan merely because
            # the presentation-side job summary could not be refreshed.
            return

    def _planning_context(self, job_id: str) -> dict[str, Any]:
        provider = self._planning_context_provider
        if provider is None:
            return {"jobId": job_id, "available": False}
        try:
            value = provider(job_id)
        except Exception:
            return {"jobId": job_id, "available": False}
        return copy.deepcopy(value) if isinstance(value, dict) else {"jobId": job_id, "available": False}

    # ------------------------------------------------------ skill registry

    def seed_skills(self, source_root: Path) -> None:
        if not source_root.is_dir():
            return
        # Drop builtin skills stored under the pre-rename "cliptalk-" prefix so
        # upgraded data directories do not carry duplicate enabled entries.
        for legacy in self.store.list(
            "skills",
            predicate=lambda item: str(item.get("source") or "") == "builtin"
            and str(item.get("id") or "").startswith("cliptalk-"),
        ):
            self.store.delete("skills", str(legacy["id"]))
        for skill_file in sorted(source_root.glob("*/SKILL.md")):
            markdown = skill_file.read_text(encoding="utf-8")
            fields = parse_skill_markdown(markdown)
            skill_id = fields["name"]
            existing = self.store.get("skills", skill_id)
            digest = content_hash(markdown)
            if existing and existing.get("contentHash") == digest:
                continue
            self.store.save("skills", {
                "id": skill_id, "name": skill_id, "description": fields["description"],
                "status": "enabled", "source": "builtin", "version": fields.get("version") or "1.0.0",
                "contentHash": digest, "skillMarkdown": markdown,
                "path": str(skill_file.parent.resolve()), "allowedTools": fields.get("allowedTools") or [],
                "workflowProfile": fields.get("workflowProfile") or "",
                "validation": {"valid": True, "errors": []},
            })

    def install_skill(
        self, *, markdown: str, source: str, status: str = "validated", path: str = "",
    ) -> dict[str, Any]:
        fields = parse_skill_markdown(markdown)
        digest = content_hash(markdown)
        skill_id = fields["name"]
        available_tools = {item["name"] for item in tool_catalog()}
        missing_tools = sorted(set(fields.get("allowedTools") or []) - available_tools)
        workflow_profile = str(fields.get("workflowProfile") or "")
        invalid_profile = bool(workflow_profile and workflow_profile not in WORKFLOW_PROFILE_KINDS)
        validation: dict[str, Any] = {"valid": not missing_tools and not invalid_profile, "errors": []}
        if missing_tools:
            validation["errors"].append(f"缺少工具：{', '.join(missing_tools)}")
            status = "draft"
        if invalid_profile:
            validation["errors"].append(f"未知工作流档案：{workflow_profile}")
            status = "draft"
        existing = self.store.get("skills", skill_id)
        declared_version = str(fields.get("version") or "1.0.0")
        version = declared_version
        if existing and source != "builtin" and declared_version == "1.0.0":
            major = int(str(existing.get("version") or "0").split(".")[0] or 0) + 1
            version = f"{major}.0.0"
        return self.store.save("skills", {
            "id": skill_id, "name": skill_id, "description": fields["description"],
            "status": status, "source": source, "version": version,
            "contentHash": digest, "skillMarkdown": markdown, "path": path,
            "allowedTools": fields.get("allowedTools") or [],
            "workflowProfile": workflow_profile, "validation": validation,
        })

    def generate_skill(self, *, request: str) -> dict[str, Any]:
        result = self.planner_backend.generate_skill(
            {
                "request": request,
                "toolCatalog": tool_catalog(),
            },
            model_config=self.model_config_resolver(),
        )
        markdown = str(result.get("skillMarkdown") or "")
        skill = self.install_skill(markdown=markdown, source="generated", status="draft")
        profile = profile_for_skill(skill)
        profile_valid = bool(skill.get("workflowProfile")) and bool(profile.get("managed"))
        skill["simulation"] = result.get("simulation") or {}
        simulation_valid = bool((skill["simulation"] or {}).get("valid"))
        skill["validation"] = {
            "valid": bool((skill.get("validation") or {}).get("valid")) and simulation_valid and profile_valid,
            "errors": list((skill.get("validation") or {}).get("errors") or [])
            + ([] if simulation_valid else ["模拟规划未通过"])
            + ([] if profile_valid else ["生成 Skill 必须声明有效的 workflow-profile，并完整包含该档案所需工具"]),
        }
        skill["status"] = "validated" if skill["validation"]["valid"] else "draft"
        return self.store.save("skills", skill)

    def enable_skill(self, skill_id: str, expected_hash: str) -> dict[str, Any]:
        skill = self.store.get("skills", skill_id)
        if not skill:
            raise KeyError(skill_id)
        if str(skill.get("contentHash") or "") != expected_hash:
            raise ValueError("Skill 内容已经变化，请重新审核")
        if not bool((skill.get("validation") or {}).get("valid")):
            raise ValueError("Skill 校验未通过")
        skill["status"] = "enabled"
        return self.store.save("skills", skill)

    def tool_catalog(self) -> list[dict[str, Any]]:
        return tool_catalog()

    def _enabled_skill_for_kind(self, kind: str) -> dict[str, Any] | None:
        return next((
            item for item in self.store.list(
                "skills", newest_first=False,
                predicate=lambda candidate: candidate.get("status") == "enabled",
            )
            if str(profile_for_skill(item).get("kind") or "") == str(kind)
        ), None)

    def _enabled_skill(self, skill_id: str | None) -> dict[str, Any]:
        if skill_id:
            skill = self.store.get("skills", skill_id)
            if not skill or skill.get("status") != "enabled":
                raise ValueError("指定的 Skill 不存在或尚未启用")
            return skill
        enabled = self.store.list(
            "skills", newest_first=False,
            predicate=lambda item: item.get("status") == "enabled",
        )
        if not enabled:
            raise ValueError("当前没有已启用的 Skill")
        raise ValueError("自动 Skill 路由需要提供当前目标")

    def _resolve_skill_chain(
        self, primary: dict[str, Any], raw_chain: Any,
    ) -> list[dict[str, Any]]:
        resolved = [primary]
        for value in raw_chain if isinstance(raw_chain, list) else []:
            skill_id = str(value.get("id") or "") if isinstance(value, dict) else ""
            if not skill_id or skill_id == primary.get("id"):
                continue
            skill = self.store.get("skills", skill_id)
            if skill and skill.get("status") == "enabled":
                resolved.append(skill)
        return resolved

    # ---------------------------------------------------------- workspaces

    def create_workspace(self, *, job_id: str, title: str = "") -> dict[str, Any]:
        for existing in self.store.list("workspaces"):
            if existing.get("jobId") == job_id:
                return existing
        workspace = self.store.save("workspaces", {
            "id": f"ws_{uuid.uuid4().hex}", "jobId": job_id,
            "title": title or "智能剪辑工作区", "revision": 1,
            "status": "ready", "activePlanId": None,
        })
        self.store.append_event(workspace["id"], "workspace.created", {"workspace": workspace})
        self._notify_workspace_state(workspace)
        return workspace

    def workspace_for_job(self, job_id: str) -> dict[str, Any] | None:
        return next((
            item for item in self.store.list("workspaces")
            if str(item.get("jobId") or "") == str(job_id)
        ), None)

    def bind_workspace_to_job(
        self, *, workspace_id: str, job_id: str, source_job_id: str = "",
    ) -> dict[str, Any]:
        workspace = self.store.get("workspaces", workspace_id)
        if not workspace:
            raise KeyError(workspace_id)
        workspace["jobId"] = job_id
        if source_job_id:
            workspace.setdefault("sourceJobId", source_job_id)
        workspace = self.store.save("workspaces", workspace)
        plan = self.store.get("plans", str(workspace.get("activePlanId") or ""))
        self._notify_workspace_state(workspace, plan)
        return workspace

    def active_plan_for_workspace(self, workspace: dict[str, Any]) -> dict[str, Any] | None:
        plan_id = str(workspace.get("activePlanId") or "")
        if plan_id:
            plan = self.store.get("plans", plan_id)
            if plan and str(plan.get("status") or "") in ACTIVE_PLAN_STATUSES:
                return plan
        return next((
            item for item in self.store.list(
                "plans", predicate=lambda candidate: candidate.get("workspaceId") == workspace.get("id"),
            )
            if str(item.get("status") or "") in ACTIVE_PLAN_STATUSES
        ), None)

    # -------------------------------------------------------------- routing

    def route_skill(self, goal: str, *, planning_context: dict[str, Any] | None = None) -> dict[str, Any]:
        enabled = self.store.list(
            "skills", newest_first=False,
            predicate=lambda item: item.get("status") == "enabled",
        )
        if not enabled:
            raise ValueError("当前没有已启用的 Skill")
        context = planning_context if isinstance(planning_context, dict) else {}
        brief = editing_brief(goal, context)
        eligible = [
            item for item in enabled
            if skill_routing_eligibility(item, brief=brief, context=context)[0]
        ]
        if not eligible:
            raise ValueError("当前没有满足素材前置条件的 Skill；请先生成或选择一个可审核成片")
        editing = context.get("editing") if isinstance(context.get("editing"), dict) else {}
        source_edit_requested = str(brief.get("delivery") or "") == "timeline" and bool(
            brief.get("retrievalQuery") or brief.get("sourceRange") or brief.get("removedSourceRanges")
            or _goal_mentions_highlights(goal)
        )
        revision_requested = bool(editing.get("hasActiveSession") or editing.get("hasOutputs")) and bool(
            brief.get("durationSource") == "relative"
            or brief.get("anchorStart")
            or brief.get("sourceRange")
            or brief.get("removedSourceRanges")
            or _goal_mentions_revision(goal)
        )
        preferred_kind = _preferred_skill_kind(brief, goal, editing, source_edit_requested, revision_requested)
        if preferred_kind:
            deterministic = next((
                item for item in eligible
                if str(profile_for_skill(item).get("kind") or "") == preferred_kind
            ), None)
            if deterministic:
                return deterministic
        result = self.planner_backend.route_skill(
            {
                "goal": goal,
                "skills": [{
                    "id": item["id"], "description": item["description"],
                    "version": item["version"], "contentHash": item["contentHash"],
                } for item in eligible],
                "routingHints": {
                    "sourceEditRequested": bool(brief.get("retrievalQuery")) and str(brief.get("delivery") or "") == "timeline",
                    "hasExistingOutputs": bool((context.get("editing") or {}).get("hasOutputs")),
                    "socialDelivery": _social_delivery(goal),
                },
            },
            model_config=self.model_config_resolver(),
        )
        selected_id = str(result.get("skillId") or "")
        selected = next((item for item in eligible if item["id"] == selected_id), None)
        if not selected:
            raise AgentPlannerError("Agent 没有选择有效的 Skill")
        return selected

    # ------------------------------------------------------- plan requests

    def prepare_plan_request(
        self, *, workspace_id: str, goal: str, skill_id: str | None = None,
        execution_mode: str = AUTONOMOUS_REVIEW,
        input_context: dict[str, Any] | None = None, replaces_plan_id: str | None = None,
        message_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        import re

        normalized_mode = str(execution_mode or AUTONOMOUS_REVIEW)
        if normalized_mode not in VALID_EXECUTION_MODES:
            raise ValueError("Agent 执行方式无效")
        planning_brief = editing_brief(goal, {})
        planning_surface = "editor" if (
            str(planning_brief.get("delivery") or "") == "timeline"
            or bool(planning_brief.get("anchorStart"))
            or bool(planning_brief.get("sourceRange"))
            or bool(planning_brief.get("removedSourceRanges"))
            or bool(planning_brief.get("durationSource") == "relative")
            or bool(re.search(
                r"当前成片|已有成片|时间线|二次精剪|返修|重排|缩短|恢复|加字幕|添加文本|添加文字|加文字|叠加文字",
                str(goal or ""),
            ))
        ) else "agent"
        with self._execution_lock:
            workspace = self.store.get("workspaces", workspace_id)
            if not workspace:
                raise KeyError(workspace_id)
            active_plan = self.active_plan_for_workspace(workspace)
            replacing = bool(active_plan and active_plan["id"] == replaces_plan_id and active_plan.get("status") == "awaiting_confirmation")
            if (active_plan and not replacing) or workspace.get("planningRequestId"):
                raise ValueError("当前已有正在生成、等待确认或执行中的 Agent 计划；请先在计划面板完成、停止或取消该计划")
            previous_status = str(workspace.get("status") or "ready")
            workspace.update({
                "status": "planning", "planningRequestId": f"planning_{uuid.uuid4().hex}",
                "planningStartedAt": now_iso(), "planningExecutionMode": normalized_mode,
                "planningSurface": planning_surface,
                "planningPreviousStatus": previous_status,
                "planningInputContext": copy.deepcopy(input_context or {}),
                "replacesPlanId": replaces_plan_id if replacing else None,
                "planningMessageId": message_id,
            })
            workspace = self.store.save("workspaces", workspace)
        try:
            self._notify_workspace_state(workspace)
            planning_context = self._planning_context(str(workspace.get("jobId") or ""))
            planning_context["inputContext"] = copy.deepcopy(input_context or {})
            if (input_context or {}).get("outputAspect"):
                planning_context["delivery"] = {"outputAspect": input_context["outputAspect"], "outputFit": input_context.get("outputFit", "blur")}
            ranges = (input_context or {}).get("sourceRanges") or []
            if len(ranges) == 1:
                planning_context.update({"sourceScope": "custom", "sourceRange": ranges[0]})
            brief = editing_brief(goal, planning_context)
            if brief.get("unresolvedRelativeDuration"):
                raise ValueError("找不到当前成片的可用时长，无法理解“再短一点”。请先选择一个成片版本，或直接说明目标秒数。")
            skill = self._enabled_skill(skill_id) if skill_id else self.route_skill(goal, planning_context=planning_context)
            eligible, reason = skill_routing_eligibility(skill, brief=brief, context=planning_context)
            # An explicitly selected Skill is allowed to produce a structured
            # prerequisite result. Automatic routing still excludes it so an
            # inapplicable editor can never displace a valid source workflow.
            if not eligible and not skill_id:
                raise ValueError(f"当前 Skill 不适用于该目标：{reason}")
            profile = profile_for_skill(skill)
            skills = compose_skills(
                skill, brief=brief, context=planning_context,
                enabled_skill_for_kind=self._enabled_skill_for_kind,
            )
            model = self.model_config_resolver()
            catalog = tool_catalog()
        except Exception:
            self.release_plan_request(
                workspace_id, request_id=str(workspace.get("planningRequestId") or ""),
            )
            raise
        allowed_tools = sorted({
            tool for item in skills for tool in profile_for_skill(item).get("tools", set())
        })
        payload = {
            "requestId": str(workspace.get("planningRequestId") or ""),
            "workspaceId": workspace_id,
            "goal": goal, "skill": {
                "id": skill["id"], "version": skill["version"],
                "contentHash": skill["contentHash"], "markdown": skill["skillMarkdown"],
            },
            "skills": [{
                "id": item["id"], "version": item["version"],
                "contentHash": item["contentHash"], "markdown": item["skillMarkdown"],
                "role": "primary" if index == 0 else "addon",
                "workflowProfile": profile_for_skill(item)["kind"],
            } for index, item in enumerate(skills)],
            "workspace": {"jobId": workspace["jobId"], "revision": workspace["revision"]},
            "planningContext": planning_context, "brief": brief,
            "executionMode": normalized_mode,
            "profile": {"kind": profile["kind"], "managed": bool(profile["managed"]), "allowedTools": allowed_tools},
            "toolCatalog": catalog, "model": model,
        }
        return workspace, skill, payload

    def release_plan_request(self, workspace_id: str, *, request_id: str = "") -> None:
        """Clear an in-flight planning reservation after stream completion/error."""
        with self._execution_lock:
            workspace = self.store.get("workspaces", workspace_id)
            if not workspace:
                return
            current_request_id = str(workspace.get("planningRequestId") or "")
            if not current_request_id or (request_id and current_request_id != request_id):
                return
            workspace.pop("planningRequestId", None)
            workspace.pop("planningStartedAt", None)
            workspace.pop("planningExecutionMode", None)
            workspace.pop("planningSurface", None)
            previous_status = str(workspace.pop("planningPreviousStatus", "") or "")
            if previous_status and previous_status != "planning":
                workspace["status"] = previous_status
            elif not workspace.get("activePlanId"):
                workspace["status"] = "ready"
            else:
                previous = self.store.get("plans", workspace["activePlanId"])
                if previous and previous.get("status") == "awaiting_confirmation":
                    workspace["status"] = "awaiting_plan_confirmation"
            workspace.pop("planningInputContext", None)
            workspace.pop("planningMessageId", None)
            workspace.pop("replacesPlanId", None)
            workspace = self.store.save("workspaces", workspace)
        self._notify_workspace_state(workspace)

    def recover_stale_planning_requests(self, *, max_age_seconds: float | None = None) -> int:
        """Release reservations whose owning HTTP planning stream no longer exists."""
        maximum_age = max(
            30.0,
            float(max_age_seconds if max_age_seconds is not None else self.timeout_seconds + 30.0),
        )
        now = datetime.now(timezone.utc)
        recovered = 0
        with self._execution_lock:
            workspaces = self.store.list(
                "workspaces",
                predicate=lambda item: bool(item.get("planningRequestId")),
            )
            for workspace in workspaces:
                try:
                    started = datetime.fromisoformat(str(workspace.get("planningStartedAt") or ""))
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    age = (now - started.astimezone(timezone.utc)).total_seconds()
                except (TypeError, ValueError):
                    age = maximum_age + 1.0
                if age <= maximum_age:
                    continue
                request_id = str(workspace.pop("planningRequestId", "") or "")
                workspace.pop("planningStartedAt", None)
                workspace.pop("planningExecutionMode", None)
                workspace.pop("planningSurface", None)
                previous_status = str(workspace.pop("planningPreviousStatus", "") or "")
                if previous_status and previous_status != "planning":
                    workspace["status"] = previous_status
                elif not workspace.get("activePlanId"):
                    workspace["status"] = "ready"
                workspace = self.store.save("workspaces", workspace)
                self.store.append_event(workspace["id"], "planning.recovered", {
                    "requestId": request_id,
                    "reason": "规划连接已超时，工作区已恢复为可重试状态",
                })
                recovered += 1
                self._notify_workspace_state(workspace)
        return recovered

    def record_planning_progress(self, workspace_id: str, update: dict[str, Any]) -> None:
        """Persist an auditable planning milestone for polling and reloads."""
        phase = str(update.get("phase") or "").strip()[:80]
        title = str(update.get("title") or "正在生成 Agent 执行计划").strip()[:160]
        detail = str(update.get("detail") or "正在整理可确认的剪辑步骤。").strip()[:500]
        with self._execution_lock:
            workspace = self.store.get("workspaces", workspace_id)
            if not workspace or str(workspace.get("status") or "") != "planning":
                return
            workspace["planningProgress"] = {
                "phase": phase, "title": title, "detail": detail,
                "updatedAt": now_iso(),
            }
            workspace = self.store.save("workspaces", workspace)
        self._notify_workspace_state(workspace)

    # ------------------------------------------------------- plan creation

    def persist_plan_result(
        self, *, workspace: dict[str, Any], skill: dict[str, Any],
        goal: str, result: dict[str, Any], execution_mode: str = AUTONOMOUS_REVIEW,
    ) -> dict[str, Any]:
        workspace_id = str(workspace["id"])
        raw_plan = result.get("plan") if isinstance(result.get("plan"), dict) else result
        context = raw_plan.get("planningContext") if isinstance(raw_plan, dict) and isinstance(raw_plan.get("planningContext"), dict) else self._planning_context(str(workspace.get("jobId") or ""))
        context = copy.deepcopy(context)
        frozen = copy.deepcopy(workspace.get("planningInputContext") or {})
        context["inputContext"] = frozen
        if frozen.get("outputAspect"):
            context["delivery"] = {"outputAspect": frozen["outputAspect"], "outputFit": frozen.get("outputFit", "blur")}
        ranges = frozen.get("sourceRanges") or []
        if len(ranges) == 1:
            context.update({"sourceScope": "custom", "sourceRange": ranges[0]})
        brief = editing_brief(goal, context)
        if bool(profile_for_skill(skill).get("managed")):
            # Built-in workflow composition is server-owned. A planning model
            # must not turn words such as “旧封面/旧输出” in an audit request
            # into media-producing add-ons.
            skills = compose_skills(
                skill, brief=brief, context=context,
                enabled_skill_for_kind=self._enabled_skill_for_kind,
            )
        else:
            skills = self._resolve_skill_chain(
                skill, raw_plan.get("skillChain") if isinstance(raw_plan, dict) else None,
            )
        compiled = compile_profile_plan(
            raw_plan, skill=skill, skills=skills, goal=goal, context=context,
            execution_mode=execution_mode,
        )
        from .brief import plan_understanding

        compiled["understanding"] = plan_understanding(brief, context)
        plan = normalize_plan(
            compiled, workspace=workspace, skill=skill, skills=skills, goal=goal,
        )
        plan["inputContext"] = frozen
        plan["replacesPlanId"] = workspace.get("replacesPlanId")
        # Bind review approval to the actual selection and output version.
        if frozen:
            plan["planHash"] = content_hash(plan["planHash"] + json.dumps(frozen, sort_keys=True, ensure_ascii=False))
        with self._execution_lock:
            latest_workspace = self.store.get("workspaces", workspace_id)
            expected_request = str(workspace.get("planningRequestId") or "")
            if expected_request and (not latest_workspace or latest_workspace.get("planningRequestId") != expected_request):
                raise ValueError("这次规划请求已失效，请查看当前计划")
            if latest_workspace:
                workspace = latest_workspace
            replaced = self.store.get("plans", str(workspace.get("replacesPlanId") or ""))
            if workspace.get("replacesPlanId") and (not replaced or replaced.get("status") != "awaiting_confirmation"):
                raise ValueError("原方案状态已变化，请重新查看；未替换原方案")
            plan = self.store.save("plans", plan)
            if replaced:
                replaced.update({"status": "cancelled", "supersededBy": plan["id"], "completedAt": now_iso()})
                self.store.save("plans", replaced)
            messages = list(workspace.get("messages") or [])
            if not workspace.get("planningMessageId"):
                messages.extend([
                    {"role": "user", "text": goal, "createdAt": now_iso()},
                    {"role": "assistant", "kind": "plan", "text": plan["summary"], "planId": plan["id"], "createdAt": now_iso()},
                ])
            latest_workspace = self.store.get("workspaces", workspace_id)
            if latest_workspace:
                workspace = latest_workspace
            workspace.update({
                "activePlanId": plan["id"], "status": "awaiting_plan_confirmation",
                "messages": messages, "selectedSkillId": skill["id"],
                "executionMode": plan["executionMode"],
            })
            workspace.pop("planningRequestId", None)
            workspace.pop("planningStartedAt", None)
            workspace.pop("planningExecutionMode", None)
            workspace.pop("planningSurface", None)
            workspace.pop("planningPreviousStatus", None)
            workspace.pop("planningInputContext", None)
            workspace.pop("planningMessageId", None)
            workspace.pop("replacesPlanId", None)
            self.store.save("workspaces", workspace)
            self.store.append_event(workspace_id, "plan.created", {"plan": plan})
            self.store.append_event(workspace_id, "plan.confirmation_required", {
                "planId": plan["id"], "planHash": plan["planHash"],
            })
            self._notify_workspace_state(workspace, plan)
            return plan

    def persist_fallback_plan(
        self, *, workspace: dict[str, Any], skill: dict[str, Any], goal: str,
        execution_mode: str, error: Exception,
    ) -> dict[str, Any]:
        """Compile a deterministic plan only for first-party managed Skills."""
        if not bool(profile_for_skill(skill).get("managed")):
            raise error
        warning = str(error)[:500]
        plan = self.persist_plan_result(
            workspace=workspace, skill=skill, goal=goal,
            result={
                "plan": {
                    "summary": goal,
                    "strategy": {},
                    "planningSource": "deterministic_fallback",
                    "planningWarning": warning,
                },
                "events": [{
                    "phase": "plan_ready", "title": "已切换本地确定性规划",
                    "detail": "在线规划服务暂不可用，已按内置 Skill 工作流生成可确认计划。",
                }],
            },
            execution_mode=execution_mode,
        )
        self.store.append_event(str(workspace["id"]), "planning.fallback", {
            "planId": plan["id"], "reason": warning,
        })
        return plan

    # --------------------------------------------------------- graph entry

    def _graph_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }

    def _thread_lock(self, thread_id: str) -> threading.RLock:
        with self._execution_lock:
            return self._thread_locks.setdefault(thread_id, threading.RLock())

    def _make_stream_emit(self, workspace_id: str) -> Callable[[dict[str, Any]], None]:
        writer = None
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:  # noqa: BLE001 - non-streaming invocation
            writer = None

        def emit(event: dict[str, Any]) -> None:
            if not isinstance(event, dict) or not event.get("type"):
                return
            try:
                self.store.append_event(workspace_id, "agent.message_update", copy.deepcopy(event))
                if event.get("type") == "planning.progress":
                    self.record_planning_progress(workspace_id, event)
            except Exception:  # noqa: BLE001 - progress events must not break planning
                pass
            if writer is not None:
                try:
                    writer(copy.deepcopy(event))
                except Exception:  # noqa: BLE001
                    pass

        return emit

    def create_plan(
        self, *, workspace_id: str, goal: str, skill_id: str | None = None,
        execution_mode: str = AUTONOMOUS_REVIEW,
        input_context: dict[str, Any] | None = None, replaces_plan_id: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        workspace, skill, payload = self.prepare_plan_request(
            workspace_id=workspace_id, goal=goal, skill_id=skill_id,
            execution_mode=execution_mode,
            input_context=input_context, replaces_plan_id=replaces_plan_id, message_id=message_id,
        )
        thread_id = f"agent:{payload['requestId']}"
        try:
            try:
                final_state = self._invoke_plan_graph(
                    thread_id, workspace, skill, payload, goal, execution_mode,
                )
                plan_id = str((final_state or {}).get("plan_id") or "")
                plan = self.store.get("plans", plan_id) if plan_id else None
                if not plan:
                    raise AgentPlannerError("规划没有生成可确认计划")
                return plan
            except (AgentPlannerError, ValueError) as error:
                return self.persist_fallback_plan(
                    workspace=workspace, skill=skill, goal=goal,
                    execution_mode=execution_mode, error=error,
                )
        finally:
            self.release_plan_request(
                workspace_id, request_id=str(workspace.get("planningRequestId") or ""),
            )

    def stream_plan(
        self, *, workspace_id: str = "", goal: str = "", skill_id: str | None = None,
        execution_mode: str = AUTONOMOUS_REVIEW,
        input_context: dict[str, Any] | None = None, replaces_plan_id: str | None = None,
        message_id: str | None = None,
        prepared: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
    ):
        """Drive one planning run through the graph, yielding wire events.

        Yields ``{"event": name, "data": payload}`` dicts; the SSE endpoint
        forwards them verbatim so the frontend contract stays unchanged.
        ``prepared`` reuses a synchronous ``prepare_plan_request`` result so
        validation errors map to HTTP status codes before streaming starts.
        """
        if prepared is None:
            workspace, skill, payload = self.prepare_plan_request(
                workspace_id=workspace_id, goal=goal, skill_id=skill_id,
                execution_mode=execution_mode,
                input_context=input_context, replaces_plan_id=replaces_plan_id,
                message_id=message_id,
            )
        else:
            workspace, skill, payload = prepared
        thread_id = f"agent:{payload['requestId']}"
        try:
            emitted_plan = False
            try:
                with self._thread_lock(thread_id):
                    for mode, chunk in self.graph.stream(
                        self._plan_input(workspace, skill, payload, goal, execution_mode),
                        self._graph_config(thread_id),
                        stream_mode=["custom", "updates"],
                    ):
                        if mode == "custom" and isinstance(chunk, dict) and chunk.get("type"):
                            yield {"event": str(chunk["type"]), "data": chunk}
                snapshot = self.graph.get_state(self._graph_config(thread_id))
                plan_id = str((snapshot.values or {}).get("plan_id") or "")
                plan = self.store.get("plans", plan_id) if plan_id else None
                if plan:
                    emitted_plan = True
                    yield {"event": "plan", "data": {"action": "plan_confirmation", "plan": plan}}
                else:
                    yield {"event": "error", "data": {"message": "规划没有生成可确认计划"}}
            except Exception as error:  # noqa: BLE001 - mirrors the old SSE fallback
                try:
                    plan = self.persist_fallback_plan(
                        workspace=workspace, skill=skill, goal=goal,
                        execution_mode=execution_mode, error=error,
                    )
                except Exception:  # noqa: BLE001
                    yield {"event": "error", "data": {"message": str(error)}}
                else:
                    emitted_plan = True
                    yield {
                        "event": "plan",
                        "data": {
                            "action": "plan_confirmation", "plan": plan,
                            "warning": "在线规划服务暂不可用，已使用内置确定性规划。",
                        },
                    }
            if not emitted_plan:
                yield {"event": "error", "data": {"message": "Pi 规划流程意外结束且没有返回计划"}}
        finally:
            self.release_plan_request(
                workspace_id, request_id=str(workspace.get("planningRequestId") or ""),
            )

    def _plan_input(
        self, workspace: dict[str, Any], skill: dict[str, Any], payload: dict[str, Any],
        goal: str, execution_mode: str,
    ) -> dict[str, Any]:
        return {
            "op": "plan",
            "workspace_id": str(workspace["id"]),
            "goal": goal,
            "skill_id": str(skill["id"]),
            "execution_mode": execution_mode,
            "workspace": copy.deepcopy(workspace),
            "skill": copy.deepcopy(skill),
            "payload": copy.deepcopy(payload),
        }

    def _invoke_plan_graph(
        self, thread_id: str, workspace: dict[str, Any], skill: dict[str, Any],
        payload: dict[str, Any], goal: str, execution_mode: str,
    ) -> dict[str, Any] | None:
        with self._thread_lock(thread_id):
            return self.graph.invoke(
                self._plan_input(workspace, skill, payload, goal, execution_mode),
                self._graph_config(thread_id),
            )

    def _plan_thread_id(self, plan: dict[str, Any]) -> str:
        return str(plan.get("threadId") or f"plan:{plan.get('id') or uuid.uuid4().hex}")

    def _kick_plan(self, plan_id: str) -> None:
        """Resume a parked graph thread, or start execution on a fresh one."""
        plan = self.store.get("plans", plan_id)
        if not plan or str(plan.get("status") or "") not in {"running", "approved"}:
            return
        thread_id = self._plan_thread_id(plan)
        config = self._graph_config(thread_id)
        with self._thread_lock(thread_id):
            try:
                snapshot = self.graph.get_state(config)
            except Exception:  # noqa: BLE001 - unknown thread
                snapshot = None
            try:
                if snapshot is not None and snapshot.next:
                    self.graph.invoke(Command(resume={"action": "kick"}), config)
                else:
                    self.graph.invoke({
                        "op": "execute", "plan_id": plan_id,
                        "workspace_id": str(plan.get("workspaceId") or ""),
                        "goal": str(plan.get("goal") or ""),
                    }, config)
            except Exception:  # noqa: BLE001 - graph resumes must not crash callers
                LOGGER.exception("Agent 计划 %s 恢复执行失败", plan_id)
                self._fail_plan_quietly(plan_id)

    def _fail_plan_quietly(self, plan_id: str) -> None:
        plan = self.store.get("plans", plan_id)
        if plan and str(plan.get("status") or "") not in TERMINAL_PLAN_STATUSES:
            LOGGER.exception("Agent 计划 %s 的图执行异常，已标记为失败", plan_id)
            self._finish_plan(plan, "failed")

    def probe(self, model_config: dict[str, Any]) -> dict[str, Any]:
        return self.planner_backend.probe(model_config)

    # ------------------------------------------------------ plan lifecycle

    def approve_plan(self, plan_id: str, *, expected_hash: str) -> dict[str, Any]:
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan:
                raise KeyError(plan_id)
            if plan.get("status") != "awaiting_confirmation":
                raise ValueError("计划当前不能确认")
            if plan.get("planHash") != expected_hash:
                raise ValueError("计划已经变化，请重新审核")
            brief = plan.get("brief") if isinstance(plan.get("brief"), dict) else {}
            if str(brief.get("coverSourceStatus") or "") == "requires_external_asset":
                subject = str(brief.get("coverSubject") or "指定人物").strip()
                raise ValueError(
                    f"当前无法获取“{subject}”的外部封面图片；请修改要求，改用本次视频画面制作封面"
                )
            workspace = self.store.get("workspaces", str(plan["workspaceId"]))
            if workspace and workspace.get("planningRequestId"):
                raise ValueError("正在准备修改方案，请等待完成后确认")
            validator = getattr(self, "_reference_validator", None)
            if validator:
                validator(plan)
            if not workspace or workspace.get("revision") != plan.get("workspaceRevision"):
                raise ValueError("素材工作区已经变化，请重新规划")
            plan["threadId"] = self._plan_thread_id(plan)
            plan["status"] = "approved"
            plan["approval"] = {
                "approvedAt": now_iso(), "planHash": expected_hash,
                "workspaceRevision": workspace["revision"],
                "executionMode": str(plan.get("executionMode") or STEPWISE_REVIEW),
                "allowedTools": sorted({step["tool"] for step in plan["steps"]}),
                "allowedSideEffects": sorted({step["sideEffect"] for step in plan["steps"]}),
                "skillHash": plan["skillHash"],
                "skillHashes": {
                    str(item.get("id") or ""): str(item.get("contentHash") or "")
                    for item in plan.get("skills") or [] if str(item.get("id") or "")
                },
            }
            plan = self.store.save("plans", plan)
            run = self.store.save("runs", {
                "id": f"run_{uuid.uuid4().hex}", "workspaceId": workspace["id"],
                "planId": plan_id, "status": "running", "startedAt": now_iso(),
            })
            plan["runId"] = run["id"]
            plan["status"] = "running"
            plan = self.store.save("plans", plan)
            workspace.update({
                "status": "running", "activePlanId": plan_id,
                "executionMode": str(plan.get("executionMode") or STEPWISE_REVIEW),
            })
            self.store.save("workspaces", workspace)
            self.store.append_event(workspace["id"], "plan.approved", {"planId": plan_id, "runId": run["id"]})
        self._notify_workspace_state(workspace, plan)
        self._kick_plan(plan_id)
        return self.store.get("plans", plan_id) or plan

    def cancel_plan(self, plan_id: str) -> dict[str, Any]:
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan:
                raise KeyError(plan_id)
            if plan.get("status") in {"preview_ready", "no_result", "failed", "cancelled"}:
                return plan
            plan["status"] = "cancelled"
            plan["completedAt"] = now_iso()
            run = self.store.get("runs", str(plan.get("runId") or "")) if plan.get("runId") else None
            if run:
                run.update({"status": "cancelled", "completedAt": plan["completedAt"]})
                self.store.save("runs", run)
            for step in plan["steps"]:
                if step["status"] in {"pending", "running", "waiting_operation", "action_required"}:
                    step["status"] = "cancelled"
            plan = self.store.save("plans", plan)
            workspace = self.store.get("workspaces", str(plan["workspaceId"]))
            if workspace:
                workspace["status"] = "cancelled"
                workspace = self.store.save("workspaces", workspace)
            self.store.append_event(plan["workspaceId"], "plan.cancelled", {"planId": plan_id})
        self._cancel_plan_operations(plan_id)
        if workspace:
            self._notify_workspace_state(workspace, plan)
        return plan

    def resolve_action(
        self, plan_id: str, *, approved: bool, value: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan:
                raise KeyError(plan_id)
            receipt = content_hash(json.dumps({"approved": approved, "value": value}, sort_keys=True, ensure_ascii=False))
            if approved and receipt in (plan.get("actionResolutionReceipts") or []):
                return plan
            if plan.get("status") != "action_required":
                raise ValueError("计划当前没有待确认操作")
            step = next((item for item in plan["steps"] if item["status"] == "action_required"), None)
            if not step:
                raise ValueError("待确认步骤不存在")
            if not approved:
                step.update({"status": "failed", "error": "用户拒绝了该步骤", "completedAt": now_iso()})
                plan["status"] = "failed"
                self.store.save("plans", plan)
                self._finish_plan(plan, "failed")
                return self.store.get("plans", plan_id) or plan
            workspace = self.store.get("workspaces", str(plan["workspaceId"]))
            if not workspace:
                raise ValueError("Agent Workspace 不存在")
            resolved_value = self._validate_action_resolution(workspace, step, value)
            if self._action_resolution_validator:
                resolved_value = self._action_resolution_validator(workspace, step, resolved_value)
            if (step.get("result") or {}).get("action") == "content_evidence_review":
                binder = getattr(self, "_reference_review_binder", None)
                if binder:
                    binder(plan, workspace, resolved_value)
                # Evidence approval precedes the proposal: replay, don't skip it.
                step.update({"status": "pending", "reviewConfirmation": resolved_value})
                step.pop("result", None)
                step.pop("completedAt", None)
            else:
                step.update({
                    "status": "completed", "completedAt": now_iso(),
                    "result": {"userConfirmed": True, "value": resolved_value},
                })
            plan["status"] = "running"
            plan["actionResolutionReceipts"] = [*(plan.get("actionResolutionReceipts") or []), receipt][-64:]
            plan = self.store.save("plans", plan)
            self.store.append_event(plan["workspaceId"], "action.resolved", {
                "planId": plan_id, "stepId": step["id"], "approved": True,
            })
            if workspace:
                self._notify_workspace_state(workspace, plan)
        self._kick_plan(plan_id)
        return self.store.get("plans", plan_id) or plan

    def retry_action(self, plan_id: str, *, cover_revision: dict[str, Any] | None = None) -> dict[str, Any]:
        """Re-run a recoverable action step whose artifact was never created."""
        pending_plan = self.store.get("plans", plan_id)
        if cover_revision is not None:
            cover_step = next((s for s in (pending_plan or {}).get("steps", []) if s.get("tool") == "review_cover_variants"), None)
            if not cover_step or (pending_plan or {}).get("status") not in {"action_required", "preview_ready", "completed"}:
                raise ValueError("请等待当前处理完成后修改封面")
        retry_failed_chain = False
        retry_qc_chain = False
        retry_cover_chain = False
        step: dict[str, Any] | None = None
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan:
                raise KeyError(plan_id)
            if cover_revision is not None:
                if plan.get("status") not in {"action_required", "preview_ready", "completed"}:
                    raise ValueError("任务状态已变化，请刷新后重试")
                waiting = next((s for s in plan["steps"] if s.get("status") == "action_required"), None)
                if waiting and waiting.get("tool") != "review_cover_variants":
                    raise ValueError("请先完成当前待确认事项，再修改封面")
                plan["status"] = "action_required"
                next(s for s in plan["steps"] if s.get("tool") == "review_cover_variants")["status"] = "action_required"
            qc_step = next((
                item for item in reversed(plan.get("steps") or [])
                if str(item.get("tool") or "") == "run_delivery_qc"
            ), None)
            qc_result = qc_step.get("result") if isinstance((qc_step or {}).get("result"), dict) else {}
            qc_artifact = qc_result.get("artifact") if isinstance(qc_result.get("artifact"), dict) else {}
            brief = plan.get("brief") if isinstance(plan.get("brief"), dict) else {}
            target = float(brief.get("targetSeconds") or 0)
            tolerance = float(brief.get("durationToleranceSeconds") or 0)
            preview_durations: list[float] = []
            for candidate_step in plan.get("steps") or []:
                candidate_result = candidate_step.get("result") if isinstance(candidate_step.get("result"), dict) else {}
                artifact = candidate_result.get("artifact") if isinstance(candidate_result.get("artifact"), dict) else {}
                if artifact.get("kind") == "social_reframe_preview" and isinstance(artifact.get("output"), dict):
                    preview_durations.append(float(artifact["output"].get("duration") or 0))
                elif artifact.get("kind") == "review_preview_batch":
                    preview_durations.extend(
                        float(item.get("duration") or 0)
                        for item in artifact.get("previews") or [] if isinstance(item, dict)
                    )
            composition_retryable = bool(
                target and preview_durations
                and any(duration < target - tolerance or duration > target + tolerance for duration in preview_durations)
            )
            repair = quality_repair_plan(qc_artifact, plan.get("steps") or [], plan.get("inputContext"))
            if (
                plan.get("status") == "preview_ready"
                and qc_artifact.get("passed") is False
                and (composition_retryable or repair.get("available"))
            ):
                replay_tools = {str(repair["replayFromTool"])} if repair.get("available") else {"review_content_evidence", "propose_timeline_edit"}
                replay_index = next((
                    index for index, item in enumerate(plan["steps"])
                    if str(item.get("tool") or "") in replay_tools
                ), None)
                if replay_index is None:
                    raise ValueError("当前质检问题没有可安全重建的时间线步骤")
                plan.setdefault("qualityRepairHistory", []).append({
                    "createdAt": now_iso(), "repair": repair, "qualityReport": copy.deepcopy(qc_artifact),
                })
                plan["qualityRepairHistory"] = plan["qualityRepairHistory"][-5:]
                for item in plan["steps"][replay_index:]:
                    item["status"] = "pending"
                    for key in ("completedAt", "error", "result", "operationId"):
                        item.pop(key, None)
                plan["status"] = "running"
                plan["replanCount"] = 0
                plan.pop("completedAt", None)
                plan = self.store.save("plans", plan)
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if workspace:
                    workspace["status"] = "running"
                    workspace = self.store.save("workspaces", workspace)
                self.store.append_event(plan["workspaceId"], "plan.retry_started", {
                    "planId": plan_id, "failedStepId": str((qc_step or {}).get("id") or ""),
                    "replayFromTool": plan["steps"][replay_index]["tool"],
                    "reason": "delivery_qc_failed",
                })
                if workspace:
                    self._notify_workspace_state(workspace, plan)
                retry_qc_chain = True
            elif plan.get("status") == "no_result":
                terminal_step = next((
                    item for item in plan["steps"]
                    if isinstance(item.get("result"), dict)
                    and str(item["result"].get("terminalStatus") or "") == "no_result"
                ), None)
                terminal_artifact = (
                    terminal_step.get("result", {}).get("artifact")
                    if isinstance((terminal_step or {}).get("result"), dict) else {}
                )
                reason_code = str((terminal_artifact or {}).get("reasonCode") or "")
                terminal_tool = str((terminal_step or {}).get("tool") or "")
                replay_tool = (
                    "propose_timeline_edit"
                    if reason_code in {"insufficient_coverage", "duration_constraint_unmet"}
                    and terminal_tool == "propose_timeline_edit"
                    else terminal_tool
                    if reason_code == "ambiguous_identity"
                    and terminal_tool in {"select_people", "select_speakers"}
                    else "search_content"
                )
                replay_index = next((
                    index for index, item in enumerate(plan["steps"])
                    if str(item.get("tool") or "") == replay_tool
                ), None)
                if replay_index is None:
                    raise ValueError("当前无结果计划没有可安全重跑的步骤")
                if (
                    replay_tool == "propose_timeline_edit"
                    and reason_code in {"insufficient_coverage", "duration_constraint_unmet"}
                    and isinstance(plan.get("brief"), dict)
                    and plan["brief"].get("durationExplicit")
                ):
                    refreshed_brief = editing_brief(
                        str(plan.get("goal") or ""),
                        plan.get("planningContext") if isinstance(plan.get("planningContext"), dict) else {},
                    )
                    if not refreshed_brief.get("durationExplicit") and refreshed_brief.get("targetSeconds") is None:
                        plan["brief"].update({
                            "targetSeconds": None,
                            "durationExplicit": False,
                            "durationSource": "none",
                        })
                        for candidate_step in plan["steps"]:
                            arguments = candidate_step.get("arguments")
                            if not isinstance(arguments, dict):
                                continue
                            if str(candidate_step.get("tool") or "") == "propose_timeline_edit":
                                arguments.pop("targetSeconds", None)
                                arguments.pop("toleranceSeconds", None)
                                arguments.pop("durationSource", None)
                                instruction = str(arguments.get("instruction") or "")
                                import re as _re

                                instruction = _re.sub(
                                    r"[；;]\s*目标\s*\d+(?:\.\d+)?\s*秒，允许浮动\s*±\s*\d+(?:\.\d+)?\s*秒",
                                    "",
                                    instruction,
                                )
                                arguments["instruction"] = instruction[:500]
                            elif str(candidate_step.get("tool") or "") == "run_delivery_qc":
                                arguments.pop("targetSeconds", None)
                                arguments.pop("toleranceSeconds", None)
                for item in plan["steps"][replay_index:]:
                    item["status"] = "pending"
                    for key in ("completedAt", "error", "result", "operationId"):
                        item.pop(key, None)
                plan["status"] = "running"
                plan["replanCount"] = 0
                plan.pop("completedAt", None)
                plan = self.store.save("plans", plan)
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if workspace:
                    workspace["status"] = "running"
                    workspace = self.store.save("workspaces", workspace)
                self.store.append_event(plan["workspaceId"], "plan.retry_started", {
                    "planId": plan_id,
                    "replayFromTool": plan["steps"][replay_index]["tool"],
                    "reason": reason_code or "no_result",
                })
                if workspace:
                    self._notify_workspace_state(workspace, plan)
                retry_failed_chain = True
            elif plan.get("status") == "failed":
                failed_step = next(
                    (item for item in plan["steps"] if item.get("status") == "failed"),
                    None,
                )
                replay_tools = replan_force_replay_tools(failed_step or {})
                replay_index = next((
                    index for index, item in enumerate(plan["steps"])
                    if str(item.get("tool") or "") in replay_tools
                ), None)
                if failed_step is None or replay_index is None:
                    raise ValueError("当前失败步骤不能安全地自动重跑")
                for item in plan["steps"][replay_index:]:
                    item["status"] = "pending"
                    for key in ("completedAt", "error", "result", "operationId"):
                        item.pop(key, None)
                plan["status"] = "running"
                plan["replanCount"] = 0
                plan.pop("completedAt", None)
                plan = self.store.save("plans", plan)
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if workspace:
                    workspace["status"] = "running"
                    workspace = self.store.save("workspaces", workspace)
                self.store.append_event(plan["workspaceId"], "plan.retry_started", {
                    "planId": plan_id, "failedStepId": failed_step["id"],
                    "replayFromTool": plan["steps"][replay_index]["tool"],
                })
                if workspace:
                    self._notify_workspace_state(workspace, plan)
                retry_failed_chain = True
            elif plan.get("status") != "action_required":
                raise ValueError("计划当前没有可重试的审核步骤")
            if retry_failed_chain or retry_qc_chain:
                step = None
            else:
                step = next((item for item in plan["steps"] if item["status"] == "action_required"), None)
            action_tool = str((step or {}).get("tool") or "")
            if action_tool == "review_cover_variants":
                replay_index = next((
                    index for index, item in enumerate(plan["steps"])
                    if str(item.get("tool") or "") == "propose_cover_candidates"
                ), None)
                if replay_index is None:
                    raise ValueError("当前封面计划缺少可重新生成的候选步骤")
                refreshed_brief = editing_brief(
                    str(plan.get("goal") or ""),
                    plan.get("planningContext") if isinstance(plan.get("planningContext"), dict) else {},
                )
                brief = plan.get("brief") if isinstance(plan.get("brief"), dict) else {}
                for key in (
                    "coverRequested", "coverAspect", "coverTitle", "coverSourceKind",
                    "coverSubject", "coverIdentityPolicy", "coverSourceTime", "coverSourceStatus",
                ):
                    brief[key] = copy.deepcopy(refreshed_brief.get(key))
                if cover_revision is not None:
                    brief.update(copy.deepcopy(cover_revision))
                    brief.update({"coverRequested": True, "coverSourceKind": "source_frame", "coverSourceStatus": "available"})
                    plan["coverRevision"] = copy.deepcopy(cover_revision)
                elif isinstance(plan.get("coverRevision"), dict):
                    brief.update(copy.deepcopy(plan["coverRevision"]))
                    brief.update({"coverRequested": True, "coverSourceKind": "source_frame", "coverSourceStatus": "available"})
                plan["brief"] = brief
                candidate_step = plan["steps"][replay_index]
                arguments = candidate_step.get("arguments") if isinstance(candidate_step.get("arguments"), dict) else {}
                arguments.update({
                    "aspectRatios": [str(brief.get("coverAspect") or "16:9")],
                    "focus": str(plan.get("goal") or "")[:240],
                })
                if isinstance(plan.get("coverRevision"), dict):
                    arguments["focus"] = "；".join(filter(None, [
                        str(brief.get("coverSubject") or ""),
                        str(brief.get("coverTitle") or ""),
                    ]))[:240] or "从当前视频选择清晰的封面画面"
                subject = str(brief.get("coverSubject") or "").strip()
                if subject:
                    arguments["subject"] = subject[:120]
                else:
                    arguments.pop("subject", None)
                if brief.get("coverSourceTime") is not None:
                    arguments["sourceTime"] = float(brief["coverSourceTime"])
                else:
                    arguments.pop("sourceTime", None)
                candidate_step["arguments"] = arguments
                for cover_step in plan["steps"]:
                    if cover_step.get("tool") == "render_cover_variants":
                        cover_step.setdefault("arguments", {}).update({"titleText": str(brief.get("coverTitle") or ""), "aspectRatios": [str(brief.get("coverAspect") or "16:9")]})
                for replay_step in plan["steps"][replay_index:]:
                    if replay_step.get("tool") in {"search_content", "analyze_video", "inspect_workspace", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "layout_subtitles"} and replay_step.get("status") == "completed":
                        continue
                    replay_step["status"] = "pending"
                    for key in ("completedAt", "error", "result", "operationId"):
                        replay_step.pop(key, None)
                plan["status"] = "running"
                plan.pop("completedAt", None)
                plan = self.store.save("plans", plan)
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if not workspace:
                    raise ValueError("Agent Workspace 不存在")
                workspace["status"] = "running"
                workspace = self.store.save("workspaces", workspace)
                self.store.append_event(plan["workspaceId"], "action.retried", {
                    "planId": plan_id,
                    "stepId": str((step or {}).get("id") or ""),
                    "replayFromTool": "propose_cover_candidates",
                    "reason": "cover_candidates_do_not_match_requirement",
                })
                self._notify_workspace_state(workspace, plan)
                retry_cover_chain = True
                step = None
            stale_autonomous_subtitle = bool(
                action_tool == "prepare_subtitle_review"
                and str(plan.get("executionMode") or "") == AUTONOMOUS_REVIEW
            )
            stale_subtitle_layout = action_tool == "layout_subtitles"
            if not (retry_failed_chain or retry_qc_chain or retry_cover_chain) and (
                not step or (
                    action_tool != "propose_timeline_edit"
                    and not stale_autonomous_subtitle
                    and not stale_subtitle_layout
                )
            ):
                raise ValueError("当前审核步骤不能自动重试")
            if retry_failed_chain or retry_qc_chain or retry_cover_chain:
                pass
            elif stale_subtitle_layout:
                # Older plans could mark subtitle preparation completed even
                # though its durable draft/session state was lost. Never let a
                # generic confirmation skip layout_subtitles: replay from the
                # subtitle preparation step so the requested layout is really
                # applied before rendering the review sample.
                replay_index = next((
                    index for index, item in enumerate(plan["steps"])
                    if str(item.get("tool") or "") == "prepare_subtitle_review"
                    and int(item.get("index") or index) < int(step.get("index") or len(plan["steps"]))
                ), None)
                if replay_index is None:
                    raise ValueError("字幕排版缺少字幕草稿步骤，请修改规划后重试")
                for replay_step in plan["steps"][replay_index:]:
                    replay_step["status"] = "pending"
                    for key in ("completedAt", "error", "result", "operationId"):
                        replay_step.pop(key, None)
                plan["status"] = "running"
                plan.pop("completedAt", None)
                plan = self.store.save("plans", plan)
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if not workspace:
                    raise ValueError("Agent Workspace 不存在")
                workspace["status"] = "running"
                workspace = self.store.save("workspaces", workspace)
                self.store.append_event(plan["workspaceId"], "action.retried", {
                    "planId": plan_id, "stepId": step["id"],
                    "replayFromTool": "prepare_subtitle_review",
                    "reason": "subtitle_layout_missing_draft",
                })
                self._notify_workspace_state(workspace, plan)
            elif stale_autonomous_subtitle:
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if not workspace:
                    raise ValueError("Agent Workspace 不存在")
                step["arguments"] = {
                    **(step.get("arguments") if isinstance(step.get("arguments"), dict) else {}),
                    "requireConfirmedDraft": False,
                    "autoReview": True,
                }
                # Hand the step back to the dispatch loop as pending; the
                # graph owns the running/attempts transition.
                step.update({"status": "pending"})
                step.pop("completedAt", None)
                step.pop("error", None)
                step.pop("result", None)
                plan["status"] = "running"
                plan = self.store.save("plans", plan)
                self.store.append_event(plan["workspaceId"], "action.retried", {
                    "planId": plan_id, "stepId": step["id"],
                    "reason": "migrate_manual_subtitle_gate_to_automatic_review",
                })
                self._notify_workspace_state(workspace, plan)
            else:
                result = step.get("result") if isinstance(step.get("result"), dict) else {}
                if str(result.get("sessionId") or ""):
                    raise ValueError("时间线草案已经生成，请直接打开审核")
                workspace = self.store.get("workspaces", str(plan["workspaceId"]))
                if not workspace:
                    raise ValueError("Agent Workspace 不存在")
                step.update({"status": "pending"})
                step.pop("completedAt", None)
                step.pop("error", None)
                step.pop("result", None)
                plan["status"] = "running"
                plan = self.store.save("plans", plan)
                self.store.append_event(plan["workspaceId"], "action.retried", {
                    "planId": plan_id, "stepId": step["id"],
                })
                self._notify_workspace_state(workspace, plan)
        self._kick_plan(plan_id)
        return self.store.get("plans", plan_id) or plan

    # -------------------------------------------------- execution internals

    def _mark_action_required(
        self, plan: dict[str, Any], step: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._execution_lock:
            current = self.store.get("plans", str(plan["id"]))
            if not current or current.get("status") != "running":
                return
            current_step = next((item for item in current["steps"] if item["id"] == step["id"]), None)
            if not current_step or current_step.get("status") not in {"pending", "running"}:
                return
            current_step["status"] = "action_required"
            current_step["result"] = result or {
                "message": action_required_message(current_step),
                "action": "structured_review",
            }
            current["status"] = "action_required"
            self.store.save("plans", current)
            self.store.append_event(current["workspaceId"], "action.required", {
                "planId": current["id"], "step": current_step,
                "reason": "该步骤需要结构化用户确认",
            })
            workspace = self.store.get("workspaces", str(current["workspaceId"]))
            if workspace:
                self._notify_workspace_state(workspace, current)

    def _execute_tool_step(self, plan: dict[str, Any], step: dict[str, Any]) -> str:
        """Run one tool step; returns "operation"/"action"/"" for graph routing."""
        plan_id = str(plan["id"])
        step_id = str(step["id"])
        with self._execution_lock:
            current = self.store.get("plans", plan_id)
            if not current or current.get("status") != "running":
                return ""
            current_step = next((item for item in current["steps"] if item["id"] == step_id), None)
            if not current_step or current_step.get("status") != "pending":
                return ""
            current_step["status"] = "running"
            current_step["attempts"] = int(current_step.get("attempts") or 0) + 1
            current = self.store.save("plans", current)
            self.store.append_event(current["workspaceId"], "step.started", {
                "planId": plan_id, "step": current_step,
            })
            workspace = self.store.get("workspaces", str(current["workspaceId"]))
            if workspace:
                self._notify_workspace_state(workspace, current)
        attempt = int(current_step.get("attempts") or 0)
        try:
            if self._dispatch_tool is None:
                raise RuntimeError("媒体工具调度器尚未初始化")
            result = self._dispatch_tool(workspace or {}, str(current_step.get("tool") or ""), current_step.get("arguments") or {})
        except Exception as error:  # noqa: BLE001 - step failure is product state
            self._complete_step(plan_id, step_id, status="failed", error=str(error), expected_attempt=attempt)
            return ""
        future = result.pop("future", None) if isinstance(result, dict) else None
        cancel_operation = result.pop("cancel", None) if isinstance(result, dict) else None
        if isinstance(future, Future):
            with self._execution_lock:
                refreshed = self.store.get("plans", plan_id)
                refreshed_step = next((item for item in (refreshed or {}).get("steps", []) if item["id"] == step_id), None)
                if (not refreshed or refreshed.get("status") != "running" or not refreshed_step
                        or refreshed_step.get("status") != "running"
                        or int(refreshed_step.get("attempts") or 0) != attempt):
                    if callable(cancel_operation):
                        cancel_operation()
                    future.cancel()
                    return ""
                operation_id = str(result.get("operationId") or f"operation_{uuid.uuid4().hex}")
                refreshed_step["status"] = "waiting_operation"
                refreshed_step["operationId"] = operation_id
                refreshed_step["result"] = {
                    **result,
                    "operationOwner": "agent",
                    "planId": plan_id,
                    "runId": str(refreshed.get("runId") or ""),
                    "stepId": step_id,
                    "planRevision": int(refreshed.get("revision") or 1),
                }
                self._operation_handles[(plan_id, step_id)] = {
                    "future": future,
                    "cancel": cancel_operation if callable(cancel_operation) else None,
                    "operationId": operation_id,
                }
                self.store.save("plans", refreshed)
                self._notify_workspace_state(workspace or {}, refreshed)
            future.add_done_callback(
                lambda completed: self._graph_driver.submit(
                    self._future_finished, plan_id, step_id, operation_id, completed,
                )
            )
            return "operation"
        if isinstance(result, dict) and result.get("actionRequired"):
            self._mark_action_required(plan, step, result)
            return "action"
        self._complete_step(plan_id, step_id, status="completed", result=result, expected_attempt=attempt)
        return ""

    def _future_finished(
        self, plan_id: str, step_id: str, operation_id: str, future: Future[Any],
    ) -> None:
        with self._execution_lock:
            handle = self._operation_handles.get((plan_id, step_id))
            if not handle or handle.get("future") is not future:
                return
            self._operation_handles.pop((plan_id, step_id), None)
            current = self.store.get("plans", plan_id)
            current_step = next((
                item for item in (current or {}).get("steps") or [] if item.get("id") == step_id
            ), None)
            if (
                not current or str(current.get("status") or "") != "running"
                or not current_step or str(current_step.get("status") or "") != "waiting_operation"
                or str(current_step.get("operationId") or "") != operation_id
            ):
                return
        if future.cancelled():
            outcome = {"apply": True, "status": "failed", "error": "后台操作已取消"}
        else:
            error = future.exception()
            if error is not None:
                outcome = {"apply": True, "status": "failed", "error": str(error)}
            else:
                # Preserve durable handoff information from dispatch and merge any
                # artifact produced by the worker (for example a QC report).
                completed_result = future.result()
                outcome = {
                    "apply": True, "status": "completed",
                    "result": completed_result if isinstance(completed_result, dict) else {},
                }
        self._deliver_operation_outcome(plan_id, outcome)

    def _deliver_operation_outcome(self, plan_id: str, outcome: dict[str, Any]) -> None:
        """Resume the graph thread parked at the operation gate with an outcome."""
        plan = self.store.get("plans", plan_id)
        if not plan:
            return
        thread_id = self._plan_thread_id(plan)
        config = self._graph_config(thread_id)
        with self._thread_lock(thread_id):
            try:
                snapshot = self.graph.get_state(config)
            except Exception:  # noqa: BLE001 - thread may be finished
                snapshot = None
            if snapshot is not None and snapshot.next:
                try:
                    self.graph.invoke(Command(resume=outcome), config)
                    return
                except Exception:  # noqa: BLE001
                    self._fail_plan_quietly(plan_id)
                    return
        # No parked thread (e.g. restart adopted the future after the graph
        # run already ended): apply the outcome directly and keep going.
        step = next(
            (item for item in plan.get("steps") or [] if item.get("status") == "waiting_operation"),
            None,
        )
        if step is not None:
            self._apply_operation_outcome(plan_id, str(step["id"]), outcome)
        if str((self.store.get("plans", plan_id) or {}).get("status") or "") == "running":
            self._kick_plan(plan_id)

    def _apply_operation_outcome(
        self, plan_id: str, step_id: str, outcome: dict[str, Any],
    ) -> None:
        if not isinstance(outcome, dict):
            return
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            step = next((item for item in (plan or {}).get("steps", []) if item.get("id") == step_id), None)
            if not plan or not step or str(step.get("status") or "") != "waiting_operation":
                return
            attempt = int(step.get("attempts") or 0)
            operation_id = str(step.get("operationId") or "")
        prior_result = step.get("result") if isinstance(step.get("result"), dict) else {}
        completed_payload = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
        merged = {**prior_result, **completed_payload, "operationCompleted": True}
        if str(outcome.get("status") or "") == "failed":
            self._complete_step(
                plan_id, step_id, status="failed",
                error=str(outcome.get("error") or "后台操作失败"),
                expected_attempt=attempt, expected_operation_id=operation_id,
            )
            return
        status, result, error = self._background_operation_resolution(plan, step, merged)
        self._complete_step(
            plan_id, step_id, status=status, result=result, error=error,
            expected_attempt=attempt, expected_operation_id=operation_id,
        )

    def _background_operation_resolution(
        self, plan: dict[str, Any], step: dict[str, Any], result: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        """Map a completed worker handoff to the Agent step's real terminal state.

        Media workers normally finish at a product checkpoint.  For Agent-owned
        search steps that checkpoint is not a user-visible pause; the Agent must
        either continue with candidates or stop the downstream timeline/render
        chain when no reliable candidates exist.
        """
        explicit_terminal = str(result.get("terminalStatus") or "").strip()
        if explicit_terminal == "no_result":
            return "completed", result, ""
        if str(step.get("tool") or "") != "search_content":
            return "completed", result, ""

        job_id = self._operation_result_job_id(plan, result)
        if not job_id:
            return "completed", result, ""
        context = self._planning_context(job_id)
        if not bool(context.get("available", True)):
            return "completed", result, ""
        job_status = str(context.get("jobStatus") or "").strip()
        if job_status == "failed":
            message = str(context.get("error") or "内容检索失败")
            return "failed", result, message

        evidence = context.get("evidence") if isinstance(context.get("evidence"), dict) else {}
        search_status = str(evidence.get("lastSearchStatus") or "").strip()
        candidate_count = int(
            evidence.get("contentCandidateCount")
            if "contentCandidateCount" in evidence else evidence.get("candidateCount") or 0
        )
        clarification = evidence.get("lastSearchClarification") if isinstance(evidence.get("lastSearchClarification"), dict) else {}
        message = str(
            clarification.get("message")
            or clarification.get("question")
            or evidence.get("lastSearchMessage")
            or "没有找到可用于自动剪辑的可靠候选。"
        )
        if search_status in {"queued", "indexing", "scanning", "running"}:
            return "completed", result, ""
        if candidate_count <= 0 and search_status in {
            "needs_clarification", "ready", "completed", "awaiting_content_confirmation", "no_result",
        }:
            no_match = {
                "kind": "no_match",
                "reasonCode": str(clarification.get("kind") or search_status or "no_match"),
                "message": message[:1000],
                "candidateCount": 0,
                "searchStatus": search_status,
                "coverageComplete": bool(evidence.get("coverageComplete")),
            }
            return "completed", {
                **result,
                "terminalStatus": "no_result",
                "artifact": no_match,
            }, ""
        return "completed", result, ""

    @staticmethod
    def _operation_result_job_id(plan: dict[str, Any], result: dict[str, Any]) -> str:
        artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
        job = result.get("job") if isinstance(result.get("job"), dict) else {}
        workspace = plan.get("workspace") if isinstance(plan.get("workspace"), dict) else {}
        return str(
            artifact.get("jobId")
            or result.get("jobId")
            or job.get("id")
            or workspace.get("jobId")
            or ""
        ).strip()

    def _cancel_plan_operations(self, plan_id: str) -> None:
        """Cancel only Futures and media operations started by this Agent plan."""
        with self._execution_lock:
            handles = [
                handle for (owner_plan_id, _step_id), handle in self._operation_handles.items()
                if owner_plan_id == plan_id
            ]
            self._operation_handles = {
                key: handle for key, handle in self._operation_handles.items()
                if key[0] != plan_id
            }
        for handle in handles:
            cancel = handle.get("cancel")
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            future = handle.get("future")
            if isinstance(future, Future):
                future.cancel()

    def _complete_step(
        self, plan_id: str, step_id: str, *, status: str,
        result: Any = None, error: str = "", expected_attempt: int | None = None,
        expected_operation_id: str | None = None,
    ) -> None:
        no_result = bool(
            status == "completed" and isinstance(result, dict)
            and str(result.get("terminalStatus") or "") == "no_result"
        )
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan or plan.get("status") != "running":
                return
            step = next((item for item in plan["steps"] if item["id"] == step_id), None)
            if not step or step.get("status") not in {"running", "waiting_operation"}:
                return
            if expected_attempt is not None and int(step.get("attempts") or 0) != expected_attempt:
                return
            if expected_operation_id is not None and step.get("operationId") != expected_operation_id:
                return
            step.update({"status": status, "completedAt": now_iso()})
            if result is not None:
                step["result"] = result
            if error:
                step["error"] = error[:2000]
            if no_result:
                artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
                terminal_message = str(
                    artifact.get("message") or result.get("message")
                    or "上游步骤未形成可继续执行的可靠结果。"
                )[:1000]
                terminal_reason = str(artifact.get("reasonCode") or "no_result")
                step["outcome"] = "no_result"
                step["outcomeMessage"] = terminal_message
                terminal_tool = str(step.get("tool") or "")
                skip_reason = (
                    "上游步骤未形成可应用的时间线，后续字幕、封面、画幅预览或质检未执行。"
                    if terminal_tool == "propose_timeline_edit" else
                    "上游内容检索或候选筛选未形成可靠结果，后续编排与渲染步骤未执行。"
                    if terminal_tool in {"search_content", "review_content_evidence", "select_multi_topic_evidence"} else
                    "上游步骤未形成可继续执行的结果，后续步骤未执行。"
                )
                for downstream in plan["steps"]:
                    if downstream.get("status") == "pending":
                        downstream.update({
                            "status": "skipped", "completedAt": now_iso(),
                            "skipReason": skip_reason,
                            "blockedByStepId": step_id,
                            "blockedByTool": str(step.get("tool") or ""),
                            "blockedByReason": terminal_reason,
                            "blockedByMessage": terminal_message,
                        })
            self.store.save("plans", plan)
            self.store.append_event(plan["workspaceId"], f"step.{status}", {
                "planId": plan_id, "step": step,
            })
            workspace = self.store.get("workspaces", str(plan["workspaceId"]))
            if workspace:
                self._notify_workspace_state(workspace, plan)
            if no_result:
                self._finish_plan(plan, "no_result")
                return

    def _attempt_replan(self, plan_id: str, failed_step_id: str) -> str:
        """Re-plan once after a failed step; returns the resulting graph phase."""
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            if not plan or int(plan.get("replanCount") or 0) >= 2:
                return "finished"
            workspace = self.store.get("workspaces", str(plan["workspaceId"]))
            skill = self.store.get("skills", str(plan["skillId"]))
            failed_step = next((item for item in plan["steps"] if item["id"] == failed_step_id), None)
            if not workspace or not skill or not failed_step:
                return "finished"
            block_reason = automatic_replan_block_reason(failed_step)
            if block_reason:
                self.store.append_event(plan["workspaceId"], "plan.replan_skipped", {
                    "planId": plan_id, "failedStepId": failed_step_id,
                    "reason": block_reason, "error": str(failed_step.get("error") or "")[:1000],
                })
                return "finished"
            previous_steps = copy.deepcopy(plan["steps"])
            selected_skills = [skill]
            for item in plan.get("skills") or []:
                skill_id = str(item.get("id") or "") if isinstance(item, dict) else ""
                if not skill_id or skill_id == str(skill.get("id") or ""):
                    continue
                selected = self.store.get("skills", skill_id)
                if selected:
                    selected_skills.append(selected)
            context = (
                copy.deepcopy(plan.get("planningContext"))
                if isinstance(plan.get("planningContext"), dict)
                else self._planning_context(str(workspace.get("jobId") or ""))
            )
            brief = editing_brief(str(plan.get("goal") or ""), context)
            profile = profile_for_skill(skill)
            allowed_tools = sorted({
                tool
                for selected in selected_skills
                for tool in profile_for_skill(selected).get("tools", set())
            })
            payload = {
                "workspaceId": workspace["id"],
                "goal": plan["goal"],
                "skill": {
                    "id": skill["id"], "version": skill["version"],
                    "contentHash": skill["contentHash"], "markdown": skill["skillMarkdown"],
                },
                "skills": [{
                    "id": selected["id"], "version": selected["version"],
                    "contentHash": selected["contentHash"], "markdown": selected["skillMarkdown"],
                    "role": "primary" if index == 0 else "addon",
                    "workflowProfile": profile_for_skill(selected)["kind"],
                } for index, selected in enumerate(selected_skills)],
                "workspace": {"jobId": workspace["jobId"], "revision": workspace["revision"]},
                "planningContext": context, "brief": brief,
                "executionMode": str(plan.get("executionMode") or STEPWISE_REVIEW),
                "profile": {
                    "kind": profile["kind"], "managed": bool(profile["managed"]),
                    "allowedTools": allowed_tools,
                },
                "toolCatalog": tool_catalog(),
                "model": self.model_config_resolver(),
                "replan": {
                    "completedSteps": [
                        {"id": item["id"], "tool": item["tool"], "result": item.get("result")}
                        for item in previous_steps if item["status"] == "completed"
                    ],
                    "failedStep": {
                        "id": failed_step["id"], "tool": failed_step["tool"],
                        "arguments": failed_step["arguments"], "error": failed_step.get("error"),
                    },
                    "approvedEnvelope": plan.get("approval") or {},
                },
            }
        try:
            result = self.planner_backend.plan(
                payload, model_config=self.model_config_resolver(), emit=None,
            )
            raw_plan = result.get("plan") if isinstance(result.get("plan"), dict) else result
            compiled = compile_profile_plan(
                raw_plan, skill=skill, skills=selected_skills, goal=plan["goal"], context=context,
                execution_mode=str(plan.get("executionMode") or STEPWISE_REVIEW),
            )
            replacement = normalize_plan(
                compiled, workspace=workspace, skill=skill, skills=selected_skills, goal=plan["goal"],
            )
        except Exception as error:  # noqa: BLE001 - replan failure keeps the plan
            self.store.append_event(plan["workspaceId"], "plan.replan_failed", {
                "planId": plan_id, "error": str(error)[:1000],
            })
            return "finished"
        previous_signatures = step_execution_signatures(previous_steps)
        replacement_signatures = step_execution_signatures(replacement["steps"])
        completed_by_signature = {
            previous_signatures.get(str(item.get("id") or "")): item
            for item in previous_steps
            if item.get("status") == "completed"
            and previous_signatures.get(str(item.get("id") or ""))
        }
        force_replay_tools = replan_force_replay_tools(failed_step)
        for step in replacement["steps"]:
            if str(step.get("tool") or "") in force_replay_tools:
                continue
            previous = completed_by_signature.get(
                replacement_signatures.get(str(step.get("id") or ""))
            )
            if previous:
                step.update({
                    "status": "completed",
                    "attempts": int(previous.get("attempts") or 1),
                    "completedAt": previous.get("completedAt") or now_iso(),
                    "result": copy.deepcopy(previous.get("result")),
                })
        approval = plan.get("approval") if isinstance(plan.get("approval"), dict) else {}
        new_tools = {step["tool"] for step in replacement["steps"]}
        new_effects = {step["sideEffect"] for step in replacement["steps"]}
        old_seconds = sum(int(step.get("estimatedSeconds") or 0) for step in previous_steps)
        new_seconds = sum(int(step.get("estimatedSeconds") or 0) for step in replacement["steps"])
        minor = (
            new_tools.issubset(set(approval.get("allowedTools") or []))
            and new_effects.issubset(set(approval.get("allowedSideEffects") or []))
            and all(
                dict(approval.get("skillHashes") or {}).get(str(item.get("id") or ""))
                == str(item.get("contentHash") or "")
                for item in replacement.get("skills") or []
            )
            and new_seconds <= max(old_seconds + 60, int(old_seconds * 1.25))
        )
        with self._execution_lock:
            current = self.store.get("plans", plan_id)
            if not current or current.get("status") != "running":
                return "finished"
            history = list(current.get("revisionHistory") or [])
            history.append({
                "revision": current.get("revision"), "planHash": current.get("planHash"),
                "steps": previous_steps, "replannedAt": now_iso(),
            })
            current.update({
                "summary": replacement["summary"], "steps": replacement["steps"],
                "planHash": replacement["planHash"], "agent": replacement.get("agent") or {},
                "skills": replacement.get("skills") or current.get("skills") or [],
                "brief": replacement.get("brief") or current.get("brief") or {},
                "planningContext": replacement.get("planningContext") or current.get("planningContext") or {},
                "decisionRecord": replacement.get("decisionRecord") or [],
                "profile": replacement.get("profile") or current.get("profile") or "",
                "revision": int(current.get("revision") or 1) + 1,
                "replanCount": int(current.get("replanCount") or 0) + 1,
                "revisionHistory": history[-10:],
            })
            if minor:
                current["status"] = "running"
                current["approval"]["lastAutomaticReplanAt"] = now_iso()
            else:
                current["status"] = "awaiting_confirmation"
                current["approval"] = None
            self.store.save("plans", current)
            self.store.append_event(current["workspaceId"], "plan.replanned", {
                "planId": plan_id, "revision": current["revision"], "material": not minor,
                "failedStepId": failed_step_id,
            })
            if not minor:
                self.store.append_event(current["workspaceId"], "plan.confirmation_required", {
                    "planId": plan_id, "planHash": current["planHash"], "reason": "重大重新规划",
                })
            if workspace:
                self._notify_workspace_state(workspace, current)
        return "executing" if minor else "awaiting_approval"

    def _finish_plan(self, plan: dict[str, Any], status: str) -> None:
        if str(plan.get("status") or "") in TERMINAL_PLAN_STATUSES and plan.get("completedAt"):
            return
        plan["status"] = status
        plan["completedAt"] = now_iso()
        self.store.save("plans", plan)
        # Bound per-plan lock growth; late callbacks re-acquire through a
        # fresh lock and are rejected by the terminal status guards anyway.
        with self._execution_lock:
            self._thread_locks.pop(self._plan_thread_id(plan), None)
        run_id = str(plan.get("runId") or "")
        if run_id:
            run = self.store.get("runs", run_id)
            if run:
                run.update({"status": status, "completedAt": now_iso()})
                self.store.save("runs", run)
        workspace = self.store.get("workspaces", plan["workspaceId"])
        if workspace:
            workspace["status"] = status
            result_id = f"plan-result:{plan['id']}"
            messages = [m for m in workspace.get("messages") or [] if m.get("id") != result_id]
            artifacts = [(step.get("result") or {}).get("artifact") or {} for step in plan.get("steps") or []]
            has_preview = any("preview" in str(a.get("kind") or "") for a in artifacts)
            quality_failed = any(a.get("kind") == "delivery_qc_report" and a.get("passed") is False for a in artifacts)
            result_text = (
                "审核样片已生成；检查存在提醒，请查看本次方案的检查详情。" if quality_failed else
                "审核样片已生成。" if has_preview else "本次方案已执行完成，可查看对应结果。"
            ) if status == "preview_ready" else {
                "failed": "本次方案执行失败。请查看失败步骤。",
                "no_result": "本次未找到可用片段，可以调整条件重新检索。",
                "cancelled": "本次方案已停止。",
            }.get(status, "本次方案已结束。")
            messages.append({"id": result_id, "role": "assistant", "kind": "plan_result", "planId": plan["id"], "text": result_text, "createdAt": now_iso()})
            workspace["messages"] = messages
            workspace = self.store.save("workspaces", workspace)
        event_type = {
            "preview_ready": "preview.ready",
            "no_result": "plan.no_result",
            "cancelled": "plan.cancelled",
        }.get(status, "plan.failed")
        self.store.append_event(plan["workspaceId"], event_type, {"planId": plan["id"]})
        if workspace:
            self._notify_workspace_state(workspace, plan)
        if status in {"failed", "no_result", "cancelled"}:
            self._cancel_plan_operations(str(plan.get("id") or ""))
        finished = getattr(self, "_conversation_finished", None)
        if finished:
            finished(plan["workspaceId"])

    # ------------------------------------------------------------- recovery

    def adopt_recovered_operation(
        self, plan_id: str, step_id: str, future: Future[Any], operation_id: str,
    ) -> bool:
        """Reconnect a durable media Future to its Agent step after restart."""
        with self._execution_lock:
            plan = self.store.get("plans", plan_id)
            step = next((item for item in (plan or {}).get("steps", []) if item.get("id") == step_id), None)
            if not plan or not step or str(step.get("status") or "") != "waiting_operation":
                return False
            self._operation_handles[(plan_id, step_id)] = {
                "future": future, "cancel": None, "operationId": operation_id,
            }
        future.add_done_callback(
            lambda completed: self._graph_driver.submit(
                self._future_finished, plan_id, step_id, operation_id, completed,
            )
        )
        return True

    def recover_completed_operations(self) -> int:
        """Settle Agent steps whose durable media operation finished while detached."""
        recovered: list[tuple[str, str, str, dict[str, Any], str]] = []
        with self._execution_lock:
            plans = self.store.list(
                "plans",
                predicate=lambda item: item.get("status") == "running",
            )
            for plan in plans:
                workspace = self.store.get("workspaces", str(plan.get("workspaceId") or ""))
                for step in plan.get("steps") or []:
                    if str(step.get("status") or "") != "waiting_operation":
                        continue
                    result = copy.deepcopy(step.get("result") if isinstance(step.get("result"), dict) else {})
                    if workspace and not self._operation_result_job_id(plan, result):
                        result["jobId"] = str(workspace.get("jobId") or "")
                    job_id = self._operation_result_job_id(plan, result)
                    if not job_id:
                        continue
                    context = self._planning_context(job_id)
                    evidence = context.get("evidence") if isinstance(context.get("evidence"), dict) else {}
                    job_status = str(context.get("jobStatus") or "").strip()
                    search_status = str(evidence.get("lastSearchStatus") or "").strip()
                    if job_status in {"queued", "running", "cancelling", "briefing"} or search_status in {
                        "queued", "indexing", "scanning", "running",
                    }:
                        continue
                    status, resolved, error = self._background_operation_resolution(
                        plan, step, {**result, "operationCompleted": True},
                    )
                    recovered.append((str(plan["id"]), str(step["id"]), status, resolved, error))
        count = 0
        for plan_id, step_id, status, result, error in recovered:
            self._complete_step(plan_id, step_id, status=status, result=result, error=error)
            count += 1
        for plan_id, _step_id, _status, _result, _error in recovered:
            plan = self.store.get("plans", plan_id)
            if plan and str(plan.get("status") or "") == "running":
                self._kick_plan(plan_id)
        return count

    # ----------------------------------------------- action resolution gate

    @staticmethod
    def _validate_action_resolution(
        workspace: dict[str, Any], step: dict[str, Any], value: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("请提交本次审核面板产生的结构化确认结果")
        context = value.get("context") if isinstance(value.get("context"), dict) else {}
        if (
            str(context.get("jobId") or "") != str(workspace.get("jobId") or "")
            or str(context.get("stepId") or "") != str(step.get("id") or "")
        ):
            raise ValueError("确认结果与当前 Agent 步骤不匹配，请在对应审核面板完成选择")
        selection = value.get("selection") if isinstance(value.get("selection"), dict) else {}
        required_key = {
            "select_people": "personIds",
            "select_speakers": "speakerRefs",
            "review_cover_variants": "variantIds",
        }.get(str(step.get("tool") or ""))
        if required_key:
            selected = selection.get(required_key)
            if not isinstance(selected, list) or not [item for item in selected if str(item).strip()]:
                label = "人物" if required_key == "personIds" else "说话人" if required_key == "speakerRefs" else "封面"
                raise ValueError(f"请先在审核面板选择至少一个{label}，再继续计划")
        tool_name = str(step.get("tool") or "")
        if tool_name == "layout_subtitles":
            raise ValueError("字幕排版不能通过空确认跳过；请重新生成字幕草稿并执行排版")
        if tool_name == "propose_timeline_edit" and (step.get("result") or {}).get("action") != "content_evidence_review":
            if not str(selection.get("editSessionId") or "") or not str(selection.get("proposalId") or ""):
                raise ValueError("请先在精剪时间线生成待审核的时间线草案，再继续计划")
        if tool_name == "confirm_timeline_edit":
            revision = selection.get("revision")
            if not str(selection.get("editSessionId") or "") or not isinstance(revision, int) or isinstance(revision, bool):
                raise ValueError("请先在精剪时间线应用并保存审核后的草案，再继续计划")
        return copy.deepcopy(value)

    # Compatibility delegates used by the media kernel (main.py).
    _social_delivery = staticmethod(_social_delivery)
    _cover_intro_requested = staticmethod(_cover_intro_requested)
    _cover_intro_duration = staticmethod(_cover_intro_duration)
    _retrieval_query = staticmethod(_retrieval_query)
    _editing_brief = staticmethod(editing_brief)


def _goal_mentions_highlights(goal: str) -> bool:
    import re

    return bool(re.search(r"高光|最精彩|精彩部分", str(goal or "")))


def _goal_mentions_revision(goal: str) -> bool:
    import re

    return bool(re.search(
        r"当前成片|已有成片|时间线|二次精剪|返修|重排|缩短|恢复|加字幕|添加文本|添加文字|加文字|叠加文字",
        str(goal or ""),
    ))


def _preferred_skill_kind(
    brief: dict[str, Any], goal: str, editing: dict[str, Any],
    source_edit_requested: bool, revision_requested: bool,
) -> str:
    import re

    if bool(brief.get("diagnosticsRequested")):
        return "edit-diagnostics"
    if (
        re.search(r"新任务|同一源|同源|复用|沿用|旧任务|串(?:任务|用)|上个任务|上一次", str(goal or ""))
        and re.search(r"检查|核查|校验|验证|排查|是否|有没有|误用|串用", str(goal or ""))
    ):
        return "source-provenance"
    if bool(brief.get("coverIntroRequested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "cover-intro"
    if bool(brief.get("coverRequested")) and str(brief.get("delivery") or "") == "artifact" and not source_edit_requested:
        return "cover"
    if revision_requested and (
        brief.get("durationSource") == "relative"
        or brief.get("anchorStart")
        or brief.get("sourceRange")
        or brief.get("removedSourceRanges")
    ):
        return "revision"
    if bool(brief.get("subtitleAssetRequested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "caption-layout"
    if bool(brief.get("draftExportRequested")) and not source_edit_requested:
        return "local-draft"
    if bool(brief.get("motionGraphicsRequested")) and not source_edit_requested:
        return "local-motion"
    if bool(brief.get("audioPolishRequested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "audio-polish"
    if revision_requested and not source_edit_requested and (
        bool(brief.get("subtitleRequested"))
        or bool(brief.get("graphicsRequested"))
        or bool(brief.get("brollRequested"))
    ):
        if bool(brief.get("subtitleRequested")):
            return "caption-layout"
        if bool(brief.get("graphicsRequested")):
            return "graphics-package"
        return "broll-overlay"
    if bool(brief.get("deliveryExportRequested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "platform-delivery"
    if bool(brief.get("socialDelivery", {}).get("requested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "dynamic-reframe"
    if bool(brief.get("deliveryQcRequested")) and bool(editing.get("hasOutputs")) and not source_edit_requested:
        return "delivery-qc"
    if revision_requested:
        if bool(brief.get("subtitleRequested")):
            return "caption-layout"
        if bool(brief.get("graphicsRequested")):
            return "graphics-package"
        if bool(brief.get("brollRequested")):
            return "broll-overlay"
        return "revision"
    if bool(brief.get("multiTopicRequested")) and bool(brief.get("retrievalQuery")):
        return "multi-topic"
    if bool(brief.get("speakerTargeted")):
        return "speaker"
    if bool(brief.get("personTargeted")):
        return "person"
    if bool(brief.get("interview")):
        return "interview"
    if bool(brief.get("shortForm")):
        return "shortform"
    if bool(brief.get("retrievalQuery")):
        return "content"
    if str(brief.get("delivery") or "") == "timeline":
        # A format/aspect request that also asks for a new cut is a
        # highlight edit unless the user actually identifies a person,
        # speaker, interview structure, or semantic retrieval target.
        # Leaving this to model routing allowed words such as “主播” to
        # select the identity editor and manufacture an unnecessary gate.
        return "highlight"
    return ""
