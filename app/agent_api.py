from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import AgentPlatform
from .agent.planner import AgentPlannerError
from .agent_store import content_hash, parse_skill_markdown
from .assistant_interaction import AssistantInteraction


class WorkspaceCreateRequest(BaseModel):
    jobId: str = Field(min_length=1, max_length=96)
    title: str = Field(default="", max_length=160)


class AgentMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    skillId: str | None = Field(default=None, max_length=64)
    executionMode: Literal["autonomous_review", "stepwise_review"] = "autonomous_review"
    clientMessageId: str | None = Field(default=None, max_length=96)
    uiContext: dict[str, Any] | None = None
    replyToActionId: str | None = Field(default=None, max_length=96)
    confirmationValue: dict[str, Any] | None = None


class PendingChangeRequest(BaseModel):
    choice: Literal["stop", "after", "discard", "prepare"]


class PlanApprovalRequest(BaseModel):
    planHash: str = Field(min_length=64, max_length=64)


class SkillGenerateRequest(BaseModel):
    request: str = Field(min_length=10, max_length=4000)


class SkillEnableRequest(BaseModel):
    contentHash: str = Field(min_length=64, max_length=64)


class ActionResolutionRequest(BaseModel):
    approved: bool
    value: dict[str, Any] | None = None


class CoverRevisionRequest(BaseModel):
    coverSourceTime: float | None = Field(default=None, ge=0)
    coverSubject: str = Field(default="", max_length=120)
    coverTitle: str = Field(default="", max_length=200)
    coverAspect: Literal["16:9", "9:16", "1:1"] = "16:9"


class ActionRetryRequest(BaseModel):
    cover: CoverRevisionRequest | None = None


def _safe_extract(archive: Path, destination: Path, *, max_uncompressed: int = 50 * 1024 * 1024) -> None:
    total = 0
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            total += max(0, int(member.file_size))
            if total > max_uncompressed:
                raise ValueError("压缩包解压后超过 50MB 限制")
            member_path = (destination / member.filename).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise ValueError("压缩包包含越界路径")
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _single_package_root(path: Path, required_name: str) -> Path:
    if (path / required_name).is_file():
        return path
    candidates = [item for item in path.iterdir() if item.is_dir() and (item / required_name).is_file()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"压缩包根目录必须包含 {required_name}")


def build_agent_router(
    *, platform: AgentPlatform,
    job_getter: Callable[[str], dict[str, Any] | None],
) -> APIRouter:
    router = APIRouter(prefix="/api/agent", tags=["agent"])
    conversation = AssistantInteraction(platform, job_getter)

    def workspace_detail(workspace: dict[str, Any]) -> dict[str, Any]:
        workspace_id = str(workspace["id"])
        plans = platform.store.list(
            "plans", predicate=lambda item: item.get("workspaceId") == workspace_id,
        )
        runs = platform.store.list(
            "runs", predicate=lambda item: item.get("workspaceId") == workspace_id,
        )
        return {"workspace": workspace, "plans": plans, "runs": runs}

    @router.get("/health")
    def agent_health() -> dict[str, Any]:
        model = platform.model_config_resolver()
        configured = all(str(model.get(key) or "").strip() for key in ("apiKey", "model", "baseUrl"))
        return {
            "status": "ok" if configured else "degraded",
            "runtime": {"name": "langgraph", "inProcess": True},
            "skills": len(platform.store.list("skills")),
            "model": {
                "configured": configured,
                "source": str(model.get("configSource") or "agent_settings"),
                "provider": str(model.get("provider") or ""),
                "name": str(model.get("model") or ""),
            },
        }

    @router.get("/workspaces")
    def list_workspaces() -> dict[str, Any]:
        return {"workspaces": platform.store.list("workspaces")}

    @router.post("/workspaces", status_code=201)
    def create_workspace(request: WorkspaceCreateRequest) -> dict[str, Any]:
        job = job_getter(request.jobId)
        if not job:
            raise HTTPException(404, "素材任务不存在")
        workspace = platform.create_workspace(
            job_id=request.jobId, title=request.title or str(job.get("filename") or "智能剪辑工作区"),
        )
        return {"workspace": workspace}

    @router.get("/workspaces/by-job/{job_id}")
    def get_workspace_by_job(job_id: str) -> dict[str, Any]:
        workspace = platform.workspace_for_job(job_id)
        if not workspace:
            raise HTTPException(404, "该任务尚未创建 Agent Workspace")
        return workspace_detail(workspace)

    @router.get("/workspaces/{workspace_id}")
    def get_workspace(workspace_id: str) -> dict[str, Any]:
        workspace = platform.store.get("workspaces", workspace_id)
        if not workspace:
            raise HTTPException(404, "Agent Workspace 不存在")
        return workspace_detail(workspace)

    @router.post("/workspaces/{workspace_id}/messages", status_code=201)
    def send_message(workspace_id: str, request: AgentMessageRequest) -> dict[str, Any]:
        try:
            if request.clientMessageId or request.uiContext is not None:
                return conversation.handle(workspace_id, request.model_dump())
            plan = platform.create_plan(
                workspace_id=workspace_id, goal=request.text.strip(), skill_id=request.skillId,
                execution_mode=request.executionMode,
            )
        except KeyError as error:
            raise HTTPException(404, "Agent Workspace 不存在") from error
        except AgentPlannerError as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {"action": "plan_confirmation", "plan": plan}

    @router.post("/workspaces/{workspace_id}/pending-changes")
    def pending_changes(workspace_id: str, request: PendingChangeRequest) -> dict[str, Any]:
        try:
            return conversation.change(workspace_id, request.choice)
        except KeyError as error:
            raise HTTPException(404, "工作区不存在") from error
        except (ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/workspaces/{workspace_id}/messages/stream")
    def stream_message_plan(
        workspace_id: str, request: AgentMessageRequest,
    ) -> StreamingResponse:
        if request.clientMessageId or request.uiContext is not None:
            if not platform.store.get("workspaces", workspace_id):
                raise HTTPException(404, "工作区不存在")

            async def conversation_events():
                task = asyncio.create_task(asyncio.to_thread(conversation.handle, workspace_id, request.model_dump()))
                task.add_done_callback(lambda completed: None if completed.cancelled() else completed.exception())
                planning_announced = False
                # Shield the durable request from an HTTP disconnect.
                try:
                    yield 'event: planning.progress\ndata: {"phase":"context_loading","title":"正在核对目标与素材范围"}\n\n'
                    while not task.done():
                        await asyncio.wait({task}, timeout=2)
                        if not task.done():
                            state = platform.store.get("workspaces", workspace_id) or {}
                            if state.get("planningRequestId") and not planning_announced:
                                planning_announced = True
                                yield 'event: planning.progress\ndata: {"phase":"decomposing_goal","title":"正在整理待确认方案","detail":"确认前不会执行媒体分析或渲染"}\n\n'
                            yield ": heartbeat\n\n"
                    result = task.result()
                    event = "plan" if result.get("action") == "plan_confirmation" else "assistant.result"
                    yield f"event: {event}\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
                except (ValueError, RuntimeError, KeyError) as error:
                    yield f"event: error\ndata: {json.dumps({'message': str(error)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(conversation_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        # Validate synchronously so preparation failures map to real status
        # codes instead of an aborted event stream.
        try:
            prepared = platform.prepare_plan_request(
                workspace_id=workspace_id, goal=request.text.strip(), skill_id=request.skillId,
                execution_mode=request.executionMode,
            )
        except KeyError as error:
            raise HTTPException(404, "Agent Workspace 不存在") from error
        except (AgentPlannerError, ValueError) as error:
            raise HTTPException(503 if isinstance(error, AgentPlannerError) else 400, str(error)) from error

        async def plan_events():
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            def produce() -> None:
                try:
                    for event in platform.stream_plan(prepared=prepared):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except Exception as error:  # noqa: BLE001 - surfaced as an SSE error event
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"event": "error", "data": {"message": str(error)}},
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=produce, daemon=True, name=f"agent-plan-{workspace_id}").start()
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

        return StreamingResponse(
            plan_events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/workspaces/{workspace_id}/events")
    def stream_events(
        workspace_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        if not platform.store.get("workspaces", workspace_id):
            raise HTTPException(404, "Agent Workspace 不存在")

        async def events():
            try:
                cursor = max(after, int(request.headers.get("Last-Event-ID") or 0))
            except ValueError:
                cursor = after
            idle_ticks = 0
            while True:
                if await request.is_disconnected():
                    return
                rows = await asyncio.to_thread(
                    platform.store.events_after, workspace_id, cursor,
                )
                if rows:
                    idle_ticks = 0
                    for event in rows:
                        cursor = int(event["sequence"])
                        data = json.dumps(event, ensure_ascii=False)
                        yield f"id: {cursor}\nevent: {event['type']}\ndata: {data}\n\n"
                else:
                    idle_ticks += 1
                    if idle_ticks % 15 == 0:
                        yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/plans/{plan_id}")
    def get_plan(plan_id: str) -> dict[str, Any]:
        plan = platform.store.get("plans", plan_id)
        if not plan:
            raise HTTPException(404, "执行计划不存在")
        return {"plan": plan}

    @router.post("/plans/{plan_id}/confirm")
    def confirm_plan(plan_id: str, request: PlanApprovalRequest) -> dict[str, Any]:
        try:
            plan = platform.approve_plan(plan_id, expected_hash=request.planHash)
        except KeyError as error:
            raise HTTPException(404, "执行计划不存在") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"plan": plan}

    @router.post("/plans/{plan_id}/cancel")
    def cancel_plan(plan_id: str) -> dict[str, Any]:
        try:
            return {"plan": platform.cancel_plan(plan_id)}
        except KeyError as error:
            raise HTTPException(404, "执行计划不存在") from error

    @router.post("/plans/{plan_id}/actions/resolve")
    def resolve_plan_action(plan_id: str, request: ActionResolutionRequest) -> dict[str, Any]:
        try:
            plan = platform.resolve_action(
                plan_id, approved=request.approved, value=request.value,
            )
        except KeyError as error:
            raise HTTPException(404, "执行计划不存在") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"plan": plan}

    @router.post("/plans/{plan_id}/actions/retry")
    def retry_plan_action(plan_id: str, request: ActionRetryRequest | None = None) -> dict[str, Any]:
        try:
            cover = request.cover.model_dump() if request and request.cover else None
            return {"plan": platform.retry_action(plan_id, cover_revision=cover)} if cover is not None else {"plan": platform.retry_action(plan_id)}
        except KeyError as error:
            raise HTTPException(404, "执行计划不存在") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/skills")
    def list_skills() -> dict[str, Any]:
        return {"skills": platform.store.list("skills")}

    @router.post("/skills/install", status_code=201)
    async def install_skill(file: UploadFile = File(...)) -> dict[str, Any]:
        if not str(file.filename or "").lower().endswith(".zip"):
            raise HTTPException(400, "请上传 ZIP 格式的标准 Skill 包")
        payload = await file.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(413, "Skill 包不能超过 10MB")
        with tempfile.TemporaryDirectory(prefix="chatclip-skill-") as temporary:
            temporary_path = Path(temporary)
            archive = temporary_path / "skill.zip"
            archive.write_bytes(payload)
            extracted = temporary_path / "extracted"
            extracted.mkdir()
            try:
                _safe_extract(archive, extracted)
                package_root = _single_package_root(extracted, "SKILL.md")
                markdown = (package_root / "SKILL.md").read_text(encoding="utf-8")
                fields = parse_skill_markdown(markdown)
                digest = content_hash(payload)
                target = platform.skills_root / fields["name"] / digest
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(package_root, target)
                skill = platform.install_skill(
                    markdown=markdown, source="upload", status="validated", path=str(target),
                )
            except (OSError, ValueError, zipfile.BadZipFile, UnicodeError) as error:
                raise HTTPException(400, str(error)) from error
        return {"skill": skill}

    @router.post("/skills/generate", status_code=201)
    def generate_skill(request: SkillGenerateRequest) -> dict[str, Any]:
        try:
            return {"skill": platform.generate_skill(request=request.request.strip())}
        except (AgentPlannerError, ValueError) as error:
            raise HTTPException(503 if isinstance(error, AgentPlannerError) else 400, str(error)) from error

    @router.post("/skills/{skill_id}/enable")
    def enable_skill(skill_id: str, request: SkillEnableRequest) -> dict[str, Any]:
        try:
            return {"skill": platform.enable_skill(skill_id, request.contentHash)}
        except KeyError as error:
            raise HTTPException(404, "Skill 不存在") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/skills/{skill_id}/disable")
    def disable_skill(skill_id: str) -> dict[str, Any]:
        skill = platform.store.get("skills", skill_id)
        if not skill:
            raise HTTPException(404, "Skill 不存在")
        skill["status"] = "disabled"
        return {"skill": platform.store.save("skills", skill)}

    return router
