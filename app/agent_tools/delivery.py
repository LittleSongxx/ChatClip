"""Delivery, preview and media-composition Agent tool handlers."""
from __future__ import annotations

from typing import Any
from pathlib import Path

from .context import ToolContext


def render_review_preview(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    if ctx.autonomous:
        batch = ctx.snapshot.get("agentTimelineBatch") if isinstance(ctx.snapshot.get("agentTimelineBatch"), dict) else {}
        variants = [item for item in batch.get("variants") or [] if isinstance(item, dict)]
        ready_variants: list[dict[str, Any]] = []
        for item in variants:
            session = next((
                value for value in ctx.snapshot.get("editSessions") or []
                if str(value.get("id") or "") == str(item.get("sessionId") or "")
            ), None)
            if session and int(session.get("revision") or 0) > 0 and not isinstance(session.get("pendingProposal"), dict):
                ready_variants.append({
                    "sessionId": str(session["id"]),
                    "revision": int(session.get("revision") or 0),
                    "title": str(item.get("title") or session.get("title") or "审核方案"),
                })
        if not ready_variants:
            raise RuntimeError("自动时间线尚未应用，无法生成审核样片")
        future = _main.submit_render_task(ctx.job_id, _main.run_agent_review_batch, ready_variants)
        return {
            "operationId": f'{ctx.job_id}:agent_review_batch:{_main.uuid.uuid4().hex[:8]}',
            "operation": "agent_review_batch", "accepted": True,
            "variantCount": len(ready_variants), "future": future,
            "cancel": lambda: _main.cancel_job(ctx.job_id),
        }
    session_id = str(ctx.snapshot.get("activeEditSessionId") or "")
    session = next((
        item for item in ctx.snapshot.get("editSessions") or []
        if str(item.get("id") or "") == session_id
    ), None)
    if session and int(session.get("revision") or 0) > 0 and not isinstance(session.get("pendingProposal"), dict):
        revision = int(session.get("revision") or 0)
        future = _main.submit_render_task(ctx.job_id, _main.run_agent_review_batch, [{
            "sessionId": session_id, "revision": revision,
            "title": str(session.get("title") or "审核方案"),
        }])
        return {
            "operationId": f'{ctx.job_id}:agent_review_render:{session_id}:{revision}',
            "operation": "agent_review_render", "accepted": True, "future": future,
            "cancel": lambda: _main.cancel_job(ctx.job_id),
        }
    outputs = _main.public_job(ctx.snapshot).get("outputs") or []
    if outputs:
        return {"artifact": {"kind": "review_preview", "jobId": ctx.job_id, "outputs": outputs,
                             "message": "已复用当前审核输出。"}}
    return {"actionRequired": True, "action": "preview_generation",
            "message": "请先应用并保存时间线草案，再继续生成审核输出。"}


def analyze_reframe_safe_areas(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    aspect = str(arguments.get("aspect") or "9:16")
    fit = str(arguments.get("fit") or "blur")
    output, path = ctx.current_output_reference()
    if path is None or not path.is_file():
        return {"actionRequired": True, "action": "preview_generation", "message": "当前没有可用于画幅分析的成片或审核样片。"}
    info = _main.probe_video(path, _main.settings.ffprobe)
    source_ratio = round(info.width / max(1, info.height), 4)
    target_size = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080), "16:9": (1920, 1080)}.get(aspect)
    target_ratio = round(target_size[0] / target_size[1], 4) if target_size else source_ratio
    recommended_fit = "blur" if abs(source_ratio - target_ratio) > .03 and fit != "crop" else fit
    return {
        "artifact": {
            "kind": "reframe_safe_area_report", "jobId": ctx.job_id,
            "sourceFilename": str((output or {}).get("filename") or path.name),
            "source": {"width": info.width, "height": info.height, "ratio": source_ratio},
            "target": {"aspect": aspect, "ratio": target_ratio},
            "fit": recommended_fit,
            "protections": ["complete_frame", "subtitles", "screen_text", "faces", "product_subject"],
            "message": "建议完整保留原画面并使用同画面虚化背景补齐。" if recommended_fit == "blur" else "按用户指定方式进行画幅适配。",
        },
    }


def propose_broll_overlay(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    query = str(arguments.get("query") or "").strip()
    max_overlays = max(1, min(12, int(arguments.get("maxOverlays") or 6)))
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        session_id = str((current or {}).get("activeEditSessionId") or "")
        session = _main.find_edit_session(current, session_id) if current and session_id else None
        search = current.get("contentSearch") if current and isinstance(current.get("contentSearch"), dict) else {}
        review = search.get("reviewDraft") if isinstance(search.get("reviewDraft"), dict) else {}
        selected_ids = [str(value) for value in review.get("orderedMatchIds") or review.get("selectedMatchIds") or [] if str(value)]
        candidates = {
            str(item.get("id") or ""): item for item in search.get("candidates") or []
            if isinstance(item, dict) and float(item.get("end") or 0) > float(item.get("start") or 0)
        }
        if not current or not session:
            return {"actionRequired": True, "action": "timeline_confirmation", "message": "请先建立当前时间线，再添加辅助画面。"}
        primary_clips = [item for item in session.get("clips") or [] if isinstance(item, dict)]
        if not primary_clips:
            return ctx.no_result("empty_timeline", "当前时间线为空，无法插入辅助画面。", query=query)
        schedule = _main.composition_schedule([
            {"start": item.get("sourceStart"), "end": item.get("sourceEnd"), "playbackRate": item.get("playbackRate") or 1, "transitionIn": item.get("transitionIn") or {"type": "cut", "duration": 0}}
            for item in primary_clips
        ])
        overlays = []
        for index, match_id in enumerate(selected_ids[:max_overlays]):
            candidate = candidates.get(match_id)
            if not candidate:
                continue
            primary_index = min(index, len(primary_clips) - 1)
            primary_id = str(primary_clips[primary_index].get("id") or "")
            overlay_start = float(schedule[primary_index].get("outputStart") or 0) + .2
            duration = min(2.5, max(.2, float(candidate.get("end") or 0) - float(candidate.get("start") or 0)))
            overlays.append({
                "id": f'broll_{_main.uuid.uuid4().hex[:10]}', "primarySegmentId": primary_id,
                "sourceStart": round(float(candidate.get("start") or 0), 3),
                "sourceEnd": round(float(candidate.get("start") or 0) + duration, 3),
                "outputOffset": round(overlay_start - float(schedule[primary_index].get("outputStart") or 0), 3),
                "muted": True, "sourceRef": {"kind": "content_match", "id": match_id},
                "title": str(candidate.get("title") or candidate.get("summary") or "辅助画面")[:100],
            })
        if not overlays:
            return ctx.no_result("no_broll_candidates", "没有可用于辅助画面插入的已确认候选。", query=query, candidate_count=len(candidates))
        session.setdefault("cutaways", [])
        session["cutaways"].extend(overlays)
        session["revision"] = int(session.get("revision") or 0) + 1
        session["renderedVersionId"] = None
        session["updatedAt"] = _main.now_iso()
        current["updatedAt"] = _main.now_iso()
        _main.save_job(current)
    return {"artifact": {"kind": "broll_overlay_plan", "jobId": ctx.job_id, "sessionId": session_id, "cutaways": overlays, "message": f"已加入 {len(overlays)} 个辅助画面插入点。"}}


def render_graphics_package(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    text = str(arguments.get("text") or "").strip()[:500]
    placement = str(arguments.get("placement") or "top").strip().lower()
    vertical = "bottom" if placement in {"bottom", "底部", "下方"} else "middle" if placement in {"middle", "center"} else "top"
    if not text:
        text = "视频重点"
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        session_id = str((current or {}).get("activeEditSessionId") or "")
        session = _main.find_edit_session(current, session_id) if current and session_id else None
        if not current or not session:
            return {"actionRequired": True, "action": "timeline_confirmation", "message": "请先建立当前时间线，再添加图文包装。"}
        duration = max(.5, float(session.get("duration") or 0))
        layer = {
            "id": f'edit_text_{_main.uuid.uuid4().hex[:12]}', "text": text,
            "start": 0.0, "end": min(duration, 5.0),
            "style": _main.normalize_subtitle_layout({
                "preset": "bold", "vertical": vertical, "horizontal": "center",
                "fontSizeRatio": .052 if vertical != "middle" else .060,
            }, "bold"),
            "source": "agent_graphics_package",
        }
        session.setdefault("textLayers", []).append(layer)
        session["revision"] = int(session.get("revision") or 0) + 1
        session["renderedVersionId"] = None
        session["updatedAt"] = _main.now_iso()
        current["updatedAt"] = _main.now_iso()
        _main.save_job(current)
    return {"artifact": {"kind": "graphics_package_layer", "jobId": ctx.job_id, "sessionId": session_id, "textLayer": layer, "message": "已添加当前任务图文包装层。"}}


def render_motion_graphics(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    title = str(arguments.get("title") or "").strip()[:120] or "精彩内容"
    subtitle = str(arguments.get("subtitle") or "").strip()[:220]
    aspect = str(arguments.get("aspect") or "9:16").strip()
    duration = max(0.5, min(8.0, float(arguments.get("duration") or 1.5)))
    fps = 24
    width, height = _main.motion_canvas_size(aspect)
    output_name = f"agent-motion-{aspect.replace(':', 'x')}-{_main.uuid.uuid4().hex[:8]}.mp4"
    work_root = Path(str(ctx.snapshot.get("workDirectory") or ""))
    output_root = Path(str(ctx.snapshot.get("outputDirectory") or ""))
    render_id = f'motion_{_main.uuid.uuid4().hex[:12]}'
    render_root = work_root / "local-motion" / render_id
    html_path = render_root / "index.html"
    frames_dir = render_root / "frames"
    output_path = output_root / output_name
    renderer_script = Path(_main.__file__).resolve().parent.parent / "tools" / "render_html_motion.mjs"

    def motion_worker() -> dict[str, Any]:
        render_root.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            _main.build_motion_graphics_html(
                title=title,
                subtitle=subtitle,
                label="ChatClip",
                aspect=aspect,
                theme=str(arguments.get("theme") or "chatclip"),
            ),
            encoding="utf-8",
        )
        _main.render_html_motion_video(
            html_path=html_path,
            output_path=output_path,
            frames_dir=frames_dir,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            ffmpeg=_main.settings.ffmpeg,
            renderer_script=renderer_script,
        )
        rendered = _main.probe_video(output_path, _main.settings.ffprobe)
        output = {
            "filename": output_name,
            "title": title,
            "duration": round(rendered.duration, 3),
            "width": rendered.width,
            "height": rendered.height,
            "hasAudio": rendered.has_audio,
            "previewOnly": True,
            "outputKind": "local_motion_graphics",
            "motionGraphics": {
                "renderer": "html-playwright-ffmpeg",
                "aspect": aspect,
                "duration": duration,
                "title": title,
            },
        }
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                output_path.unlink(missing_ok=True)
                raise RuntimeError("本地图文动效生成完成时任务已不存在")
            _main.normalize_output_versions(current)
            current_version = _main.find_output_version(current, str(current.get("currentOutputVersionId") or ""))
            if current_version:
                current_version.setdefault("previewOutputs", []).append(output)
            else:
                current.setdefault("agentPreviewOutputs", []).append(output)
            current["lastAgentMotionGraphicsFilename"] = output_name
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "local_motion_graphics",
                "jobId": ctx.job_id,
                "output": {
                    **output,
                    "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{output_name}',
                    "previewUrl": f'/api/jobs/{ctx.job_id}/outputs/{output_name}',
                },
                "message": "本地图文动效视频已生成。",
            },
        }

    future = _main.output_preview_executor.submit(motion_worker)
    return {
        "operationId": f'{ctx.job_id}:local_motion:{output_name}',
        "operation": "local_motion_graphics", "accepted": True,
        "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


def export_editing_draft(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    export_format = str(arguments.get("format") or "chatclip-json").strip().lower()
    if export_format not in {"chatclip-json", "jianying-bridge"}:
        export_format = "chatclip-json"
    with _main.jobs_lock:
        current_snapshot = _main.copy.deepcopy(_main.jobs.get(ctx.job_id) or ctx.snapshot)
    has_content = bool(
        current_snapshot.get("activeEditSessionId")
        or current_snapshot.get("editSessions")
        or current_snapshot.get("outputVersions")
        or current_snapshot.get("outputs")
    )
    draft_name = f'editing-draft-{export_format}-{_main.uuid.uuid4().hex[:8]}.json'
    draft_path = Path(str(current_snapshot.get("outputDirectory") or ctx.snapshot.get("outputDirectory") or "")) / draft_name
    _main.write_editing_draft_package(current_snapshot, draft_path)
    artifact = {
        "kind": "editing_draft_package",
        "jobId": ctx.job_id,
        "format": export_format,
        "filename": draft_name,
        "downloadUrl": f'/api/jobs/{ctx.job_id}/draft-exports/{draft_name}',
        "mode": "timeline" if has_content else "source-only",
        "message": (
            "已导出 ChatClip 本地草稿包；当前不会直接写入剪映目录。"
            if has_content
            else "当前还没有时间线或成片，已导出仅包含源素材与任务元数据的本地草稿包。"
        ),
    }
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        if current:
            current.setdefault("draftExports", []).append({
                **artifact,
                "createdAt": _main.now_iso(),
                "path": str(draft_path),
            })
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
    return {"artifact": artifact}


def compose_motion_intro(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    intro_filename = str(arguments.get("introFilename") or ctx.snapshot.get("lastAgentMotionGraphicsFilename") or "").strip()
    if Path(intro_filename).name != intro_filename or not intro_filename:
        return {"actionRequired": True, "action": "preview_generation", "message": "请先生成本地图文动效片头。"}
    output, source_path = ctx.current_output_reference(filename)
    intro_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / intro_filename
    if source_path is None or not source_path.is_file():
        return {"actionRequired": True, "action": "preview_generation", "message": "当前没有可合入动态图文片头的成片。"}
    if not intro_path.is_file():
        return {"actionRequired": True, "action": "preview_generation", "message": "本地图文动效片头文件不存在，请重新生成。"}
    target_name = f'motion-intro-{_main.uuid.uuid4().hex[:8]}.mp4'
    target_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / target_name

    def compose_motion_worker() -> dict[str, Any]:
        media = _main.compose_motion_intro_video(
            intro_path=intro_path,
            source_path=source_path,
            output_path=target_path,
            ffmpeg=_main.settings.ffmpeg,
            ffprobe=_main.settings.ffprobe,
        )
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                target_path.unlink(missing_ok=True)
                raise RuntimeError("动态图文片头合成完成时任务已不存在")
            _main.normalize_output_versions(current)
            current_output, current_version, _position = _main.output_download_context(current, str((output or {}).get("filename") or filename or source_path.name)) or ({}, {}, 0)
            version_id, version_number = _main.next_output_version(current)
            new_output = _main.copy.deepcopy(current_output or output or {})
            new_output.update({
                "filename": target_name,
                "title": f"{str((current_output or output or {}).get('title') or '成片')}（动态图文片头）",
                "duration": float(media["duration"]),
                "width": int(media["width"]),
                "height": int(media["height"]),
                "hasAudio": bool(media["hasAudio"]),
                "previewOnly": False,
                "versionId": version_id,
                "versionNumber": version_number,
                "versionCreatedAt": _main.now_iso(),
                "motionIntro": {
                    "enabled": True,
                    "introFilename": intro_filename,
                    "introDuration": float(media["introDuration"]),
                    "sourceFilename": str((current_output or output or {}).get("filename") or source_path.name),
                },
            })
            new_version = {
                "id": version_id,
                "number": version_number,
                "createdAt": _main.now_iso(),
                "outputs": [new_output],
                "previewOnly": False,
                "variantKind": "motion_intro_export",
                "parentVersionId": str((current_version or {}).get("id") or current.get("currentOutputVersionId") or ""),
            }
            current.setdefault("outputVersions", []).append(new_version)
            current["currentOutputVersionId"] = version_id
            current["outputs"] = [new_output]
            current.update({
                "status": "completed", "stage": "completed", "progress": 1.0,
                "stageProgress": 1.0, "progressMode": "completed",
                "detail": "动态图文片头成片已生成",
                "currentAction": "动态图文片头成片已生成",
                "error": None, "updatedAt": _main.now_iso(),
            })
            _main._sync_output_manifest(current)
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "motion_intro_output",
                "jobId": ctx.job_id,
                "output": {
                    **new_output,
                    "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{target_name}',
                    "previewUrl": f'/api/jobs/{ctx.job_id}/outputs/{target_name}',
                },
                "message": "已生成带本地图文动效片头的新成片版本。",
            },
        }

    future = _main.output_preview_executor.submit(compose_motion_worker)
    return {
        "operationId": f'{ctx.job_id}:compose_motion_intro:{target_name}',
        "operation": "compose_motion_intro", "accepted": True,
        "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


def render_social_preview(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    aspect = str(arguments.get("aspect") or "9:16").strip()
    fit = str(arguments.get("fit") or "blur").strip().lower()
    focus_x = max(0.0, min(1.0, float(arguments.get("focusX", .5))))
    focus_y = max(0.0, min(1.0, float(arguments.get("focusY", .5))))
    available = [item for item in _main.all_job_outputs(ctx.snapshot) if isinstance(item, dict) and item.get("filename")]
    if filename:
        source_output = next((item for item in available if str(item.get("filename")) == filename), None)
        if not source_output:
            raise RuntimeError("指定的审核样片或版本不存在，未改用其他视频")
    else:
        source_output = next((item for item in reversed(available) if not item.get("socialReframe")), None)
    source_path: Path | None = None
    source_session: dict[str, Any] | None = None
    source_filename = ""
    if source_output:
        source_filename = str(source_output["filename"])
        if Path(source_filename).name != source_filename:
            raise RuntimeError("成片文件名无效")
        source_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / source_filename
        linked_session_id = str(
            source_output.get("sourceEditSessionId")
            or source_output.get("editSessionId")
            or source_output.get("sessionId")
            or ""
        )
        source_session = next((
            item for item in ctx.snapshot.get("editSessions") or []
            if str(item.get("id") or "") == linked_session_id
        ), None)
        if not source_path.is_file() and source_session and str(source_session.get("previewPath") or ""):
            candidate_path = Path(source_session["previewPath"])
            if candidate_path.name == source_filename and candidate_path.is_file():
                source_path = candidate_path
    elif ctx.autonomous:
        batch = ctx.snapshot.get("agentTimelineBatch") if isinstance(ctx.snapshot.get("agentTimelineBatch"), dict) else {}
        first = next((item for item in batch.get("variants") or [] if isinstance(item, dict)), None)
        source_session = next((
            item for item in ctx.snapshot.get("editSessions") or []
            if first and str(item.get("id") or "") == str(first.get("sessionId") or "")
        ), None)
        candidate_path = Path(str((source_session or {}).get("previewPath") or ""))
        if candidate_path.is_file():
            source_path = candidate_path
            source_filename = candidate_path.name
    if not source_output:
        if source_path is None:
            return {
                "actionRequired": True,
                "action": "preview_generation",
                "message": "当前没有可用于画幅重构的审核样片。",
            }
    if source_path is None or not source_path.is_file():
        raise RuntimeError("画幅重构所需的成片文件不存在")
    aspect_slug = aspect.replace(":", "x")
    preview_filename = f'agent-social-{aspect_slug}-{_main.uuid.uuid4().hex[:8]}.mp4'
    preview_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / preview_filename
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    source_session_id = str((source_session or {}).get("id") or "")
    source_session_revision = int((source_session or {}).get("revision") or 0)
    source_session_fingerprint = (
        _main._edit_session_preview_fingerprint(ctx.snapshot, source_session)
        if source_session else ""
    )

    def render_social_worker() -> dict[str, Any]:
        source_info = _main.probe_video(source_path, _main.settings.ffprobe)
        rendered = _main.create_social_reframe_preview(
            source_path, preview_path, aspect=aspect, fit=fit,
            focus_x=focus_x, focus_y=focus_y, has_audio=source_info.has_audio,
            ffmpeg=_main.settings.ffmpeg, ffprobe=_main.settings.ffprobe,
        )
        fit_label = "虚化背景" if fit == "blur" else "留边" if fit == "pad" else "裁切"
        output = {
            "filename": preview_filename, "title": f"{aspect} {fit_label}审核预览",
            "duration": round(rendered.duration, 3), "width": rendered.width,
            "height": rendered.height, "previewOnly": True,
            "outputKind": "social_reframe_preview", "socialReframe": True,
            "sourceOutputFilename": source_filename,
            "sourceEditSessionId": str((source_session or {}).get("id") or ""),
            "reframe": {"aspect": aspect, "fit": fit, "focusX": focus_x, "focusY": focus_y},
            "segments": _main.copy.deepcopy(
                (source_output or {}).get("segments")
                or (source_session or {}).get("clips") or []
            ),
            "subtitleMode": (
                (source_output or {}).get("subtitleMode")
                or ("burn" if (source_session or {}).get("subtitleEnabled") else "none")
            ),
            "overlayVerification": _main.copy.deepcopy(
                (source_output or {}).get("overlayVerification")
                or (source_session or {}).get("previewOverlayVerification")
                or {}
            ),
            "reason": (
                "完整保留原画面，并使用同画面虚化背景填充社媒画布。"
                if fit == "blur" else
                "完整保留原画面并使用留边填充社媒画布。"
                if fit == "pad" else
                "按指定焦点裁切为社媒画幅。"
            ),
        }
        if isinstance((source_output or {}).get("coverIntro"), dict):
            output["coverIntro"] = _main.copy.deepcopy((source_output or {}).get("coverIntro"))
        output["planId"] = ctx.workspace.get("activePlanId")
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                preview_path.unlink(missing_ok=True)
                raise RuntimeError("画幅预览完成时素材任务已不存在")
            _main.normalize_output_versions(current)
            live_session = next((
                item for item in current.get("editSessions") or []
                if str(item.get("id") or "") == source_session_id
            ), None)
            session_is_current = bool(
                live_session
                and int(live_session.get("revision") or 0) == source_session_revision
                and _main._edit_session_preview_fingerprint(current, live_session) == source_session_fingerprint
            )
            current_version = _main.find_output_version(current, str(current.get("currentOutputVersionId") or ""))
            if current_version:
                current_version.setdefault("previewOutputs", []).append(output)
            else:
                current.setdefault("agentPreviewOutputs", []).append(output)
            current["lastAgentSocialPreviewFilename"] = preview_filename
            if session_is_current and live_session:
                live_session["reframe"] = _main.copy.deepcopy(output["reframe"])
                reframe_fingerprint = _main._edit_session_preview_fingerprint(current, live_session)
                live_session.update({
                    "previewStatus": "ready",
                    "previewRevision": source_session_revision,
                    "previewFingerprint": reframe_fingerprint,
                    "previewPath": str(preview_path),
                    "previewUrl": f'/api/jobs/{ctx.job_id}/edit-sessions/{source_session_id}/preview?r={source_session_revision}',
                    "previewError": None,
                    "renderPlanFingerprint": reframe_fingerprint,
                    "previewOverlayVerification": {
                        **_main.copy.deepcopy(output.get("overlayVerification") or {}),
                        "renderPipelineVersion": 3,
                        "reframe": _main.copy.deepcopy(output["reframe"]),
                    },
                    "updatedAt": _main.now_iso(),
                })
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
            public_output = {
                **_main.copy.deepcopy(output),
                "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{preview_filename}',
                "previewUrl": f'/api/jobs/{ctx.job_id}/outputs/{preview_filename}',
            }
        return {
            "artifact": {
                "kind": "social_reframe_preview", "jobId": ctx.job_id,
                "output": public_output,
                "message": "社媒画幅审核预览已生成。",
            },
        }

    future = _main.output_preview_executor.submit(render_social_worker)
    return {
        "operationId": f'{ctx.job_id}:social_reframe:{preview_filename}',
        "operation": "social_reframe_preview", "accepted": True, "future": future,
        "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


def polish_audio_mix(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    noise_reduction = bool(arguments.get("noiseReduction", True))
    output, source_path = ctx.current_output_reference(filename)
    if source_path is None or not source_path.is_file():
        return {"actionRequired": True, "action": "preview_generation", "message": "当前没有可用于音频优化的成片或审核样片。"}
    output_name = f'agent-audio-polish-{_main.uuid.uuid4().hex[:8]}.mp4'
    target_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / output_name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def audio_polish_worker() -> dict[str, Any]:
        info = _main.probe_video(source_path, _main.settings.ffprobe)
        if not info.has_audio:
            return ctx.no_result("missing_audio", "当前输出没有音轨，无法进行音频优化。", candidate_count=1)
        filters = []
        if noise_reduction:
            filters.append("afftdn")
        filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")
        filters.append("alimiter=limit=0.95")
        temporary = target_path.with_suffix(".tmp.mp4")
        temporary.unlink(missing_ok=True)
        _main.subprocess.run([
            _main.settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path), "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-af", ",".join(filters),
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            str(temporary),
        ], check=True, timeout=max(180.0, info.duration * 3.0))
        temporary.replace(target_path)
        rendered = _main.probe_video(target_path, _main.settings.ffprobe)
        public_output = {
            **_main.copy.deepcopy(output or {}),
            "filename": output_name, "title": "音频优化审核版本",
            "duration": round(rendered.duration, 3), "width": rendered.width,
            "height": rendered.height, "hasAudio": rendered.has_audio,
            "previewOnly": True, "outputKind": "audio_polish_preview",
            "sourceOutputFilename": str((output or {}).get("filename") or source_path.name),
            "audioPolish": {"noiseReduction": noise_reduction, "voiceFirst": bool(arguments.get("voiceFirst", True))},
        }
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                target_path.unlink(missing_ok=True)
                raise RuntimeError("音频优化完成时任务已不存在")
            _main.normalize_output_versions(current)
            current_version = _main.find_output_version(current, str(current.get("currentOutputVersionId") or ""))
            if current_version:
                current_version.setdefault("previewOutputs", []).append(public_output)
            else:
                current.setdefault("agentPreviewOutputs", []).append(public_output)
            current["lastAgentAudioPolishFilename"] = output_name
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {"artifact": {"kind": "audio_polish_preview", "jobId": ctx.job_id, "output": {**public_output, "previewUrl": f'/api/jobs/{ctx.job_id}/outputs/{output_name}', "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{output_name}'}, "message": "音频优化审核版本已生成。"}}

    future = _main.output_preview_executor.submit(audio_polish_worker)
    return {
        "operationId": f'{ctx.job_id}:audio_polish:{output_name}',
        "operation": "audio_polish_preview", "accepted": True,
        "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


def export_delivery_master(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    output, source_path = ctx.current_output_reference(filename)
    if source_path is None or not source_path.is_file():
        return {"actionRequired": True, "action": "preview_generation", "message": "当前没有可用于正式交付的成片或审核样片。"}
    platform_name = _main.re.sub(r"[^A-Za-z0-9_-]+", "-", str(arguments.get("platform") or "generic")).strip("-") or "generic"
    export_name = f'delivery-{platform_name}-{_main.uuid.uuid4().hex[:8]}.mp4'
    target_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / export_name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def export_worker() -> dict[str, Any]:
        _main.shutil.copy2(source_path, target_path)
        rendered = _main.probe_video(target_path, _main.settings.ffprobe)
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                target_path.unlink(missing_ok=True)
                raise RuntimeError("正式交付完成时任务已不存在")
            version_id, version_number = _main.next_output_version(current)
            new_output = {
                **_main.copy.deepcopy(output or {}),
                "filename": export_name, "title": f"{platform_name} 正式交付版",
                "duration": round(rendered.duration, 3), "width": rendered.width,
                "height": rendered.height, "hasAudio": rendered.has_audio,
                "previewOnly": False, "outputKind": "delivery_master",
                "sourceOutputFilename": str((output or {}).get("filename") or source_path.name),
                "versionId": version_id, "versionNumber": version_number,
                "versionCreatedAt": _main.now_iso(),
                "delivery": {"platform": platform_name, "aspect": str(arguments.get("aspect") or "")},
            }
            new_version = {
                "id": version_id, "number": version_number, "createdAt": _main.now_iso(),
                "outputs": [new_output], "previewOnly": False,
                "variantKind": "delivery_master",
                "parentVersionId": str((output or {}).get("versionId") or current.get("currentOutputVersionId") or ""),
            }
            current.setdefault("outputVersions", []).append(new_version)
            current["currentOutputVersionId"] = version_id
            current["outputs"] = [new_output]
            current.update({
                "status": "completed", "stage": "completed", "progress": 1.0,
                "stageProgress": 1.0, "progressMode": "completed",
                "detail": "正式交付版本已生成", "currentAction": "正式交付版本已生成",
                "error": None, "updatedAt": _main.now_iso(),
            })
            _main._sync_output_manifest(current)
            _main.save_job(current)
        return {"artifact": {"kind": "delivery_master", "jobId": ctx.job_id, "output": {**new_output, "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{export_name}'}, "message": "正式交付版本已生成。"}}

    future = _main.output_preview_executor.submit(export_worker)
    return {
        "operationId": f'{ctx.job_id}:delivery_export:{export_name}',
        "operation": "delivery_master_export", "accepted": True,
        "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


def run_delivery_qc(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    strict = bool(arguments.get("strict", False))
    raw_target_seconds = arguments.get("targetSeconds")
    expected_aspect = str(arguments.get("expectedAspect") or "")
    brief = ctx.snapshot.get("brief") if isinstance(ctx.snapshot.get("brief"), dict) else {}
    cover_intro_required = bool(arguments.get("requireCoverIntro") or brief.get("coverIntroRequested"))
    target_seconds = (
        max(0.0, float(raw_target_seconds))
        if isinstance(raw_target_seconds, (int, float)) and not isinstance(raw_target_seconds, bool)
        else None
    )
    tolerance_seconds = max(0.0, float(arguments.get("toleranceSeconds") or 0))
    available = [item for item in _main.all_job_outputs(ctx.snapshot) if isinstance(item, dict) and item.get("filename")]
    if filename:
        selected = [item for item in available if str(item.get("filename")) == filename]
    else:
        preferred_value = (
            ctx.snapshot.get("lastAgentCoverIntroPreviewFilename")
            if cover_intro_required else
            ctx.snapshot.get("lastAgentSocialPreviewFilename")
            or ctx.snapshot.get("lastAgentCoverIntroPreviewFilename")
        )
        preferred = str(preferred_value or "")
        selected = [item for item in available if str(item.get("filename")) == preferred]
        if not selected:
            selected = available[-1:] if available else []
    if not selected:
        return {
            "actionRequired": True,
            "action": "preview_generation",
            "message": "当前没有可检查的成片，请先完成一个审核样片或正式输出。",
        }

    subtitle_required = bool(
        brief.get("subtitleRequested")
        or str(brief.get("subtitlePreference") or "").strip().lower()
        not in {"", "none", "off", "false"}
    )
    graphics_required = bool(brief.get("graphicsRequested"))
    cover_required = bool(arguments.get("requireCover") or brief.get("coverRequested"))
    current_cover_id = str(ctx.snapshot.get("currentCoverVersionId") or "")
    current_cover = next((
        value for value in ctx.snapshot.get("coverVersions") or []
        if isinstance(value, dict) and str(value.get("id") or "") == current_cover_id
    ), None)
    cover_draft = ctx.snapshot.get("coverDraft") if isinstance(ctx.snapshot.get("coverDraft"), dict) else {}
    required_cover_title = str(arguments.get("expectedCoverTitle") or brief.get("coverTitle") or cover_draft.get("titleText") or "").strip()

    def deliverable_issue(code: str, message: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "severity": "error", "code": code, "message": message,
            "evidence": _main.copy.deepcopy(evidence),
        }

    def delivery_qc_worker() -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        for item in selected:
            selected_filename = str(item["filename"])
            if Path(selected_filename).name != selected_filename:
                raise RuntimeError("成片文件名无效")
            media_path = Path(str(ctx.snapshot.get("outputDirectory") or "")) / selected_filename
            if not media_path.is_file():
                raise RuntimeError(f"成片文件不存在：{selected_filename}")
            expected = item.get("duration")
            report = _main.analyze_rendered_media(
                media_path, ffmpeg=_main.settings.ffmpeg, ffprobe=_main.settings.ffprobe,
                expected_duration=float(expected) if isinstance(expected, (int, float)) else None,
                expect_audio=None,
            )
            report["filename"] = selected_filename
            report.pop("path", None)
            aspect_issue = _main.aspect_check((report.get("media") or {}).get("width"),
                                        (report.get("media") or {}).get("height"), expected_aspect)
            report["aspectCheck"] = {"expectedAspect": expected_aspect or None,
                                     "status": aspect_issue["status"] if aspect_issue else "passed" if expected_aspect else "not_requested"}
            if aspect_issue:
                report.setdefault("issues", []).append(aspect_issue)
                report["passed"] = False
            source_session_id = str(item.get("sourceEditSessionId") or item.get("sessionId") or "")
            source_session = next((s for s in ctx.snapshot.get("editSessions") or [] if str(s.get("id")) == source_session_id), None)
            if source_session is None and any(s.get("sourceContentContract") for s in item.get("segments") or []):
                source_segments = _main.copy.deepcopy(item["segments"])
                source_session = {
                    "id": f'qc_{_main.uuid.uuid4().hex[:12]}',
                    "contentBinding": _main.selection_binding({"id": selected_filename, "recordType": "assembly",
                        "basketSnapshot": {"outputFilename": selected_filename}}, source_segments),
                    "clips": [{"id": s["id"], "sourceRef": {"id": s["id"]},
                               "sourceStart": s["start"], "sourceEnd": s["end"],
                               "playbackRate": s.get("playbackRate", 1), "transitionIn": s.get("transitionIn")}
                              for s in source_segments],
                }
                _main.refresh_edit_session(source_session, ctx.snapshot)
            if source_session:
                content_report = _main._check_rendered_content(ctx.snapshot, source_session, media_path)
                report["contentVerification"] = content_report
                report.setdefault("issues", []).extend(content_report["issues"])
                if not content_report["passed"]:
                    report["passed"] = False
            overlay = item.get("overlayVerification") if isinstance(item.get("overlayVerification"), dict) else {}
            overlay_verified = bool(
                overlay.get("applied")
                and int(overlay.get("renderPipelineVersion") or 0) >= 2
            )
            subtitle_ready = bool(overlay_verified and int(overlay.get("subtitleCueCount") or 0) > 0)
            graphics_ready = bool(overlay_verified and int(overlay.get("textLayerCount") or 0) > 0)
            cover_title_ready = bool(
                current_cover
                and (
                    not required_cover_title
                    or (
                        str(current_cover.get("titleText") or "").strip() == required_cover_title
                        and bool(current_cover.get("titleLines"))
                    )
                )
            )
            cover_source = cover_draft.get("source") if isinstance(cover_draft.get("source"), dict) else {}
            cover_source_ready = bool(
                current_cover
                and (
                    str(cover_source.get("kind") or "") != "accepted_timeline"
                    or (
                        bool(str(cover_source.get("editSessionId") or ""))
                        and bool(current_cover.get("evidenceRefs"))
                    )
                )
            )
            cover_ready = bool(current_cover and cover_title_ready and cover_source_ready)
            cover_intro = item.get("coverIntro") if isinstance(item.get("coverIntro"), dict) else {}
            cover_intro_ready = bool(
                cover_intro.get("enabled")
                and current_cover_id
                and str(cover_intro.get("coverVersionId") or "") == current_cover_id
            )
            report["deliverables"] = {
                "subtitle": {"required": subtitle_required, "passed": subtitle_ready, "verification": _main.copy.deepcopy(overlay)},
                "graphicsText": {"required": graphics_required, "passed": graphics_ready, "verification": _main.copy.deepcopy(overlay)},
                "cover": {
                    "required": cover_required, "passed": cover_ready,
                    "currentCoverVersionId": current_cover_id or None,
                    "requiredTitle": required_cover_title or None,
                    "actualTitle": str((current_cover or {}).get("titleText") or "") or None,
                    "sourceKind": str(cover_source.get("kind") or "") or None,
                },
                "coverIntro": {
                    "required": cover_intro_required, "passed": cover_intro_ready,
                    "currentCoverVersionId": current_cover_id or None,
                    "outputCoverVersionId": str(cover_intro.get("coverVersionId") or "") or None,
                },
            }
            if subtitle_required and not subtitle_ready:
                report.setdefault("issues", []).append(deliverable_issue(
                    "requested_subtitles_not_rendered",
                    "任务要求字幕，但成片没有可验证的已烧录字幕。",
                    report["deliverables"]["subtitle"],
                ))
                report["passed"] = False
            if graphics_required and not graphics_ready:
                report.setdefault("issues", []).append(deliverable_issue(
                    "requested_text_not_rendered",
                    "任务要求添加文字，但成片没有可验证的已渲染文字图层。",
                    report["deliverables"]["graphicsText"],
                ))
                report["passed"] = False
            if cover_required and not cover_ready:
                report.setdefault("issues", []).append(deliverable_issue(
                    "requested_cover_not_validated",
                    "任务要求封面，但当前封面未通过标题、来源或版本校验。",
                    report["deliverables"]["cover"],
                ))
                report["passed"] = False
            if cover_intro_required and not cover_intro_ready:
                report.setdefault("issues", []).append(deliverable_issue(
                    "requested_cover_intro_not_rendered",
                    "任务要求把封面作为片头，但当前成片未绑定并合入本任务的封面版本。",
                    report["deliverables"]["coverIntro"],
                ))
                report["passed"] = False
            if target_seconds:
                actual_seconds = float((report.get("media") or {}).get("duration") or 0)
                minimum_seconds = max(0.0, target_seconds - tolerance_seconds)
                maximum_seconds = target_seconds + tolerance_seconds
                duration_passed = minimum_seconds <= actual_seconds <= maximum_seconds
                report["targetDuration"] = {
                    "targetSeconds": round(target_seconds, 3),
                    "toleranceSeconds": round(tolerance_seconds, 3),
                    "minimumSeconds": round(minimum_seconds, 3),
                    "maximumSeconds": round(maximum_seconds, 3),
                    "actualSeconds": round(actual_seconds, 3),
                    "passed": duration_passed,
                }
                if not duration_passed:
                    report.setdefault("issues", []).append({
                        "severity": "error",
                        "code": "target_duration_mismatch",
                        "message": (
                            f"成片时长 {actual_seconds:.2f} 秒不在目标范围 "
                            f"{minimum_seconds:.2f}–{maximum_seconds:.2f} 秒内。"
                        ),
                        "evidence": _main.copy.deepcopy(report["targetDuration"]),
                    })
                    report["passed"] = False
            report["strictPassed"] = bool(report["passed"]) and (
                not strict or not any(issue.get("severity") == "warning" for issue in report["issues"])
            )
            reports.append(report)
        current_plan = _main.agent_platform.store.get("plans", str(ctx.workspace.get("activePlanId") or "")) or {}
        repair = _main.quality_repair_plan({"reports": reports}, current_plan.get("steps") or [], current_plan.get("inputContext"))
        return {
            "artifact": {
                "kind": "delivery_qc_report", "jobId": ctx.job_id,
                "passed": all(report["strictPassed"] if strict else report["passed"] for report in reports),
                "strict": strict, "reports": reports, "repair": repair,
            },
        }

    future = _main.output_preview_executor.submit(delivery_qc_worker)
    return {
        "operationId": f'{ctx.job_id}:delivery_qc', "operation": "delivery_qc",
        "accepted": True, "future": future,
        "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


