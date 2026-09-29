"""Cover Agent tool handlers."""
from __future__ import annotations

from typing import Any
from pathlib import Path

from .context import ToolContext


def propose_cover_candidates(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    source_scope = str(arguments.get("sourceScope") or "accepted_cut")
    candidate_budget = max(3, min(24, int(arguments.get("candidateBudget") or 16)))
    aspects = [
        str(value) for value in arguments.get("aspectRatios") or ["16:9"]
        if str(value) in _main.COVER_ASPECT_SIZES
    ] or ["16:9"]
    title_text = str(arguments.get("titleText") or "")[:80]
    focus = str(arguments.get("focus") or "")[:240]
    subject = str(arguments.get("subject") or "")[:120].strip()
    requested_source_time = arguments.get("sourceTime")
    source_path, source_metadata, evidence = _main._cover_source(ctx.snapshot, source_scope)
    draft_id = f'cover_draft_{_main.uuid.uuid4().hex[:12]}'
    work_root = Path(str(ctx.snapshot.get("workDirectory") or ""))
    cover_root = work_root / "cover-director" / draft_id
    frames_directory = cover_root / "candidates"

    def propose_cover_worker() -> dict[str, Any]:
        points = _main.cover_sample_points(
            float(source_metadata.get("duration") or 0), evidence,
            budget=candidate_budget,
            include_uniform=str(source_metadata.get("kind") or "") != "accepted_timeline",
        )
        if requested_source_time is not None:
            source_time = max(0.0, min(
                max(0.0, float(source_metadata.get("duration") or 0) - .001),
                float(requested_source_time),
            ))
            duration = max(0.0, float(source_metadata.get("duration") or 0))
            nearby_times = []
            for offset in (0.0, -.5, .5):
                value = max(0.0, min(max(0.0, duration - .001), source_time + offset))
                if all(abs(value - existing) >= .05 for existing in nearby_times):
                    nearby_times.append(value)
            points = [{
                "time": value,
                "evidenceRefs": [],
                "evidenceStrength": 1.0,
                "evidenceText": f"用户指定封面时间 {source_time:.3f} 秒附近",
            } for value in nearby_times]
        frames = _main.extract_frames_at_times(
            source_path, frames_directory, [float(item["time"]) for item in points],
            ffmpeg=_main.settings.ffmpeg,
        )
        raw_frames = []
        for index, (frame, point) in enumerate(zip(frames, points), 1):
            raw_frames.append({
                "id": f"cover_candidate_{index:02d}", "path": frame.path,
                "sourceTime": round(frame.time, 3),
                "evidenceRefs": list(point.get("evidenceRefs") or []),
                "evidenceStrength": float(point.get("evidenceStrength") or .35),
                "evidenceText": str(point.get("evidenceText") or ""),
            })
        selected, rejected = _main.score_cover_frames(
            raw_frames, request_focus=focus, require_person=bool(subject), limit=candidate_budget,
        )
        if requested_source_time is not None:
            requested_number = float(requested_source_time)
            selected = [
                item for item in selected
                if abs(float(item.get("sourceTime") or 0) - requested_number) <= .75
            ]
        if not selected:
            if requested_source_time is not None:
                subject_hint = f"且包含“{subject}”" if subject else ""
                raise RuntimeError(
                    f"没有找到 {float(requested_source_time):.1f} 秒附近{subject_hint}的可用封面画面；"
                    "请调整时间点或封面要求"
                )
            if subject and any(item.get("rejectionReason") == "missing_person" for item in rejected):
                raise RuntimeError(f"没有找到包含人物的封面画面，无法确认“{subject}”；请更换时间点或封面要求")
            raise RuntimeError("没有找到可用的非黑、可解码封面画面")
        candidates: list[dict[str, Any]] = []
        for item in selected:
            path = Path(str(item.pop("path")))
            candidates.append({
                **item,
                "subjectVerification": {
                    "subject": subject,
                    "status": "identity_unverified" if subject else "not_required",
                    "personDetected": (item.get("metrics") or {}).get("faceCount", None) not in {0, None},
                },
                "artifactFile": str(path.relative_to(work_root)),
                "previewUrl": f"/api/jobs/{ctx.job_id}/cover-artifacts/{item['id']}",
            })
        draft = {
            "schemaVersion": _main.COVER_SCHEMA_VERSION, "id": draft_id, "jobId": ctx.job_id,
            "status": "candidates_ready", "source": source_metadata,
            "sourceAssetId": str(ctx.snapshot.get("sourceAssetId") or ctx.snapshot.get("sourceHash") or ""),
            "sourceSearchId": str((ctx.snapshot.get("contentSearch") or {}).get("id") or "") if isinstance(ctx.snapshot.get("contentSearch"), dict) else "",
            "sourceQuery": str(((ctx.snapshot.get("contentSearch") or {}).get("intent") or {}).get("query") or (ctx.snapshot.get("contentSearch") or {}).get("instruction") or "")[:500] if isinstance(ctx.snapshot.get("contentSearch"), dict) else "",
            "aspectRatios": aspects, "titleText": title_text, "focus": focus,
            "subject": subject,
            "requestedSourceTime": float(requested_source_time) if requested_source_time is not None else None,
            "candidates": candidates, "variants": [],
            "rejectedSummary": {
                reason: sum(1 for item in rejected if item.get("rejectionReason") == reason)
                for reason in {str(item.get("rejectionReason") or "unknown") for item in rejected}
            },
            "createdAt": _main.now_iso(), "updatedAt": _main.now_iso(),
        }
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                raise RuntimeError("封面候选完成时素材任务已不存在")
            current["coverDraft"] = draft
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "cover_candidates", "jobId": ctx.job_id, "draftId": draft_id,
                "source": _main.copy.deepcopy(source_metadata),
                "candidates": [_main._public_cover_artifact(ctx.job_id, item) for item in candidates],
                "rejectedSummary": _main.copy.deepcopy(draft["rejectedSummary"]),
            },
        }

    future = _main.output_preview_executor.submit(propose_cover_worker)
    return {
        "operationId": f'{ctx.job_id}:cover_candidates:{draft_id}',
        "operation": "cover_candidates", "accepted": True, "future": future,
        "cancel": lambda: future.cancel(),
    }


def render_cover_variants(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    draft = ctx.snapshot.get("coverDraft") if isinstance(ctx.snapshot.get("coverDraft"), dict) else {}
    candidates = [item for item in draft.get("candidates") or [] if isinstance(item, dict)]
    if not draft.get("id") or not candidates:
        raise RuntimeError("请先生成可用的封面候选")
    directions = [
        str(value) for value in arguments.get("directions") or _main.COVER_DIRECTIONS
        if str(value) in _main.COVER_DIRECTIONS
    ]
    directions = list(dict.fromkeys(directions))[:3] or list(_main.COVER_DIRECTIONS)
    aspects = [
        str(value) for value in arguments.get("aspectRatios") or draft.get("aspectRatios") or ["16:9"]
        if str(value) in _main.COVER_ASPECT_SIZES
    ] or ["16:9"]
    title_text = str(arguments.get("titleText") or draft.get("titleText") or "")[:80]
    work_root = Path(str(ctx.snapshot.get("workDirectory") or ""))
    variants_directory = work_root / "cover-director" / str(draft["id"]) / "variants"
    font_path = Path(_main.__file__).resolve().parent.parent / "fonts" / "SourceHanSansSC-Bold.otf"

    def render_cover_worker() -> dict[str, Any]:
        variants: list[dict[str, Any]] = []
        for aspect in aspects:
            for index, direction in enumerate(directions):
                candidate = candidates[index % len(candidates)]
                source_frame = (work_root / str(candidate.get("artifactFile") or "")).resolve()
                try:
                    source_frame.relative_to(work_root.resolve())
                except ValueError as error:
                    raise RuntimeError("封面候选路径无效") from error
                if not source_frame.is_file():
                    raise RuntimeError("封面候选画面不存在")
                signature = _main.hashlib.sha256(
                    f"{draft['id']}:{candidate.get('id')}:{aspect}:{direction}:{title_text}".encode("utf-8")
                ).hexdigest()[:12]
                variant_id = f"cover_variant_{signature}"
                output = variants_directory / f"{variant_id}.jpg"
                rendered = _main.render_cover_variant(
                    source_frame, output, aspect=aspect, direction=direction,
                    title=title_text,
                    font_path=font_path,
                )
                variants.append({
                    "schemaVersion": _main.COVER_SCHEMA_VERSION,
                    "variantId": variant_id, "status": "preview",
                    "direction": direction, "aspectRatio": aspect,
                    "width": rendered["width"], "height": rendered["height"],
                    "titleText": title_text,
                    "titleLines": rendered["titleLines"],
                    "sourceCandidateId": str(candidate.get("id") or ""),
                    "sourceTime": float(candidate.get("sourceTime") or 0),
                    "evidenceRefs": _main.copy.deepcopy(candidate.get("evidenceRefs") or []),
                    "score": _main.copy.deepcopy(candidate.get("score") or {}),
                    "subjectVerification": _main.copy.deepcopy(candidate.get("subjectVerification") or {}),
                    "contentHash": rendered["contentHash"],
                    "artifactFile": str(output.relative_to(work_root)),
                    "previewUrl": f'/api/jobs/{ctx.job_id}/cover-artifacts/{variant_id}',
                    "provenance": {
                        "kind": "source_frame_composite", "renderer": "chatclip-cover-v1",
                        "createdAt": _main.now_iso(),
                    },
                })
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            current_draft = current.get("coverDraft") if current and isinstance(current.get("coverDraft"), dict) else {}
            if not current or str(current_draft.get("id") or "") != str(draft.get("id") or ""):
                raise RuntimeError("封面渲染完成前候选版本已经变化")
            current_draft.update({
                "status": "review_ready", "variants": variants,
                "selectedVariantId": variants[0]["variantId"] if variants else "",
                "updatedAt": _main.now_iso(),
            })
            current["coverTimelineDraft"] = {
                "schemaVersion": "cover-timeline-draft-v1",
                "activeVariantId": variants[0]["variantId"] if variants else "",
                "variants": {
                    str(item["variantId"]): {"duration": 1.0, "updatedAt": _main.now_iso()}
                    for item in variants
                },
                "updatedAt": _main.now_iso(),
            }
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "cover_variant_set", "jobId": ctx.job_id,
                "draftId": str(draft["id"]),
                "variants": [_main._public_cover_artifact(ctx.job_id, item) for item in variants],
                "defaultVariantId": variants[0]["variantId"] if variants else "",
            },
        }

    future = _main.output_preview_executor.submit(render_cover_worker)
    return {
        "operationId": f"{ctx.job_id}:cover_variants:{draft['id']}",
        "operation": "cover_variants", "accepted": True, "future": future,
        "cancel": lambda: future.cancel(),
    }


def review_cover_variants(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    if ctx.autonomous and not str((ctx.snapshot.get("coverDraft") or {}).get("subject") or "").strip():
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            draft = current.get("coverDraft") if current and isinstance(current.get("coverDraft"), dict) else {}
            variants = [item for item in draft.get("variants") or [] if isinstance(item, dict)]
            if not variants:
                raise RuntimeError("尚未生成可用的封面候选版本")
            required_title = str(draft.get("titleText") or "").strip()
            required_aspects = {
                str(value) for value in draft.get("aspectRatios") or [] if str(value)
            }
            accepted_timeline = str((draft.get("source") or {}).get("kind") or "") == "accepted_timeline"
            eligible = [
                item for item in variants
                if (not required_title or (
                    str(item.get("titleText") or "").strip() == required_title
                    and bool(item.get("titleLines"))
                ))
                and (not required_aspects or str(item.get("aspectRatio") or "") in required_aspects)
                and (not accepted_timeline or bool(item.get("evidenceRefs")))
            ]
            if not eligible:
                raise RuntimeError("封面候选未满足当前任务的标题、画幅或来源要求，已停止自动选择")
            # Candidate scoring is deterministic and already performed by
            # render_cover_variants. Keep the highest ranked candidate as
            # the automatic default, while leaving the full set available
            # for a user override before final delivery.
            def cover_score(item: dict[str, Any]) -> float:
                score = item.get("score") if isinstance(item.get("score"), dict) else {}
                return float(score.get("total") or sum(
                    float(score.get(key) or 0)
                    for key in ("requestAlignment", "subjectReadability", "emotionOrAction", "visualClarity", "titleSafeSpace", "distinctiveness")
                ))
            selected = max(eligible, key=cover_score)
            draft["selectedVariantId"] = str(selected.get("variantId") or "")
            draft["selectionMode"] = "auto_best_validated"
            draft["status"] = "auto_selected"
            draft["updatedAt"] = _main.now_iso()
            current["coverDraft"] = draft
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "cover_auto_selection", "jobId": ctx.job_id,
                "selectedVariantId": str(selected.get("variantId") or ""),
                "variants": [_main._public_cover_artifact(ctx.job_id, item) for item in variants],
                "message": "已按当前任务主题和画幅自动选择最佳封面，可在正式成片生成前替换。",
            },
        }
    subject = str((ctx.snapshot.get("coverDraft") or {}).get("subject") or "").strip()
    return {
        "actionRequired": True, "action": "cover_review",
        "message": (
            f"请核对候选画面中的人物是否为“{subject}”。系统只确认画面中检测到人物，不能仅凭画面自动证明具体身份。"
            if subject else
            "封面候选已放入主时间轴。选择后会保存为当前任务封面，不会重新生成视频。"
        ),
    }


def confirm_cover(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        draft = current.get("coverDraft") if current and isinstance(current.get("coverDraft"), dict) else {}
        selected_id = str(draft.get("selectedVariantId") or "")
    public_version, bound_outputs, revision = _main._activate_cover_variant(ctx.job_id, selected_id)
    # Confirming a cover is an atomic cover operation.  It must not
    # silently turn into a full video export: doing so previously made a
    # successfully saved cover appear to fail when an unrelated export
    # precondition (for example source-subtitle acknowledgement) blocked
    # the downstream render.  Cover-intro composition and formal export
    # are represented by their own explicit plan steps.
    _main.append_cover_result(ctx.job_id, bound_outputs)
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        session_id = str((current or {}).get("activeEditSessionId") or "")
        session = next((
            item for item in (current or {}).get("editSessions") or []
            if isinstance(item, dict) and str(item.get("id") or "") == session_id
        ), None)
        revision_number = int((session or {}).get("revision") or 0)
        brief = current.get("brief") if current and isinstance(current.get("brief"), dict) else {}
        goal_text = " ".join(
            str(message.get("text") or "")
            for message in (current or {}).get("messages") or []
            if isinstance(message, dict)
        )
    if ctx.autonomous:
        cover_intro_requested = bool(brief.get("coverIntroRequested")) or _main.AgentPlatform._cover_intro_requested(goal_text)
        intro_duration = float(
            brief.get("coverIntroDurationSeconds")
            or _main.AgentPlatform._cover_intro_duration(goal_text)
        )
        active_plan = _main.agent_platform.store.get("plans", str(ctx.workspace.get("activePlanId") or "")) or {}
        has_explicit_intro_step = any(
            str(step.get("tool") or "") == "compose_cover_intro"
            for step in active_plan.get("steps") or []
            if isinstance(step, dict)
        )
        if cover_intro_requested and not has_explicit_intro_step and session_id and revision_number > 0:
            def cover_intro_preview_worker() -> dict[str, Any]:
                with _main.jobs_lock:
                    current_job = _main.jobs.get(ctx.job_id)
                    if not current_job:
                        raise RuntimeError("封面片头生成时任务已不存在")
                    current_session = _main.find_edit_session(current_job, session_id)
                    preview_path = Path(str(current_session.get("previewPath") or ""))
                    cover = _main.current_cover_version(current_job)
                    cover_path = _main.cover_version_path(current_job, cover) if cover else None
                    output_dir = Path(str(current_job.get("outputDirectory") or ""))
                if not preview_path.is_file():
                    raise RuntimeError("当前审核样片不存在，无法合成封面片头")
                if not cover_path or not cover_path.is_file():
                    raise RuntimeError("当前任务封面文件不存在，无法合成封面片头")
                output_dir.mkdir(parents=True, exist_ok=True)
                target_path = output_dir / f'agent-cover-intro-preview-{_main.uuid.uuid4().hex[:8]}.mp4'
                rendered = _main.render_cover_intro(
                    preview_path, cover_path, target_path, duration=intro_duration,
                    ffmpeg=_main.settings.ffmpeg, ffprobe=_main.settings.ffprobe,
                )
                output = {
                    "filename": target_path.name,
                    "title": "封面片头审核预览",
                    "duration": round(float(rendered["duration"]), 3),
                    "width": int(rendered["width"]),
                    "height": int(rendered["height"]),
                    "hasAudio": bool(rendered.get("hasAudio")),
                    "previewOnly": True,
                    "outputKind": "cover_intro_review_preview",
                    "sourceOutputFilename": preview_path.name,
                    "sourceEditSessionId": session_id,
                    "overlayVerification": _main.copy.deepcopy(
                        current_session.get("previewOverlayVerification") or {}
                    ),
                    "subtitleMode": (
                        "burn"
                        if int((current_session.get("previewOverlayVerification") or {}).get("subtitleCueCount") or 0) > 0
                        else "none"
                    ),
                    "coverIntro": {
                        "enabled": True,
                        "duration": float(rendered.get("introDuration") or intro_duration),
                        "coverVersionId": str((cover or {}).get("id") or ""),
                    },
                }
                with _main.jobs_lock:
                    current_job = _main.jobs.get(ctx.job_id)
                    if current_job:
                        current_job.setdefault("agentPreviewOutputs", []).append(output)
                        current_job["updatedAt"] = _main.now_iso()
                        _main.save_job(current_job)
                return {
                    "artifact": {
                        "kind": "review_preview",
                        "jobId": ctx.job_id,
                        "output": _main.copy.deepcopy(output),
                        "message": "已生成带当前任务封面片头的审核预览。",
                    },
                }
            future = _main.output_preview_executor.submit(cover_intro_preview_worker)
            return {
                "operationId": f'{ctx.job_id}:cover_intro_preview:{session_id}:{revision_number}',
                "operation": "cover_intro_preview", "accepted": True,
                "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
            }
        return {
            "artifact": {
                "kind": "current_cover", "jobId": ctx.job_id,
                "cover": public_version,
                "thumbnailUrl": f'/api/jobs/{ctx.job_id}/thumbnail?revision={revision}',
                "boundOutputs": bound_outputs,
                "message": _main.copy_messages.COVER_SAVED_AUTO_REVIEW,
            },
        }
    return {
        "artifact": {
            "kind": "current_cover", "jobId": ctx.job_id,
            "cover": public_version,
            "thumbnailUrl": f'/api/jobs/{ctx.job_id}/thumbnail?revision={revision}',
            "boundOutputs": bound_outputs,
            "message": _main.copy_messages.cover_saved(len(bound_outputs)),
        },
    }


def compose_cover_intro(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    duration = max(.5, min(5.0, float(arguments.get("duration") or 1.5)))
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        if not current:
            raise RuntimeError("任务不存在")
        cover_version = _main.current_cover_version(current)
        cover_path = _main.cover_version_path(current, cover_version) if cover_version else None
        if not cover_version or not cover_path or not cover_path.is_file():
            raise RuntimeError("请先生成并确认当前任务封面")
        _main.normalize_output_versions(current)
        output_dir = Path(str(current.get("outputDirectory") or ""))
        source_filename = ""
        source_output: dict[str, Any] | None = None
        requested_filename = Path(str(arguments.get("filename") or "")).name
        available = [
            item for item in _main.all_job_outputs(current)
            if isinstance(item, dict) and str(item.get("filename") or "")
        ]

        def cover_intro_source_rank(item: dict[str, Any], index: int) -> tuple[int, int]:
            kind = str(item.get("outputKind") or "")
            if kind == "cover_intro_review_preview":
                return (-1, index)
            if kind == "social_reframe_preview" or item.get("socialReframe"):
                return (5, index)
            if item.get("previewOnly"):
                return (4, index)
            return (3, index)

        if requested_filename:
            source_output = next((
                item for item in available
                if Path(str(item.get("filename") or "")).name == requested_filename
            ), None)
        else:
            ranked = sorted(
                enumerate(available),
                key=lambda pair: cover_intro_source_rank(pair[1], pair[0]),
                reverse=True,
            )
            source_output = next((item for _index, item in ranked), None)
        if source_output:
            candidate = Path(str(source_output.get("filename") or "")).name
            candidate_path = output_dir / candidate
            if candidate and candidate_path.is_file():
                source_filename = candidate
                source_path = candidate_path
            else:
                source_path = None
        else:
            source_path = None
        if not source_filename:
            raise RuntimeError("请先生成审核样片，再添加封面片头")
        source_kind = "agent_preview" if bool((source_output or {}).get("previewOnly")) else "output"
        current["coverIntroDraft"] = {
            "schemaVersion": "cover-intro-timeline-v1", "enabled": True,
            "duration": duration, "coverVersionId": str(cover_version.get("id") or ""),
            "updatedAt": _main.now_iso(),
        }
        current["coverIntroOperation"] = {
            "status": "queued", "filename": source_filename,
            "sourceKind": source_kind, "duration": duration, "queuedAt": _main.now_iso(),
        }
        current.update({
            "status": "running", "stage": "rendering", "progress": .9,
            "stageProgress": 0.0, "progressMode": "indeterminate",
            "detail": "正在生成封面片头审核样片", "currentAction": "正在把当前任务封面合入审核样片",
            "model": "FFmpeg", "error": None, "updatedAt": _main.now_iso(),
        })
        _main.save_job(current)
    target_name = f'agent-cover-intro-preview-{_main.uuid.uuid4().hex[:8]}.mp4'
    target_path = Path(str(ctx.snapshot.get("outputDirectory") or output_dir)) / target_name

    def cover_intro_agent_worker() -> dict[str, Any]:
        if source_path is None or not source_path.is_file():
            raise RuntimeError("封面片头合成源样片不存在")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = _main.render_cover_intro(
            source_path, cover_path, target_path, duration=duration,
            ffmpeg=_main.settings.ffmpeg, ffprobe=_main.settings.ffprobe,
        )
        output = _main.copy.deepcopy(source_output or {})
        output.update({
            "filename": target_name,
            "title": "带封面片头的审核样片",
            "duration": round(float(rendered["duration"]), 3),
            "width": int(rendered["width"]),
            "height": int(rendered["height"]),
            "hasAudio": bool(rendered.get("hasAudio")),
            "previewOnly": True,
            "outputKind": "cover_intro_review_preview",
            "sourceOutputFilename": source_filename,
            "coverIntro": {
                "enabled": True,
                "duration": float(rendered.get("introDuration") or duration),
                "coverVersionId": str(cover_version.get("id") or ""),
                "sourceFilename": source_filename,
                "sourceKind": source_kind,
            },
        })
        with _main.jobs_lock:
            current_job = _main.jobs.get(ctx.job_id)
            if not current_job:
                target_path.unlink(missing_ok=True)
                raise RuntimeError("封面片头审核样片生成完成时任务已不存在")
            current_job.setdefault("agentPreviewOutputs", []).append(output)
            current_job["lastAgentCoverIntroPreviewFilename"] = target_name
            current_job["coverIntroOperation"] = {
                "status": "completed",
                "filename": target_name,
                "sourceFilename": source_filename,
                "sourceKind": source_kind,
                "completedAt": _main.now_iso(),
            }
            current_job["coverIntroDraft"] = {
                "schemaVersion": "cover-intro-timeline-v1",
                "enabled": True,
                "duration": float(rendered.get("introDuration") or duration),
                "coverVersionId": str(cover_version.get("id") or ""),
                "renderedFilename": target_name,
                "updatedAt": _main.now_iso(),
            }
            current_job.update({
                "status": "completed", "stage": "completed", "progress": 1.0,
                "stageProgress": 1.0, "progressMode": "determinate",
                "detail": "封面片头审核样片已生成",
                "currentAction": "封面片头审核样片已生成",
                "model": "FFmpeg", "error": None, "updatedAt": _main.now_iso(),
            })
            _main.save_job(current_job)
        public_output = {
            **_main.copy.deepcopy(output),
            "videoUrl": f'/api/jobs/{ctx.job_id}/outputs/{target_name}',
            "previewUrl": f'/api/jobs/{ctx.job_id}/outputs/{target_name}',
        }
        return {
            "artifact": {
                "kind": "cover_intro_review_preview",
                "jobId": ctx.job_id,
                "output": public_output,
                "message": "已生成带当前任务封面片头的最终审核样片。",
            },
        }

    future = _main.output_preview_executor.submit(cover_intro_agent_worker)
    return {
        "operationId": f'{ctx.job_id}:cover_intro_preview:{source_filename}:{duration:.1f}',
        "operation": "cover_intro_preview", "accepted": True,
        "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
    }


