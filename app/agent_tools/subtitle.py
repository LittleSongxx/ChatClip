"""Subtitle Agent tool handlers."""
from __future__ import annotations

from typing import Any

from .context import ToolContext


def prepare_subtitle_review(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    session_id = str(ctx.snapshot.get("activeEditSessionId") or "")
    session = next((
        item for item in ctx.snapshot.get("editSessions") or []
        if str(item.get("id") or "") == session_id
    ), None)
    if ctx.autonomous:
        if not session or not session.get("clips"):
            raise RuntimeError("请先建立并应用时间线，再自动生成字幕草稿")
        draft_id = str(session.get("subtitleDraftId") or "")
        if session.get("subtitleEnabled") and draft_id:
            try:
                existing_draft = _main._subtitle_draft_for_job(ctx.snapshot, draft_id)
            except (_main.HTTPException, OSError, ValueError):
                existing_draft = None
            if existing_draft and str(existing_draft.get("status") or "") in {"confirmed", "auto_reviewed"}:
                return {"artifact": {
                    "kind": "subtitle_review_draft", "mode": "reuse_reviewed",
                    "sessionId": session_id, "subtitleDraftId": draft_id,
                    "cueCount": len(existing_draft.get("cues") or []),
                    "autoReviewed": str(existing_draft.get("status") or "") == "auto_reviewed",
                    "message": "已复用当前时间线的字幕稿；审核样片生成后仍可继续修改。",
                }}
        subtitle_style = _main.normalize_subtitle_style(str(arguments.get("style") or "clean"))
        future = _main.output_preview_executor.submit(
            _main.run_agent_auto_subtitle_review, ctx.job_id, session_id, subtitle_style,
        )
        return {
            "operationId": f'{ctx.job_id}:agent_subtitle_review:{session_id}:{_main.uuid.uuid4().hex[:8]}',
            "operation": "agent_subtitle_review",
            "accepted": True,
            "sessionId": session_id,
            "future": future,
            "cancel": lambda: _main.cancel_job(ctx.job_id),
        }
    return {
        "actionRequired": True,
        "action": "subtitle_review",
        "sessionId": session_id or None,
        "message": (
            "请在精剪时间线中建立并确认字幕草稿（文字、断句、说话人标签），再继续生成样片。"
            if session and session.get("clips") else "请先建立并应用精剪时间线，再生成字幕审核稿。"
        ),
    }


def layout_subtitles(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    position = str(arguments.get("position") or "bottom").strip().lower()
    vertical = "top" if position in {"top", "顶部", "上方"} else "middle" if position in {"middle", "center", "居中"} else "bottom"
    style = _main.normalize_subtitle_style(str(arguments.get("style") or "clean"))
    raw_size = arguments.get("fontSizeRatio")
    layout_input: dict[str, Any] = {"preset": style, "vertical": vertical, "horizontal": "center"}
    if raw_size is not None:
        layout_input["fontSizeRatio"] = raw_size
    layout = _main.normalize_subtitle_layout(layout_input, style)
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        session_id = str((current or {}).get("activeEditSessionId") or "")
        session = _main.find_edit_session(current, session_id) if current and session_id else None
        if not current or not session:
            return {"actionRequired": True, "action": "subtitle_review", "message": "请先建立并应用精剪时间线，再进行字幕排版。"}
        draft_id = str(session.get("subtitleDraftId") or "")
        if not bool(session.get("subtitleEnabled")) or not draft_id:
            return {
                "actionRequired": True, "action": "subtitle_review",
                "message": "请先生成并确认当前时间线的字幕草稿，再调整字幕排版。",
            }
        try:
            draft = _main._subtitle_draft_for_job(current, draft_id)
        except (_main.HTTPException, OSError, ValueError):
            return {
                "actionRequired": True, "action": "subtitle_review",
                "message": "当前字幕草稿不可用，请重新生成并确认后再调整排版。",
            }
        if str(draft.get("status") or "") not in {"confirmed", "auto_reviewed"}:
            return {
                "actionRequired": True, "action": "subtitle_review",
                "message": "字幕文字与断句尚未确认；请先完成字幕审核，再调整排版。",
            }
        cues = [
            _main.copy.deepcopy(cue) for cue in draft.get("cues") or []
            if isinstance(cue, dict) and str(cue.get("text") or "").strip()
        ]
        if not cues:
            return {"actionRequired": True, "action": "subtitle_review", "message": "当前字幕草稿没有有效文字，请先补全并确认字幕。"}
        draft["globalStyle"] = layout
        draft["revision"] = int(draft.get("revision") or 0) + 1
        draft["updatedAt"] = _main.now_iso()
        _main.save_subtitle_draft_file(str(current.get("workDirectory") or ""), draft)
        session.update({
            "subtitleEnabled": True, "subtitleDraftId": draft_id,
            "subtitleStyle": style, "agentSubtitleLayout": layout,
            "revision": int(session.get("revision") or 0) + 1,
            "renderedVersionId": None, "updatedAt": _main.now_iso(),
        })
        current["updatedAt"] = _main.now_iso()
        _main.save_job(current)
    return {
        "artifact": {
            "kind": "subtitle_layout_applied", "jobId": ctx.job_id,
            "sessionId": session_id, "subtitleDraftId": draft_id,
            "layout": layout, "cueCount": len(cues),
            "message": (
                f"已在自动校对稿上应用 {len(cues)} 条字幕的{vertical}部布局；成片阶段仍可修改。"
                if str(draft.get("status") or "") == "auto_reviewed" else
                f"已在确认稿上应用 {len(cues)} 条字幕的{vertical}部布局。"
            ),
        },
    }


def export_subtitles(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    filename = str(arguments.get("filename") or "").strip()
    fmt = str(arguments.get("format") or "srt").lower()
    if fmt not in {"srt", "vtt"}:
        raise RuntimeError("字幕格式仅支持 srt 或 vtt")
    output, _source_path = ctx.current_output_reference(filename)
    if not output or not output.get("filename"):
        return {"actionRequired": True, "action": "select_output", "message": "当前没有可导出字幕的成片或审核样片。"}
    output_filename = str(output.get("filename") or "")
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        if not current:
            raise RuntimeError("任务不存在")
        context = _main.output_download_context(current, output_filename)
        if not context:
            return ctx.no_result("missing_output", "当前成片不存在，无法导出字幕。", artifact_kind="subtitle_export")
        checked_output, version, position = context
        frozen_cues = checked_output.get("subtitleCues") if isinstance(checked_output.get("subtitleCues"), list) else None
        draft_id = str(checked_output.get("subtitleDraftId") or version.get("subtitleDraftId") or "")
        draft = _main._subtitle_draft_for_job(current, draft_id) if draft_id and frozen_cues is None else None
        cues = (
            _main.copy.deepcopy(frozen_cues) if frozen_cues is not None else ([
                _main.copy.deepcopy(cue) for cue in (draft.get("cues") or [])
                if int(cue.get("outputIndex") or 0) == position - 1 and str(cue.get("text") or "").strip()
            ] if draft else _main._subtitle_cues(current, checked_output))
        )
    if not cues:
        return ctx.no_result("missing_subtitles", "当前成片没有可用对白字幕，无法导出字幕文件。", artifact_kind="subtitle_export")
    download_url = f'/api/jobs/{ctx.job_id}/outputs/{_main.quote(output_filename)}/subtitles?format={fmt}'
    return {
        "artifact": {
            "kind": "subtitle_export",
            "jobId": ctx.job_id,
            "filename": output_filename,
            "format": fmt,
            "downloadUrl": download_url,
            "cueCount": len(cues),
            "message": f"已准备 {fmt.upper()} 字幕下载链接；没有生成或修改视频。",
        },
    }


