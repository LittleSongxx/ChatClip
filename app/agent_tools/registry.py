"""Registry mapping Agent tool names to their extracted handlers."""
from __future__ import annotations

from typing import Any, Callable

from . import content, cover, delivery, subtitle, timeline
from .context import ToolContext

TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], dict[str, Any]]] = {
    "inspect_workspace": content.inspect_workspace,
    "validate_task_provenance": content.validate_task_provenance,
    "diagnose_edit_failure": content.diagnose_edit_failure,
    "select_multi_topic_evidence": content.select_multi_topic_evidence,
    "propose_cover_candidates": cover.propose_cover_candidates,
    "render_cover_variants": cover.render_cover_variants,
    "review_cover_variants": cover.review_cover_variants,
    "confirm_cover": cover.confirm_cover,
    "cancel_operation": timeline.cancel_operation,
    "select_people": content.select_identity,
    "select_speakers": content.select_identity,
    "review_content_evidence": content.review_content_evidence,
    "compose_cover_intro": cover.compose_cover_intro,
    "propose_timeline_edit": timeline.propose_timeline_edit,
    "confirm_timeline_edit": timeline.confirm_timeline_edit,
    "prepare_subtitle_review": subtitle.prepare_subtitle_review,
    "render_review_preview": delivery.render_review_preview,
    "analyze_reframe_safe_areas": delivery.analyze_reframe_safe_areas,
    "layout_subtitles": subtitle.layout_subtitles,
    "propose_broll_overlay": delivery.propose_broll_overlay,
    "render_graphics_package": delivery.render_graphics_package,
    "render_motion_graphics": delivery.render_motion_graphics,
    "export_editing_draft": delivery.export_editing_draft,
    "compose_motion_intro": delivery.compose_motion_intro,
    "render_social_preview": delivery.render_social_preview,
    "polish_audio_mix": delivery.polish_audio_mix,
    "export_subtitles": subtitle.export_subtitles,
    "export_delivery_master": delivery.export_delivery_master,
    "run_delivery_qc": delivery.run_delivery_qc,
    "analyze_highlights": content.content_workflow,
    "search_content": content.content_workflow,
    "discover_people": content.content_workflow,
    "discover_speakers": content.content_workflow,
}


def dispatch(ctx: ToolContext, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise RuntimeError(f"未注册的 Agent 核心工具：{tool_name}")
    return handler(ctx, arguments)
