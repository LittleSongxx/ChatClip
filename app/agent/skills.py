"""Skill profiles: the executable workflow registry behind SKILL.md files.

A Skill supplies editorial policy; this registry owns executable workflow
shape. Generated Skills fall back to their explicit allow-list when their
declared ``workflow-profile`` fully covers the profile's required tools.
"""

from __future__ import annotations

import re
from typing import Any, Callable



# A Skill supplies editorial policy; this registry owns executable workflow
# shape. Generated Skills fall back to their explicit allow-list, while trusted
# Plugins can later register profiles through the same contract.
SKILL_PROFILES: dict[str, dict[str, Any]] = {
    "chatclip-highlight-director": {"kind": "highlight", "tools": {"inspect_workspace", "analyze_highlights", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-content-extractor": {"kind": "content", "tools": {"inspect_workspace", "search_content", "review_content_evidence", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-interview-editor": {"kind": "interview", "tools": {"inspect_workspace", "discover_speakers", "select_speakers", "search_content", "review_content_evidence", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-person-editor": {"kind": "person", "tools": {"inspect_workspace", "discover_people", "select_people", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-speaker-editor": {"kind": "speaker", "tools": {"inspect_workspace", "discover_speakers", "select_speakers", "search_content", "review_content_evidence", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-revision-editor": {"kind": "revision", "tools": {"inspect_workspace", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-shortform-hook-director": {"kind": "shortform", "tools": {"inspect_workspace", "analyze_highlights", "search_content", "review_content_evidence", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-delivery-qc": {"kind": "delivery-qc", "tools": {"inspect_workspace", "run_delivery_qc"}},
    "chatclip-social-reframe-exporter": {"kind": "social-reframe", "tools": {"inspect_workspace", "render_social_preview", "run_delivery_qc"}},
    "chatclip-smart-reframe": {"kind": "social-reframe", "tools": {"inspect_workspace", "render_social_preview", "run_delivery_qc"}},
    "chatclip-cover-director": {"kind": "cover", "tools": {"inspect_workspace", "propose_cover_candidates", "render_cover_variants", "review_cover_variants", "confirm_cover"}},
    "chatclip-subtitle-editor": {"kind": "revision", "tools": {"inspect_workspace", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview"}},
    "chatclip-source-provenance-guard": {"kind": "source-provenance", "tools": {"inspect_workspace", "validate_task_provenance"}},
    "chatclip-multi-topic-assembler": {"kind": "multi-topic", "tools": {"inspect_workspace", "search_content", "select_multi_topic_evidence", "propose_timeline_edit", "confirm_timeline_edit", "prepare_subtitle_review", "render_review_preview", "render_social_preview", "run_delivery_qc"}},
    "chatclip-cover-intro-composer": {"kind": "cover-intro", "tools": {"inspect_workspace", "propose_cover_candidates", "render_cover_variants", "review_cover_variants", "confirm_cover", "compose_cover_intro", "run_delivery_qc"}},
    "chatclip-dynamic-reframe-director": {"kind": "dynamic-reframe", "tools": {"inspect_workspace", "analyze_reframe_safe_areas", "render_social_preview", "run_delivery_qc"}},
    "chatclip-caption-layout-director": {"kind": "caption-layout", "tools": {"inspect_workspace", "layout_subtitles", "prepare_subtitle_review", "render_review_preview", "export_subtitles"}},
    "chatclip-audio-polish-mixer": {"kind": "audio-polish", "tools": {"inspect_workspace", "polish_audio_mix", "run_delivery_qc"}},
    "chatclip-broll-overlay-editor": {"kind": "broll-overlay", "tools": {"inspect_workspace", "search_content", "review_content_evidence", "propose_broll_overlay", "confirm_timeline_edit", "render_review_preview"}},
    "chatclip-graphics-packager": {"kind": "graphics-package", "tools": {"inspect_workspace", "render_graphics_package", "render_review_preview"}},
    "chatclip-local-motion-renderer": {"kind": "local-motion", "tools": {"inspect_workspace", "render_motion_graphics", "compose_motion_intro", "run_delivery_qc"}},
    "chatclip-local-draft-exporter": {"kind": "local-draft", "tools": {"inspect_workspace", "export_editing_draft"}},
    "chatclip-platform-delivery-exporter": {"kind": "platform-delivery", "tools": {"inspect_workspace", "export_delivery_master", "run_delivery_qc"}},
    "chatclip-edit-diagnostics": {"kind": "edit-diagnostics", "tools": {"inspect_workspace", "diagnose_edit_failure"}},
}
WORKFLOW_PROFILE_KINDS = frozenset(profile["kind"] for profile in SKILL_PROFILES.values())


def profile_for_skill(skill: dict[str, Any]) -> dict[str, Any]:
    known = SKILL_PROFILES.get(str(skill.get("id") or ""))
    if known:
        return {"kind": known["kind"], "tools": set(known["tools"]), "managed": True}
    allowed = {str(item) for item in skill.get("allowedTools") or [] if str(item)}
    requested_kind = str(skill.get("workflowProfile") or "").strip().lower()
    requested = next((value for value in SKILL_PROFILES.values() if value["kind"] == requested_kind), None)
    if requested and set(requested["tools"]).issubset(allowed):
        return {"kind": requested["kind"], "tools": set(requested["tools"]), "managed": True}
    return {"kind": "custom", "tools": allowed, "managed": False}


def supports_autonomous_review(skills: list[dict[str, Any]]) -> bool:
    """Only first-party workflow Skills may silently cross review gates."""
    return bool(skills) and all(
        str(skill.get("id") or "") in SKILL_PROFILES
        and str(skill.get("source") or "") in {"builtin", "test"}
        and not skill.get("pluginId")
        for skill in skills
    )


def skill_chain_payload(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": item["id"], "version": item.get("version") or "1.0.0",
        "contentHash": item.get("contentHash") or "",
        "role": "primary" if index == 0 else "addon",
        "workflowProfile": item.get("workflowProfile") or "",
    } for index, item in enumerate(skills)]


def local_motion_requires_current_output(brief: dict[str, Any]) -> bool:
    """Only require an output when the user explicitly asks to modify one.

    A title card or motion intro can be rendered as a standalone preview
    from a source task.  Merely using the word “片头” must not turn that
    creation request into an existing-output edit.
    """
    goal = str(brief.get("goal") or "")
    return bool(re.search(
        r"(?:合入|并入|接入|追加到|添加到|加到).{0,12}(?:当前|已有|现有)?成片|"
        r"(?:当前|已有|现有)成片.{0,16}(?:合入|并入|追加|添加|加上|片头)",
        goal,
    ))


def skill_routing_eligibility(
    skill: dict[str, Any], *, brief: dict[str, Any], context: dict[str, Any],
) -> tuple[bool, str]:
    """Reject a format-conversion Skill when the user still needs a source edit.

    A platform name or aspect ratio is delivery metadata.  It must never
    displace an explicit source-content query such as “find X and make a
    cut”.  Social reframing also has a hard media precondition: an
    existing accepted output to use as its source.
    """
    profile = profile_for_skill(skill)
    kind = str(profile.get("kind") or "")
    editing = context.get("editing") if isinstance(context.get("editing"), dict) else {}
    has_outputs = bool(editing.get("hasOutputs"))
    source_edit_requested = str(brief.get("delivery") or "") == "timeline" and bool(
        brief.get("retrievalQuery") or brief.get("sourceRange") or brief.get("removedSourceRanges")
    )
    if kind in {"social-reframe", "dynamic-reframe"} and source_edit_requested:
        return False, "请求仍需从源片检索并组合内容，不能直接改画幅"
    if kind in {"social-reframe", "dynamic-reframe"} and not has_outputs:
        return False, "画幅转换需要当前任务已有可审核成片；不能拿源视频或其他任务输出代替"
    if kind == "audio-polish" and not has_outputs:
        return False, "音频优化需要当前任务已有可试听成片；不能拿源视频或其他任务输出代替"
    if kind == "platform-delivery" and not has_outputs:
        return False, "正式交付需要当前任务已有审核通过的成片；不能导出源视频或其他任务输出"
    if kind == "cover-intro" and not has_outputs:
        return False, "封面片头需要当前任务已有成片；不能先生成无归属封面或复用其他任务封面"
    if kind == "local-motion" and local_motion_requires_current_output(brief) and not has_outputs:
        return False, "动态图文片头合成需要当前任务已有成片；不能先生成无法合入的独立动效"
    if kind == "delivery-qc" and source_edit_requested:
        return False, "请求仍需先生成内容剪辑，不能只运行交付质检"
    if kind == "delivery-qc" and not has_outputs:
        return False, "交付质检需要已有成片"
    if kind == "local-draft":
        return True, ""
    has_timeline = bool(editing.get("hasActiveSession"))
    if (
        (kind == "caption-layout" or str(skill.get("id") or "") == "chatclip-subtitle-editor")
        and bool(brief.get("subtitleAssetRequested"))
        and not has_outputs
    ):
        return False, "字幕文件导出需要当前任务已有带字幕的成片或审核样片"
    if (
        (kind == "caption-layout" or str(skill.get("id") or "") == "chatclip-subtitle-editor")
        and not bool(brief.get("subtitleAssetRequested"))
        and not has_timeline
    ):
        return False, "该编辑需要当前任务已有已确认时间线；不能重新分析源视频或复用其他任务的编辑结果"
    if (
        kind in {"graphics-package", "broll-overlay"}
        and not has_timeline
    ):
        return False, "该编辑需要当前任务已有已确认时间线；不能重新分析源视频或复用其他任务的编辑结果"
    return True, ""


def skill_precondition_descriptor(
    skill: dict[str, Any], *, brief: dict[str, Any], context: dict[str, Any],
) -> dict[str, str] | None:
    eligible, reason = skill_routing_eligibility(skill, brief=brief, context=context)
    if eligible:
        return None
    kind = str(profile_for_skill(skill).get("kind") or "")
    if (kind == "caption-layout" or str(skill.get("id") or "") == "chatclip-subtitle-editor") and bool(brief.get("subtitleAssetRequested")):
        required_state, code = "current_output", "missing_current_output"
    elif kind in {"caption-layout", "graphics-package", "broll-overlay"} or str(skill.get("id") or "") == "chatclip-subtitle-editor":
        required_state, code = "active_timeline", "missing_current_timeline"
    elif kind in {"social-reframe", "dynamic-reframe", "audio-polish", "platform-delivery", "cover-intro", "delivery-qc"} or (
        kind == "local-motion" and local_motion_requires_current_output(brief)
    ):
        required_state, code = "current_output", "missing_current_output"
    else:
        required_state, code = "compatible_request", "incompatible_skill_goal"
    return {
        "requiredState": required_state,
        "preconditionCode": code,
        "preconditionMessage": reason,
    }


SkillLookup = Callable[[str], dict[str, Any] | None]


def compose_skills(
    primary: dict[str, Any], *, brief: dict[str, Any], context: dict[str, Any],
    enabled_skill_for_kind: SkillLookup,
) -> list[dict[str, Any]]:
    """Build one auditable Skill chain without letting add-ons steal the edit goal."""
    skills = [primary]
    kinds = {str(profile_for_skill(primary).get("kind") or "custom")}

    def append_kind(kind: str, *, required: bool = True) -> None:
        if kind in kinds:
            return
        skill = enabled_skill_for_kind(kind)
        if skill is None:
            if required:
                raise ValueError(f"当前缺少已启用的 {kind} Skill，无法完整覆盖用户要求")
            return
        skills.append(skill)
        kinds.add(kind)

    def append_reframe_kind(*, required: bool = True) -> None:
        if "social-reframe" in kinds or "dynamic-reframe" in kinds:
            return
        for candidate_kind in ("social-reframe", "dynamic-reframe"):
            skill = enabled_skill_for_kind(candidate_kind)
            if skill is not None:
                skills.append(skill)
                kinds.add(candidate_kind)
                return
        if required:
            raise ValueError("当前缺少已启用的 social-reframe 或 dynamic-reframe Skill，无法完整覆盖用户要求")

    primary_kind = next(iter(kinds))
    # Audit and diagnostic Skills mention many artifact names while
    # describing what they inspect. Those nouns are not requests to create
    # a cover, export, reframe, or edit, so their workflow must stay read-only.
    if primary_kind in {"source-provenance", "edit-diagnostics", "delivery-qc"}:
        return skills
    if primary_kind == "revision" and brief.get("retrievalQuery"):
        append_kind("content")
    if (
        bool(brief.get("socialDelivery", {}).get("requested"))
        and primary_kind not in {"social-reframe", "dynamic-reframe", "local-motion"}
    ):
        append_reframe_kind()
    if bool(brief.get("coverIntroRequested")) and "cover-intro" not in kinds:
        append_kind("cover-intro", required=False)
    explicit_caption_position = bool(re.search(
        r"字幕.{0,16}(?:顶部|上方|顶端|底部|下方|底端)|(?:顶部|上方|顶端|底部|下方|底端).{0,16}字幕",
        str(brief.get("goal") or ""),
    ))
    if (
        bool(brief.get("subtitleRequested"))
        and (primary_kind == "revision" or explicit_caption_position)
        and "caption-layout" not in kinds
    ):
        append_kind("caption-layout", required=False)
    if bool(brief.get("audioPolishRequested")) and "audio-polish" not in kinds:
        append_kind("audio-polish", required=False)
    if bool(brief.get("brollRequested")) and "broll-overlay" not in kinds:
        append_kind("broll-overlay", required=False)
    if bool(brief.get("graphicsRequested")) and "graphics-package" not in kinds:
        append_kind("graphics-package", required=False)
    if (
        bool(brief.get("deliveryQcRequested"))
        and "delivery-qc" not in kinds
        and "social-reframe" not in kinds
        and "dynamic-reframe" not in kinds
    ):
        append_kind("delivery-qc")
    if bool(brief.get("coverRequested")) and "cover" not in kinds and primary_kind != "cover-intro":
        append_kind("cover")
    if bool(brief.get("deliveryExportRequested")) and "platform-delivery" not in kinds:
        append_kind("platform-delivery", required=False)
    return skills
