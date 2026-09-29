"""Deterministic plan compiler and validators.

For managed (built-in) Skills the compiler — not the model — owns workflow
shape: step skeletons, confirmation ordering and safety constraints are
compiled from the brief and planning context, while the planning model's
prose survives only as ``summary``/``strategy`` metadata. Custom Skills keep
their LLM-authored step DAG but pass through the same validators.
Ported verbatim from the pre-LangGraph ``AgentPlatform``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from typing import Any

from .brief import editing_brief
from .catalog import (
    AUTONOMOUS_REVIEW,
    STEPWISE_REVIEW,
    VALID_EXECUTION_MODES,
    VALID_SIDE_EFFECTS,
    tool_catalog,
    tool_catalog_fingerprint,
)
from .prompts import PROMPT_VERSION
from .skills import (
    profile_for_skill,
    skill_chain_payload,
    skill_precondition_descriptor,
    supports_autonomous_review,
)


# Conservative planning-time estimates (seconds) used by the replan budget
# check; a replacement plan that grows beyond +25% of these forces
# re-approval instead of silently continuing as a minor revision.
TOOL_ESTIMATED_SECONDS: dict[str, int] = {
    "inspect_workspace": 2, "analyze_highlights": 300, "search_content": 240,
    "review_content_evidence": 60, "discover_people": 600, "select_people": 30,
    "discover_speakers": 300, "select_speakers": 30, "select_multi_topic_evidence": 60,
    "propose_timeline_edit": 180, "confirm_timeline_edit": 15,
    "prepare_subtitle_review": 240, "layout_subtitles": 15,
    "render_review_preview": 240, "render_social_preview": 180,
    "propose_cover_candidates": 120, "render_cover_variants": 180,
    "review_cover_variants": 30, "confirm_cover": 10, "compose_cover_intro": 120,
    "analyze_reframe_safe_areas": 60, "polish_audio_mix": 240,
    "propose_broll_overlay": 120, "render_graphics_package": 60,
    "render_motion_graphics": 240, "compose_motion_intro": 180,
    "export_subtitles": 15, "export_editing_draft": 15, "export_delivery_master": 240,
    "run_delivery_qc": 90, "validate_task_provenance": 5,
    "diagnose_edit_failure": 15, "cancel_operation": 5,
}


def replan_force_replay_tools(failed_step: dict[str, Any]) -> set[str]:
    """Invalidate an upstream artifact known to have caused the failure."""
    failed_tool = str(failed_step.get("tool") or "")
    if failed_tool in {
        "analyze_highlights",
        "search_content",
        "discover_people",
        "discover_speakers",
        "propose_timeline_edit",
        "render_review_preview",
        "run_delivery_qc",
        # Cover activation is idempotent: retrying it reuses the selected
        # content hash/current version.  This also repairs legacy plans
        # where the cover was saved before an unrelated hidden export
        # failed and incorrectly marked this step as failed.
        "confirm_cover",
    }:
        return {failed_tool}
    if (
        failed_tool == "review_content_evidence"
        and any(
            message in str(failed_step.get("error") or "")
            for message in (
                "没有生成可用于自动编排的有效候选",
                "自动编排缺少必要类别的候选",
            )
        )
    ):
        return {"search_content"}
    return set()


def automatic_replan_block_reason(failed_step: dict[str, Any]) -> str:
    """Return why rebuilding the same DAG cannot repair this failure."""
    error = str(failed_step.get("error") or "")
    if (
        str(failed_step.get("tool") or "") == "review_content_evidence"
        and "自动编排缺少必要类别的候选" in error
    ):
        return "missing_required_evidence_category"
    return ""


def compile_profile_plan(
    raw: dict[str, Any], *, skill: dict[str, Any], goal: str, context: dict[str, Any],
    skills: list[dict[str, Any]] | None = None,
    execution_mode: str = AUTONOMOUS_REVIEW,
) -> dict[str, Any]:
    """Compile a stable plan skeleton from facts; keep LLM prose as metadata.

    This removes conditional workflow choice from the model for built-in
    Skills. Custom Skills remain constrained by their declared tool set.
    """
    result = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    strategy = result.get("strategy") if isinstance(result.get("strategy"), dict) else {}
    selected_skills = skills or [skill]
    profile = profile_for_skill(skill)
    mode = (
        str(execution_mode) if str(execution_mode) in VALID_EXECUTION_MODES
        else AUTONOMOUS_REVIEW
    )
    if not supports_autonomous_review(selected_skills):
        mode = STEPWISE_REVIEW
    profile_kinds = {str(profile_for_skill(item).get("kind") or "custom") for item in selected_skills}
    # A standalone cover task ends in an explicit creative choice. Keep
    # that task in review mode even when cover generation is an automatic
    # add-on to a larger editing workflow.
    if str(profile.get("kind") or "") == "cover":
        mode = STEPWISE_REVIEW
    brief = editing_brief(goal, context)
    # Plans created before the duration-inference fix may still carry the
    # legacy 30s short-form default. Content retrieval must never inherit
    # that implicit cap unless the user explicitly requested a duration.
    if (not brief.get("durationExplicit")
            and brief.get("specificContentTarget")
            and brief.get("targetSeconds") is not None):
        brief["targetSeconds"] = None
    result["brief"] = brief
    result["planningContext"] = context
    result["profile"] = profile["kind"]
    result["executionMode"] = mode
    result["skillChain"] = skill_chain_payload(selected_skills)
    if not profile["managed"]:
        return result

    tool_map = {item["name"]: item for item in tool_catalog()}
    steps: list[dict[str, Any]] = []
    decision: list[dict[str, Any]] = []

    def add(tool: str, title: str, arguments: dict[str, Any], output: str, dependencies: list[str] | None = None) -> str:
        referenced_output = (context.get("inputContext") or {}).get("outputFilename")
        if referenced_output and tool == "render_social_preview" and not any(s["tool"] == "render_review_preview" for s in steps):
            arguments = {**arguments, "filename": referenced_output}
        source_range = brief.get("sourceRange")
        if tool in {"analyze_highlights", "search_content", "discover_people", "discover_speakers"} and source_range:
            if source_range.get("requiresDuration"):
                raise ValueError("尚未读取素材时长，无法确定末尾范围，请在素材就绪后重试")
            arguments = {**arguments, "sourceScopeKind": "custom", "sourceScopeStart": source_range["start"], "sourceScopeEnd": source_range["end"]}
        step_id = f"step_{len(steps) + 1}_{tool}"
        steps.append({
            "id": step_id, "title": title, "tool": tool, "arguments": arguments,
            "dependencies": list(dependencies) if dependencies is not None else ([steps[-1]["id"]] if steps else []), "expectedOutput": output,
            "sideEffect": tool_map[tool]["sideEffect"],
            "estimatedSeconds": TOOL_ESTIMATED_SECONDS.get(tool, 60), "optional": False,
        })
        return step_id

    def qc_arguments() -> dict[str, Any]:
        arguments: dict[str, Any] = {"strict": True}
        delivery_aspect = (brief.get("socialDelivery") or {}).get("aspect")
        expected_aspect = (
            delivery_aspect
            if (brief.get("socialDelivery") or {}).get("requested") else
            brief.get("coverAspect") if brief.get("coverIntroRequested") else ""
        )
        if expected_aspect in {"9:16", "16:9", "1:1", "4:5"}:
            arguments["expectedAspect"] = expected_aspect
        if brief.get("coverRequested"):
            arguments.update({"requireCover": True, "expectedCoverTitle": str(brief.get("coverTitle") or "")})
        if brief.get("coverIntroRequested"):
            arguments["requireCoverIntro"] = True
        if isinstance(brief.get("targetSeconds"), (int, float)):
            arguments["targetSeconds"] = float(brief["targetSeconds"])
            arguments["toleranceSeconds"] = float(
                brief.get("durationToleranceSeconds") or 0
            )
        return arguments

    def cover_candidate_source_scope() -> str:
        editing_context = context.get("editing") if isinstance(context.get("editing"), dict) else {}
        has_adopted_source = bool(
            editing_context.get("hasOutputs")
            or editing_context.get("hasActiveSession")
            or context.get("currentOutputVersionId")
        )
        if not has_adopted_source:
            return "source_video"
        if brief.get("coverSourceTime") is not None and re.search(
            r"源视频|原视频|本次视频|素材|第\s*\d|(?:\d+(?:\.\d+)?)\s*(?:秒钟|秒|s)",
            str(brief.get("goal") or ""),
            re.IGNORECASE,
        ):
            return "source_video"
        return "accepted_cut"

    kind = str(profile["kind"])
    precondition = skill_precondition_descriptor(
        skill, brief=brief, context=context,
    )
    if precondition:
        add(
            "inspect_workspace", "检查所选编辑能力的前置条件",
            precondition,
            "当前任务前置条件状态；缺失时返回结构化说明并停止后续操作",
        )
        reason = precondition["preconditionMessage"]
        result["summary"] = f"先检查当前任务是否满足所选编辑能力的前置条件。{reason}。不会分析源视频、复用其他任务结果或启动渲染。"
        result["steps"] = steps
        result["decisionRecord"] = [{
            "rule": "skill_precondition",
            "outcome": "stop_with_clear_limitation",
            "reason": reason,
        }]
        return result
    if bool(brief.get("coverRequested")) and str(brief.get("coverSourceKind") or "") == "external_image":
        subject = str(brief.get("coverSubject") or "指定人物").strip()
        reason = (
            f"封面要求使用“{subject}”的外部图片，但当前版本只能从本次视频中抽帧制作封面，"
            "尚不支持联网找图或导入独立封面图片"
        )
        add(
            "inspect_workspace", "等待外部封面素材",
            {
                "requiredState": "external_cover_asset",
                "preconditionCode": "missing_external_cover_asset",
                "preconditionMessage": reason,
            },
            "获得明确来源的封面图片后才可继续，不会用源视频画面替代",
        )
        result["summary"] = f"{reason}。请修改要求为使用本次视频画面；当前不会开始检索、剪辑或生成错误封面。"
        result["planningWarning"] = reason
        result["steps"] = steps
        result["decisionRecord"] = [{
            "rule": "external_cover_capability",
            "outcome": "stop_without_silent_fallback",
            "reason": reason,
        }]
        return result
    add("inspect_workspace", "核查可复用素材与分析结果", {}, "素材、转写、证据和已有编辑状态")
    speaker = context.get("speaker") if isinstance(context.get("speaker"), dict) else {}
    people = context.get("people") if isinstance(context.get("people"), dict) else {}
    has_candidates = bool((context.get("evidence") or {}).get("hasCandidates")) if isinstance(context.get("evidence"), dict) else False
    # The deterministic brief owns the evidence query.  A planning model
    # may describe the strategy, but it must not expand a concise target
    # into a long list of guessed objects/actions that makes retrieval both
    # slower and less precise.
    explicit_retrieval_query = str(
        brief.get("retrievalQuery") or strategy.get("searchQuery") or ""
    ).strip()
    retrieval_query = explicit_retrieval_query or str(brief["goal"]).strip()
    if brief.get("excludedContent"):
        retrieval_query += "；不要" + "、".join(brief["excludedContent"])
    needs_timeline = (
        str(brief.get("delivery") or "timeline") == "timeline"
        and kind not in {
            "delivery-qc", "social-reframe", "cover", "source-provenance",
            "dynamic-reframe", "cover-intro", "caption-layout", "audio-polish",
            "broll-overlay", "graphics-package", "local-motion", "local-draft",
            "platform-delivery", "edit-diagnostics",
        }
    )
    requested_variants = max(1, min(4, int(brief.get("variantCount") or 1)))
    timeline_step_id: str | None = None
    base_delivery_step_id: str | None = None

    if kind == "source-provenance":
        add("validate_task_provenance", "校验当前任务素材边界", {"strict": True}, "当前任务、源素材、范围和可复用产物的归属报告")
        decision.append({"rule": "task_provenance", "outcome": "validate", "reason": "目标要求核对任务隔离或防止旧状态复用"})
    elif kind == "edit-diagnostics":
        add("diagnose_edit_failure", "诊断当前剪辑流程问题", {"focus": brief["goal"][:240]}, "失败原因、受影响产物和可执行下一步")
        decision.append({"rule": "diagnostics", "outcome": "inspect", "reason": "目标要求解释失败、卡住或界面结果异常"})
    elif kind == "local-motion":
        editing = context.get("editing") if isinstance(context.get("editing"), dict) else {}
        social = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
        aspect = str(social.get("aspect") or brief.get("coverAspect") or "9:16")
        duration = max(0.5, min(8.0, float(brief.get("targetSeconds") or 1.5)))
        title_text = str(brief.get("coverTitle") or brief.get("retrievalQuery") or brief.get("goal") or "")[:120]
        motion_step = add(
            "render_motion_graphics",
            "生成本地图文动效视频",
            {
                "title": title_text,
                "subtitle": str(brief.get("goal") or "")[:180],
                "aspect": aspect,
                "duration": duration,
                "theme": "chatclip",
            },
            "由本地 HTML/Playwright/FFmpeg 管线生成的动效视频",
        )
        qc_dependency = motion_step
        if bool(brief.get("motionIntroRequested")) and bool(editing.get("hasOutputs")):
            qc_dependency = add(
                "compose_motion_intro",
                "合成动态图文片头成片",
                {},
                "带本地图文动效片头的新成片版本",
                [motion_step],
            )
        add("run_delivery_qc", "检查动效视频质量", qc_arguments(), "动效视频质检报告", [qc_dependency])
        decision.append({"rule": "local_motion", "outcome": "html_video", "reason": "目标要求本地图文动效、标题卡或片头"})
    elif kind == "local-draft":
        add(
            "export_editing_draft",
            "导出本地剪辑草稿包",
            {"format": "chatclip-json"},
            "包含当前时间线、输出版本和封面引用的本地草稿包",
        )
        decision.append({"rule": "local_draft", "outcome": "chatclip_json", "reason": "目标要求导出草稿或外部编辑器工程"})
    elif kind == "dynamic-reframe":
        delivery = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
        aspect = str(delivery.get("aspect") or "9:16")
        fit = str(delivery.get("fit") or "blur")
        safe_step = add("analyze_reframe_safe_areas", "分析画幅安全区", {"aspect": aspect, "fit": fit}, "人物、字幕、文字和主体保护策略")
        social_step = add(
            "render_social_preview", f"生成 {aspect} 画幅审核预览",
            {"aspect": aspect, "fit": fit, "focusX": float(delivery.get("focusX", .5)), "focusY": float(delivery.get("focusY", .5))},
            f"{aspect} 审核预览", [safe_step],
        )
        base_delivery_step_id = social_step
        decision.append({"rule": "dynamic_reframe", "outcome": fit, "reason": f"按当前成片生成 {aspect} 画幅预览"})
    elif kind == "cover-intro":
        aspect = str(brief.get("coverAspect") or "16:9")
        title_text = str(brief.get("coverTitle") or "")[:80]
        candidate_arguments: dict[str, Any] = {
            "sourceScope": cover_candidate_source_scope(), "candidateBudget": 16,
            "aspectRatios": [aspect], "focus": str(brief.get("goal") or "")[:240],
        }
        if brief.get("coverSubject"):
            candidate_arguments["subject"] = str(brief["coverSubject"])[:120]
        if brief.get("coverSourceTime") is not None:
            candidate_arguments["sourceTime"] = float(brief["coverSourceTime"])
        if title_text:
            candidate_arguments["titleText"] = title_text
        cover_candidate_step = add("propose_cover_candidates", "提取并评分封面候选", candidate_arguments, "当前任务内可追溯封面候选")
        render_arguments: dict[str, Any] = {
            "aspectRatios": [aspect],
            "directions": ["source_clean", "source_editorial", "source_cinematic"],
        }
        if title_text:
            render_arguments["titleText"] = title_text
        cover_variant_step = add("render_cover_variants", "生成封面预览版本", render_arguments, "不覆盖当前封面的封面预览", [cover_candidate_step])
        cover_review_step = add("review_cover_variants", "选择当前任务封面", {}, "已选择的当前任务封面", [cover_variant_step])
        cover_confirm_step = add("confirm_cover", "保存当前任务封面", {}, "已绑定当前任务的封面版本", [cover_review_step])
        intro_step = add(
            "compose_cover_intro", "合成封面片头",
            {"duration": float(brief.get("coverIntroDurationSeconds") or 1.5)},
            "带当前封面片头的新成片版本", [cover_confirm_step],
        )
        add("run_delivery_qc", "检查封面片头成片质量", qc_arguments(), "封面片头成片质检报告", [intro_step])
        decision.append({"rule": "cover_intro", "outcome": "compose", "reason": "目标要求封面出现在视频开头"})
    elif kind == "caption-layout":
        if bool(brief.get("subtitleAssetRequested")):
            add(
                "export_subtitles",
                "导出字幕文件",
                {
                    "format": "vtt" if re.search(r"\bVTT\b|WebVTT", str(brief.get("goal") or ""), re.IGNORECASE) else "srt",
                },
                "当前成片或审核样片对应的字幕文件；不生成视频",
            )
            decision.append({
                "rule": "subtitle_asset_export",
                "outcome": "export_subtitles",
                "reason": "目标明确要求字幕文件而不是视频成片",
            })
            result["summary"] = "从当前成片或审核样片导出字幕文件；不生成、不修改视频。"
            result["steps"] = steps
            result["decisionRecord"] = decision
            return result
        position = "top" if re.search(r"顶部|上方|顶端", str(brief.get("goal") or "")) else "bottom"
        subtitle_step = add(
            "prepare_subtitle_review",
            "自动生成并校对字幕草稿" if mode == AUTONOMOUS_REVIEW else "生成字幕校对稿",
            {
                "style": "clean",
                "requireConfirmedDraft": mode != AUTONOMOUS_REVIEW,
                "autoReview": mode == AUTONOMOUS_REVIEW,
            },
            (
                "与当前时间线对应、可在成片阶段继续修改的自动校对字幕草稿"
                if mode == AUTONOMOUS_REVIEW else
                "与当前时间线对应、待人工校对的字幕草稿"
            ),
        )
        layout_step = add(
            "layout_subtitles", "应用字幕排版策略",
            {"position": position, "style": "clean"},
            "在已确认字幕草稿上应用布局，不改动字幕文字",
            [subtitle_step],
        )
        add("render_review_preview", "生成带字幕最终审核样片", {"subtitleMode": "burned_in_review_watermarked_low_bitrate"}, "带已确认字幕与布局的最终审核样片", [layout_step])
        decision.append({
            "rule": "caption_layout", "outcome": position,
            "reason": (
                "自动生成并校对字幕草稿，应用排版后直接生成带字幕审核样片；成片阶段仍可修改"
                if mode == AUTONOMOUS_REVIEW else
                "生成字幕校对稿，经人工确认后应用排版并生成带字幕样片"
            ),
        })
    elif kind == "audio-polish":
        polish_step = add(
            "polish_audio_mix", "生成音频优化版本",
            {"noiseReduction": bool(brief.get("audioPolishRequested")), "voiceFirst": True},
            "音频优化版本",
        )
        add("run_delivery_qc", "检查音频优化版本质量", qc_arguments(), "音频优化版本质检报告", [polish_step])
        decision.append({"rule": "audio_polish", "outcome": "normalize", "reason": "目标要求音频优化或响度处理"})
    elif kind == "broll-overlay":
        add("search_content", "检索辅助画面素材", {"query": retrieval_query[:500] or brief["goal"][:500]}, "可作为 B-roll 的视觉候选")
        review_step = add(
            "review_content_evidence",
            "自动筛选辅助画面候选" if mode == AUTONOMOUS_REVIEW else "确认辅助画面候选",
            {"query": retrieval_query[:500] or brief["goal"][:500], "minimumSelection": 1},
            "已确认的辅助画面候选",
        )
        broll_step = add("propose_broll_overlay", "加入辅助画面插入轨", {"query": retrieval_query[:500] or brief["goal"][:500], "maxOverlays": 6}, "当前时间线的 B-roll 插入草稿", [review_step])
        add("render_review_preview", "生成辅助画面审核预览", {"subtitleMode": "none"}, "带辅助画面的审核样片", [broll_step])
        decision.append({"rule": "broll_overlay", "outcome": "cutaways", "reason": "目标要求穿插或覆盖辅助画面"})
    elif kind == "graphics-package":
        graphics_step = add(
            "render_graphics_package", "添加图文包装层",
            {"text": str(brief.get("graphicsText") or brief.get("retrievalQuery") or "")[:120], "placement": "top", "style": "clean"},
            "当前时间线的可编辑图文层",
        )
        add("render_review_preview", "生成图文包装审核预览", {"subtitleMode": "none"}, "带图文包装的审核样片", [graphics_step])
        decision.append({"rule": "graphics_package", "outcome": "text_layers", "reason": "目标要求标题、标签、参数卡或水印"})
    elif kind == "platform-delivery":
        delivery = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
        export_step = add(
            "export_delivery_master", "导出正式交付版本",
            {"platform": "generic", "aspect": str(delivery.get("aspect") or "")},
            "需要明确授权的正式交付文件",
        )
        add("run_delivery_qc", "检查正式交付质量", qc_arguments(), "正式交付文件质检报告", [export_step])
        decision.append({"rule": "delivery_export", "outcome": "requires_approval", "reason": "正式导出必须保留明确确认点"})
    elif kind == "multi-topic":
        add(
            "search_content", "提取多主题剪辑证据",
            {"query": retrieval_query[:500]},
            "每个必需主题的候选时间段、证据与可用上下文",
        )
        if needs_timeline:
            add(
                "select_multi_topic_evidence",
                "自动筛选多主题候选片段" if mode == AUTONOMOUS_REVIEW else "确认多主题候选片段",
                {"query": retrieval_query[:500], "minimumPerTopic": 1},
                "每个主题至少一个可靠候选，缺失则返回无结果",
            )
        decision.append({"rule": "multi_topic", "outcome": "balanced_selection", "reason": "目标包含多个必需主题或分段配额"})
    elif kind in {"highlight", "shortform"} and not has_candidates:
        if bool(brief.get("formatOnly")):
            decision.append({
                "rule": "source_passthrough",
                "outcome": "full_source_timeline",
                "reason": "仅改变画幅或安全区，直接以完整素材建立时间线，不执行高光筛选",
            })
        elif kind == "shortform" and bool(brief["specificContentTarget"]):
            add("search_content", "提取指定主题的 Hook 证据", {
                "query": retrieval_query[:500],
            }, "可作为短视频开场和完整观点的内容证据")
            decision.append({"rule": "shortform_evidence", "outcome": "search", "reason": "目标指定了主题或观点"})
            if needs_timeline:
                add(
                    "review_content_evidence",
                    "自动筛选短视频候选片段" if mode == AUTONOMOUS_REVIEW else "确认短视频候选片段",
                    {"query": retrieval_query[:500], "minimumSelection": 1},
                    "用户确认的 Hook 与正文候选",
                )
        else:
            highlight_arguments: dict[str, Any] = {"instruction": brief["goal"]}
            if brief.get("targetSeconds") is not None:
                highlight_arguments["targetSeconds"] = brief["targetSeconds"]
            add(
                "analyze_highlights", "提取高光候选",
                highlight_arguments, "按内容质量排序的高光证据",
            )
            decision.append({"rule": "highlight_evidence", "outcome": "analyze", "reason": "未发现可复用高光候选"})
    elif (kind in {"content", "speaker", "revision"}
          and any(p.get("kind") == "speech.voice_identity" for p in
                  (((context.get("evidence") or {}).get("contentConstraint") or {}).get("contract") or {}).get("predicates", []))
          and re.search(r"(?:当前|这些).{0,30}片段.{0,12}(?:合成|生成)", str(brief["goal"]))
          and not re.search(r"重新检索|换一个|另一个|另外|排除", str(brief["goal"]))):
        add("review_content_evidence", "试听并确认当前声纹匹配片段",
            {"query": str((context.get("evidence") or {}).get("lastQuery") or ""), "minimumSelection": 1},
            "当前声纹匹配候选的审核选择，不重新选择匿名说话人")
        decision.append({"rule": "existing_voice_candidates", "outcome": "review_existing",
                         "reason": "合成当前声纹检索结果，保留原身份约束"})
    elif kind in {"content", "interview", "speaker"} or (kind == "revision" and "content" in profile_kinds):
        requires_speaker_scope = kind == "speaker" or bool(brief["speakerTargeted"])
        if requires_speaker_scope and bool(speaker.get("needsDiscovery")):
            add("discover_speakers", "补全说话人证据", {}, "说话人分段与角色置信度")
            decision.append({"rule": "speaker_evidence", "outcome": "discover", "reason": "现有说话人证据不可用"})
        needs_speaker_confirmation = requires_speaker_scope and (
            bool(speaker.get("needsConfirmation"))
            or (not bool(speaker.get("selectedCount")) and kind == "speaker")
        )
        if needs_speaker_confirmation and mode != AUTONOMOUS_REVIEW:
            add("select_speakers", "确认目标说话人与保留方式", {"mode": brief.get("selectionMode") or "include"}, "已保存的说话人范围")
            decision.append({"rule": "speaker_identity", "outcome": "confirm", "reason": str(speaker.get("confirmationReason") or "目标涉及特定声音或角色不确定")})
        elif needs_speaker_confirmation:
            add(
                "select_speakers", "核定目标说话人",
                {
                    "mode": brief.get("selectionMode") or "include",
                    "label": brief.get("speakerTargetLabel") or "",
                },
                "可靠证据支持的说话人范围，无法消歧时返回无结果",
            )
            decision.append({
                "rule": "speaker_identity", "outcome": "evidence_rank_or_no_result",
                "reason": "自动模式不插入说话人确认点；仅采用可靠证据，无法消歧时返回无结果",
            })
        search_query = retrieval_query
        review_query = retrieval_query
        if kind == "interview":
            # Subtitle delivery is downstream rendering, not evidence that
            # the source contains useful screen text. Keep interview
            # discovery on spoken Q&A so a request to add subtitles does
            # not trigger a full-source OCR/visual scan.
            if explicit_retrieval_query:
                search_query = (
                    f"仅根据对白检索：分别检索以下回答主题：{explicit_retrieval_query}；"
                    "各主题作为独立候选，不要求同一片段同时命中；"
                    "口头提问仅作为回答的可选上下文补齐，不作为必选主题；"
                    "不使用画面或屏幕文字作为主题证据"
                )[:500]
                review_query = explicit_retrieval_query
            else:
                search_query = (
                    "仅根据对白检索：访谈中的完整回答、观点、经历和问答内容；"
                    "口头提问仅作为回答的可选上下文补齐，不作为必选主题；"
                    "去重、长停顿、字幕和生成样片属于剪辑要求，不作为检索主题；"
                    "不使用画面或屏幕文字作为主题证据"
                )[:500]
                review_query = "访谈完整回答和必要问题上下文"
        if brief.get("anchorStart") and re.search(
            r"从\s*(?:讲到|讲|提到|介绍|说到|说)", str(brief.get("goal") or "")
        ) and not re.search(r"画面|屏幕|镜头|字幕|文字|出现|展示", str(brief.get("goal") or "")):
            search_query = f"仅根据对白检索：{brief['anchorStart']['query']}；定位相关发言，不使用画面或屏幕文字作为证据"
        if brief.get("sourceEndAnchor") and brief.get("anchorStart"):
            search_query = (
                f"{brief['anchorStart'].get('query') or ''} 到 "
                f"{brief['sourceEndAnchor'].get('query') or ''}"
            ).strip(" 到")
            review_query = search_query
        add(
            "search_content",
            "检索目标内容" if not needs_timeline else "提取剪辑证据",
            {"query": search_query[:500]},
            "与目标相关的候选时间段、证据与可用上下文",
        )
        decision.append({
            "rule": "semantic_query", "outcome": "search",
            "reason": f"从任务描述中抽取检索目标“{search_query[:80]}”",
        })
        if needs_timeline and bool(brief.get("requiresEvidenceReview")):
            review_arguments: dict[str, Any] = {
                "query": review_query[:500], "minimumSelection": 1,
            }
            if brief.get("anchorStart"):
                review_arguments["selectionPolicy"] = "unique_or_review"
            add(
                "review_content_evidence",
                "核定起剪位置" if brief.get("anchorStart") else
                "自动筛选用于组合的候选片段" if mode == AUTONOMOUS_REVIEW else
                "确认用于组合的候选片段",
                review_arguments,
                (
                    "自动选定的候选片段及其排列范围"
                    if mode == AUTONOMOUS_REVIEW else "用户确认的候选片段及其排列范围"
                ),
            )
            decision.append({
                "rule": "evidence_review",
                "outcome": "unique_or_review" if brief.get("anchorStart") else
                "auto_select" if mode == AUTONOMOUS_REVIEW else "confirm",
                "reason": (
                    "自动模式依据有效证据筛选并保存候选"
                    if mode == AUTONOMOUS_REVIEW else "用户同时要求检索与组合成片，需先审核候选"
                ),
            })
    elif kind == "person":
        if bool(people.get("needsDiscovery")):
            add("discover_people", "发现画面人物", {}, "人物簇与出镜区间")
        if (
            bool(people.get("needsConfirmation", not bool(people.get("selectedCount"))))
            and mode != AUTONOMOUS_REVIEW
        ):
            add("select_people", "确认目标人物与保留方式", {"mode": brief.get("selectionMode") or "include"}, "已保存的人物范围")
        elif bool(people.get("needsConfirmation", not bool(people.get("selectedCount")))):
            add(
                "select_people", "核定目标人物",
                {
                    "mode": brief.get("selectionMode") or "include",
                    "description": brief.get("personDescription") or "",
                },
                "可靠证据支持的人物范围，无法消歧时返回无结果",
            )
            decision.append({
                "rule": "person_identity", "outcome": "evidence_rank_or_no_result",
                "reason": "自动模式不插入人物确认点；仅采用可靠证据，无法消歧时返回无结果",
            })
    elif kind == "revision":
        decision.append({"rule": "source_analysis", "outcome": "reuse", "reason": "返修流程不重新分析源素材"})
    elif kind == "delivery-qc":
        add("run_delivery_qc", "检查成片交付质量", qc_arguments(), "逐文件质检报告与可追踪问题")
        decision.append({"rule": "delivery_qc", "outcome": "analyze", "reason": "对现有成片运行确定性媒体检查"})
    elif kind == "social-reframe":
        delivery = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
        aspect = str(delivery.get("aspect") or "9:16")
        fit = str(delivery.get("fit") or "blur")
        base_delivery_step_id = add(
            "render_social_preview", f"生成 {aspect} 社媒审核预览",
            {"aspect": aspect, "fit": fit, "focusX": float(delivery.get("focusX", .5)), "focusY": float(delivery.get("focusY", .5))},
            f"{aspect} 审核预览",
        )
        decision.append({"rule": "social_reframe", "outcome": fit, "reason": f"按请求生成 {aspect} 预览"})
    elif kind == "cover":
        aspect = str(brief.get("coverAspect") or "16:9")
        title_text = str(brief.get("coverTitle") or "")[:80]
        candidate_arguments = {
            "sourceScope": cover_candidate_source_scope(),
            "candidateBudget": 16,
            "aspectRatios": [aspect],
            "focus": str(brief.get("goal") or "")[:240],
        }
        if brief.get("coverSubject"):
            candidate_arguments["subject"] = str(brief["coverSubject"])[:120]
        if brief.get("coverSourceTime") is not None:
            candidate_arguments["sourceTime"] = float(brief["coverSourceTime"])
        if title_text:
            candidate_arguments["titleText"] = title_text
        cover_candidate_step = add(
            "propose_cover_candidates", "提取并评分封面候选",
            candidate_arguments,
            "带源时间、证据、六维评分和去重结果的封面候选集",
        )
        render_arguments = {
            "aspectRatios": [aspect],
            "directions": ["source_clean", "source_editorial", "source_cinematic"],
        }
        if title_text:
            render_arguments["titleText"] = title_text
        cover_variant_step = add(
            "render_cover_variants", "生成三种本地封面预览",
            render_arguments,
            "三种不覆盖当前封面的可追溯封面预览",
            [cover_candidate_step],
        )
        cover_review_step = add("review_cover_variants", "自动优选当前任务封面", {}, "按当前主题和画幅自动选择的封面候选", [cover_variant_step])
        add("confirm_cover", "保存当前任务封面", {}, "已绑定当前任务的封面版本", [cover_review_step])
        decision.append({
            "rule": "cover_generation", "outcome": "local_source_variants",
            "reason": f"从当前任务生成 {aspect} 的三种源帧封面方向，不调用外部生成服务",
        })

    # Managed profiles own the executable timeline instruction.  The model
    # may describe a strategy, but it must not turn a render/QC recovery
    # into a different edit (for example by inventing speed changes).
    # Keeping this value deterministic also lets a replan safely reuse an
    # already applied timeline when its source request did not change.
    instruction = str(brief["goal"] or strategy.get("timelineInstruction") or "")
    if brief["targetSeconds"]:
        instruction = f"{instruction}；目标 {brief['targetSeconds']} 秒，允许浮动 ±{brief['durationToleranceSeconds']} 秒"
    if brief.get("anchorStart"):
        anchor_query = str(brief["anchorStart"].get("query") or "")
        instruction = (
            f"{instruction}；以已核定的“{anchor_query}”匹配片段作为新时间线起点，"
            "移除此前内容，保留后续可用编辑内容"
        )
    if brief.get("sourceEndAnchor"):
        end_query = str((brief.get("sourceEndAnchor") or {}).get("query") or "")
        if end_query:
            instruction = f"{instruction}；在“{end_query}”对应位置结束，移除之后内容"
    if brief.get("sourceRange"):
        scope = brief["sourceRange"]
        instruction = f"{instruction}；仅使用源视频 {float(scope['start']):g} 到 {float(scope['end']):g} 秒范围"
    for removal in brief.get("removedSourceRanges") or []:
        if isinstance(removal, dict):
            instruction = f"{instruction}；移除源视频 {float(removal.get('start') or 0):g} 到 {float(removal.get('end') or 0):g} 秒内容"
    if brief.get("graphicsRequested"):
        overlay_text = str(brief.get("overlayText") or brief.get("graphicsText") or "").strip()
        duration = brief.get("overlayDurationSeconds")
        duration_text = f"，显示 {float(duration):g} 秒" if isinstance(duration, (int, float)) else ""
        if overlay_text:
            instruction = f"{instruction}；添加文字层“{overlay_text}”{duration_text}"
    if kind == "shortform":
        instruction = f"{instruction}；前 1–3 秒必须进入明确 Hook，保留完整观点或动作，不使用片头、空白铺垫或重复表达"
    if needs_timeline:
        timeline_title = (
            f"建立精剪时间线并准备 {requested_variants} 种结构方向"
            if requested_variants > 1 else "建立可审阅精剪时间线"
        )
        timeline_output = (
            f"以已确认候选为依据，可生成 {requested_variants} 种结构方向的待审核草案"
            if requested_variants > 1 else "尚未应用的可编辑时间线草案"
        )
        timeline_arguments: dict[str, Any] = {
            "instruction": instruction[:500], "variantCount": requested_variants,
        }
        if brief.get("targetSeconds") is not None:
            timeline_arguments.update({
                "targetSeconds": float(brief["targetSeconds"]),
                "toleranceSeconds": float(brief["durationToleranceSeconds"]),
                "durationSource": str(brief.get("durationSource") or "explicit"),
            })
        if brief.get("anchorStart"):
            timeline_arguments["anchorStartQuery"] = str(
                brief["anchorStart"].get("query") or ""
            )[:240]
        if brief.get("distinctSourceAcrossVariants"):
            timeline_arguments["distinctSourceAcrossVariants"] = True
        variant_directions = [str(item)[:240] for item in strategy.get("variantDirections") or []][:requested_variants]
        if variant_directions:
            timeline_arguments["variantDirections"] = variant_directions
        add("propose_timeline_edit", timeline_title, timeline_arguments, timeline_output)
        timeline_step_id = add(
            "confirm_timeline_edit",
            "验证并应用时间线方案" if mode == AUTONOMOUS_REVIEW else "确认并应用时间线草案",
            {}, "已应用并保存的时间线",
        )
    subtitle_dependency_id = timeline_step_id
    subtitle_requested = bool(brief.get("subtitleRequested")) and kind != "caption-layout"
    subtitle_position = "top" if re.search(
        r"字幕.{0,16}(?:顶部|上方|顶端)|(?:顶部|上方|顶端).{0,16}字幕",
        str(brief.get("goal") or ""),
    ) else "bottom"
    if subtitle_requested:
        subtitle_dependency_id = add(
            "prepare_subtitle_review",
            "自动生成并校对字幕草稿" if mode == AUTONOMOUS_REVIEW else "生成字幕校对稿",
            {
                "style": "clean",
                "requireConfirmedDraft": mode != AUTONOMOUS_REVIEW,
                "autoReview": mode == AUTONOMOUS_REVIEW,
            },
            (
                "与已应用时间线对应、可在成片阶段继续修改的自动校对字幕草稿"
                if mode == AUTONOMOUS_REVIEW else
                "与已应用时间线对应、待人工校对的字幕草稿"
            ),
            [timeline_step_id] if timeline_step_id else None,
        )
        if "caption-layout" in profile_kinds:
            subtitle_dependency_id = add(
                "layout_subtitles",
                "应用顶部字幕排版" if subtitle_position == "top" else "应用字幕排版策略",
                {"position": subtitle_position, "style": "clean"},
                "在已确认字幕草稿上应用布局，不改动字幕文字",
                [subtitle_dependency_id],
            )
            decision.append({
                "rule": "caption_layout",
                "outcome": subtitle_position,
                "reason": "目标明确要求字幕位置，先生成并确认字幕，再调整布局并渲染审核样片",
            })
    preview_dependency_id = subtitle_dependency_id or timeline_step_id
    review_preview_step_id: str | None = None
    if subtitle_requested:
        review_preview_step_id = add(
            "render_review_preview", "生成带字幕最终审核样片",
            {"subtitleMode": "burned_in_review_watermarked_low_bitrate"},
            "带已确认字幕与布局的最终审核样片",
            [preview_dependency_id] if preview_dependency_id else None,
        )
    elif (
        kind != "cover-intro"
        and (bool(brief.get("reviewPreviewRequested")) or (mode == AUTONOMOUS_REVIEW and needs_timeline))
    ):
        review_preview_step_id = add(
            "render_review_preview", "准备低码率审阅样片",
            {"subtitleMode": "burned_in_review_watermarked_low_bitrate"},
            "带轻水印的审阅样片",
            [preview_dependency_id] if preview_dependency_id else None,
        )
    cover_confirm_step_id: str | None = None
    if kind != "cover" and bool(brief.get("coverRequested")) and "cover" in profile_kinds:
        aspect = str(brief.get("coverAspect") or "16:9")
        title_text = str(brief.get("coverTitle") or "")[:80]
        candidate_arguments = {
            "sourceScope": cover_candidate_source_scope(),
            "candidateBudget": 16,
            "aspectRatios": [aspect],
            "focus": str(brief.get("goal") or "")[:240],
        }
        if brief.get("coverSubject"):
            candidate_arguments["subject"] = str(brief["coverSubject"])[:120]
        if brief.get("coverSourceTime") is not None:
            candidate_arguments["sourceTime"] = float(brief["coverSourceTime"])
        if title_text:
            candidate_arguments["titleText"] = title_text
        cover_candidate_step = add(
            "propose_cover_candidates", "提取并评分封面候选",
            candidate_arguments,
            "带源时间、证据、六维评分和去重结果的封面候选集",
            [timeline_step_id] if timeline_step_id else None,
        )
        render_arguments = {
            "aspectRatios": [aspect],
            "directions": ["source_clean", "source_editorial", "source_cinematic"],
        }
        if title_text:
            render_arguments["titleText"] = title_text
        cover_variant_step = add(
            "render_cover_variants", "生成三种本地封面预览",
            render_arguments,
            "三种不覆盖当前封面的可追溯封面预览",
            [cover_candidate_step],
        )
        cover_review_step = add("review_cover_variants", "自动优选当前任务封面", {}, "按当前主题和画幅自动选择的封面候选", [cover_variant_step])
        cover_confirm_step_id = add("confirm_cover", "保存当前任务封面", {}, "已绑定当前任务的封面版本", [cover_review_step])
        decision.append({
            "rule": "cover_generation", "outcome": "local_source_variants",
            "reason": f"从当前成片生成 {aspect} 的三种源帧封面方向，不调用外部生成服务",
        })
    social = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
    final_delivery_step_id: str | None = base_delivery_step_id or review_preview_step_id
    delivery_qc_needed = bool(base_delivery_step_id)
    if kind not in {"social-reframe", "dynamic-reframe", "local-motion", "delivery-qc"} and bool(social.get("requested")):
        if not bool(brief.get("reviewPreviewRequested")) and mode != AUTONOMOUS_REVIEW:
            review_preview_step_id = add(
                "render_review_preview", "生成画幅转换所需的审核输出",
                {"subtitleMode": "none"}, "不覆盖原版本的已确认时间线输出",
                [preview_dependency_id] if preview_dependency_id else None,
            )
        aspect = str(social.get("aspect") or "9:16")
        social_dependency_id = review_preview_step_id or preview_dependency_id
        social_step = add(
            "render_social_preview", f"生成 {aspect} 社媒审核预览",
            {"aspect": aspect, "fit": str(social.get("fit") or "blur"),
             "focusX": float(social.get("focusX", .5)), "focusY": float(social.get("focusY", .5))},
            f"{aspect} 审核预览",
            [social_dependency_id] if social_dependency_id else None,
        )
        final_delivery_step_id = social_step
        delivery_qc_needed = True
    if (
        kind != "cover-intro"
        and bool(brief.get("coverIntroRequested"))
        and "cover-intro" in profile_kinds
        and cover_confirm_step_id
    ):
        intro_dependencies = [cover_confirm_step_id]
        if final_delivery_step_id:
            intro_dependencies.append(final_delivery_step_id)
        elif preview_dependency_id:
            intro_dependencies.append(preview_dependency_id)
        intro_step = add(
            "compose_cover_intro", "合成封面片头审核样片",
            {"duration": float(brief.get("coverIntroDurationSeconds") or 1.5)},
            "带当前任务封面片头的最终审核样片",
            intro_dependencies,
        )
        final_delivery_step_id = intro_step
        delivery_qc_needed = True
        decision.append({
            "rule": "cover_intro",
            "outcome": "compose_review_preview",
            "reason": "目标要求封面出现在视频开头，因此封面确认后继续合成审核样片",
        })
    if delivery_qc_needed and final_delivery_step_id:
        qc_dependencies = [final_delivery_step_id]
        if (
            cover_confirm_step_id
            and not brief.get("coverIntroRequested")
            and cover_confirm_step_id not in qc_dependencies
        ):
            qc_dependencies.append(cover_confirm_step_id)
        add(
            "run_delivery_qc",
            "检查最终审核样片质量" if bool(brief.get("coverIntroRequested")) else "检查社媒预览质量",
            qc_arguments(),
            "最终审核样片的完整质检报告",
            qc_dependencies,
        )
    elif kind not in {"social-reframe", "dynamic-reframe", "local-motion", "delivery-qc"} and bool(brief.get("deliveryQcRequested")):
        if needs_timeline and not bool(brief.get("reviewPreviewRequested")):
            add("render_review_preview", "生成质检所需的审核输出", {"subtitleMode": "none"}, "已确认时间线的审核输出")
        add("run_delivery_qc", "检查成片交付质量", qc_arguments(), "逐文件质检报告与可追踪问题")
    target = (f"目标区间约 {brief['targetSeconds']}±{brief['durationToleranceSeconds']} 秒（实际时长将在候选与时间线确认后确定）"
              if brief["targetSeconds"] else "不设时长上限，完整保留符合条件的片段（完成后统计实际时长）")
    if kind == "delivery-qc":
        result["summary"] = "检查已有成片的可解码性、时长、音画流、黑帧、冻结、静音和响度；不修改或导出媒体。"
    elif kind == "social-reframe":
        result["summary"] = "从已有成片生成不覆盖原文件的社媒画幅审核预览，使用双遍响度规范化，并在完成后运行交付质检。"
    elif kind == "cover":
        result["summary"] = (
            f"从已有成片优先提取并评分封面候选，生成 {brief.get('coverAspect') or '16:9'} 的三种本地审核预览。"
            "选择后保存为新的封面版本；不修改视频、不覆盖历史封面，也不发布到外部平台。"
        )
    elif kind == "source-provenance":
        result["summary"] = "校验当前 Agent 工作区、源素材、范围和可复用产物是否都属于本次任务；不修改媒体。"
    elif kind == "edit-diagnostics":
        result["summary"] = "诊断当前剪辑流程的失败或异常状态，输出具体原因和下一步；不修改媒体。"
    elif kind == "dynamic-reframe":
        result["summary"] = "从当前成片生成目标画幅审核预览，默认完整保留原画面并用虚化背景补齐，同时运行质检。"
    elif kind == "cover-intro":
        result["summary"] = "从当前任务生成并确认封面，然后把封面合成为当前成片片头并质检；不会复用其他任务封面。"
    elif kind == "caption-layout":
        result["summary"] = "为当前时间线生成字幕排版草稿并渲染审核预览，优先满足位置、安全区和可读性要求。"
    elif kind == "audio-polish":
        result["summary"] = "基于当前成片生成音频优化版本，进行响度规范化和保守降噪后运行质检。"
    elif kind == "broll-overlay":
        result["summary"] = "检索并加入相关辅助画面插入轨，保留主音频，完成后生成审核预览。"
    elif kind == "graphics-package":
        result["summary"] = "在当前时间线添加可编辑图文包装层，并生成审核预览。"
    elif kind == "local-motion":
        result["summary"] = "生成动态图文、标题卡或片头视频。"
    elif kind == "local-draft":
        result["summary"] = "导出当前任务本地剪辑草稿包，供外部编辑器映射使用；不会写入剪映或其他第三方目录。"
    elif kind == "platform-delivery":
        result["summary"] = "确认效果后生成可下载的成片。"
    elif kind == "multi-topic":
        result["summary"] = (
            f"按多个必需主题整理素材，{target}。"
            "保留各主题的相关片段，缺少内容时提示你补充。"
        )
    elif not needs_timeline:
        result["summary"] = (
            f"查找与“{retrieval_query[:100]}”相关的片段，供你预览和选择。"
        )
    elif kind == "interview" and mode == STEPWISE_REVIEW:
        result["summary"] = (
            f"将访谈素材剪辑为{target}的主题精华：一次性检索核心回答、必要提问、自然转场和可删重复，"
            "按主题保留完整观点并删除冗余表达。确认片段后生成剪辑预览。"
        )
    else:
        variant_clause = (
            f"自动选择片段，生成 {requested_variants} 个剪辑预览。"
            if mode == AUTONOMOUS_REVIEW and requested_variants > 1 else
            "自动选择片段并生成剪辑预览。"
            if mode == AUTONOMOUS_REVIEW else
            f"先确认片段，再比较 {requested_variants} 种剪辑方案。"
            if requested_variants > 1 else "先确认片段，再安排播放顺序。"
        )
        delivery_clause = (
            f"随后自动生成 {social.get('aspect')} 社媒审核预览并质检。"
            if bool(social.get("requested")) and mode == AUTONOMOUS_REVIEW else
            f"确认后生成 {social.get('aspect')} 预览。"
            if bool(social.get("requested")) else ""
        )
        final_clause = (
            "同时生成封面供你选择。"
            if mode == AUTONOMOUS_REVIEW and bool(brief.get("coverRequested"))
            else ""
        )
        result["summary"] = (
            f"将素材剪辑为{target}的视频。{variant_clause}"
            f"{delivery_clause}{final_clause}"
        )
    constraints = []
    if brief.get("sourceRange"):
        scope = brief["sourceRange"]
        constraints.append(f"素材范围 {scope['start']:g}–{scope['end']:g} 秒")
    if brief.get("excludedContent"):
        constraints.append("排除：" + "、".join(brief["excludedContent"]))
    if brief.get("subtitleRequested"):
        constraints.append("生成字幕")
    if brief.get("distinctSourceAcrossVariants"):
        constraints.append("各版本不得复用源片段；无法满足时停止并说明原因")
    if constraints:
        result["summary"] += " 要求：" + "；".join(constraints) + "。"
    result["steps"] = steps
    result["decisionRecord"] = decision
    return result


def validate_tool_arguments(
    step_id: str, tool: dict[str, Any], arguments: dict[str, Any],
) -> None:
    schema = tool.get("parameters")
    if not isinstance(schema, dict):
        return
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    missing = [str(name) for name in required if str(name) not in arguments]
    if missing:
        raise ValueError(f"步骤 {step_id} 缺少工具参数：{', '.join(missing)}")
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"步骤 {step_id} 包含未声明参数：{', '.join(unknown)}")
    expected_types: dict[str, tuple[type[Any], ...]] = {
        "string": (str,), "number": (int, float), "integer": (int,),
        "boolean": (bool,), "object": (dict,), "array": (list,),
    }
    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict) or not property_schema.get("type"):
            continue
        expected = expected_types.get(str(property_schema["type"]))
        if expected and (isinstance(value, bool) and str(property_schema["type"]) in {"number", "integer"} or not isinstance(value, expected)):
            raise ValueError(f"步骤 {step_id} 的参数 {name} 类型无效")
        if "enum" in property_schema and value not in property_schema["enum"]:
            raise ValueError(f"步骤 {step_id} 的参数 {name} 不在允许范围内")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError(f"步骤 {step_id} 的参数 {name} 必须为有限数值")
            for boundary, invalid in (
                ("minimum", lambda limit: value < limit),
                ("maximum", lambda limit: value > limit),
                ("exclusiveMinimum", lambda limit: value <= limit),
                ("exclusiveMaximum", lambda limit: value >= limit),
            ):
                if boundary in property_schema and invalid(property_schema[boundary]):
                    raise ValueError(f"步骤 {step_id} 的参数 {name} 超出允许范围")
    if "sourceScopeStart" in arguments and "sourceScopeEnd" in arguments:
        if arguments["sourceScopeEnd"] <= arguments["sourceScopeStart"]:
            raise ValueError(f"步骤 {step_id} 的素材结束时间必须大于开始时间")


def validate_acyclic(steps: list[dict[str, Any]]) -> None:
    dependencies = {step["id"]: set(step["dependencies"]) for step in steps}
    ready = [step_id for step_id, items in dependencies.items() if not items]
    visited: set[str] = set()
    while ready:
        current = ready.pop()
        if current in visited:
            continue
        visited.add(current)
        for step_id, items in dependencies.items():
            items.discard(current)
            if not items and step_id not in visited:
                ready.append(step_id)
    if len(visited) != len(steps):
        raise ValueError("执行计划包含循环依赖")


def step_execution_signatures(steps: list[dict[str, Any]]) -> dict[str, str]:
    """Hash a step together with the exact dependency chain it consumes.

    A confirmation tool commonly has no arguments of its own.  Matching
    only ``tool + arguments`` therefore reused an old confirmation after a
    replan had produced a different timeline proposal.  Dependency-aware
    signatures make reuse transitive: a step is reusable only when both it
    and every upstream input are unchanged.
    """
    by_id = {
        str(step.get("id") or ""): step
        for step in steps if str(step.get("id") or "")
    }
    signatures: dict[str, str] = {}
    visiting: set[str] = set()

    def signature(step_id: str) -> str:
        if step_id in signatures:
            return signatures[step_id]
        if step_id in visiting or step_id not in by_id:
            return ""
        visiting.add(step_id)
        step = by_id[step_id]
        payload = {
            "tool": str(step.get("tool") or ""),
            "arguments": step.get("arguments") or {},
            "pluginIdentity": [step.get(key) for key in ("pluginId", "pluginVersion", "contentHash", "treeHash")],
            "dependencies": [
                signature(str(dependency))
                for dependency in step.get("dependencies") or []
            ],
        }
        visiting.discard(step_id)
        digest = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str,
            ).encode("utf-8")
        ).hexdigest()
        signatures[step_id] = digest
        return digest

    for step_id in by_id:
        signature(step_id)
    return signatures


def validate_timeline_review_gate(steps: list[dict[str, Any]]) -> None:
    """Keep an AI timeline draft separate from its user-approved edit state."""
    subtitle_review = next((step for step in steps if step["tool"] == "prepare_subtitle_review"), None)
    for layout in (step for step in steps if step["tool"] == "layout_subtitles"):
        if subtitle_review is None:
            raise ValueError("字幕排版前必须先生成并确认字幕审核稿")
        if int(layout["index"]) <= int(subtitle_review["index"]):
            raise ValueError("字幕排版必须在字幕审核稿生成并确认后执行")
        if subtitle_review["id"] not in set(layout["dependencies"]):
            raise ValueError("字幕排版步骤必须依赖已确认的字幕审核稿")
    proposal = next((step for step in steps if step["tool"] == "propose_timeline_edit"), None)
    if proposal is None:
        return
    confirmation = next((step for step in steps if step["tool"] == "confirm_timeline_edit"), None)
    if confirmation is None:
        raise ValueError("时间线提案后必须加入确认时间线步骤，再生成字幕或审核样片")
    if proposal["id"] not in set(confirmation["dependencies"]):
        raise ValueError("确认时间线步骤必须依赖时间线提案")
    for step in steps:
        if step["tool"] not in {"prepare_subtitle_review", "layout_subtitles", "render_review_preview"}:
            continue
        if int(step["index"]) <= int(confirmation["index"]):
            raise ValueError("字幕和审核样片必须在时间线确认后执行")


def normalize_plan(
    raw: dict[str, Any], *, workspace: dict[str, Any],
    skill: dict[str, Any], goal: str, skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tool_map = {item["name"]: item for item in tool_catalog()}
    selected_skills = skills or [skill]
    profile = profile_for_skill(skill)
    execution_mode = str(raw.get("executionMode") or STEPWISE_REVIEW)
    if execution_mode not in VALID_EXECUTION_MODES or not supports_autonomous_review(selected_skills):
        execution_mode = STEPWISE_REVIEW
    allowed_tools: set[str] = set()
    for selected in selected_skills:
        selected_profile = profile_for_skill(selected)
        declared_tools = {str(item) for item in selected.get("allowedTools") or [] if str(item)}
        allowed_tools.update(set(selected_profile["tools"]) if selected_profile["managed"] else declared_tools)
    raw_steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    if not raw_steps or len(raw_steps) > 24:
        raise ValueError("Agent 必须生成 1–24 个计划步骤")
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    for index, value in enumerate(raw_steps, 1):
        if not isinstance(value, dict):
            raise ValueError("计划步骤格式无效")
        step_id = str(value.get("id") or f"step_{index}")[:64]
        tool_name = str(value.get("tool") or "")
        if step_id in seen:
            raise ValueError("计划步骤 ID 重复")
        if tool_name not in tool_map:
            raise ValueError(f"计划引用了未安装的工具：{tool_name}")
        if allowed_tools and tool_name not in allowed_tools:
            raise ValueError(f"计划引用了当前 Skill 未授权的工具：{tool_name}")
        seen.add(step_id)
        dependencies = [str(item) for item in value.get("dependencies") or []]
        declared_effect = str(tool_map[tool_name].get("sideEffect") or "analysis")
        requested_effect = str(value.get("sideEffect") or declared_effect)
        if requested_effect != declared_effect:
            raise ValueError(f"步骤 {step_id} 不能改变工具声明的副作用级别")
        effect = declared_effect
        if effect not in VALID_SIDE_EFFECTS:
            raise ValueError(f"步骤 {step_id} 的副作用级别无效")
        arguments = value.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError(f"步骤 {step_id} 的工具参数必须是对象")
        validate_tool_arguments(step_id, tool_map[tool_name], arguments)
        steps.append({
            "id": step_id, "index": index,
            "title": str(value.get("title") or tool_name)[:160],
            "tool": tool_name, "arguments": arguments,
            "dependencies": dependencies, "expectedOutput": str(value.get("expectedOutput") or "")[:500],
            "sideEffect": effect, "estimatedSeconds": min(86400, max(0, int(value.get("estimatedSeconds") or 0))),
            "optional": bool(value.get("optional")), "status": "pending", "attempts": 0,
            "pluginId": tool_map[tool_name].get("pluginId"),
            "pluginVersion": tool_map[tool_name].get("pluginVersion"),
            "contentHash": tool_map[tool_name].get("contentHash"),
            "treeHash": tool_map[tool_name].get("treeHash"),
        })
    for step in steps:
        unknown = set(step["dependencies"]) - seen
        if unknown:
            raise ValueError(f"步骤 {step['id']} 引用了不存在的依赖")
    validate_acyclic(steps)
    validate_timeline_review_gate(steps)
    canonical = {
        "workspaceId": workspace["id"], "workspaceRevision": workspace["revision"],
        "executionMode": execution_mode,
        "goal": goal, "skillId": skill["id"], "skillVersion": skill["version"],
        "skillHash": skill["contentHash"], "skills": skill_chain_payload(selected_skills),
        "toolCatalogFingerprint": tool_catalog_fingerprint(),
        "promptVersion": PROMPT_VERSION,
        "steps": [
            {key: value for key, value in step.items() if key not in {"status", "attempts"}}
            for step in steps
        ],
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "id": f"plan_{uuid.uuid4().hex}", **canonical,
        "summary": str(raw.get("summary") or goal)[:1000],
        "status": "awaiting_confirmation", "revision": 1,
        "planHash": digest, "steps": steps,
        "approval": None, "agent": raw.get("agent") or {},
        "brief": raw.get("brief") if isinstance(raw.get("brief"), dict) else {},
        "understanding": raw.get("understanding") if isinstance(raw.get("understanding"), dict) else {},
        "planningContext": raw.get("planningContext") if isinstance(raw.get("planningContext"), dict) else {},
        "decisionRecord": raw.get("decisionRecord") if isinstance(raw.get("decisionRecord"), list) else [],
        "profile": str(raw.get("profile") or profile["kind"]),
        "planningSource": str(raw.get("planningSource") or "agent_service"),
        "planningWarning": str(raw.get("planningWarning") or "")[:500],
    }
