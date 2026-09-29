"""Media tool catalog: the LLM-facing capability surface of the agent.

The catalog carries JSON-schema parameter declarations plus the side-effect
class of every tool. Side-effect classes drive the human gates:
``export``/``delete`` steps never run autonomously (except
``export_editing_draft``), and ``identity``/``review`` steps pause for
structured user confirmation in stepwise mode.
"""

from __future__ import annotations

import copy
from typing import Any


CORE_TOOL_CATALOG: tuple[dict[str, Any], ...] = (
    {"name": "inspect_workspace", "description": "读取当前素材、任务状态和已有分析结果；也可在执行前核验当前任务是否具备所选 Skill 的必要成片或时间线", "sideEffect": "read", "parameters": {"type": "object", "properties": {"requiredState": {"type": "string"}, "preconditionCode": {"type": "string"}, "preconditionMessage": {"type": "string"}}, "additionalProperties": False}},
    {"name": "analyze_highlights", "description": "运行多模态高光分析并生成候选", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"instruction": {"type": "string"}, "targetSeconds": {"type": "number"}, "focus": {"type": "string"}}, "additionalProperties": False}},
    {"name": "search_content", "description": "根据语义、字幕和画面证据搜索内容片段", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "review_content_evidence", "description": "请求用户审核并保存将用于组合的内容候选", "sideEffect": "review", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "minimumSelection": {"type": "integer"}, "selectionPolicy": {"type": "string", "enum": ["all_reliable", "unique_or_review"]}}, "required": ["query"], "additionalProperties": False}},
    {"name": "discover_people", "description": "发现素材中的人物和出现区间", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "select_people", "description": "请求用户确认需要保留或排除的人物", "sideEffect": "identity", "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "description": {"type": "string"}}, "additionalProperties": False}},
    {"name": "discover_speakers", "description": "发现当前素材内的匿名说话人", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"expectedSpeakers": {"type": "integer"}}, "additionalProperties": False}},
    {"name": "select_speakers", "description": "请求用户确认说话人及保留方式", "sideEffect": "identity", "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "label": {"type": "string"}}, "additionalProperties": False}},
    {"name": "propose_timeline_edit", "description": "根据已确认的证据建立一到四个可审阅、尚未应用的时间线修改", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"instruction": {"type": "string"}, "variantCount": {"type": "integer"}, "variantDirections": {"type": "array"}, "targetSeconds": {"type": "number"}, "toleranceSeconds": {"type": "number"}, "durationSource": {"type": "string"}, "anchorStartQuery": {"type": "string"}}, "required": ["instruction"], "additionalProperties": False}},
    {"name": "confirm_timeline_edit", "description": "请求用户审核并确认已应用的时间线修改", "sideEffect": "preview", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "prepare_subtitle_review", "description": "生成已确认时间线对应的可审阅字幕草稿；自动模式仅应用低风险校对并继续生成样片", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"style": {"type": "string"}, "requireConfirmedDraft": {"type": "boolean"}, "autoReview": {"type": "boolean"}}, "additionalProperties": False}},
    {"name": "render_review_preview", "description": "渲染带轻水印、低码率和审阅字幕的审核样片；不做正式导出", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"subtitleMode": {"type": "string"}}, "additionalProperties": False}},
    {"name": "render_social_preview", "description": "从已有成片生成指定社媒画幅的审核预览；无论目标比例为何，默认完整保留原始画面并使用同画面虚化背景补足画布，只有用户明确要求时才裁切或留黑边，不覆盖原成片", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "aspect": {"type": "string"}, "fit": {"type": "string"}, "focusX": {"type": "number"}, "focusY": {"type": "number"}}, "required": ["aspect", "fit"], "additionalProperties": False}},
    {"name": "propose_cover_candidates", "description": "从已有成片或源视频抽取、去重并评分可追溯的封面候选帧，不修改当前封面", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"sourceScope": {"type": "string"}, "candidateBudget": {"type": "integer"}, "aspectRatios": {"type": "array"}, "titleText": {"type": "string"}, "focus": {"type": "string"}, "subject": {"type": "string"}, "sourceTime": {"type": "number"}}, "required": ["sourceScope", "candidateBudget", "aspectRatios"], "additionalProperties": False}},
    {"name": "render_cover_variants", "description": "从已评分候选帧生成不覆盖现有封面的本地审核预览，保留来源、评分和内容哈希", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"aspectRatios": {"type": "array"}, "directions": {"type": "array"}, "titleText": {"type": "string"}}, "required": ["aspectRatios", "directions"], "additionalProperties": False}},
    {"name": "review_cover_variants", "description": "审核封面候选；自动模式按当前主题和画幅自动选优，分步模式请求用户选择", "sideEffect": "review", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "confirm_cover", "description": "只保存并绑定当前任务封面版本；如需把封面合入视频，必须由独立的片头合成步骤执行", "sideEffect": "preview", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "run_delivery_qc", "description": "对已有成片运行完整解码、目标画幅、封面与片头要求、目标时长、音轨、黑帧、冻结、静音和响度检查，不修改媒体", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "strict": {"type": "boolean"}, "targetSeconds": {"type": "number"}, "toleranceSeconds": {"type": "number"}, "expectedAspect": {"type": "string", "enum": ["9:16", "16:9", "1:1", "4:5"]}, "requireCover": {"type": "boolean"}, "requireCoverIntro": {"type": "boolean"}, "expectedCoverTitle": {"type": "string"}}, "additionalProperties": False}},
    {"name": "validate_task_provenance", "description": "校验当前工作区、源素材、时间范围、候选、时间线、封面、字幕和输出均属于当前任务", "sideEffect": "read", "parameters": {"type": "object", "properties": {"strict": {"type": "boolean"}}, "additionalProperties": False}},
    {"name": "select_multi_topic_evidence", "description": "从内容检索结果中按多个必需主题选择可靠候选；缺少任一主题时返回结构化无结果", "sideEffect": "review", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "minimumPerTopic": {"type": "integer"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "compose_cover_intro", "description": "把当前任务已确认封面合成为当前成片的短片头，不复用其他任务封面", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"duration": {"type": "number"}}, "additionalProperties": False}},
    {"name": "analyze_reframe_safe_areas", "description": "分析当前输出画幅转换的安全策略，保护人物、字幕、屏幕文字和产品主体", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {"aspect": {"type": "string"}, "fit": {"type": "string"}}, "required": ["aspect"], "additionalProperties": False}},
    {"name": "layout_subtitles", "description": "更新当前精剪时间线已确认字幕稿的布局，支持顶部/底部、安全区、字号和双语等要求；不会自行生成字幕文字", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"position": {"type": "string"}, "style": {"type": "string"}, "fontSizeRatio": {"type": "number"}}, "additionalProperties": False}},
    {"name": "export_subtitles", "description": "从当前成片或审核样片导出字幕文件，支持 SRT/VTT，不生成或修改视频", "sideEffect": "export", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "format": {"type": "string", "enum": ["srt", "vtt"]}}, "additionalProperties": False}},
    {"name": "polish_audio_mix", "description": "基于当前任务输出生成音频优化版本，支持响度规范化、保守降噪和人声优先", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "noiseReduction": {"type": "boolean"}, "voiceFirst": {"type": "boolean"}}, "additionalProperties": False}},
    {"name": "propose_broll_overlay", "description": "把已确认的辅助画面作为静音插入镜头加入当前时间线，保留主音频", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "maxOverlays": {"type": "integer"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "render_graphics_package", "description": "在当前精剪时间线添加可编辑图文层，如标题、标签、参数卡、价格、Logo、水印或 CTA", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "placement": {"type": "string"}, "style": {"type": "string"}}, "additionalProperties": False}},
    {"name": "render_motion_graphics", "description": "使用本地 HTML/React 风格渲染管线生成动态图文、标题卡、片头或封面动效视频，不依赖外部 API", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "subtitle": {"type": "string"}, "aspect": {"type": "string"}, "duration": {"type": "number"}, "theme": {"type": "string"}}, "additionalProperties": False}},
    {"name": "compose_motion_intro", "description": "把本地图文动效合成为当前任务成片片头，生成新版本，不覆盖原成片", "sideEffect": "preview", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "introFilename": {"type": "string"}}, "additionalProperties": False}},
    {"name": "export_editing_draft", "description": "导出当前任务的本地剪辑草稿包，供外部编辑器或剪映映射器使用；不写入第三方软件目录", "sideEffect": "export", "parameters": {"type": "object", "properties": {"format": {"type": "string"}}, "additionalProperties": False}},
    {"name": "export_delivery_master", "description": "将当前确认输出生成正式交付版本或平台包；需要明确授权，不覆盖历史版本", "sideEffect": "export", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "platform": {"type": "string"}, "aspect": {"type": "string"}}, "additionalProperties": False}},
    {"name": "diagnose_edit_failure", "description": "诊断当前 Agent 计划、素材证据、时间线、画幅、封面、字幕、预览和导出问题", "sideEffect": "read", "parameters": {"type": "object", "properties": {"focus": {"type": "string"}}, "additionalProperties": False}},
    {"name": "cancel_operation", "description": "取消当前计划启动的后台操作", "sideEffect": "analysis", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
)

FORBIDDEN_AUTONOMOUS_EFFECTS = frozenset({"export", "delete"})
SAFE_AUTONOMOUS_EXPORT_TOOLS = frozenset({"export_editing_draft"})
ACTION_REQUIRED_EFFECTS = frozenset({"identity", "review"})
VALID_SIDE_EFFECTS = frozenset({"read", "analysis", "preview", "identity", "review", "export", "delete"})
AUTONOMOUS_REVIEW = "autonomous_review"
STEPWISE_REVIEW = "stepwise_review"
VALID_EXECUTION_MODES = frozenset({AUTONOMOUS_REVIEW, STEPWISE_REVIEW})

ACTIVE_PLAN_STATUSES = frozenset({
    "awaiting_confirmation", "approved", "running", "action_required",
})


for _tool in CORE_TOOL_CATALOG:
    if _tool["name"] in {"analyze_highlights", "search_content", "discover_people", "discover_speakers"}:
        _tool["parameters"]["properties"].update({
            "sourceScopeKind": {"type": "string", "enum": ["all", "custom"]},
            "sourceScopeStart": {"type": "number", "minimum": 0},
            "sourceScopeEnd": {"type": "number", "exclusiveMinimum": 0},
        })
    if _tool["name"] == "propose_timeline_edit":
        _tool["parameters"]["properties"]["distinctSourceAcrossVariants"] = {"type": "boolean"}


def tool_catalog() -> list[dict[str, Any]]:
    """The executable tool surface exposed to planning and execution."""
    return [copy.deepcopy(item) for item in CORE_TOOL_CATALOG]


def action_required_message(step: dict[str, Any]) -> str:
    tool = str(step.get("tool") or "")
    return {
        "review_content_evidence": "请在内容候选面板勾选要用于组合的片段；保存选择后再继续。",
        "select_multi_topic_evidence": "请在内容候选面板确认每个主题的可用片段；保存后再继续。",
        "select_people": "请在人物面板完成目标人物选择后再继续。",
        "select_speakers": "请在说话人面板完成目标声音选择后再继续。",
        "review_cover_variants": "请在封面审核面板选择并保存当前任务封面；如任务要求片头或目标画幅，后续会继续生成最终审核样片。",
        "export_editing_draft": "将导出当前任务本地草稿包；请确认后再继续。",
        "export_delivery_master": "正式交付会生成新的可下载版本；请确认后再导出。",
    }.get(tool, "该步骤需要你在审核面板完成结构化确认。")
