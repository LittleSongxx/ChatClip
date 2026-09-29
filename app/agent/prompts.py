"""System prompts for the LangChain planner.

Ported from the retired Node agent-service; wording is behavioural
contract, the frontend renders progress phases named here.
"""

from __future__ import annotations

import json
from typing import Any


# Bump when planner behaviour guidance changes materially; planHash binds
# this so plans approved under older prompt semantics require re-approval.
PROMPT_VERSION = 2


def plan_system_prompt(payload: dict[str, Any]) -> str:
    managed = payload.get("profile", {}).get("managed") is not False and str(payload.get("profile", {}).get("kind") or "") != "custom"
    replan_context = (
        f"\n\n这是失败后的重新规划。已完成步骤是不可重复的事实；针对失败原因给出替代路径。\n{json.dumps(payload['replan'], ensure_ascii=False, indent=2)}"
        if payload.get("replan") else ""
    )
    return (
        "你是 ChatClip 智能剪辑 Planner。你的职责是解释目标、命名剪辑结构并调用 submit_plan；"
        "平台会依据事实快照与 Skill 能力档案编译最终可执行步骤。\n\n"
        "硬性规则：\n"
        "- 当前是规划阶段，禁止执行媒体分析、渲染、导出、删除或身份绑定。\n"
        "- 只能引用组合 Skill 能力档案允许的真实工具；不得绕过 allowed-tools。\n"
        "- 当前执行器一次只运行一个步骤，计划不得声称并行。\n"
        "- 使用 brief 的 targetSeconds 与 durationToleranceSeconds；brief 未给出明确时长时不得自行套用固定时长默认（短视频 Hook 目标可按 30 秒以内把握节奏）。\n"
        "- 身份选择仅在 brief 明确限定人物/声音或 planningContext 表明置信度不足时出现一次。\n"
        "- 时间线必须先作为未应用草案提出，再经 confirm_timeline_edit 由用户确认已应用。"
        "字幕审核和审阅样片必须在时间线确认之后；正式导出不属于本计划。\n"
        "- 对主题访谈，把主题、回答、必要提问、转场、重复与可删内容合并为一次语义检索。\n"
        "- 对短视频 Hook 目标：无指定主题时优先高光证据；指定主题、观点或引用时优先语义检索。"
        "目标未给出时默认 30 秒；开头 1–3 秒必须进入明确 Hook。\n"
        "- summary 中解释结构、删重策略和必要人工边界；不得输出计划 JSON 文本，必须且只调用一次 submit_plan。\n"
        + (
            "- 这是平台托管 Skill：不要编写 DAG 或步骤，只提交检索扩展、时间线策略和多版本差异；平台负责确定步骤与确认点。\n"
            if managed else
            "- 这是自定义 Skill：提交 1–24 个仅使用授权工具的串行 DAG 步骤。\n"
        )
        + f"\n当前 Skill：\n{payload['skill']['markdown']}\n\n"
        + f"组合 Skill：\n{json.dumps(payload.get('skills') or [], ensure_ascii=False, indent=2)}\n\n"
        + f"能力档案：\n{json.dumps(payload.get('profile') or {}, ensure_ascii=False, indent=2)}\n\n"
        + f"剪辑 Brief：\n{json.dumps(payload.get('brief') or {}, ensure_ascii=False, indent=2)}\n\n"
        + f"素材事实快照：\n{json.dumps(payload.get('planningContext') or {}, ensure_ascii=False, indent=2)}\n\n"
        + f"可用工具：\n{json.dumps(payload.get('toolCatalog'), ensure_ascii=False, indent=2)}\n\n"
        + f"工作区：{json.dumps(payload.get('workspace') or {}, ensure_ascii=False)}"
        + replan_context
        + "\n"
    )


PLAN_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 64},
        "title": {"type": "string", "minLength": 1, "maxLength": 160},
        "tool": {"type": "string", "minLength": 1, "maxLength": 100},
        "arguments": {"type": "object", "additionalProperties": True},
        "dependencies": {"type": "array", "items": {"type": "string", "maxLength": 64}, "maxItems": 24},
        "expectedOutput": {"type": "string", "maxLength": 500},
        "sideEffect": {
            "type": "string",
            "enum": ["read", "analysis", "preview", "identity", "review", "export", "delete"],
        },
        "estimatedSeconds": {"type": "integer", "minimum": 0, "maximum": 86400},
        "optional": {"type": "boolean"},
    },
    "required": ["id", "title", "tool", "arguments", "dependencies", "expectedOutput", "sideEffect", "estimatedSeconds", "optional"],
    "additionalProperties": False,
}


def submit_plan_tool_schema(managed: bool) -> dict[str, Any]:
    if managed:
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
                "strategy": {
                    "type": "object",
                    "properties": {
                        "searchQuery": {"type": "string", "maxLength": 500},
                        "timelineInstruction": {"type": "string", "maxLength": 500},
                        "orderingPolicy": {"type": "string", "maxLength": 120},
                        "preserveContext": {"type": "boolean"},
                        "removeRepetition": {"type": "boolean"},
                        "variantDirections": {"type": "array", "items": {"type": "string", "maxLength": 240}, "maxItems": 4},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        }
    else:
        parameters = {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
                "steps": {"type": "array", "items": PLAN_STEP_SCHEMA, "minItems": 1, "maxItems": 24},
            },
            "required": ["summary", "steps"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "提交完整的、尚未执行的剪辑任务 DAG，等待用户确认。",
            "parameters": parameters,
        },
    }


SUBMIT_SKILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_skill",
        "description": "提交尚未启用的标准 SKILL.md 与模拟规划结论。",
        "parameters": {
            "type": "object",
            "properties": {
                "skillMarkdown": {"type": "string", "minLength": 40, "maxLength": 16000},
                "simulationSummary": {"type": "string", "minLength": 1, "maxLength": 1000},
                "requiredTools": {"type": "array", "items": {"type": "string", "maxLength": 100}, "maxItems": 24},
            },
            "required": ["skillMarkdown", "simulationSummary", "requiredTools"],
            "additionalProperties": False,
        },
    },
}


SELECT_SKILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "select_skill",
        "description": "选择最适合当前目标的一个已安装 Skill。",
        "parameters": {
            "type": "object",
            "properties": {
                "skillId": {"type": "string", "minLength": 1, "maxLength": 64},
                "reason": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            "required": ["skillId", "reason"],
            "additionalProperties": False,
        },
    },
}


PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "agent_capability_probe",
        "description": "回传指定 challenge，用于验证模型确实支持 Tool Calling。",
        "parameters": {
            "type": "object",
            "properties": {
                "challenge": {"type": "string", "minLength": 1, "maxLength": 100},
            },
            "required": ["challenge"],
            "additionalProperties": False,
        },
    },
}


def skill_system_prompt(payload: dict[str, Any]) -> str:
    workflow_kinds = "highlight、content、interview、person、speaker、revision、shortform、delivery-qc、social-reframe"
    return (
        "你是 ChatClip Skill Creator。根据用户描述创建一个精确、可复用的智能剪辑 Skill，然后调用 submit_skill。\n\n"
        "要求：\n"
        "- 遵循 Agent Skills 标准，生成带 name、description、allowed-tools 和 workflow-profile frontmatter 的 SKILL.md。\n"
        f"- workflow-profile 只能是 {workflow_kinds} 之一；它让平台接管步骤顺序、确认点和安全约束。\n"
        "- 选择 workflow-profile 后，allowed-tools 必须完整包含该档案所需工具；涉及时间线的 Skill 必须包含 confirm_timeline_edit。\n"
        "- name 使用小写字母、数字、连字符，最多 64 字符。\n"
        "- description 必须清楚说明何时适用，避免吸引无关任务。\n"
        "- 正文只包含会改变 Agent 决策的任务拆解、质量标准和真实约束。\n"
        "- Skill 不得要求 Bash、任意文件读写、直接导出、删除或绕过用户确认。\n"
        "- 只能使用下列工具，不得声称不存在的能力：\n"
        f"{json.dumps(payload.get('toolCatalog'), ensure_ascii=False, indent=2)}\n"
        "- 必须调用 submit_skill，不要只输出 Markdown。"
    )


def router_system_prompt(payload: dict[str, Any]) -> str:
    return (
        "你是 ChatClip Skill Router。根据用户目标与 description 选择唯一最合适的 Skill，并调用 select_skill。"
        "不得选择目录之外的 ID。\n\n"
        f"路由约束（必须遵守）：\n{json.dumps(payload.get('routingHints') or {}, ensure_ascii=False, indent=2)}\n\n"
        f"Skill 目录：\n{json.dumps(payload.get('skills'), ensure_ascii=False, indent=2)}"
    )


def probe_system_prompt(challenge: str) -> str:
    return (
        "你正在执行模型能力探测。必须调用 agent_capability_probe，"
        f"并将 challenge 设置为 {challenge}。不要输出其他内容。"
    )
