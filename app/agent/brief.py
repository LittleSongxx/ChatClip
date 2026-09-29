"""Deterministic editing-brief extraction from Chinese editing goals.

The brief is the product's own NLP layer: durations, source ranges, anchors,
deliverables and identity targets are parsed with explicit rules so a planning
model may only describe strategy, never invent facts. Ported verbatim from the
pre-LangGraph ``AgentPlatform`` static helpers to keep behaviour identical.
"""

from __future__ import annotations

import copy
import re
from typing import Any


def _chinese_number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text == "半":
        return .5
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if all(character in digits for character in text):
        return float("".join(str(digits[character]) for character in text))
    if "十" in text:
        left, right = text.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return float(tens * 10 + ones)
    return None


def _split_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"[，,。；;\n]+", str(text or ""))
        if clause.strip()
    ]


def _strip_negative_clauses(text: str) -> str:
    clauses: list[str] = []
    for clause in _split_clauses(text):
        if re.search(r"^(?:但|但是|并且|同时|也)?\s*(?:不要|不能|不应|无需|不需要|别|禁止)", clause):
            continue
        clauses.append(clause)
    return "，".join(clauses)


def _time_token_pattern() -> str:
    return (
        r"(?:"
        r"(?:\d{1,2}:)?\d{1,2}:\d{1,2}"
        r"|(?:第\s*)?\d+(?:\.\d+)?\s*(?:秒钟|秒|s(?![a-z]))"
        r")"
    )


def _parse_duration_amount(value: str, unit: str) -> float | None:
    try:
        amount = float(value)
    except ValueError:
        amount = _chinese_number(value)
    if amount is None or amount <= 0:
        return None
    return amount * (60 if unit in {"分钟", "分"} else 1)


def _source_time_range(text: str, context: dict[str, Any]) -> dict[str, Any] | None:
    token = _time_token_pattern()
    explicit = re.search(
        rf"从\s*({token})\s*(?:开始|起)?\s*(?:剪|保留|截取|提取)?\s*(?:到|至|~|～|-|—)\s*({token})",
        str(text or ""),
        re.IGNORECASE,
    )
    if explicit:
        start = _parse_timecode_seconds(explicit.group(1))
        end = _parse_timecode_seconds(explicit.group(2))
        if start is not None and end is not None and end > start:
            return {
                "kind": "custom", "start": round(start, 3), "end": round(end, 3),
                "description": explicit.group(0), "requiresDuration": False,
                "source": "explicit_range",
            }

    removal = re.search(
        r"(?:剪掉|删掉|删除|去掉|移除|裁掉)\s*(前|开头)\s*([0-9.零〇一二两三四五六七八九十半]+)\s*(分钟|分|秒钟|秒)",
        str(text or ""),
    )
    if removal:
        seconds = _parse_duration_amount(removal.group(2), removal.group(3))
        if seconds is not None:
            return {
                "kind": "remove", "start": 0.0, "end": round(seconds, 3),
                "description": removal.group(0), "requiresDuration": False,
                "source": "remove_prefix",
            }

    # “前 3 秒进入 Hook”“前三秒突出核心亮点”描述的是成片开头该做什么，
    # 不是素材选取范围。金额紧跟成片谓语动词时不作为 sourceRange。
    include = re.search(
        r"(?<!剪掉)(?<!删掉)(?<!删除)(?<!去掉)(?<!移除)(?<!裁掉)"
        r"(最后|结尾|末尾|开头|前)\s*([0-9.零〇一二两三四五六七八九十半]+)\s*(分钟|分|秒钟|秒)"
        r"(?!\s*(?:要|应(?:该)?|需要|必须|尽量|最好|先)?\s*"
        r"(?:突出|展示|呈现|强调|显示|出现|放出|给出|交代|点明|点题|聚焦|抓住|切入|进入|抛出|引出|露|印|放))",
        str(text or ""),
    )
    if include:
        seconds = _parse_duration_amount(include.group(2), include.group(3))
        if seconds is not None:
            duration = float(context.get("duration") or context.get("sourceDuration") or 0)
            tail = include.group(1) in {"最后", "结尾", "末尾"}
            return {
                "kind": "custom", "start": max(0, duration - seconds) if tail else 0,
                "end": duration if tail else min(seconds, duration) if duration else seconds,
                "description": include.group(0), "requiresDuration": tail and duration <= 0,
                "source": "relative_source_range",
            }
    return None


def _source_range(text: str, context: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _source_time_range(text, context)
    if parsed and str(parsed.get("kind") or "") != "remove":
        return parsed
    existing = context.get("sourceRange")
    return copy.deepcopy(existing) if isinstance(existing, dict) else None


def _duration_target(text: str) -> tuple[int | None, bool]:
    source_text = str(text or "")
    normalized = _strip_timeline_or_cover_time_refs(source_text)
    normalized = re.sub(
        r"(?:最后|结尾|末尾|开头|前)\s*[0-9.零〇一二两三四五六七八九十半]+\s*(?:分钟|分|秒钟|秒)",
        "",
        normalized,
    )
    # A duration mentioned as something to avoid is not a requested
    # target. For example, “不要因为默认 30 秒目标丢片段” used to be
    # compiled into an explicit 30-second cut.
    normalized = re.sub(
        r"(?:不要|不能|不应|无需|不需要)[^，,。；;]{0,80}?"
        r"\d+(?:\.\d+)?\s*(?:秒钟|秒|s(?![a-z])|分钟|分)"
        r"[^，,。；;]{0,80}(?=[，,。；;]|$)",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    # Parse an explicit target/tolerance pair before the generic duration
    # patterns. In text such as ``60±6 秒`` the generic expression can
    # only attach the unit to ``6`` and would mistake the tolerance for the
    # target duration.
    tolerance_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:±|\+\s*/\s*-|\+\s*-)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))",
        normalized,
        re.IGNORECASE,
    )
    if tolerance_match:
        return round(float(tolerance_match.group(1))), True
    total_match = re.search(
        r"(?:总(?:时长|长度)|组合成|合成为|拼接(?:合成)?(?:成|为)|拼成|剪成|做成|最终(?:视频|成片)?(?:为|要)?)"
        r".{0,12}?(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))",
        normalized,
        re.IGNORECASE,
    )
    if total_match:
        return round(float(total_match.group(1))), True
    minute_match = re.search(r"(?:约|大约|约为|做成)?\s*(\d+(?:\.\d+)?)\s*(?:分钟|分)", normalized)
    second_match = re.search(
        r"(?:约|大约|约为|做成)?\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))",
        normalized,
        re.IGNORECASE,
    )
    if minute_match:
        minutes = float(minute_match.group(1))
        trailing = re.search(rf"{re.escape(minute_match.group(0))}\s*(\d+)\s*秒", normalized)
        return round(minutes * 60 + (float(trailing.group(1)) if trailing else 0)), True
    if second_match:
        return round(float(second_match.group(1))), True
    chinese_minute = re.search(r"([零〇一二两三四五六七八九十半]+)\s*(?:分钟|分)(半)?", normalized)
    if chinese_minute:
        minutes = _chinese_number(chinese_minute.group(1))
        if minutes is not None:
            return round((minutes + (.5 if chinese_minute.group(2) else 0)) * 60), True
    chinese_second = re.search(r"([零〇一二两三四五六七八九十]+)\s*(?:秒钟|秒)", normalized)
    if chinese_second:
        seconds = _chinese_number(chinese_second.group(1))
        if seconds is not None:
            return round(seconds), True
    return None, False


def _duration_tolerance(text: str) -> int | None:
    text = _strip_timeline_or_cover_time_refs(str(text or ""))
    match = re.search(
        r"\d+(?:\.\d+)?\s*(?:±|\+\s*/\s*-|\+\s*-)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))",
        text,
        re.IGNORECASE,
    )
    return max(0, round(float(match.group(1)))) if match else None


def _strip_timeline_or_cover_time_refs(text: str) -> str:
    """Remove timestamp references that locate material, not output length."""
    value = str(text or "")
    if not value:
        return ""
    time_token = r"(?:第\s*)?\d+(?:\.\d+)?\s*(?:秒钟|秒|s(?![a-z]))"
    # Durations attached to text/graphic overlays describe the element
    # lifetime, not the whole output duration.
    value = re.sub(
        rf"(?:文字|文本|文案|标题|贴纸|水印|logo|Logo|字幕|图文|角标|标签)"
        rf"[^，,。；;\n]{{0,40}}?(?:显示|持续|停留|保留)?\s*{time_token}",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        rf"(?:显示|持续|停留|保留)\s*{time_token}"
        rf"[^，,。；;\n]{{0,40}}?(?:文字|文本|文案|标题|贴纸|水印|logo|Logo|字幕|图文|角标|标签)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    if re.search(r"文字(?!幕)|文本|文案|标题|贴纸|水印|logo|Logo|图文|角标|标签|添加[^，,。；;\n]{0,24}[“\"'‘]", value):
        value = re.sub(rf"(?:显示|持续|停留|保留)\s*{time_token}", "", value, flags=re.IGNORECASE)
    # A duration attached to a cover intro belongs to that intro card,
    # not to the full video.  Keep it available to _cover_intro_duration
    # by parsing that field from the original goal, while excluding it
    # from targetSeconds here.
    value = re.sub(rf"{time_token}\s*(?:的)?片头", "片头", value, flags=re.IGNORECASE)
    value = re.sub(rf"片头[^，,。；;\n]{{0,12}}?{time_token}", "片头", value, flags=re.IGNORECASE)
    # “54s 出现的人/第54秒画面/54秒处” is a source timestamp, especially
    # when used to pick a cover frame. It must not become targetSeconds.
    value = re.sub(
        rf"{time_token}\s*(?:左右|附近|前后)?\s*(?:出现|处|位置|时间点|画面|帧|这一帧|那一帧|的人|的那个人)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    clause_parts = re.split(r"([，,。；;\n])", value)
    cleaned: list[str] = []
    for index in range(0, len(clause_parts), 2):
        clause = clause_parts[index]
        delimiter = clause_parts[index + 1] if index + 1 < len(clause_parts) else ""
        if (
            re.search(time_token, clause, flags=re.IGNORECASE)
            and re.search(r"封面|缩略图|海报|截图|截帧|取帧|选取|选择|使用|用|出现|人物|这个人|那个人|画面|帧", clause)
            and not re.search(r"目标|总(?:时长|长度)|成片(?:时长|长度)?|视频(?:时长|长度)?|剪成|做成|合成.{0,6}\d", clause)
        ):
            clause = re.sub(time_token, "", clause, flags=re.IGNORECASE)
        cleaned.append(clause + delimiter)
    return "".join(cleaned)


def _parse_timecode_seconds(value: str) -> float | None:
    text = str(value or "").strip()
    colon = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", text)
    if colon:
        parts = [int(item) for item in colon.groups(default="0")]
        return float(parts[0] * 60 + parts[1]) if colon.group(3) is None else float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    match = re.fullmatch(r"(?:第\s*)?(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _relative_duration_request(text: str) -> bool:
    return bool(re.search(
        r"(?:再短一点|再短些|更短(?:一点|些)?|稍微短(?:一点|些)?|"
        r"再精简(?:一点|些)?|再紧凑(?:一点|些)?)",
        str(text or ""),
    ))


def _anchor_start(text: str) -> dict[str, Any] | None:
    semantic_range = _semantic_anchor_range(text)
    if semantic_range:
        return copy.deepcopy(semantic_range.get("start"))
    time_match = re.search(
        r"从\s*((?:\d{1,2}:)?\d{1,2}:\d{1,2}|(?:第\s*)?\d+(?:\.\d+)?\s*(?:秒钟|秒|s(?![a-z])))"
        r"\s*(?:左右|附近|前后)?\s*(?:出现|开始|处|位置|时间点)?\s*"
        r"(.{0,80}?)(?:的地方|的位置|那里|那儿|处)?\s*(?:开始|起)(?:剪|保留|播放|做|生成|编辑)?",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if time_match:
        seconds = _parse_timecode_seconds(time_match.group(1))
        query = str(time_match.group(2) or "").strip(" 　：:，,。；;")
        query = re.sub(r"^(?:出现|看到|有|是)\s*", "", query).strip()
        query = re.sub(r"(?:的)?(?:部分|内容|片段|画面|镜头)$", "", query).strip()
        if query:
            payload: dict[str, Any] = {"query": query[:240], "selectionPolicy": "unique_or_review"}
            if seconds is not None:
                payload["sourceTimeSeconds"] = round(seconds, 3)
            return payload
    match = re.search(
        r"从\s*(?:讲到|讲|提到|介绍|说到|说)?\s*"
        r"(.{1,100}?)(?:的地方|的位置|那里|那儿|处)?\s*"
        r"(?:开始|起)(?:剪|保留|播放|做|生成|编辑)?",
        str(text or ""),
    )
    if not match:
        return None
    query = str(match.group(1) or "").strip(" 　：:，,。；;")
    query = re.sub(r"(?:的)?(?:部分|内容|片段|画面)$", "", query).strip()
    if re.fullmatch(r"[\d\s:.：零一二三四五六七八九十百]+(?:秒|分钟|分)?", query):
        return None
    return {"query": query[:240], "selectionPolicy": "unique_or_review"} if query else None


def _semantic_anchor_range(text: str) -> dict[str, dict[str, Any]] | None:
    match = re.search(
        r"(?:从|把|将)?\s*(?:讲到|讲|提到|介绍|说到|说)?\s*(.{1,60}?)(?:的地方|的位置|那里|那儿|处)?\s*"
        r"(?:开始|起)\s*(?:到|至|直到)\s*"
        r"(?:讲到|讲|提到|介绍|说到|说)?\s*(.{1,60}?)(?:的地方|的位置|那里|那儿|处)?\s*"
        r"(?:结束|为止|之前|前)?(?:剪出来|截出来|保留|剪|$)",
        str(text or ""),
    )
    if not match:
        return None
    start = re.sub(r"(?:的)?(?:部分|内容|片段|画面|镜头)$", "", str(match.group(1) or "")).strip(" 　：:，,。；;")
    end = re.sub(r"(?:的)?(?:部分|内容|片段|画面|镜头)$", "", str(match.group(2) or "")).strip(" 　：:，,。；;")
    if not start or not end:
        return None
    return {
        "start": {"query": start[:240], "selectionPolicy": "unique_or_review"},
        "end": {"query": end[:240], "selectionPolicy": "unique_or_review"},
    }


def _generic_cover_subject(text: str) -> bool:
    value = re.sub(r"\s+", "", str(text or ""))
    if not value:
        return True
    return bool(re.fullmatch(
        r"(?:最)?(?:具有|有)?(?:冲击性|视觉冲击|张力|感染力|吸引力|代表性|高级感|电影感|质感|氛围感|美感|"
        r"精彩|好看|漂亮|清晰|醒目|震撼|燃|酷|帅|关键|重要|合适|适合|好|最佳|最好|最棒|亮眼|出彩)"
        r"(?:的)?",
        value,
    ))


def _cover_intro_requested(text: str) -> bool:
    """Return whether the cover must be rendered into the start of the video."""
    return bool(re.search(
        r"片头|(?:最)?开头.{0,30}封面"
        r"|封面.{0,24}(?:放进|合入|加入|插入|作为(?:视频|成片)?开头)"
        r"|封面.{0,16}(?:直接)?(?:和|与|同).{0,10}(?:成片|视频|样片).{0,10}(?:进行)?(?:合成|拼接)"
        r"|(?:将|把)?(?:这个|该|当前)?封面.{0,16}(?:合成|拼接)(?:到|进|入|至)?(?:当前)?(?:成片|视频|样片)",
        str(text or ""),
    ))


def _cover_intro_duration(text: str, *, default: float = 1.5) -> float:
    """Read a duration attached to a cover/intro phrase, otherwise use the product default."""
    match = re.search(
        r"(?:封面|片头)[^，,。；;\n]{0,20}?(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))"
        r"|(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))[^，,。；;\n]{0,20}?(?:封面|片头)",
        str(text or ""),
        re.IGNORECASE,
    )
    value = float(next((item for item in (match.groups() if match else ()) if item), default))
    return max(.5, min(5.0, value))


def _requested_variant_count(goal: str) -> int:
    text = str(goal or "")
    short = re.search(r"(?:做|生成|给我|提供)\s*([2-4两二三四])\s*版(?:[，,。；;\s]|$)", text)
    if short:
        return int(short[1]) if short[1].isdigit() else {"两": 2, "二": 2, "三": 3, "四": 4}[short[1]]
    # Explicit counts win; generic “different versions/cuts” intentionally
    # defaults to three choices so the promise is observable and useful.
    match = re.search(r"(?:生成|给我|做|提供)?\s*([2-4])\s*(?:个|种|版)?\s*(?:不同的?)?(?:成片|版本|剪辑方案|剪法)", text)
    if match:
        return int(match.group(1))
    chinese = re.search(r"(?:两|二|三|四)\s*(?:个|种|版)?\s*(?:不同的?)?(?:成片|版本|剪辑方案|剪法)", text)
    if chinese:
        return {"两": 2, "二": 2, "三": 3, "四": 4}.get(chinese.group(0)[0], 3)
    if re.search(r"不同(?:的)?(?:成片|版本|剪辑方案|剪法)|多个(?:成片|版本|剪辑方案)|多版", text):
        return 3
    return 1


def _social_delivery(goal: str) -> dict[str, Any]:
    """Return a delivery-format request without confusing it with the edit goal."""
    text = str(goal or "").lower()
    if "9:16" in text or re.search(
        r"(?:合成|生成|输出|做(?:成|一个|一条)?|剪成|制作|转换为|改成).{0,18}竖屏|竖屏(?:的)?(?:视频|成片|输出|版本)|(?:视频|成片|输出|版本|样片).{0,8}竖屏",
        text,
    ):
        aspect = "9:16"
    elif "1:1" in text or re.search(r"方(?:形|屏)", text):
        aspect = "1:1"
    elif "4:5" in text:
        aspect = "4:5"
    elif "16:9" in text or re.search(
        r"(?:合成|生成|输出|做成|剪成|制作|转换为|改成).{0,18}横屏|横屏(?:的)?(?:视频|成片|输出|版本)|(?:视频|成片|输出|版本|样片).{0,8}横屏",
        text,
    ):
        aspect = "16:9"
    elif re.search(r"小红书|抖音|reels?|shorts?|视频号|竖屏", text, re.IGNORECASE):
        aspect = "9:16"
    else:
        aspect = ""
    if re.search(r"(?:不要|不能|不应|无需|不需要|别|禁止).{0,8}(?:裁切|裁剪|crop)|保留完整画面|完整保留", text):
        fit = "blur"
    elif re.search(r"中心裁切|居中裁切|裁满|裁切铺满|放大裁切|\bcrop\b", text):
        fit = "crop"
    elif re.search(r"留黑|黑边|黑色留边|\bpad\b", text):
        fit = "pad"
    else:
        # Any unspecified aspect conversion must preserve the complete
        # source composition. Fill the unused canvas with a blurred copy
        # instead of silently cropping content from any edge.
        fit = "blur"
    focus_x = .25 if re.search(r"主体靠左|焦点靠左|人物靠左", text) else .75 if re.search(r"主体靠右|焦点靠右|人物靠右", text) else .5
    focus_y = .25 if re.search(r"主体靠上|焦点靠上", text) else .75 if re.search(r"主体靠下|焦点靠下", text) else .5
    return {"requested": bool(aspect), "aspect": aspect, "fit": fit, "focusX": focus_x, "focusY": focus_y}


def editing_brief(goal: str, context: dict[str, Any]) -> dict[str, Any]:
    text = str(goal or "").strip()
    # Parse requested deliverables only from affirmative clauses. Artifact
    # names inside “不要创建成片/不能沿用旧封面” are constraints, not work
    # the Agent is authorized to perform.
    affirmative_text = _strip_negative_clauses(text)
    # Keep the retrieval instruction separate from the conversational
    # wrapper and the requested deliverable.  Sending the whole sentence
    # (for example “upload this video, find X, then make several cuts”)
    # to semantic retrieval used to pollute the evidence query with
    # planning language and made every request look like the same edit.
    semantic_anchor_range = _semantic_anchor_range(text)
    anchor_start = _anchor_start(text)
    anchor_end = copy.deepcopy((semantic_anchor_range or {}).get("end"))
    retrieval_query = str((anchor_start or {}).get("query") or _retrieval_query(text))
    source_time_range = _source_time_range(text, context)
    removed_source_ranges = [source_time_range] if isinstance(source_time_range, dict) and str(source_time_range.get("kind") or "") == "remove" else []
    source_range = _source_range(text, context)
    excluded_clauses = re.findall(r"(?:不要|不保留|删除|去掉|排除|剔除|移除)([^，,。；;\n]+)", text)
    target_seconds, explicit_duration = _duration_target(text)
    explicit_tolerance = _duration_tolerance(text)
    relative_duration = target_seconds is None and _relative_duration_request(text)
    input_context = context.get("inputContext") if isinstance(context.get("inputContext"), dict) else {}
    editing_context = context.get("editing") if isinstance(context.get("editing"), dict) else {}
    current_duration = input_context.get("outputDurationSeconds")
    current_duration_source = "referenced_output" if isinstance(current_duration, (int, float)) else ""
    if not isinstance(current_duration, (int, float)):
        current_duration = editing_context.get("currentDurationSeconds")
        current_duration_source = str(editing_context.get("currentDurationSource") or "current_edit")
    unresolved_relative_duration = bool(relative_duration and not (
        isinstance(current_duration, (int, float)) and not isinstance(current_duration, bool)
        and float(current_duration) > 0
    ))
    duration_source = "explicit" if explicit_duration else ""
    if relative_duration and not unresolved_relative_duration:
        target_seconds = max(4, round(float(current_duration) * .8))
        explicit_duration = True
        duration_source = "relative"
    elif target_seconds is None and isinstance(context.get("targetSeconds"), (int, float)):
        target_seconds = int(context["targetSeconds"])
        duration_source = "context"
    if relative_duration and not re.search(r"找出|找到|查找|搜索|检索|定位|提取|截取|介绍|讲解|讲到|提到|讨论|说到", text):
        # This is a timeline adjustment, not a request to search for the
        # literal phrase “再短一点” inside the video.
        retrieval_query = ""
    interview = bool(re.search(r"访谈|采访|问答|播客|证言", text))
    cover_intro_requested = _cover_intro_requested(affirmative_text)
    cover_intro_duration = _cover_intro_duration(affirmative_text)
    cover_intro_only_opening = bool(
        cover_intro_requested
        and not re.search(r"短视频|短片|reel|hook|爆点|口播|切片", text, re.IGNORECASE)
    )
    short_form = bool(
        re.search(r"短视频|短片|reel|hook|爆点|口播|切片", text, re.IGNORECASE)
        or (re.search(r"开头", text) and not cover_intro_only_opening)
    )
    # “短视频/Hook” describes an editorial form, not a duration. Only an
    # explicit user duration may become a hard fitting constraint.
    variants = _requested_variant_count(text)
    subtitle_asset_requested = bool(re.search(
        r"(?:只|仅)?\s*(?:导出|下载|生成|保存)\s*(?:字幕文件|字幕\s*SRT|SRT|VTT)|(?:字幕文件|SRT|VTT)\s*(?:导出|下载|保存)",
        affirmative_text,
        re.IGNORECASE,
    ))
    subtitle_requested = bool(re.search(
        r"(?:加上?|添加|生成|制作|配上?|烧录|导出|需要|要有|带)(?:对应的?|顶部|底部|双语|中英|中文|英文|汉语)?字幕"
        r"|字幕(?:稿|文件|样式|校对|审核)|SRT|双语字幕",
        affirmative_text,
        re.IGNORECASE,
    ))
    preview_requested = bool(re.search(r"审阅样片|审核样片|低码率|审阅预览|预览样片", affirmative_text))
    qc_requested = bool(re.search(r"质检|交付检查|检查(?:成片|输出|视频).*(?:质量|交付)|是否可以交付", text))
    social_delivery = _social_delivery(text)
    cover_requested = bool(re.search(r"封面|缩略图|海报帧|poster\s*frame|thumbnail", affirmative_text, re.IGNORECASE))
    audio_polish_requested = bool(re.search(r"降噪|人声增强|声音增强|音频优化|音量|响度|爆音|削波|静音|背景音乐|混音|音乐压低", affirmative_text))
    broll_requested = bool(re.search(r"穿插|补充画面|补画面|覆盖画面|叠加画面|b-?roll|B-?roll|空镜|产品画面覆盖", affirmative_text, re.IGNORECASE))
    graphics_requested = bool(re.search(
        r"标题卡|参数卡|价格卡|报价卡|角标|标签|贴纸|水印|logo|Logo|CTA|关键词高亮|图文|文字(?!幕)层"
        r"|(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}(?:价格|报价)"
        r"|(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}(?:文本|文字(?!幕)|文案|说明)",
        affirmative_text,
    ) or re.search(
        r"(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}?[“\"'‘][^。；;”\"'’\n]{1,120}[”\"'’]\s*(?:文本|文字(?!幕)|文案|说明)",
        affirmative_text,
    ))
    graphics_text_match = re.search(
        r"(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}?(?:文本|文字(?!幕)|文案|说明)"
        r"\s*(?:是|为|用|写|内容为|：|:)?\s*[“\"'‘]([^。；;”\"'’\n]{1,120})[”\"'’]",
        text,
    )
    if not graphics_text_match:
        graphics_text_match = re.search(
            r"(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}?(?:文本|文字(?!幕)|文案|说明)"
            r"\s*(?:是|为|用|写|内容为|：|:)\s*([^，,。；;\n]{1,120})",
            text,
        )
    if not graphics_text_match:
        graphics_text_match = re.search(
            r"(?:添加|加上?|叠加|写上?|显示|放上?)[^。；;\n]{0,24}?"
            r"[“\"'‘]([^。；;”\"'’\n]{1,120})[”\"'’]\s*(?:文本|文字(?!幕)|文案|说明)",
            text,
        )
    graphics_text = str(graphics_text_match.group(1) if graphics_text_match else "").strip(" \t\r\n“”\"'‘’")
    overlay_duration_match = re.search(
        r"(?:文字|文本|文案|标题|贴纸|水印|logo|Logo|字幕|图文|角标|标签)"
        r"[^，,。；;\n]{0,40}?(?:显示|持续|停留|保留)\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))"
        r"|(?:显示|持续|停留|保留)\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))"
        r"[^，,。；;\n]{0,40}?(?:文字|文本|文案|标题|贴纸|水印|logo|Logo|字幕|图文|角标|标签)",
        text,
        re.IGNORECASE,
    )
    overlay_duration_seconds = (
        float(next(value for value in overlay_duration_match.groups() if value))
        if overlay_duration_match else None
    )
    if overlay_duration_seconds is None and graphics_requested:
        overlay_duration_loose = re.search(
            r"(?:显示|持续|停留|保留)\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))",
            text,
            re.IGNORECASE,
        )
        if overlay_duration_loose:
            overlay_duration_seconds = float(overlay_duration_loose.group(1))
    motion_graphics_requested = bool(re.search(
        r"Remotion|HyperFrames?|HTML\s*(?:视频|动效)|动效|动态图文|动态字幕|动态封面|动态片头|标题动画|片头动画|可视化视频",
        text,
        re.IGNORECASE,
    ))
    motion_intro_requested = bool(
        motion_graphics_requested
        and re.search(r"当前成片|已有成片|合入|加入|插入|放到|作为|片头|开头", text)
    )
    draft_export_requested = bool(re.search(r"剪映|CapCut|草稿|工程文件|项目文件|导出草稿|导出工程|外部编辑器", affirmative_text, re.IGNORECASE))
    delivery_export_requested = bool(re.search(r"正式导出|高清导出|导出高清|最终导出|可下载|下载|交付包|发布(?!会)|出片|成品文件", affirmative_text))
    if subtitle_asset_requested and re.search(r"不要|不生成|无需|不需要", text) and re.search(r"视频|成片|样片", text):
        delivery_export_requested = False
    diagnostics_requested = bool(re.search(r"为什么|为何|哪里.{0,8}问题|诊断|排查|卡住|失败|未找到|不合理|怎么回事", text))
    multi_topic_requested = bool(re.search(
        r"分别|各(?:类|段|个|自)|每(?:类|段|个)|\d+\s*段|[一二两三四五六七八九十]\s*段|、.*、",
        retrieval_query or text,
    ))
    composition_requested = bool(re.search(
        r"组合|合成|剪辑|成片|版本|合集|精华|整理(?:成|为)|做成|剪成|剪出来|截出来|制作|编排|剪掉|删掉|裁掉|删除|去掉|排除|剔除|移除|只保留|重排|缩短|加速|返修",
        affirmative_text,
    ))
    cover_title_match = re.search(
        r"(?:封面|缩略图|海报帧)[^。；;\n]{0,20}?(?:标题|文案|文本描述|文字描述|文字)"
        r"\s*(?:仅\s*)?(?:是|为|用|写|添加|加上|改成|：|:)?\s*[“\"'‘]?([^。；;”\"'’\n]{1,120})",
        text,
    )
    if not cover_title_match:
        cover_title_match = re.search(
            r"(?:封面|缩略图|海报帧)[^。；;\n]{0,30}?"
            r"(?:(?:上\s*)?写(?:好)?上?|加上|添加|放上|配上)"
            r"\s*(?:仅\s*)?(?:是|为|用|写|内容为|：|:)?\s*[“\"'‘]?([^。；;”\"'’\n]{1,120})",
            text,
        )
    cover_title = str(cover_title_match.group(1) if cover_title_match else "").strip(" \t\r\n“”\"'‘’")
    cover_title = re.sub(r"^(?:上|好上|写上|写好上)?\s*[：:]\s*", "", cover_title).strip()
    cover_title = re.split(
        r"[，,]\s*(?=(?:(?:并|然后|再|同时|之后|接着)\s*)?"
        r"(?:(?:将|把)?(?:这个|该|当前)?封面.{0,24}(?:成片|视频|样片|片头|开头|合成|拼接|放进|合入|加入|插入)"
        r"|(?:生成|导出|制作|合成|添加|调整|检查|发布)))",
        cover_title,
        maxsplit=1,
    )[0].strip()
    external_cover_match = re.search(
        r"(?:找到|寻找|查找|搜索|网上找|使用|采用|上传|提供)\s*"
        r"(?:一张|一幅|一张合适的|合适的)?\s*"
        r"(.{1,60}?)(?:的)?(?:照片|图片|肖像|头像)\s*"
        r"(?:来)?(?:作为|用作|做成|当作)\s*(?:视频)?封面",
        text,
    )
    source_cover_match = re.search(
        r"(?:用|使用|采用|选用|选取|选择|截取)\s*"
        r"(?:第?\s*\d+(?:\.\d+)?\s*(?:秒钟|秒|s)\s*(?:左右|附近|前后)?\s*(?:出现|看到|有)?(?:的)?\s*)?"
        r"(.{1,60}?)(?:的)?(?:照片|图片|肖像|头像|画面|镜头|帧)?\s*"
        r"(?:来)?(?:作为|用作|做成|当作)\s*(?:视频)?封面",
        text,
        re.IGNORECASE,
    )
    source_cover_time_match = re.search(
        r"(?:用|使用|采用|选用|选取|选择|截取)\s*(?:第)?\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s)"
        r"[^。；;\n]{0,80}?(?:作为|用作|做成|当作)\s*(?:视频)?封面",
        text,
        re.IGNORECASE,
    )
    cover_source_kind = "external_image" if external_cover_match else "source_frame"
    cover_subject = str(
        external_cover_match.group(1)
        if external_cover_match else source_cover_match.group(1) if source_cover_match else ""
    ).strip(
        " \t\r\n，,。；;：:“”\"'‘’"
    )
    cover_subject = re.sub(
        r"^(?:出现|看到|看见|有|画面中|视频中)(?:的)?", "", cover_subject,
    ).strip(" \t\r\n，,。；;：:“”\"'‘’")
    cover_subject = "" if _generic_cover_subject(cover_subject) else cover_subject
    cover_aspect_match = re.search(
        r"(?:封面|缩略图|海报帧)[^。；;\n]{0,40}?(9:16|16:9|4:5|1:1)"
        r"|(9:16|16:9|4:5|1:1)[^。；;\n]{0,20}?(?:封面|缩略图|海报帧)",
        text,
        re.IGNORECASE,
    )
    inherited_output_aspect = str((context.get("delivery") or {}).get("outputAspect") or "")
    cover_aspect = str(
        (cover_aspect_match.group(1) or cover_aspect_match.group(2))
        if cover_aspect_match else social_delivery.get("aspect")
        or (inherited_output_aspect if inherited_output_aspect in {"9:16", "16:9", "1:1", "4:5"} else "16:9")
    )
    # A cover aspect and a video delivery aspect are independent. Keep an
    # explicit vertical/video request when both appear in the same goal;
    # only suppress social delivery for a cover-only request.
    video_format_requested = bool(re.search(
        r"(?:竖屏|横屏|方形|方屏)(?:的)?(?:[\u4e00-\u9fff]{0,8})?(?:视频|短片|成片|输出|版本|审核样片|审阅样片|样片)|社媒|抖音|视频号|小红书|reels?|shorts?"
        r"|(?:9:16|4:5|1:1|16:9)\s*(?:的)?(?:视频|成片|输出|版本)"
        r"|(?:视频|成片|输出|版本)\s*(?:为|做成|改成|转换为|设置为)\s*(?:9:16|4:5|1:1|16:9)",
        text, re.IGNORECASE,
    ) or re.search(r"(?:视频|成片|输出|版本|样片).{0,8}(?:竖屏|横屏|方形|方屏)", text, re.IGNORECASE))
    if cover_requested and not video_format_requested:
        social_delivery = {
            "requested": False, "aspect": "", "fit": "blur", "focusX": .5, "focusY": .5,
        }
    selection_mode = (
        "exclude" if re.search(
            r"(?:删除|去掉|排除|剔除|移除|不保留|不要)"
            r".{0,18}(?:已选|说话人|Speaker|主持人|嘉宾|歌手|人物|这个人|该人物|男性|女性|男生|女生|男人|女人|穿.{0,8}(?:衣|服))",
            text, re.IGNORECASE,
        )
        else "compare" if re.search(r"比较|对比|分别|各自|每个人|每位", text)
        else "include"
    )
    named_speaker_match = re.search(
        r"(?:只保留|保留|选择|找出|找到|提取|截取)\s*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9·]{1,15})\s*(?:说话|发言|回答|讲述|讲话|旁白)",
        text,
        re.IGNORECASE,
    )
    speaker_targeted = bool(re.search(
        r"主持人|嘉宾|说话人|发言人|声音|音色|声纹|旁白|(?:女性|男性|女生|男生|女人|男人|某人).{0,8}(?:说话|发言|回答|讲述|讲话)|Speaker\s*\d+",
        text, re.IGNORECASE,
    ) or named_speaker_match)
    # An interaction such as “主持人与歌手交谈” is a semantic content
    # target, not an instruction to include or exclude a voice cluster.
    if retrieval_query and re.search(r"交谈|对话|交流|谈话|聊天", retrieval_query):
        speaker_targeted = False
    speaker_label_match = re.search(
        r"(?:说话人|Speaker)\s*([A-ZＡ-Ｚ]|\d{1,2})\b",
        text,
        re.IGNORECASE,
    )
    if speaker_label_match:
        label_clause = next((clause for clause in re.split(r"[，,。；;\n]", text) if speaker_label_match.group(0) in clause), "")
        if re.search(r"只保留|保留|选择", label_clause) and not re.search(r"不要|不保留|排除|删除|去掉", label_clause):
            selection_mode = "include"
    speaker_target_label = (
        f"说话人 {str(speaker_label_match.group(1)).upper()}"
        if speaker_label_match else str(named_speaker_match.group(1)).strip() if named_speaker_match else ""
    )
    # ``人物`` in interview prose usually means respondents, not visual
    # identity tracking.  Requiring an appearance/selection cue keeps
    # requests such as “不同人物关于成长的回答” on the interview
    # workflow instead of launching a multi-thousand-frame body scan.
    person_targeted = not speaker_targeted and bool(re.search(
        r"出镜|人脸|面孔|某人|这个人|那个人|该人物|目标人物|"
        r"人物\s*[A-ZＡ-Ｚ0-9一二三四五六七八九十]+|"
        r"(?:选择|确认|指定|识别).{0,8}(?:画面)?人物|"
        r"(?:人物|有哪些人|都有谁).{0,12}(?:选|选择|确认)|"
        r"红衣|黑衣|白衣|蓝衣|穿.{0,8}(?:衣|服)|"
        r"(?:女性|男性|女生|男生|女人|男人).{0,10}(?:出镜|出现|画面)",
        text,
    ))
    person_description_match = re.search(
        r"(穿[^，,。；;]{1,40}?(?:衣|服|上衣)[^，,。；;]{0,12}?(?:人|人物))",
        text,
    )
    person_time_match = re.search(
        r"(?:第)?\s*(\d+(?:\.\d+)?)\s*(?:秒钟|秒|s(?![a-z]))\s*(?:左右|附近|前后)?\s*(?:出现|看到|看见|有)?(?:的)?\s*(这个人|那个人|该人物|这个人物|目标人物)",
        text,
        re.IGNORECASE,
    )
    person_description = str(
        person_description_match.group(1)
        if person_description_match else
        f"{person_time_match.group(1)} 秒出现的{person_time_match.group(2)}"
        if person_time_match else ""
    ).strip()
    if person_time_match and re.search(r"这个人|那个人|该人物|目标人物", retrieval_query):
        retrieval_query = ""
    # “找到 X” is a complete request in its own right.  It should end in
    # evidence review rather than silently creating a timeline, subtitles,
    # and a sample that the user did not ask for.
    search_only_requested = bool(re.search(
        r"(?:只|仅)?\s*(?:列出|展示|查看|检索|查找|找出).{0,40}(?:候选|源视频时间|时间段|时间码)"
        r"|不要.{0,20}(?:成片|视频|样片|封面)",
        text,
    ) and not re.search(r"(?:合成|生成|制作|剪成|做成).{0,20}(?:视频|成片|样片)", affirmative_text))
    cover_intro_only_request = bool(
        cover_requested
        and cover_intro_requested
        and not retrieval_query
        and not source_range
        and not removed_source_ranges
        and not subtitle_requested
        and not audio_polish_requested
        and not broll_requested
        and not graphics_requested
        and not motion_graphics_requested
        and not draft_export_requested
        and not delivery_export_requested
        and not social_delivery.get("requested")
    )
    format_only = bool(social_delivery["requested"] and not retrieval_query)
    timeline_requested = bool(not cover_intro_only_request and (
        composition_requested or short_form or variants > 1 or target_seconds or subtitle_requested
        or preview_requested or anchor_start
        or (social_delivery["requested"] and (retrieval_query or format_only))
    ))
    if search_only_requested and not composition_requested and not subtitle_asset_requested:
        timeline_requested = False
    if subtitle_asset_requested:
        timeline_requested = False
    # A project output aspect is a delivery default, not a new editing
    # request. Apply it only when this goal already asks for a timeline;
    # a search-only request must still stop at candidate review. Explicit
    # aspect wording in the current goal always wins over the default.
    project_aspect = str((context.get("delivery") or {}).get("outputAspect") or "source")
    if timeline_requested and not social_delivery["requested"] and project_aspect in {"16:9", "9:16", "1:1", "4:5"}:
        social_delivery = {
            "requested": True, "aspect": project_aspect,
            "fit": "crop" if (context.get("delivery") or {}).get("outputFit") == "crop" else "blur",
            "focusX": .5, "focusY": .5,
        }
    time_refs: list[dict[str, Any]] = []
    if isinstance(source_range, dict):
        time_refs.append({
            "kind": "source_range",
            "start": source_range.get("start"),
            "end": source_range.get("end"),
            "description": source_range.get("description") or "",
        })
    for item in removed_source_ranges:
        time_refs.append({
            "kind": "remove_range",
            "start": item.get("start"),
            "end": item.get("end"),
            "description": item.get("description") or "",
        })
    if anchor_start and anchor_start.get("sourceTimeSeconds") is not None:
        time_refs.append({
            "kind": "source_anchor_start",
            "time": anchor_start.get("sourceTimeSeconds"),
            "query": anchor_start.get("query") or "",
        })
    elif anchor_start:
        time_refs.append({
            "kind": "semantic_anchor_start",
            "query": anchor_start.get("query") or "",
        })
    if anchor_end:
        time_refs.append({
            "kind": "semantic_anchor_end",
            "query": anchor_end.get("query") or "",
        })
    if source_cover_time_match:
        time_refs.append({
            "kind": "cover_frame",
            "time": float(source_cover_time_match.group(1)),
            "query": cover_subject[:120],
        })
    if person_time_match:
        time_refs.append({
            "kind": "person_reference",
            "time": float(person_time_match.group(1)),
            "query": person_description[:120],
        })
    if overlay_duration_seconds is not None:
        time_refs.append({"kind": "overlay_duration", "duration": overlay_duration_seconds})
    operation_intent = (
        "export_subtitles" if subtitle_asset_requested else
        "search_only" if not timeline_requested and retrieval_query else
        "revise_timeline" if (graphics_requested or broll_requested or (subtitle_requested and not subtitle_asset_requested)) and not retrieval_query else
        "reframe_existing" if format_only and social_delivery.get("requested") and not (source_range or removed_source_ranges) else
        "compose_timeline" if timeline_requested else
        "cover_asset" if cover_requested else
        "inspect"
    )
    return {
        "schemaVersion": 2,
        "goal": text[:1000], "targetSeconds": target_seconds,
        "operationIntent": operation_intent,
        "timeRefs": time_refs,
        "removedSourceRanges": removed_source_ranges,
        "sourceEndAnchor": copy.deepcopy(anchor_end),
        "inheritedContentConstraint": copy.deepcopy((context.get("evidence") or {}).get("contentConstraint"))
        if not retrieval_query and (timeline_requested or social_delivery.get("requested")) else None,
        "durationExplicit": explicit_duration,
        "durationSource": duration_source or "none",
        "relativeDurationBaseSeconds": round(float(current_duration), 3)
        if relative_duration and not unresolved_relative_duration else None,
        "relativeDurationBaseSource": current_duration_source if relative_duration else "",
        "unresolvedRelativeDuration": unresolved_relative_duration,
        "durationToleranceSeconds": (
            explicit_tolerance if explicit_tolerance is not None
            else 15 if target_seconds == 180
            else max(5, round((target_seconds or 60) * .1))
        ),
        "themeOrganized": bool(re.search(r"主题|按.*回答|访谈|采访|问答", text)),
        "interview": interview,
        "removeRepetition": bool(re.search(r"重复|冗余|精简|删", text)),
        "speakerTargeted": speaker_targeted,
        "speakerTargetLabel": speaker_target_label,
        "speakerDescription": speaker_target_label,
        "personTargeted": person_targeted,
        "personDescription": person_description,
        "personSourceTime": float(person_time_match.group(1)) if person_time_match else None,
        "subjectKind": "speaker" if speaker_targeted else "person" if person_targeted else "content",
        "selectionMode": selection_mode,
        "keepQuestionContext": (interview or bool(re.search(r"提问|上下文|转场", affirmative_text))) and not any("提问" in clause or "问题" in clause for clause in excluded_clauses),
        "shortForm": short_form,
        "specificContentTarget": bool(retrieval_query) or bool(re.search(r"围绕|关于|主题|观点|案例|演示|说过|讲到", text)),
        "retrievalQuery": retrieval_query,
        "anchorStart": copy.deepcopy(anchor_start),
        # A cover supplements a video deliverable; it must never turn an
        # explicit short-video/edit request into a cover-only workflow.
        "delivery": "timeline" if timeline_requested else "artifact" if (cover_requested or subtitle_asset_requested) else "candidates",
        "variantCount": variants,
        "requiresEvidenceReview": bool(
            timeline_requested and (retrieval_query or interview or speaker_targeted)
        ),
        "subtitleRequested": subtitle_requested,
        "subtitleAssetRequested": subtitle_asset_requested,
        "reviewPreviewRequested": preview_requested,
        "deliveryQcRequested": qc_requested,
        "coverIntroRequested": cover_intro_requested,
        "coverIntroDurationSeconds": cover_intro_duration,
        "audioPolishRequested": audio_polish_requested,
        "brollRequested": broll_requested,
        "graphicsRequested": graphics_requested,
        "graphicsText": graphics_text,
        "overlayText": graphics_text,
        "overlayDurationSeconds": overlay_duration_seconds,
        "motionGraphicsRequested": motion_graphics_requested,
        "motionIntroRequested": bool(motion_intro_requested),
        "draftExportRequested": draft_export_requested,
        "deliveryExportRequested": delivery_export_requested,
        "diagnosticsRequested": diagnostics_requested,
        "multiTopicRequested": multi_topic_requested,
        "formatOnly": format_only,
        "socialDelivery": social_delivery,
        "coverRequested": cover_requested,
        "coverAspect": cover_aspect,
        "coverTitle": cover_title,
        "coverSourceKind": cover_source_kind,
        "coverSubject": cover_subject[:120],
        "coverIdentityPolicy": "verify" if cover_subject else "ignore",
        "coverSourceTime": float(source_cover_time_match.group(1)) if source_cover_time_match else None,
        "coverSourceStatus": "requires_external_asset" if external_cover_match else "available",
        "sourceScope": "custom" if source_range else str(context.get("sourceScope") or "all"),
        "sourceRange": source_range,
        "excludedContent": [clause for clause in excluded_clauses if not re.search(r"字幕|封面|复用|重复使用|默认|创建|生成|渲染|导出|时长|秒|分钟", clause)],
        "distinctSourceAcrossVariants": bool(re.search(r"(?:不要|不允许|禁止|不能|不得|不)\s*(?:重复使用|复用|重复).{0,8}(?:片段|镜头|素材)", text)),
    }


def _retrieval_query(goal: str) -> str:
    """Extract the semantic target, never the surrounding task chatter."""
    text = re.sub(r"\s+", " ", str(goal or "")).strip()
    if not text:
        return ""
    # Agent-entry messages can carry source metadata and an instruction
    # prefix.  Neither is evidence the video search should try to match.
    text = re.sub(r"(?:上传|从)\s*[^，,；;：:]{1,160}?\s*(?:，|,|；|;)?\s*(?:交给|让)?\s*(?:智能)?剪辑\s*Agent\s*[:：]?", "", text, flags=re.I)
    text = re.sub(r"(?:素材范围|范围)\s*[:：].*$", "", text, flags=re.I)
    # A generic highlight request describes an editorial ranking task,
    # not a semantic phrase that must literally occur in the media.
    if re.search(r"最精彩的部分|精彩部分|高光(?:视频|成片|片段)?", text) and not re.search(
        r"关于|围绕|介绍|讲解|讲到|提到|指定主题|找到|查找|搜索|检索|找出", text
    ):
        return ""
    keep_positive = re.search(
        r"(?:去掉|删除|排除|剔除|移除)\s*没有\s*(.{1,80}?)(?:的)?(?:部分|片段|画面|镜头|内容)"
        r".{0,24}?(?:保留|只保留|留下)\s*有\s*\1",
        text,
    )
    if keep_positive:
        return str(keep_positive.group(1) or "").strip(" ：:，,。；;")[:240]
    enumerated = re.search(
        r"(?:分别(?:是|为)|包括|包含)\s*[:：]?\s*(.{2,160}?)(?=(?:[。！？!?；;]|每(?:个|段)|各(?:个|段)|$))",
        text,
    )
    about_answer = re.search(
        r"关于\s*(.{2,160}?)\s*的(?:回答|回应|发言|观点|内容)",
        text,
    )
    natural_extract = re.search(
        r"(?:把|将)?\s*((?:介绍|讲解|讲到|提到|讨论|说到).{1,120}?)"
        r"(?:的)?(?:部分|片段|内容|画面)?\s*(?:剪出来|截出来|提取出来|保留下来)(?:[。！？!?,，]|$)",
        text,
    )
    match = enumerated or about_answer or natural_extract or re.search(
        r"(?:(?:只\s*)?(?:找出|找到|查找|搜索|检索|定位|提取|截取|列出|展示|查看)(?:并列出)?\s*[:：]?\s*|^(?:只保留|保留|只列出|列出)\s*[:：]?\s*)"
        r"(.{2,160}?)(?=(?:[，,；;。！？!?]|并(?:且|做|生成|给|合成|剪辑|制作|组合|删除|去掉|排除)?|然后|随后|再|做成|剪成|制作|组合|合成)|$)",
        text,
    )
    value = match.group(1) if match else ""
    value = re.sub(
        r"^(?:第\s*)?\d+(?:\.\d+)?\s*(?:秒钟|秒|s(?![a-z]))\s*(?:左右|附近|前后)?\s*(?:出现|看到|看见|有)?",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(r"^(?:所有|全部|视频中|素材中|其中的?)\s*", "", value).strip()
    value = re.sub(r"^(?:有|出现|看到|看见)\s*", "", value).strip()
    value = re.sub(r"^关于", "", value).strip()
    value = re.sub(r"(?:的)?(?:相关)?候选(?:片段|画面|镜头)?(?:和|及)?源视频时间$", "", value).strip()
    value = re.sub(r"(?:的)?(?:相关)?候选(?:片段|画面|镜头)?$", "", value).strip()
    value = re.sub(r"(?:的)?(?:相关)?(?:视频)?(?:片段|画面|镜头|内容|场景|部分)$", "", value).strip(" ：:，,。；;")
    if enumerated and value:
        return f"分别查找{value}，各类片段作为并列候选"[:240]
    if not value:
        # Quick-mode followups often pass only a noun phrase such as
        # “产品新老替换和核心卖点”.  Treat that as the semantic target
        # instead of falling back to the whole Agent goal or shortform
        # routing.  Exclude operational UI commands so “生成计划/重启服务”
        # does not become a media retrieval query.
        phrase = text.strip(" ：:，,。；;")
        if (
            2 <= len(phrase) <= 80
            and not re.search(
                r"上传|交给|Agent|生成|合成|剪辑|剪一个|导出|输出|封面|字幕|竖屏|横屏|方形|方屏|"
                r"做一个|做一条|小红书|抖音|视频号|审核|审阅|预览|样片|短视频|短片|Hook|hook|成片|交付|检查|计划|规划|重启|"
                r"剪掉|删掉|删除|去掉|移除|裁掉|保留后|修复|优化|为什么|为何|怎么|如何|打不开|卡住|失败|测试|9:16|4:5|1:1|16:9",
                phrase,
                re.IGNORECASE,
            )
        ):
            value = phrase
    return value[:240]


def format_seconds_label(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""
    minutes = int(value // 60)
    remainder = value - minutes * 60
    if minutes:
        return f"{minutes}:{int(round(remainder)):02d}"
    return f"{value:g} 秒"


def plan_understanding(brief: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Public, user-facing interpretation of an Agent goal.

    This deliberately mirrors the execution brief but avoids internal flags.
    The UI shows it before approval so users can catch ambiguity such as a
    timestamp being mistaken for output duration.
    """
    context = context if isinstance(context, dict) else {}
    items: list[dict[str, str]] = []

    def add(label: str, value: Any, *, kind: str = "") -> None:
        text = str(value or "").strip()
        if text:
            items.append({"label": label, "value": text[:240], **({"kind": kind} if kind else {})})

    retrieval = str(brief.get("retrievalQuery") or "").strip()
    if retrieval:
        add("检索目标", retrieval, kind="retrieval")

    social = brief.get("socialDelivery") if isinstance(brief.get("socialDelivery"), dict) else {}
    output_bits: list[str] = []
    delivery = str(brief.get("delivery") or "")
    if bool(brief.get("subtitleAssetRequested")):
        output_bits.append("导出字幕文件")
    elif delivery == "timeline":
        output_bits.append("生成可审核剪辑版本")
    elif delivery == "artifact":
        output_bits.append("生成素材资产")
    if social.get("requested"):
        output_bits.append(f"{social.get('aspect') or '目标'} 画幅")
        output_bits.append({"blur": "完整保留画面并虚化补边", "crop": "裁切铺满", "pad": "保留黑边"}.get(str(social.get("fit") or ""), ""))
    if output_bits:
        add("输出方式", " · ".join(bit for bit in output_bits if bit), kind="output")

    if isinstance(brief.get("targetSeconds"), (int, float)):
        tolerance = brief.get("durationToleranceSeconds")
        add(
            "成片时长",
            f"目标 {float(brief['targetSeconds']):g} 秒"
            + (f" · 允许 ±{float(tolerance):g} 秒" if isinstance(tolerance, (int, float)) else ""),
            kind="duration",
        )

    anchor = brief.get("anchorStart") if isinstance(brief.get("anchorStart"), dict) else {}
    if anchor:
        anchor_text = str(anchor.get("query") or "指定位置")
        time_label = format_seconds_label(anchor.get("sourceTimeSeconds"))
        add("起剪位置", f"{time_label} 附近 · {anchor_text}" if time_label else anchor_text, kind="anchor")

    if bool(brief.get("subtitleRequested")):
        if bool(brief.get("subtitleAssetRequested")):
            add("字幕", "导出字幕文件，不生成视频", kind="subtitle")
        else:
            add("字幕", "生成/应用字幕" + (" · 顶部排版" if re.search(r"顶部", str(brief.get("goal") or "")) else ""), kind="subtitle")

    if bool(brief.get("graphicsRequested")):
        text = str(brief.get("graphicsText") or "").strip()
        duration = brief.get("overlayDurationSeconds")
        add(
            "文字层",
            (text or "按要求添加图文/文字层")
            + (f" · 显示 {float(duration):g} 秒" if isinstance(duration, (int, float)) else ""),
            kind="graphics",
        )

    if bool(brief.get("coverRequested")):
        cover_bits = [f"{brief.get('coverAspect') or '16:9'} 封面"]
        if str(brief.get("coverTitle") or "").strip():
            cover_bits.append(f"文字：{brief.get('coverTitle')}")
        source_time_label = format_seconds_label(brief.get("coverSourceTime"))
        source_bits: list[str] = []
        if source_time_label:
            source_bits.append(f"{source_time_label}画面")
        if str(brief.get("coverSubject") or "").strip():
            source_bits.append(f"人物：{brief.get('coverSubject')}")
        if source_bits:
            cover_bits.append(" · ".join(source_bits))
        elif re.search(r"\d|第|秒|s|:", str(brief.get("goal") or ""), re.IGNORECASE):
            cover_bits.append("按指令中的时间点/画面线索取帧")
        add("封面", " · ".join(cover_bits), kind="cover")

    source_range = brief.get("sourceRange") if isinstance(brief.get("sourceRange"), dict) else {}
    if source_range:
        start = format_seconds_label(source_range.get("start")) or "0 秒"
        end = format_seconds_label(source_range.get("end"))
        add("素材范围", f"{start} → {end}" if end else str(source_range.get("description") or "指定范围"), kind="scope")
    elif brief.get("removedSourceRanges"):
        ranges = []
        for item in brief.get("removedSourceRanges") or []:
            if not isinstance(item, dict):
                continue
            start = format_seconds_label(item.get("start")) or "0 秒"
            end = format_seconds_label(item.get("end"))
            ranges.append(f"移除 {start} → {end}" if end else str(item.get("description") or "移除指定范围"))
        add("素材范围", "；".join(ranges), kind="scope")
    elif str(brief.get("sourceScope") or "") == "all":
        add("素材范围", "全片", kind="scope")

    end_anchor = brief.get("sourceEndAnchor") if isinstance(brief.get("sourceEndAnchor"), dict) else {}
    if end_anchor:
        add("结束位置", str(end_anchor.get("query") or "指定结束点"), kind="anchor")

    summary = "；".join(f"{item['label']}：{item['value']}" for item in items[:4])
    return {
        "schemaVersion": 1,
        "title": "Agent 理解",
        "summary": summary[:500],
        "items": items,
    }
