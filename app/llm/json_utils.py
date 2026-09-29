from __future__ import annotations

import json
import re
from typing import Any

from .errors import JsonResponseError


def _remove_trailing_json_commas(value: str) -> str:
    # Only remove punctuation outside strings. Never invent missing fields,
    # quotes, timestamps or closing braces in a truncated model response.
    result = []
    in_string = escaped = False
    for index, char in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ",":
            following = index + 1
            while following < len(value) and value[following].isspace():
                following += 1
            if following < len(value) and value[following] in "}]":
                continue
        result.append(char)
    return "".join(result)


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    else:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start:end + 1]
    try:
        # Vision models occasionally put a literal newline/tab inside a JSON
        # string even when explicitly asked for JSON. Python's non-strict mode
        # accepts those control characters without weakening structural checks.
        parsed = json.loads(_remove_trailing_json_commas(value), strict=False)
    except json.JSONDecodeError as error:
        raise JsonResponseError(f"视觉模型没有返回合法 JSON：{error}", retryable=True) from error
    if not isinstance(parsed, dict):
        raise JsonResponseError("视觉模型返回值必须是 JSON 对象", retryable=True)
    return parsed
