"""Provider registry, recommended presets and runtime model discovery.

ChatClip talks to OpenAI-compatible chat gateways (plus Anthropic Messages).
The provider list below is the single source of truth for the settings UI;
the preset matrices encode the product's recommended domestic model line.
"""

from __future__ import annotations

from typing import Any

import httpx

from .errors import VisionRequestError


PROVIDER_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "bailian",
        "name": "阿里云百炼",
        "description": "通义千问 Qwen 视觉模型（推荐主线：qwen3-vl-max）",
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "ark",
        "name": "火山方舟",
        "description": "豆包及方舟接入点（视觉降级：doubao-seed-2.0）",
        "baseUrl": "https://ark.cn-beijing.volces.com/api/v3",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "bigmodel",
        "name": "智谱 BigModel",
        "description": "GLM 视觉模型",
        "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "OpenAI 官方多模态模型",
        "baseUrl": "https://api.openai.com/v1",
        "baseUrlEditable": False,
        "thinkingSupported": False,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "openai_compatible",
        "name": "兼容接口",
        "description": "其他兼容 Chat Completions 的服务",
        "baseUrl": "",
        "baseUrlEditable": True,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
)


LLM_PROVIDER_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "deepseek",
        "name": "DeepSeek 开放平台",
        "description": "deepseek-flash 规划主线（DeepSeek-V4.1-Flash）",
        "protocol": "openai",
        "baseUrl": "https://api.deepseek.com/v1",
        "baseUrlEditable": False,
        "thinkingSupported": False,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "bailian",
        "name": "阿里云百炼",
        "description": "通义千问文本模型（规划降级：qwen3.8-max / Agent 主线）",
        "protocol": "openai",
        "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "bigmodel",
        "name": "智谱 BigModel",
        "description": "GLM 文本模型（Agent 降级：glm-4.6）",
        "protocol": "openai",
        "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "ark",
        "name": "火山方舟",
        "description": "豆包及方舟文本模型",
        "protocol": "openai",
        "baseUrl": "https://ark.cn-beijing.volces.com/api/v3",
        "baseUrlEditable": False,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "OpenAI 官方文本与推理模型",
        "protocol": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "baseUrlEditable": False,
        "thinkingSupported": False,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "openai_compatible",
        "name": "兼容接口",
        "description": "兼容 Chat Completions 的服务",
        "protocol": "openai",
        "baseUrl": "",
        "baseUrlEditable": True,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "description": "Anthropic 官方 Claude 模型",
        "protocol": "anthropic",
        "baseUrl": "https://api.anthropic.com",
        "baseUrlEditable": False,
        "thinkingSupported": False,
        "responseFormatDefault": "none",
    },
    {
        "id": "anthropic_compatible",
        "name": "Anthropic 兼容接口",
        "description": "兼容 Messages API 的服务",
        "protocol": "anthropic",
        "baseUrl": "",
        "baseUrlEditable": True,
        "thinkingSupported": False,
        "responseFormatDefault": "none",
    },
)


# 推荐模型主线（2026-09 调研定版）：视觉 qwen3-vl-max、规划 deepseek-flash、
# Agent qwen3.8-max；每个角色至多一个旗鼓相当的降级候选。
RECOMMENDED_PRESETS: dict[str, dict[str, str]] = {
    "vision": {"provider": "bailian", "model": "qwen3-vl-max", "thinkingType": "disabled"},
    "llm": {"provider": "deepseek", "model": "deepseek-flash", "thinkingType": ""},
    "agent": {"provider": "bailian", "model": "qwen3.8-max", "thinkingType": "enabled"},
}

FALLBACK_PRESETS: dict[str, list[dict[str, str]]] = {
    "vision": [{"provider": "ark", "model": "doubao-seed-2.0", "thinkingType": "disabled"}],
    "llm": [{"provider": "bailian", "model": "qwen3.8-max", "thinkingType": "disabled"}],
    "agent": [{"provider": "bigmodel", "model": "glm-4.6", "thinkingType": "enabled"}],
}


def vision_provider_label(provider: str) -> str:
    value = provider.strip().lower().replace("-", "_")
    return {
        "ark": "火山方舟",
        "volcengine_ark": "火山方舟",
        "bailian": "阿里云百炼",
        "bigmodel": "智谱 BigModel",
        "deepseek": "DeepSeek 开放平台",
        "openai": "OpenAI",
        "openai_compatible": "OpenAI 兼容接口",
    }.get(value, provider.strip() or "OpenAI 兼容接口")


def llm_provider_label(provider: str) -> str:
    value = provider.strip().lower().replace("-", "_")
    definition = next((item for item in LLM_PROVIDER_DEFINITIONS if item["id"] == value), None)
    return str(definition["name"] if definition else value or "兼容接口")


def _provider_definition(provider: str) -> dict[str, Any]:
    normalized = provider.strip().lower().replace("-", "_")
    return next((dict(item) for item in PROVIDER_DEFINITIONS if item["id"] == normalized), {
        "id": normalized or "openai_compatible",
        "name": vision_provider_label(normalized),
        "description": "OpenAI Chat Completions 兼容服务",
        "baseUrl": "",
        "baseUrlEditable": True,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    })


def _llm_provider_definition(provider: str) -> dict[str, Any]:
    normalized = provider.strip().lower().replace("-", "_")
    return next((dict(item) for item in LLM_PROVIDER_DEFINITIONS if item["id"] == normalized), {
        "id": normalized or "openai_compatible",
        "name": llm_provider_label(normalized),
        "description": "兼容文本模型服务",
        "protocol": "openai",
        "baseUrl": "",
        "baseUrlEditable": True,
        "thinkingSupported": True,
        "responseFormatDefault": "json_object",
    })


def _models_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return f"{value}/models"


def _is_probable_visual_model(model_id: str) -> bool:
    value = model_id.lower()
    excluded = (
        "embedding", "moderation", "whisper", "transcribe", "tts", "speech",
        "realtime", "audio", "image", "sora", "video-generation", "rerank",
    )
    if any(token in value for token in excluded):
        return False
    positive = (
        "doubao", "vision", "vlm", "multimodal", "gpt-4", "gpt-5", "o3", "o4",
        "-vl", "vl-", "qvq", "glm-4v", "glm-4.5v", "glm-4.6v", "glm-5v",
        "seed-2", "omni", "internvl",
    )
    return any(token in value for token in positive)


def _is_probable_text_model(model_id: str) -> bool:
    value = model_id.lower()
    excluded = (
        "embedding", "moderation", "whisper", "transcribe", "tts", "speech",
        "realtime", "audio", "image-generation", "gpt-image", "sora",
        "video-generation", "rerank", "3d-generation",
    )
    return not any(token in value for token in excluded)


def discover_models(*, api_key: str, base_url: str, provider: str, timeout_seconds: float = 20.0) -> list[dict[str, Any]]:
    if not api_key.strip():
        raise VisionRequestError("请先填写 API Key")
    if not base_url.strip():
        raise VisionRequestError("请先填写接口地址")
    url = _models_url(base_url)
    try:
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {api_key.strip()}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
        )
    except httpx.HTTPError as error:
        raise VisionRequestError(f"无法连接{vision_provider_label(provider)}：{error}", retryable=True) from error
    if response.status_code >= 400:
        detail = response.text[:500]
        if response.status_code in {401, 403}:
            raise VisionRequestError("API Key 验证失败，请检查密钥或账号权限")
        raise VisionRequestError(f"模型列表读取失败（HTTP {response.status_code}）：{detail}", retryable=response.status_code >= 500)
    try:
        body = response.json()
    except ValueError as error:
        raise VisionRequestError("模型服务没有返回合法的模型列表") from error
    raw_models = body.get("data") if isinstance(body, dict) else body
    if not isinstance(raw_models, list) and isinstance(body, dict):
        raw_models = body.get("models")
    if not isinstance(raw_models, list):
        raise VisionRequestError("模型服务返回格式不兼容，未找到模型数组")
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    has_explicit_modalities = any(
        isinstance(raw, dict)
        and isinstance(raw.get("modalities"), dict)
        and isinstance(raw["modalities"].get("input_modalities"), list)
        for raw in raw_models
    )
    for raw in raw_models:
        model_id = str(raw.get("id") or raw.get("name") or "").strip() if isinstance(raw, dict) else str(raw).strip()
        if not model_id or model_id in seen:
            continue
        raw_status = str(raw.get("status") or "") if isinstance(raw, dict) else ""
        if raw_status.lower() == "shutdown":
            continue
        modalities = raw.get("modalities") if isinstance(raw, dict) and isinstance(raw.get("modalities"), dict) else {}
        input_modalities = [str(item).lower() for item in modalities.get("input_modalities", []) if item]
        output_modalities = [str(item).lower() for item in modalities.get("output_modalities", []) if item]
        supports_image = "image" in input_modalities
        supports_video = "video" in input_modalities
        task_types = [str(item).lower() for item in (raw.get("task_type") or [])] if isinstance(raw, dict) else []
        domain = str(raw.get("domain") or "").lower() if isinstance(raw, dict) else ""
        explicit_visual = supports_image \
            and (not output_modalities or "text" in output_modalities) \
            and (not task_types or "visualquestionanswering" in task_types or domain == "vlm")
        probable_visual = explicit_visual if has_explicit_modalities else _is_probable_visual_model(model_id)
        # Some compatible services publish exact modality metadata. When it is
        # available, use it instead of guessing from model names.
        if has_explicit_modalities and not explicit_visual:
            continue
        if not has_explicit_modalities and not probable_visual:
            continue
        features = raw.get("features") if isinstance(raw, dict) and isinstance(raw.get("features"), dict) else {}
        structured = features.get("structured_outputs") if isinstance(features.get("structured_outputs"), dict) else {}
        seen.add(model_id)
        models.append({
            "id": model_id,
            "owner": str(raw.get("owned_by") or raw.get("owner") or "") if isinstance(raw, dict) else "",
            "recommended": probable_visual and raw_status.lower() != "retiring" and not any(token in model_id.lower() for token in ("code", "ui-tars")),
            "supportsImage": supports_image if has_explicit_modalities else None,
            "supportsVideo": supports_video if has_explicit_modalities else None,
            "supportsJson": bool(structured.get("json_object")) if structured else None,
            "status": raw_status,
        })
    if not models:
        raise VisionRequestError("当前账号没有返回可见模型")
    return sorted(models, key=lambda item: (not item["recommended"], item.get("status") == "Retiring", item["id"].lower()))


def discover_llm_models(
    *,
    api_key: str,
    base_url: str,
    provider: str,
    protocol: str = "openai",
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Read text-capable planning models without mixing in utility models."""
    if not api_key.strip():
        raise VisionRequestError("请先填写 API Key")
    if not base_url.strip():
        raise VisionRequestError("请先填写接口地址")
    normalized_protocol = protocol.strip().lower()
    if normalized_protocol == "anthropic":
        root = base_url.strip().rstrip("/")
        if provider == "anthropic_compatible" and "volces.com" in root and root.endswith("/api/compatible"):
            url = f"{root[:-len('/api/compatible')]}/api/v3/models"
            headers = {"Authorization": f"Bearer {api_key.strip()}", "Accept": "application/json"}
        else:
            url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
            headers = {
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
                "Accept": "application/json",
            }
    else:
        url = _models_url(base_url)
        headers = {"Authorization": f"Bearer {api_key.strip()}", "Accept": "application/json"}
    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
        )
    except httpx.HTTPError as error:
        raise VisionRequestError(f"无法连接{llm_provider_label(provider)}：{error}", retryable=True) from error
    if response.status_code >= 400:
        detail = response.text[:500]
        if response.status_code in {401, 403}:
            raise VisionRequestError("API Key 验证失败，请检查密钥或账号权限")
        raise VisionRequestError(f"模型列表读取失败（HTTP {response.status_code}）：{detail}", retryable=response.status_code >= 500)
    try:
        body = response.json()
    except ValueError as error:
        raise VisionRequestError("模型服务没有返回合法的模型列表") from error
    raw_models = body.get("data") if isinstance(body, dict) else body
    if not isinstance(raw_models, list) and isinstance(body, dict):
        raw_models = body.get("models")
    if not isinstance(raw_models, list):
        raise VisionRequestError("模型服务返回格式不兼容，未找到模型数组")
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_models:
        model_id = str(raw.get("id") or raw.get("name") or "").strip() if isinstance(raw, dict) else str(raw).strip()
        if not model_id or model_id in seen or not _is_probable_text_model(model_id):
            continue
        raw_status = str(raw.get("status") or "") if isinstance(raw, dict) else ""
        if raw_status.lower() == "shutdown":
            continue
        modalities = raw.get("modalities") if isinstance(raw, dict) and isinstance(raw.get("modalities"), dict) else {}
        input_modalities = [str(item).lower() for item in modalities.get("input_modalities", []) if item]
        output_modalities = [str(item).lower() for item in modalities.get("output_modalities", []) if item]
        if input_modalities and "text" not in input_modalities:
            continue
        if output_modalities and "text" not in output_modalities:
            continue
        features = raw.get("features") if isinstance(raw, dict) and isinstance(raw.get("features"), dict) else {}
        structured = features.get("structured_outputs") if isinstance(features.get("structured_outputs"), dict) else {}
        seen.add(model_id)
        models.append({
            "id": model_id,
            "owner": str(raw.get("owned_by") or raw.get("owner") or raw.get("display_name") or "") if isinstance(raw, dict) else "",
            "recommended": raw_status.lower() != "retiring" and not any(token in model_id.lower() for token in ("embedding", "code", "coder")),
            "supportsJson": bool(structured.get("json_object")) if structured else None,
            "status": raw_status,
        })
    if not models:
        raise VisionRequestError("当前 API Key 没有返回可用文本模型，可尝试手动填写模型 ID")
    return sorted(models, key=lambda item: (not item["recommended"], item.get("status") == "Retiring", item["id"].lower()))
