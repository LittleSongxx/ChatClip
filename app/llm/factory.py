"""LangChain chat-model factory driven by resolved role configuration.

A role configuration is the ``resolve()`` dict produced by the settings
stores: ``provider/protocol/apiKey/model/baseUrl/thinkingType/timeoutSeconds``.
The factory maps that onto native LangChain integrations — ``ChatDeepSeek``
for DeepSeek, ``ChatAnthropic`` for the Messages protocol, and ``ChatOpenAI``
pointed at each domestic gateway's OpenAI-compatible endpoint otherwise.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .errors import VisionRequestError

_THINKING_STATES = {"disabled", "enabled", "auto"}


def normalize_protocol(config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "").strip().lower().replace("-", "_")
    protocol = str(config.get("protocol") or "").strip().lower()
    if "anthropic" in protocol or "anthropic" in provider:
        return "anthropic"
    return "openai"


def _thinking_type(config: dict[str, Any]) -> str:
    return str(config.get("thinkingType") or "").strip().lower()


def _extra_body(provider: str, thinking_type: str) -> dict[str, Any]:
    """Provider-specific reasoning switches kept in OpenAI-compatible bodies."""
    if thinking_type not in _THINKING_STATES:
        return {}
    if provider in {"ark", "bigmodel"}:
        return {"thinking": {"type": thinking_type}}
    if provider == "bailian":
        # DashScope compatible mode gates Qwen3 hybrid reasoning globally.
        return {"enable_thinking": thinking_type != "disabled"}
    return {}


def create_chat_model(config: dict[str, Any], *, temperature: float = 0.1) -> BaseChatModel:
    """Build a LangChain chat model from a resolved role configuration."""
    api_key = str(config.get("apiKey") or "").strip()
    model = str(config.get("model") or "").strip()
    base_url = str(config.get("baseUrl") or "").strip().rstrip("/")
    provider = str(config.get("provider") or "").strip().lower().replace("-", "_")
    try:
        timeout_seconds = max(1.0, float(config.get("timeoutSeconds") or 60.0))
    except (TypeError, ValueError):
        timeout_seconds = 60.0
    if not api_key or not model or not base_url:
        raise VisionRequestError("模型连接尚未配置完整（缺少 API Key、模型或接口地址）")

    if normalize_protocol(config) == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=0,
        )
    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model,
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=0,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout_seconds,
        max_retries=0,
        extra_body=_extra_body(provider, _thinking_type(config)),
    )
