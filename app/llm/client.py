"""JSON-oriented client adapter over LangChain chat models.

Keeps the small provider-neutral contract the highlight pipeline and the
planning call sites rely on (``complete_json`` / ``analyze_image`` /
``analyze_video`` / cooperative ``cancel``) while delegating transport,
retries and message assembly to LangChain integrations from ``factory``.
"""

from __future__ import annotations

import base64
import mimetypes
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .errors import JSON_RETRY_INSTRUCTION, JsonResponseError, VisionRequestError
from .factory import create_chat_model
from .json_utils import parse_json_object
from .providers import vision_provider_label


def _error_status(error: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _to_vision_error(error: Exception, *, provider_name: str) -> VisionRequestError:
    status = _error_status(error)
    detail = str(error)[:800]
    if isinstance(error, VisionRequestError):
        return error
    if status is not None and 400 <= status < 500 and status not in (408, 429):
        return VisionRequestError(f"{provider_name}请求失败（HTTP {status}）：{detail}")
    return VisionRequestError(
        f"{provider_name}暂时不可用（HTTP {status}）：{detail}" if status
        else f"{provider_name}暂时不可用：{detail}",
        retryable=True,
    )


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content)


def _usage_payload(message: AIMessage) -> dict[str, Any]:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return {}
    for key in ("token_usage", "usage"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return {}


@runtime_checkable
class VisionModelClient(Protocol):
    """Small provider-neutral contract used by the highlight pipeline."""

    model: str

    def cancel(self) -> None: ...

    def analyze_image(
        self,
        prompt: str,
        image_path: Path,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]: ...

    def analyze_video(
        self,
        prompt: str,
        video_path: Path,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]: ...

    def complete_json(
        self,
        prompt: str,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]: ...


class LangChainJsonClient:
    """Multimodal JSON client for OpenAI-compatible roles (vision/planning)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        provider: str = "openai_compatible",
        thinking_type: str = "",
        response_format: str = "json_object",
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model = model
        self.provider = provider.strip().lower().replace("-", "_") or "openai_compatible"
        self.provider_name = vision_provider_label(self.provider)
        self.thinking_type = thinking_type.strip().lower()
        self.response_format = response_format.strip().lower()
        self._chat = create_chat_model({
            "provider": self.provider,
            "protocol": "openai",
            "apiKey": api_key,
            "model": model,
            "baseUrl": base_url,
            "thinkingType": self.thinking_type,
            "timeoutSeconds": timeout_seconds,
        })
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Cooperatively stop issuing further requests for this client."""
        self._cancelled.set()

    def analyze_image(
        self,
        prompt: str,
        image_path: Path,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]
        return self._complete(content, maximum_tokens=maximum_tokens, system_prompt=system_prompt)

    def analyze_video(
        self,
        prompt: str,
        video_path: Path,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Send the rendered review proxy as real dynamic video evidence.

        OpenAI-compatible multimodal gateways that do not implement
        ``video_url`` raise a normal request error; callers then fall back to
        the labelled contact sheet without losing the review job.
        """
        encoded = base64.b64encode(Path(video_path).read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(Path(video_path).name)[0] or "video/mp4"
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        return self._complete(content, maximum_tokens=maximum_tokens, system_prompt=system_prompt)

    def complete_json(
        self,
        prompt: str,
        *,
        maximum_tokens: int = 2200,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        return self._complete(prompt, maximum_tokens=maximum_tokens, system_prompt=system_prompt)

    def _bound_model(self, maximum_tokens: int) -> Any:
        kwargs: dict[str, Any] = {"max_tokens": maximum_tokens}
        if self.response_format in {"json_object", "json"}:
            kwargs["response_format"] = {"type": "json_object"}
        return self._chat.bind(**kwargs)

    def _complete(
        self,
        content: str | list[dict[str, Any]],
        *,
        maximum_tokens: int,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        if self._cancelled.is_set():
            raise VisionRequestError(f"{self.provider_name}请求已取消")
        messages: list[Any] = []
        if system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt.strip()))
        messages.append(HumanMessage(content=content))
        bound = self._bound_model(maximum_tokens)
        last_error: Exception | None = None
        for attempt in range(2):
            if self._cancelled.is_set():
                raise VisionRequestError(f"{self.provider_name}请求已取消")
            try:
                answer = bound.invoke(messages)
            except Exception as error:  # noqa: BLE001 - mapped below
                last_error = _to_vision_error(error, provider_name=self.provider_name)
                if self._cancelled.is_set():
                    raise VisionRequestError(f"{self.provider_name}请求已取消") from error
                if not last_error.retryable or attempt == 1:
                    raise last_error from error
                if self._cancelled.wait(1.5 * (attempt + 1)):
                    raise VisionRequestError(f"{self.provider_name}请求已取消") from error
                continue
            text = _message_text(answer)
            if getattr(answer, "response_metadata", {}).get("finish_reason") == "length":
                last_error = JsonResponseError("视觉模型响应达到长度上限，JSON 可能不完整", retryable=True)
            elif not text.strip():
                last_error = VisionRequestError(f"{self.provider_name}返回了空结果", retryable=True)
            else:
                try:
                    parsed = parse_json_object(text)
                except JsonResponseError as error:
                    last_error = error
                else:
                    parsed["_usage"] = _usage_payload(answer)
                    return parsed
            if isinstance(last_error, JsonResponseError) and attempt == 0:
                messages = list(messages) + [
                    HumanMessage(content=content),
                    HumanMessage(content=JSON_RETRY_INSTRUCTION),
                ]
                bound = self._bound_model(max(maximum_tokens, min(16384, maximum_tokens * 2)))
            if self._cancelled.is_set():
                raise VisionRequestError(f"{self.provider_name}请求已取消")
            if not getattr(last_error, "retryable", False) or attempt == 1:
                raise last_error
            if self._cancelled.wait(1.5 * (attempt + 1)):
                raise VisionRequestError(f"{self.provider_name}请求已取消") from last_error
        raise VisionRequestError(str(last_error or f"{self.provider_name}请求失败"))


def prompt_text(content: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
    )


class AnthropicJsonClient:
    """Text-planning client for the Anthropic Messages protocol."""

    def __init__(
        self,
        *,
        auth_token: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self._chat = create_chat_model({
            "provider": "anthropic",
            "protocol": "anthropic",
            "apiKey": auth_token,
            "model": model,
            "baseUrl": base_url,
            "timeoutSeconds": timeout_seconds,
        })
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def complete_json(self, prompt: str, *, maximum_tokens: int = 2200, system_prompt: str = "") -> dict[str, Any]:
        if self._cancelled.is_set():
            raise VisionRequestError("Anthropic 兼容接口请求已取消")
        messages: list[Any] = []
        if system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt.strip()))
        messages.append(HumanMessage(content=prompt))
        bound = self._chat.bind(max_tokens=maximum_tokens)
        last_error: Exception | None = None
        for attempt in range(2):
            if self._cancelled.is_set():
                raise VisionRequestError("Anthropic 兼容接口请求已取消")
            try:
                answer = bound.invoke(messages)
            except Exception as error:  # noqa: BLE001 - mapped below
                last_error = _to_vision_error(error, provider_name="Anthropic 兼容接口")
                if self._cancelled.is_set():
                    raise VisionRequestError("Anthropic 兼容接口请求已取消") from error
                if not last_error.retryable or attempt == 1:
                    raise last_error from error
                if self._cancelled.wait(1.5 * (attempt + 1)):
                    raise VisionRequestError("Anthropic 兼容接口请求已取消") from error
                continue
            text = _message_text(answer)
            if getattr(answer, "response_metadata", {}).get("stop_reason") == "max_tokens":
                last_error = JsonResponseError("视觉模型响应达到长度上限，JSON 可能不完整", retryable=True)
            elif not text.strip():
                last_error = VisionRequestError("Anthropic 兼容接口返回了空结果", retryable=True)
            else:
                try:
                    parsed = parse_json_object(text)
                except JsonResponseError as error:
                    last_error = error
                else:
                    parsed["_usage"] = _usage_payload(answer)
                    return parsed
            if isinstance(last_error, JsonResponseError) and attempt == 0:
                messages = list(messages) + [
                    HumanMessage(content=prompt),
                    HumanMessage(content=JSON_RETRY_INSTRUCTION),
                ]
                bound = self._chat.bind(max_tokens=max(16384, maximum_tokens * 2))
            if self._cancelled.is_set():
                raise VisionRequestError("Anthropic 兼容接口请求已取消")
            if not getattr(last_error, "retryable", False) or attempt == 1:
                raise last_error
            if self._cancelled.wait(1.5 * (attempt + 1)):
                raise VisionRequestError("Anthropic 兼容接口请求已取消") from last_error
        raise VisionRequestError(str(last_error or "Anthropic 兼容接口请求失败"))


# Historical class names retained so saved integrations keep importing.
OpenAICompatibleVisionClient = LangChainJsonClient
ArkRequestError = VisionRequestError


def create_vision_client(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    thinking_type: str = "",
    response_format: str = "json_object",
    timeout_seconds: float = 90.0,
    auth_token: str = "",
    **_deprecated: Any,
) -> Any:
    normalized_provider = provider.strip().lower().replace("-", "_") or "openai_compatible"
    if "anthropic" in normalized_provider:
        return AnthropicJsonClient(
            auth_token=auth_token or api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    return LangChainJsonClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=normalized_provider,
        thinking_type=thinking_type or ("disabled" if normalized_provider in {"ark", "volcengine_ark"} else ""),
        response_format=response_format,
        timeout_seconds=timeout_seconds,
    )
