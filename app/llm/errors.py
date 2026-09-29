from __future__ import annotations


class VisionRequestError(RuntimeError):
    """A model request failed; ``retryable`` distinguishes transient errors."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# Historical alias kept for saved integrations and existing callers.
LlmRequestError = VisionRequestError
ArkRequestError = VisionRequestError


class JsonResponseError(VisionRequestError):
    """A response that must be regenerated rather than treated as evidence."""


JSON_RETRY_INSTRUCTION = (
    "上一条响应不是完整有效的 JSON 对象。请重新检查原始素材并返回完整 JSON，"
    "不要 Markdown 或解释。字符串内的双引号必须转义，字段间必须有逗号，"
    "数组和对象必须闭合。保持原要求的字段和证据，不要凭空补充时间或候选；"
    "精简说明文字以确保响应完整。"
)
