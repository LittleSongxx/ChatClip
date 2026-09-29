"""LangChain model layer for ChatClip.

All remote model traffic (vision analysis, planning LLM, agent tool calling)
goes through this package. Provider metadata, the model factory and the
JSON-oriented client adapter live here so the rest of the application never
builds raw HTTP requests against model gateways.
"""

from .client import (
    AnthropicJsonClient,
    LangChainJsonClient,
    create_vision_client,
)
from .errors import JsonResponseError, VisionRequestError
from .factory import create_chat_model
from .json_utils import parse_json_object
from .providers import (
    FALLBACK_PRESETS,
    LLM_PROVIDER_DEFINITIONS,
    PROVIDER_DEFINITIONS,
    RECOMMENDED_PRESETS,
    discover_llm_models,
    discover_models,
    llm_provider_label,
    vision_provider_label,
)

__all__ = [
    "AnthropicJsonClient",
    "FALLBACK_PRESETS",
    "JsonResponseError",
    "LangChainJsonClient",
    "LLM_PROVIDER_DEFINITIONS",
    "PROVIDER_DEFINITIONS",
    "RECOMMENDED_PRESETS",
    "VisionRequestError",
    "create_chat_model",
    "create_vision_client",
    "discover_llm_models",
    "discover_models",
    "llm_provider_label",
    "parse_json_object",
    "vision_provider_label",
]
