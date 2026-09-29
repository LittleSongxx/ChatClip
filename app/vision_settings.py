"""Runtime model-role configuration stores.

One generic ``ModelRoleStore`` backs the three product roles (vision / llm /
agent); role-specific behaviour (Anthropic protocol, reuse-vision mode) is
declared through constructor flags instead of duplicated classes. Provider
metadata and model discovery come from :mod:`app.llm.providers`.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .llm.errors import VisionRequestError
from .llm.providers import (
    LLM_PROVIDER_DEFINITIONS,
    PROVIDER_DEFINITIONS,
    _llm_provider_definition,
    _provider_definition,
    discover_llm_models,
    discover_models,
    llm_provider_label,
    vision_provider_label,
)

__all__ = [
    "LLM_PROVIDER_DEFINITIONS",
    "LlmConfigurationStore",
    "PROVIDER_DEFINITIONS",
    "VisionConfigurationStore",
    "discover_llm_models",
    "discover_models",
    "llm_provider_label",
    "vision_provider_label",
]


def _mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return f"{value[:2]}****"
    return f"{value[:4]}****{value[-4:]}"


class ModelRoleStore:
    """Thread-safe persisted configuration for one model role."""

    supports_mode = False

    def __init__(self, path: Path, defaults: dict[str, Any]) -> None:
        self.path = path
        self.defaults = dict(defaults)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- state IO

    def _empty_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": 1,
            "activeProvider": self.default_provider,
        }
        if self.supports_mode:
            state = {
                "version": 1,
                "mode": str(self.defaults.get("mode") or "reuse_vision"),
                "activeProvider": self.default_provider,
            }
        state["providers"] = {}
        return state

    @property
    def default_provider(self) -> str:
        return str(self.defaults.get("provider") or "openai_compatible")

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty_state()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty_state()
        if not isinstance(value, dict):
            return self._empty_state()
        base = self._empty_state()
        value.setdefault("providers", {})
        value.setdefault("activeProvider", base["activeProvider"])
        if self.supports_mode:
            value.setdefault("mode", base["mode"])
        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)

    # --------------------------------------------------------------- resolving

    def resolve(self, provider: str | None = None, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            mode = str(snapshot.get("mode") or state.get("mode") or self.defaults.get("mode") or "reuse_vision")
            provider_id = str(
                provider or snapshot.get("provider") or state.get("activeProvider") or self.default_provider
            ).strip().lower().replace("-", "_")
            definition = self._definition(provider_id)
            saved = dict((state.get("providers") or {}).get(provider_id) or {})
            use_defaults = provider_id == self.defaults.get("provider")
            resolved = {
                "provider": provider_id,
                "apiKey": str(saved.get("apiKey") or (self.defaults.get("apiKey") if use_defaults else "") or ""),
                "model": str(saved.get("model") or (self.defaults.get("model") if use_defaults else "") or ""),
                "baseUrl": str(saved.get("baseUrl") or (self.defaults.get("baseUrl") if use_defaults else "") or definition.get("baseUrl") or "").rstrip("/"),
                "thinkingType": str(saved.get("thinkingType") if saved.get("thinkingType") is not None else (self.defaults.get("thinkingType") if use_defaults else "") or ""),
                "responseFormat": str(saved.get("responseFormat") or (self.defaults.get("responseFormat") if use_defaults else "") or definition["responseFormatDefault"]),
                "timeoutSeconds": float(saved.get("timeoutSeconds") or (self.defaults.get("timeoutSeconds") if use_defaults else 60.0) or 60.0),
                "models": list(saved.get("models") or []),
                "verifiedAt": saved.get("verifiedAt"),
                "keySource": "saved" if saved.get("apiKey") else ("environment" if use_defaults and self.defaults.get("apiKey") else "none"),
            }
            if self.supports_mode:
                resolved["mode"] = "reuse_vision" if mode == "reuse_vision" else "independent"
                resolved["protocol"] = str(saved.get("protocol") or definition["protocol"])
            override_keys = (
                [("protocol", "protocol")] if self.supports_mode else []
            ) + [
                ("model", "model"), ("baseUrl", "baseUrl"),
                ("thinkingType", "thinkingType"), ("responseFormat", "responseFormat"),
                ("timeoutSeconds", "timeoutSeconds"),
            ]
            for source_key, target_key in override_keys:
                if snapshot.get(source_key) not in (None, ""):
                    resolved[target_key] = snapshot[source_key]
            return resolved

    def snapshot(self) -> dict[str, Any]:
        resolved = self.resolve()
        result: dict[str, Any] = {}
        if self.supports_mode:
            result["mode"] = resolved["mode"]
            if resolved["mode"] == "reuse_vision":
                return result
        result.update({
            "provider": resolved["provider"],
            "providerLabel": self._label(resolved["provider"]),
        })
        if self.supports_mode:
            result["protocol"] = resolved["protocol"]
        result.update({
            "model": resolved["model"],
            "baseUrl": resolved["baseUrl"],
            "thinkingType": resolved["thinkingType"],
            "responseFormat": resolved["responseFormat"],
            "timeoutSeconds": resolved["timeoutSeconds"],
        })
        return result

    # ------------------------------------------------------------------ saving

    def save(
        self,
        *,
        provider: str = "",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        thinking_type: str = "",
        response_format: str = "json_object",
        models: list[dict[str, Any]] | None = None,
        verified_at: str | None = None,
        reuse_vision: bool = False,
    ) -> dict[str, Any]:
        if self.supports_mode and reuse_vision:
            with self._lock:
                state = self._read()
                state["mode"] = "reuse_vision"
                self._write(state)
                return self.resolve()
        provider_id = provider.strip().lower().replace("-", "_") or self.default_provider
        current = self.resolve(provider_id)
        normalized_base_url = base_url.strip().rstrip("/")
        current_base_url = str(current.get("baseUrl") or "").strip().rstrip("/")
        if not api_key.strip() and current["apiKey"] and normalized_base_url != current_base_url:
            raise VisionRequestError("接口地址已改变，请重新填写 API Key，不能复用旧地址的密钥")
        key = api_key.strip() or current["apiKey"]
        if not key:
            raise VisionRequestError("请填写 API Key")
        if not model.strip():
            raise VisionRequestError(self.model_required_message)
        if not base_url.strip():
            raise VisionRequestError("请填写接口地址")
        normalized_models = [
            {
                "id": model_id,
                "owner": str(item.get("owner") or ""),
                "recommended": bool(item.get("recommended")),
                "supportsImage": item.get("supportsImage"),
                "supportsVideo": item.get("supportsVideo"),
                "supportsJson": item.get("supportsJson"),
                "status": str(item.get("status") or ""),
            }
            for item in models or current.get("models") or []
            if isinstance(item, dict) and (model_id := str(item.get("id") or "").strip())
        ]
        definition = self._definition(provider_id)
        with self._lock:
            state = self._read()
            if self.supports_mode:
                state["mode"] = "independent"
            state["activeProvider"] = provider_id
            record: dict[str, Any] = {
                "apiKey": key,
                "model": model.strip(),
                "baseUrl": normalized_base_url,
                "thinkingType": thinking_type.strip().lower(),
                "responseFormat": response_format.strip().lower() or definition["responseFormatDefault"],
                "timeoutSeconds": current["timeoutSeconds"],
                "models": normalized_models,
                "verifiedAt": verified_at or current.get("verifiedAt"),
            }
            if self.supports_mode:
                record = {"protocol": definition["protocol"], **record}
            state.setdefault("providers", {})[provider_id] = record
            self._write(state)
        return self.resolve(provider_id)

    def mark_verified(self, *, provider: str, api_key: str, base_url: str, models: list[dict[str, Any]], verified_at: str) -> None:
        provider_id = provider.strip().lower().replace("-", "_")
        with self._lock:
            state = self._read()
            record = dict((state.get("providers") or {}).get(provider_id) or {})
            # Discovery is a user-initiated connection test. Persisting the key
            # here lets the subsequent Save action keep the masked credential.
            record.update({"apiKey": api_key, "baseUrl": base_url.rstrip("/"), "models": models, "verifiedAt": verified_at})
            state.setdefault("providers", {})[provider_id] = record
            self._write(state)

    # ------------------------------------------------------------ public state

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            state = self._read()
            active = str(state.get("activeProvider") or self.default_provider)
            providers = []
            for definition_source in self.provider_definitions:
                provider_id = str(definition_source["id"])
                definition = self._definition(provider_id)
                resolved = self.resolve(provider_id)
                providers.append({
                    **definition,
                    "active": provider_id == active,
                    "configured": bool(resolved["apiKey"] and resolved["model"] and resolved["baseUrl"]),
                    "keyConfigured": bool(resolved["apiKey"]),
                    "keyHint": _mask_key(resolved["apiKey"]),
                    "keySource": resolved["keySource"],
                    "model": resolved["model"],
                    "baseUrl": resolved["baseUrl"],
                    "thinkingType": resolved["thinkingType"],
                    "responseFormat": resolved["responseFormat"],
                    "models": resolved["models"],
                    "verifiedAt": resolved["verifiedAt"],
                })
            result: dict[str, Any] = {"activeProvider": active, "providers": providers}
            if self.supports_mode:
                mode = "reuse_vision" if state.get("mode") == "reuse_vision" else "independent"
                result = {"mode": mode, "reuseVision": mode == "reuse_vision", **result}
            return result

    # -------------------------------------------------------------- role hooks

    provider_definitions: tuple[dict[str, Any], ...] = PROVIDER_DEFINITIONS

    model_required_message = "请选择或填写视觉模型"

    def _definition(self, provider_id: str) -> dict[str, Any]:
        return _provider_definition(provider_id)

    def _label(self, provider_id: str) -> str:
        return vision_provider_label(provider_id)


class VisionConfigurationStore(ModelRoleStore):
    """Vision (multimodal analysis) role."""


class LlmConfigurationStore(ModelRoleStore):
    """Text-planning role with optional VLM reuse mode and Anthropic protocol."""

    supports_mode = True
    provider_definitions = LLM_PROVIDER_DEFINITIONS
    model_required_message = "请选择或填写剪辑规划模型"

    def _definition(self, provider_id: str) -> dict[str, Any]:
        return _llm_provider_definition(provider_id)

    def _label(self, provider_id: str) -> str:
        return llm_provider_label(provider_id)
