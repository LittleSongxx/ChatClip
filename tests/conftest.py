"""Shared test configuration.

Tests must never reach the HuggingFace Hub: model-weight downloads belong to
the installer (tools/prepare_*), and a blocked network would otherwise turn
tokenizer loads into indefinite TCP connects. Offline mode makes uncached
loads fail fast; application code already degrades those paths gracefully
(e.g. the voice semantic filter falls back to lexical matching).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
