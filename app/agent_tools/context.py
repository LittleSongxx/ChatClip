"""Shared dispatch context for Agent tool handlers.

``ToolContext`` is built once per ``dispatch_agent_tool`` call in ``app.main``
and carries the values the extracted branch bodies previously read from the
enclosing function scope.  ``no_result`` and ``current_output_reference`` are
moved here verbatim (they used to be closures over ``snapshot``); the methods
read ``self.snapshot`` at call time so the ``propose_timeline_edit`` snapshot
rebind keeps working exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def no_result(
    snapshot: dict[str, Any], reason_code: str, message: str, *, query: str = "",
    candidate_count: int = 0, reliable_count: int = 0,
    coverage_seconds: float = 0.0, artifact_kind: str = "no_match",
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "terminalStatus": "no_result",
        "artifact": {
            "kind": artifact_kind, "reasonCode": reason_code,
            "query": query[:500],
            "searchedScope": str((snapshot.get("request") or {}).get("sourceScopeKind") or "all"),
            "candidateCount": max(0, int(candidate_count)),
            "reliableCandidateCount": max(0, int(reliable_count)),
            "coverageSeconds": round(max(0.0, float(coverage_seconds)), 3),
            "suggestions": suggestions or ["调整检索目标", "补充包含目标内容的素材", "切换到分步审核模式"],
            "message": message,
        },
        "message": message,
    }


def current_output_reference(
    snapshot: dict[str, Any], filename: str = "",
) -> tuple[dict[str, Any] | None, Path | None]:
    from app import main as _main

    available = [item for item in _main.all_job_outputs(snapshot) if isinstance(item, dict) and item.get("filename")]
    if filename:
        output = next((item for item in available if str(item.get("filename")) == filename), None)
    else:
        output = next((item for item in reversed(available) if not item.get("previewOnly")), None)
        if output is None:
            output = next((item for item in reversed(available)), None)
    if not output:
        review_previews = [
            item for item in [
                *(snapshot.get("agentReviewPreviews") or []),
                *(snapshot.get("agentPreviewOutputs") or []),
            ]
            if isinstance(item, dict)
        ]
        output = next((item for item in reversed(review_previews) if item.get("sourceEditSessionId") or item.get("sessionId")), None)
    if not output:
        return None, None
    output_filename = Path(str(output.get("filename") or "")).name
    if output_filename:
        return output, Path(str(snapshot.get("outputDirectory") or "")) / output_filename
    session_id = str(output.get("sourceEditSessionId") or output.get("sessionId") or "")
    if session_id:
        session = next((
            item for item in snapshot.get("editSessions") or []
            if isinstance(item, dict) and str(item.get("id") or "") == session_id
        ), None)
        preview_path = Path(str((session or {}).get("previewPath") or ""))
        if preview_path.is_file():
            return output, preview_path
    direct_path = Path(str(output.get("previewPath") or ""))
    if direct_path.is_file():
        return output, direct_path
    return output, None


@dataclass
class ToolContext:
    """Per-dispatch call state shared by all tool handlers."""

    workspace: dict[str, Any]
    job_id: str
    autonomous: bool
    snapshot: dict[str, Any]
    tool_name: str = ""

    def no_result(
        self, reason_code: str, message: str, *, query: str = "",
        candidate_count: int = 0, reliable_count: int = 0,
        coverage_seconds: float = 0.0, artifact_kind: str = "no_match",
        suggestions: list[str] | None = None,
    ) -> dict[str, Any]:
        return no_result(
            self.snapshot, reason_code, message, query=query,
            candidate_count=candidate_count, reliable_count=reliable_count,
            coverage_seconds=coverage_seconds, artifact_kind=artifact_kind,
            suggestions=suggestions,
        )

    def current_output_reference(
        self, filename: str = "",
    ) -> tuple[dict[str, Any] | None, Path | None]:
        return current_output_reference(self.snapshot, filename)
