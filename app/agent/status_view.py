"""Four-way plan status projection.

The stored ``plan.status`` mixes execution state with quality and artifact
semantics. This projection derives the orthogonal views the UI reasons
about, while the raw field remains for compatibility:

- ``planStatus``      — execution lifecycle (mirrors the stored status)
- ``artifactStatus``  — what media artifact exists (none/draft/preview/exported)
- ``qualityStatus``   — delivery-QC verdict (unknown/passed/needs_review)
- ``userActionStatus``— what the user is being asked to do (none/confirm_plan/
                        resolve_action/review_result)
"""

from __future__ import annotations

from typing import Any

_PREVIEW_ARTIFACT_KINDS = ("review_preview", "review_preview_batch", "social_reframe_preview")


def _step_artifacts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for step in plan.get("steps") or []:
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
        if artifact:
            artifacts.append(artifact)
    return artifacts


def artifact_status(plan: dict[str, Any]) -> str:
    kinds = [str(item.get("kind") or "") for item in _step_artifacts(plan)]
    if any("export" in kind or "delivery_master" in kind for kind in kinds):
        return "exported"
    if any(any(marker in kind for marker in _PREVIEW_ARTIFACT_KINDS) for kind in kinds):
        return "preview"
    if any("timeline" in kind or "proposal" in kind or "subtitle" in kind for kind in kinds):
        return "draft"
    return "none"


def quality_status(plan: dict[str, Any]) -> str:
    reports = [
        item for item in _step_artifacts(plan)
        if str(item.get("kind") or "") == "delivery_qc_report"
    ]
    if not reports:
        return "unknown"
    return "passed" if all(report.get("passed") is not False for report in reports) else "needs_review"


def user_action_status(plan: dict[str, Any]) -> str:
    status = str(plan.get("status") or "")
    if status == "awaiting_confirmation":
        return "confirm_plan"
    if status == "action_required":
        return "resolve_action"
    if status in {"preview_ready", "no_result"}:
        return "review_result"
    return "none"


def project_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the four orthogonal status views for one plan."""
    return {
        "planStatus": str(plan.get("status") or ""),
        "artifactStatus": artifact_status(plan),
        "qualityStatus": quality_status(plan),
        "userActionStatus": user_action_status(plan),
    }
