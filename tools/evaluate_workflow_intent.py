from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "benchmarks" / "workflow-intent-v2.jsonl"
VALID_EXPECTED = {"highlight", "content_search", "person_edit", "speaker_edit", "clarification"}


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        expected = str(value.get("expected") or "")
        text = str(value.get("text") or "")
        if expected not in VALID_EXPECTED or not text:
            raise ValueError(f"{path}:{line_number} 不是有效意图样例")
        # Optional corpus fields (context / expectedKind) document skill-level
        # ground truth; the intent benchmark itself only scores expected/text.
        cases.append({"expected": expected, "text": text})
    return cases


def _rule_decision(text: str) -> dict[str, Any]:
    """Offline deterministic ladder; no service, no model credentials."""
    if "app" not in sys.modules:
        sys.path.insert(0, str(PROJECT_ROOT))
    from app.intent_router import route_editing_instruction

    decision = route_editing_instruction(text)
    actual = "clarification" if decision.needs_confirmation else str(decision.workflow_kind or "")
    return {
        "actual": actual or "clarification",
        "decision": {
            "workflowKind": decision.workflow_kind,
            "needsConfirmation": decision.needs_confirmation,
            "confidence": round(decision.confidence, 3),
            "source": decision.source,
        },
    }


def _llm_decision(client: httpx.Client, text: str) -> dict[str, Any]:
    """Raw model choice before the deterministic guard normalizes it."""
    response = client.post("/api/workflow-intent/classify", json={"text": text})
    response.raise_for_status()
    payload = response.json()
    decision = payload.get("decision") if isinstance(payload, dict) else {}
    raw = str(decision.get("rawWorkflowKind") or "")
    if not raw:
        raise ValueError("响应缺少 rawWorkflowKind（服务版本过旧）")
    return {"actual": raw, "decision": decision}


def _hybrid_decision(client: httpx.Client, text: str) -> dict[str, Any]:
    response = client.post("/api/workflow-intent/classify", json={"text": text})
    response.raise_for_status()
    payload = response.json()
    decision = payload.get("decision") if isinstance(payload, dict) else {}
    actual = "clarification" if decision.get("needsConfirmation") else str(decision.get("workflowKind") or "")
    return {"actual": actual, "decision": decision}


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases = load_cases(args.corpus)
    results = []
    http_client: httpx.Client | None = None
    if args.mode in {"llm", "hybrid"}:
        headers = {"X-ChatClip-Token": args.token} if args.token else {}
        http_client = httpx.Client(
            base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout,
        )
    router = {"rules": lambda _c, text: _rule_decision(text),
              "llm": _llm_decision, "hybrid": _hybrid_decision}[args.mode]
    try:
        for case in cases:
            try:
                outcome = router(http_client, case["text"]) if http_client else router(None, case["text"])
                actual = outcome["actual"]
                safe = actual == case["expected"] or actual == "clarification"
                results.append({
                    **case, "actual": actual, "exact": actual == case["expected"], "safe": safe,
                    "decision": outcome["decision"],
                })
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                results.append({**case, "actual": "error", "exact": False, "safe": True, "error": str(error)})
    finally:
        if http_client is not None:
            http_client.close()
    exact_count = sum(bool(item["exact"]) for item in results)
    unsafe_count = sum(not bool(item["safe"]) for item in results)
    exact_rate = exact_count / len(results) if results else 0.0
    passed = exact_rate >= args.minimum_exact_rate and unsafe_count == 0
    report = {
        "schemaVersion": 2,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "baseUrl": args.base_url if args.mode in {"llm", "hybrid"} else None,
        "corpus": str(args.corpus),
        "summary": {
            "total": len(results), "exact": exact_count,
            "exactRate": round(exact_rate, 4), "unsafe": unsafe_count,
            "minimumExactRate": args.minimum_exact_rate, "passed": passed,
        },
        "results": results,
    }
    return report, 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测意图路由：rules 离线规则梯 / llm 纯模型 / hybrid 规则守卫混合")
    parser.add_argument("--mode", choices=["rules", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--base-url", default="http://127.0.0.1:5180")
    parser.add_argument("--token", default="")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="报告输出路径，默认 test-results/workflow-intent-{mode}.json",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--minimum-exact-rate", type=float, default=.95)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = PROJECT_ROOT / "test-results" / f"workflow-intent-{args.mode}.json"
    report, exit_code = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"[{args.mode}] 意图评测：{summary['exact']}/{summary['total']} 精确"
        f"（exactRate={summary['exactRate']}），{summary['unsafe']} 条不安全自动路由，"
        f"报告：{args.output}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
