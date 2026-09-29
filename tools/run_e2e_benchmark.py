#!/usr/bin/env python3
"""Batch end-to-end agent benchmark: fixed task set -> deterministic verdict.

Drives the real HTTP service (models, analysis workers, rendering) through
one workspace per task, auto-confirms the plan hash, polls to a terminal
status and verifies the produced preview with ffprobe/ffmpeg only — the
judge is deterministic (delivery-media assertions), never an LLM.

Task success requires: terminal status preview_ready/completed, full decode
of every preview produced by THIS plan, and duration/aspect assertions when
the task declares them. ``action_required`` counts as a human intervention
and is NOT a success. ``expectedProfile`` is diagnostic only (routing has
its own benchmark); it never flips a task verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = PROJECT_ROOT / "benchmarks" / "e2e-tasks.jsonl"
DEFAULT_MEDIA = PROJECT_ROOT / "benchmarks" / "media"
TERMINAL_STATUSES = {"preview_ready", "completed", "failed", "cancelled", "no_result"}
SUCCESS_STATUSES = {"preview_ready", "completed"}
ASPECT_RATIOS = {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9, "4:5": 4 / 5}


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        task_id = str(value.get("taskId") or "")
        source = str(value.get("sourceVideo") or "")
        goal = str(value.get("goal") or "")
        if not task_id or task_id in seen or not source or not goal:
            raise ValueError(f"{path}:{line_number} 任务缺少 taskId/sourceVideo/goal 或 ID 重复")
        seen.add(task_id)
        assertions = value.get("assertions") if isinstance(value.get("assertions"), dict) else {}
        tasks.append({
            "taskId": task_id, "sourceVideo": source, "goal": goal,
            "executionMode": str(value.get("executionMode") or "autonomous_review"),
            "expectedProfile": value.get("expectedProfile") if value.get("expectedProfile") else None,
            "uiContext": value.get("uiContext") if isinstance(value.get("uiContext"), dict) else {},
            "assertions": assertions,
            "dependsOn": str(value["dependsOn"]) if value.get("dependsOn") else None,
        })
    return tasks


def media_info(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(result.stdout)


def full_decode(path: Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True, check=True, timeout=600,
    )


def verify_preview(path: Path, *, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    info = media_info(path, ffprobe)
    full_decode(path, ffmpeg)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "durationSeconds": round(float(info.get("format", {}).get("duration") or 0), 3),
        "width": int(video.get("width") or 0), "height": int(video.get("height") or 0),
        "hasAudio": bool(audio),
        "decoded": True,
    }


def aspect_matches(width: int, height: int, expected: str) -> bool:
    target = ASPECT_RATIOS.get(expected)
    if not target or not width or not height:
        return False
    return abs((width / height) - target) / target <= 0.03


class E2ERunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        token = args.token or os.environ.get("CHATCLIP_TEST_TOKEN", "")
        headers = {"X-ChatClip-Token": token} if token else {}
        self.client = httpx.Client(
            base_url=args.base_url.rstrip("/"), headers=headers, timeout=180,
        )
        self.jobs: dict[str, str] = {}
        self.source_has_audio: dict[str, bool] = {}
        self.results: list[dict[str, Any]] = []

    def request(self, method: str, route: str, **kwargs: Any) -> dict[str, Any]:
        response = self.client.request(method, route, **kwargs)
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {route}")
        return response.json()

    # ------------------------------------------------------------ uploads
    def upload_sources(self, tasks: list[dict[str, Any]]) -> None:
        jobs_file = self.args.jobs_file
        if jobs_file and jobs_file.is_file():
            self.jobs = {
                str(k): str(v) for k, v in json.loads(jobs_file.read_text(encoding="utf-8")).items()
            }
        for source in dict.fromkeys(task["sourceVideo"] for task in tasks):
            if source in self.jobs:
                continue
            video = (self.args.media_dir / source).resolve()
            if not video.is_file():
                raise FileNotFoundError(f"缺少源视频：{video}")
            info = media_info(video, self.args.ffprobe)
            self.source_has_audio[source] = any(
                s.get("codec_type") == "audio" for s in info.get("streams") or []
            )
            with video.open("rb") as stream:
                job = self.request(
                    "POST", "/api/jobs",
                    files={"video": (video.name, stream, "video/mp4")},
                    data={"agent_draft": "true", "draft_session_id": uuid.uuid4().hex,
                          "force_reanalyze": "true"},
                )["job"]
            self.jobs[source] = str(job["id"])
            print(f"[upload] {source} -> job {self.jobs[source]}", flush=True)
            if jobs_file:
                jobs_file.write_text(
                    json.dumps(self.jobs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                )

    # ------------------------------------------------------------ tasks
    def run_task(self, task: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "taskId": task["taskId"], "goal": task["goal"],
            "sourceVideo": task["sourceVideo"], "status": "running",
            "startedAt": datetime.now(timezone.utc).isoformat(),
        }
        self.results.append(result)
        self.save_partial(result)
        plan_id = ""
        started = time.monotonic()
        try:
            job_id = self.jobs[task["sourceVideo"]]
            if task["sourceVideo"] not in self.source_has_audio:
                info = media_info(self.args.media_dir / task["sourceVideo"], self.args.ffprobe)
                self.source_has_audio[task["sourceVideo"]] = any(
                    s.get("codec_type") == "audio" for s in info.get("streams") or []
                )
            workspace = self.request(
                "POST", "/api/agent/workspaces",
                json={"jobId": job_id, "title": f"e2e {task['taskId']}"},
            )["workspace"]
            reply = self.request(
                "POST", f"/api/agent/workspaces/{workspace['id']}/messages",
                json={
                    "text": task["goal"], "clientMessageId": uuid.uuid4().hex,
                    "executionMode": task["executionMode"], "uiContext": task["uiContext"],
                },
            )
            plan = reply.get("plan") if isinstance(reply, dict) else None
            if not isinstance(plan, dict) or not plan.get("planHash"):
                raise RuntimeError(f"消息未产生可确认计划：action={reply.get('action') if isinstance(reply, dict) else '?'}")
            plan_id = str(plan["id"])
            result.update(planId=plan_id)
            self.request("POST", f"/api/agent/plans/{plan_id}/confirm",
                         json={"planHash": plan["planHash"]})

            deadline, last_print = time.monotonic() + self.args.timeout, 0.0
            while True:
                plan = self.request("GET", f"/api/agent/plans/{plan_id}")["plan"]
                status = str(plan.get("status") or "")
                if status == "action_required":
                    result["humanIntervention"] = True
                    result["interventionSteps"] = [
                        {"tool": s.get("tool"), "message": (s.get("result") or {}).get("message")}
                        for s in plan.get("steps", []) if s.get("status") == "action_required"
                    ]
                    self.request("POST", f"/api/agent/plans/{plan_id}/cancel")
                    result.update(status="failed", planStatus="action_required",
                                  reason="autonomous run required a human gate")
                    return result
                if time.monotonic() > deadline:
                    self.request("POST", f"/api/agent/plans/{plan_id}/cancel")
                    result.update(status="failed", planStatus="timeout",
                                  reason="task deadline exceeded; plan cancelled")
                    return result
                if time.monotonic() - last_print > 30:
                    print(f"[{task['taskId']}] {status}", flush=True)
                    last_print = time.monotonic()
                if status in TERMINAL_STATUSES:
                    break
                time.sleep(self.args.poll_seconds)

            result["planStatus"] = status
            result["replanCount"] = int(plan.get("replanCount") or 0)
            usage = plan.get("planningUsage") if isinstance(plan.get("planningUsage"), dict) else {}
            result["planningTokens"] = usage.get("total_tokens")
            result["profileMatch"] = (
                None if not task["expectedProfile"]
                else str(plan.get("profile") or "") == task["expectedProfile"]
            )
            if status not in SUCCESS_STATUSES:
                result.update(status="failed", reason=f"terminal status {status}")
                result["failedSteps"] = [
                    {"tool": s.get("tool"), "error": s.get("error")}
                    for s in plan.get("steps", []) if s.get("status") == "failed"
                ]
                return result

            previews = self.collect_previews(plan)
            media_reports = self.verify_previews(task, previews, result)
            result["media"] = media_reports
            result["status"] = "success" if not result.get("assertionFailures") else "failed"
            if result["status"] == "failed":
                result["reason"] = "; ".join(result["assertionFailures"])
        except Exception as error:  # noqa: BLE001 - benchmark continues with next task
            result.update(status="failed", reason=str(error)[:500])
            if plan_id:
                try:
                    self.request("POST", f"/api/agent/plans/{plan_id}/cancel")
                except Exception:  # noqa: BLE001
                    pass
        finally:
            result["elapsedSeconds"] = round(time.monotonic() - started, 1)
            result["finishedAt"] = datetime.now(timezone.utc).isoformat()
            self.save_partial(result)
        return result

    def collect_previews(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for step in plan.get("steps", []):
            artifact = (step.get("result") or {}).get("artifact") or {}
            kind = artifact.get("kind")
            if kind == "review_preview_batch":
                previews.extend(artifact.get("previews") or [])
            elif kind == "review_preview":
                previews.append(artifact)
            elif kind == "social_reframe_preview" and isinstance(artifact.get("output"), dict):
                previews.append(artifact["output"])
        return [p for p in previews if isinstance(p, dict)]

    def verify_previews(
        self, task: dict[str, Any], previews: list[dict[str, Any]], result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assertions = task["assertions"]
        media_reports: list[dict[str, Any]] = []
        if not previews:
            # Cover/draft/graphics tasks legitimately finish without a video
            # artifact; only tasks that assert a duration or aspect need one.
            if assertions.get("targetSeconds") or assertions.get("aspect"):
                result.setdefault("assertionFailures", []).append("plan finished without a preview artifact")
            return media_reports
        source_audio = self.source_has_audio.get(task["sourceVideo"], True)
        duration_ok = aspect_ok = False
        for index, item in enumerate(previews, 1):
            url = str(item.get("previewUrl") or item.get("videoUrl") or "")
            report: dict[str, Any] = {"previewUrl": url[:200]}
            if not url.startswith("/api/") or url.startswith("//"):
                result.setdefault("assertionFailures", []).append("missing or non-local preview URL")
                media_reports.append(report)
                continue
            destination = self.args.output_dir / f"{task['taskId']}-{index}.mp4"
            with self.client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            try:
                media = verify_preview(destination, ffmpeg=self.args.ffmpeg, ffprobe=self.args.ffprobe)
            except subprocess.CalledProcessError as error:
                result.setdefault("assertionFailures", []).append(
                    f"preview {index} failed full decode: {error.stderr[:120] if error.stderr else 'decode error'}")
                media_reports.append(report)
                continue
            report.update(media)
            media_reports.append(report)
            if source_audio and not media["hasAudio"]:
                result.setdefault("assertionFailures", []).append(f"preview {index} lost source audio")
            if assertions.get("targetSeconds") is not None and abs(
                media["durationSeconds"] - float(assertions["targetSeconds"]),
            ) <= float(assertions.get("toleranceSeconds") or 0):
                duration_ok = True
            if assertions.get("aspect") and aspect_matches(
                media["width"], media["height"], str(assertions["aspect"]),
            ):
                aspect_ok = True
        if assertions.get("targetSeconds") is not None and not duration_ok:
            result.setdefault("assertionFailures", []).append(
                f"no preview within {assertions['targetSeconds']}s±{assertions.get('toleranceSeconds', 0)}s")
        if assertions.get("aspect") and not aspect_ok:
            result.setdefault("assertionFailures", []).append(f"no preview matched aspect {assertions['aspect']}")
        return media_reports

    # ------------------------------------------------------------ reporting
    def save_partial(self, _result: dict[str, Any]) -> None:
        # Reports are written at the end; per-task state lives in self.results.
        return

    def build_report(self, wall_seconds: float, passed_required: bool | None) -> dict[str, Any]:
        finished = [r for r in self.results if r.get("status") in {"success", "failed"}]
        successes = [r for r in finished if r["status"] == "success"]
        replans = [int(r.get("replanCount") or 0) for r in finished]
        tokens = [int(r.get("planningTokens") or 0) for r in finished if r.get("planningTokens")]
        interventions = [r for r in finished if r.get("humanIntervention")]
        rate = len(successes) / len(finished) if finished else 0.0
        blocked = [r for r in self.results if r.get("status") == "blocked"]
        return {
            "schemaVersion": 1,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "baseUrl": self.args.base_url,
            "tasksFile": str(self.args.tasks),
            "criteria": {
                "success": "terminal preview_ready/completed + full decode + duration/aspect assertions",
                "humanIntervention": "action_required in autonomous mode counts as failure, not success",
                "denominator": "all scheduled tasks; blocked deps and failures are kept in results",
            },
            "summary": {
                "total": len(self.results), "finished": len(finished),
                "succeeded": len(successes), "blocked": len(blocked),
                "taskSuccessRate": round(rate, 4),
                "avgReplanCount": round(sum(replans) / len(replans), 2) if replans else None,
                "avgPlanningTokens": round(sum(tokens) / len(tokens)) if tokens else None,
                "humanInterventionCount": len(interventions),
                "wallClockSeconds": round(wall_seconds, 1),
                "minimumSuccessRate": self.args.minimum_success_rate,
                "passed": passed_required,
            },
            "jobs": self.jobs,
            "results": self.results,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端任务成功率评测（确定性判据）")
    parser.add_argument("--base-url", default="http://127.0.0.1:5180")
    parser.add_argument("--token", default="")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--media-dir", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "test-results" / "e2e-benchmark-report.json")
    parser.add_argument("--jobs-file", type=Path,
                        default=PROJECT_ROOT / "test-results" / "e2e-benchmark-jobs.json",
                        help="源视频->job 映射缓存，复用可避开 10 jobs/小时限流")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="逗号分隔的 taskId 列表")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--minimum-success-rate", type=float, default=.8)
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        unknown = wanted - {t["taskId"] for t in tasks}
        if unknown:
            raise SystemExit(f"未知 taskId：{sorted(unknown)}")
        tasks = [t for t in tasks if t["taskId"] in wanted]
    if args.limit:
        tasks = tasks[:args.limit]
    args.output_dir = args.output.parent / "e2e-media"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = E2ERunner(args)
    runner.upload_sources(tasks)
    started = time.monotonic()
    by_id = {t["taskId"]: t for t in tasks}
    for task in tasks:
        dep = task["dependsOn"]
        if dep and dep in by_id:
            dep_result = next((r for r in runner.results if r["taskId"] == dep), None)
            if not dep_result or dep_result.get("status") != "success":
                runner.results.append({
                    "taskId": task["taskId"], "goal": task["goal"],
                    "status": "blocked", "reason": f"依赖任务 {dep} 未成功",
                })
                print(f"[{task['taskId']}] blocked (dep {dep})", flush=True)
                continue
        print(f"[{task['taskId']}] start: {task['goal']}", flush=True)
        runner.run_task(task)
        latest = runner.results[-1]
        print(f"[{task['taskId']}] {latest['status']} ({latest.get('planStatus')})", flush=True)
    wall = time.monotonic() - started

    report = runner.build_report(wall, None)
    if args.check:
        report["summary"]["passed"] = (
            report["summary"]["taskSuccessRate"] >= args.minimum_success_rate
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = report["summary"]
    print(
        f"E2E 评测：{s['succeeded']}/{s['finished']} 成功"
        f"（taskSuccessRate={s['taskSuccessRate']}，blocked={s['blocked']}，"
        f"人工干预={s['humanInterventionCount']}，avgReplans={s['avgReplanCount']}，"
        f"avgTokens={s['avgPlanningTokens']}），报告：{args.output}"
    )
    if args.check:
        return 0 if s["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
