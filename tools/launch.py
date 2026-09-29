#!/usr/bin/env python3
"""Foreground supervisor for the ChatClip web service."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.config import Settings  # noqa: E402


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=.5):
            return True
    except OSError:
        return False


def service_ready(url: str, service: str) -> bool:
    try:
        with urlopen(url, timeout=1) as response:
            return json.load(response).get("service") == service
    except (OSError, ValueError, URLError):
        return False


def json_response(url: str, *, token: str = "") -> dict:
    headers = {"X-ChatClip-Token": token} if token else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=2) as response:
            value = json.load(response)
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError, URLError, HTTPError):
        return {}


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        # Only the process group created by this launcher is in scope.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 ChatClip 网页服务；Ctrl+C 停止")
    parser.add_argument("--check", action="store_true", help="只检查运行环境，不启动服务")
    args = parser.parse_args()
    settings = Settings.from_environment()
    from tools.doctor import inspect_environment
    report = inspect_environment("visual")
    for check in report["checks"]:
        if check["status"] != "ok":
            print(f"{check['name']}：{check['detail']}", flush=True)
    if not report["ready"]:
        print("启动条件不满足，请先运行 python3 tools/setup.py。", file=sys.stderr)
        return 1
    if args.check:
        return 0
    probe_host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host
    if port_open(probe_host, settings.port):
        print(f"端口 {settings.port} 已被使用；不会停止或覆盖已有服务。", file=sys.stderr)
        return 1
    owned: list[subprocess.Popen] = []
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        web = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", settings.host,
                                "--port", str(settings.port)], cwd=ROOT, start_new_session=True)
        owned.append(web)
        deadline = time.monotonic() + 180
        announced = False
        while not stopping:
            if any(process.poll() is not None for process in owned):
                raise RuntimeError("网页服务已退出，请查看上方错误信息。")
            if not announced and service_ready(f"http://{probe_host}:{settings.port}/api/health", "chatclip"):
                base_url = f"http://{probe_host}:{settings.port}"
                web_health = json_response(base_url + "/api/health")
                agent_health = json_response(base_url + "/api/agent/health", token=settings.access_token)
                configured = bool(
                    web_health.get("visionConfigured")
                    and web_health.get("llmConfigured")
                    and (agent_health.get("model") or {}).get("configured")
                )
                if configured:
                    print(f"ChatClip 剪辑能力已就绪：{base_url}；按 Ctrl+C 停止。", flush=True)
                else:
                    print(f"ChatClip 网页服务已启动：{base_url}", flush=True)
                    print("首次配置尚未完成：请打开页面“设置”，按就绪清单配置并验证模型。", flush=True)
                announced = True
            if not announced and time.monotonic() > deadline:
                raise RuntimeError("网页服务启动超时，请检查日志和运行环境。")
            time.sleep(.3)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(owned):
            stop_owned(process)


if __name__ == "__main__":
    raise SystemExit(main())
