#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CHATCLIP_PYTHON="python3"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  CHATCLIP_PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

for CHATCLIP_ARG in "$@"; do
  if [ "$CHATCLIP_ARG" = "--check" ]; then
    exec "$CHATCLIP_PYTHON" tools/launch.py "$@"
  fi
done

if ! "$CHATCLIP_PYTHON" tools/launch.py --check; then
  if [ -t 0 ] && [ -t 1 ]; then
    printf '检测到安装尚未完成。现在运行统一安装器吗？首次安装会下载数 GB 依赖和模型。[Y/n] '
    read -r CHATCLIP_REPLY
    case "$CHATCLIP_REPLY" in
      n|N|no|NO|No)
        echo "已取消。准备好后运行：python3 tools/setup.py" >&2
        exit 1
        ;;
    esac
    "$CHATCLIP_PYTHON" tools/setup.py
    CHATCLIP_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  else
    echo "安装尚未完成；非交互环境不会自动下载。请先运行：python3 tools/setup.py" >&2
    exit 1
  fi
fi
exec "$CHATCLIP_PYTHON" tools/launch.py "$@"
