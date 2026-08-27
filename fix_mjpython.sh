#!/usr/bin/env bash
# macOS + uv 관리 Python에서 mjpython이 libpython3.12.dylib을 못 찾을 때 실행한다.
# 증상: "Library not loaded: @rpath/libpython3.12.dylib"
# 원인: uv의 Python은 libpython을 자기 lib/ 폴더에 두는데 mjpython 실행기의 탐색 경로에 없다.
# 조치: mjpython이 찾아보는 위치 중 하나인 .venv/libpython3.12.dylib 에 심링크를 만든다.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(readlink -f .venv/bin/python)"
LIB="$(dirname "$(dirname "$PY")")/lib/libpython3.12.dylib"
[ -f "$LIB" ] || { echo "libpython을 찾지 못했습니다: $LIB"; exit 1; }
ln -sf "$LIB" .venv/libpython3.12.dylib
echo "linked: .venv/libpython3.12.dylib -> $LIB"
uv run mjpython -c "import mujoco; print('mjpython OK', mujoco.__version__)"
