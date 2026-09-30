#!/usr/bin/env sh
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v python3 || command -v python)"
exec "$PY" "$HERE/cli.py" "$@"
