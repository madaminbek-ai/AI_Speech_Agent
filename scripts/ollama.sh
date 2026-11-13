#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_NO_CLOUD=true
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$TASK_ROOT/.models/ollama"
exec "$TASK_ROOT/.tools/ollama/bin/ollama" "$@"
