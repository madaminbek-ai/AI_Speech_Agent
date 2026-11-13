#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_ROOT/.tools/ollama"
curl --fail --location --retry 2 https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$TASK_ROOT/.tools/ollama-linux-amd64.tar.zst"
tar --zstd -xf "$TASK_ROOT/.tools/ollama-linux-amd64.tar.zst" -C "$TASK_ROOT/.tools/ollama"
"$TASK_ROOT/.tools/ollama/bin/ollama" --version
