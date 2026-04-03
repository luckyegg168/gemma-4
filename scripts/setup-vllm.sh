#!/usr/bin/env bash
# setup-vllm.sh — thin wrapper for setup_vllm.py
set -euo pipefail
cd "$(dirname "$0")/.."
python3 setup_vllm.py "$@"
