#!/usr/bin/env bash
# setup-llamacpp.sh — thin wrapper for setup_llamacpp.py
set -euo pipefail
cd "$(dirname "$0")/.."
python3 setup_llamacpp.py "$@"
