#!/usr/bin/env bash
# install.sh — thin wrapper for install.py
set -euo pipefail
cd "$(dirname "$0")/.."
python3 install.py "$@"
