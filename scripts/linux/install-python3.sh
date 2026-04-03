#!/bin/bash
# install-python3.sh — Install Python 3 and pip via apt (Ubuntu/Debian)
set -e

echo "============================================================"
echo " Gemma 4 - Install Python 3"
echo "============================================================"
echo ""

# --- Check root/sudo ---
if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    echo "[INFO] This script requires sudo to install packages."
fi

# --- Update apt ---
echo "[Step 1/4] Updating apt package list..."
sudo apt update
echo "[OK] apt updated."
echo ""

# --- Install Python 3 + pip + venv + dev headers ---
echo "[Step 2/4] Installing python3, python3-pip, python3-venv, python3-dev..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python-is-python3
echo "[OK] Python 3 installed."
echo ""

# --- Install build tools (needed for llama-cpp-python / flash-attn) ---
echo "[Step 3/4] Installing build tools (gcc, make, cmake)..."
sudo apt install -y \
    build-essential \
    cmake \
    curl \
    git
echo "[OK] Build tools installed."
echo ""

# --- Verify ---
echo "[Step 4/4] Verifying..."
python3 --version
pip3 --version
python --version 2>/dev/null && echo "[OK] 'python' aliased to python3." || true
echo ""

echo "============================================================"
echo " Python 3 installation complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Run ./install.py    to install Gemma 4 Python dependencies"
echo "  2. Run ./install-dep.sh  for the step-by-step bash installer"
echo ""
