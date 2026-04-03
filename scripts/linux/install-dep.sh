#!/bin/bash
# install-dep.sh — Install all Python dependencies for Gemma 4
# Uses a virtualenv at ~/gemma4-env to avoid PEP 668 system-package conflicts
set -e

VENV_DIR="$HOME/gemma4-env"

echo "============================================================"
echo " Gemma 4 - Dependency Installer"
echo "============================================================"
echo ""

# --- Check Python ---
echo "[Step 1/7] Checking Python installation..."
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Run install-python3.sh first."
    exit 1
fi
python3 --version
echo "[OK] Python found."
echo ""

# --- Create virtualenv ---
echo "[Step 2/7] Setting up virtual environment at $VENV_DIR ..."
if [ ! -f "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create venv. Try: sudo apt install python3-venv python3-full"
        exit 1
    fi
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
echo ""

# --- Upgrade pip ---
echo "[Step 3/7] Upgrading pip..."
"$PIP" install --upgrade pip
echo "[OK] pip upgraded."
echo ""

# --- Core ML packages ---
echo "[Step 4/7] Installing core packages (transformers, torch, accelerate)..."
"$PIP" install --upgrade transformers torch accelerate
echo "[OK] Core packages installed."
echo ""

# --- HuggingFace Hub ---
echo "[Step 5/7] Installing HuggingFace Hub CLI..."
"$PIP" install --upgrade huggingface_hub
echo "[OK] huggingface_hub installed."
echo ""

# --- Multimodal: vision + audio ---
echo "[Step 6/7] Installing multimodal dependencies (Pillow, librosa, soundfile)..."
"$PIP" install --upgrade Pillow librosa soundfile
echo "[OK] Multimodal dependencies installed."
echo ""

# --- Optional: flash-attn (CUDA only) ---
echo "[Step 7/7] Flash Attention (optional, CUDA only)..."
echo "  Flash Attention significantly improves throughput on NVIDIA GPUs."
read -p "  Install flash-attn? This requires CUDA + gcc and takes a while [y/N]: " FA_CHOICE
if [[ "$FA_CHOICE" =~ ^[Yy]$ ]]; then
    "$PIP" install flash-attn --no-build-isolation
    if [ $? -eq 0 ]; then
        echo "[OK] flash-attn installed."
    else
        echo "[WARNING] flash-attn installation failed. This is optional — continuing without it."
    fi
else
    echo "[SKIP] Skipped flash-attn."
fi
echo ""

echo "============================================================"
echo " All dependencies installed successfully."
echo " Virtualenv: $VENV_DIR"
echo "============================================================"
echo ""
echo "  Activate before running Gemma 4 scripts:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Next steps:"
echo "  1. Run ./download-models.sh to fetch model weights"
echo "  2. Accept license at https://huggingface.co/google/gemma-4-E2B-it"
echo "  3. Run ./start.sh to launch interactive chat"
echo ""
