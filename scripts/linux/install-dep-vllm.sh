#!/usr/bin/env bash
# =============================================================
#  Gemma 4 - Install vllm (GPU ONLY)
#  OpenAI-compatible inference server
#  GPU REQUIRED - no CPU inference support
# =============================================================
set -euo pipefail

echo "============================================================"
echo " Gemma 4 - Install vllm (GPU serving)"
echo " NOTE: vllm requires an NVIDIA GPU (CUDA 11.8+)"
echo "============================================================"
echo ""

# --- Step 1: Check Python + setup venv ---
echo "[Step 1/5] Checking Python..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found. Install Python 3.9+ first."
    exit 1
fi
VENV_DIR="$HOME/gemma4-env"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || { echo "[ERROR] venv failed. Try: sudo apt install python3-venv"; exit 1; }
    echo "[OK] Venv created."
fi
PYTHON="$VENV_DIR/bin/python"
$PYTHON --version

# --- Step 2: Check NVIDIA GPU ---
echo ""
echo "[Step 2/5] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "[WARNING] nvidia-smi not found. GPU may not be available."
    echo ""
    read -rp "Continue without verified GPU? [y/N]: " CONFIRM
    if [[ "${CONFIRM,,}" != "y" ]]; then
        echo "[INFO] Installation cancelled. vllm requires a GPU."
        exit 0
    fi
else
    echo "[OK] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
fi

# --- Step 3: Upgrade pip and install PyTorch CUDA ---
echo ""
echo "[Step 3/5] Installing PyTorch with CUDA support..."
$PYTHON -m pip install --upgrade pip

if $PYTHON -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
    echo "[OK] PyTorch CUDA 12.1 installed."
else
    echo "[INFO] CUDA 12.1 failed. Trying CUDA 11.8..."
    $PYTHON -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118
    echo "[OK] PyTorch CUDA 11.8 installed."
fi

# --- Step 4: Install vllm and dependencies ---
echo ""
echo "[Step 4/5] Installing vllm and dependencies..."
$PYTHON -m pip install vllm
$PYTHON -m pip install openai
$PYTHON -m pip install "huggingface_hub>=0.22.0" "transformers>=4.56.0,<5.0.0" accelerate
echo "[OK] vllm and dependencies installed."

# --- Step 5: Verify ---
echo ""
echo "[Step 5/5] Verifying installation..."
if $PYTHON -c "import vllm; print(f'vllm {vllm.__version__} OK')" 2>/dev/null; then
    echo "[OK] vllm is ready."
    echo ""
    echo "  Start vllm server: run start-vllm.sh"
    echo "  API endpoint will be: http://localhost:8000/v1"
else
    echo "[ERROR] vllm installation failed."
    echo "        Ensure CUDA drivers are installed and GPU is accessible."
    exit 1
fi

echo ""
echo "============================================================"
echo " vllm installation complete!"
echo " NOTE: vllm requires GPU for inference."
echo " Use llama.cpp (install-dep-llamacpp.sh) for CPU fallback."
echo "============================================================"
