#!/usr/bin/env bash
# =============================================================
#  Gemma 4 - Install llama-cpp-python (GPU CUDA)
#  GPU primary with CPU fallback
# =============================================================
set -euo pipefail

echo "============================================================"
echo " Gemma 4 - Install llama-cpp-python (GPU CUDA Acceleration)"
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

# --- Step 2: Upgrade pip ---
echo ""
echo "[Step 2/5] Upgrading pip..."
$PYTHON -m pip install --upgrade pip

# --- Step 3: Detect CUDA ---
echo ""
echo "[Step 3/5] Detecting CUDA GPU..."
GPU_FOUND=false
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] nvidia-smi found."
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    GPU_FOUND=true
elif command -v nvcc &>/dev/null; then
    echo "[OK] CUDA nvcc found: $(nvcc --version | grep release)"
    GPU_FOUND=true
else
    echo "[WARNING] No NVIDIA GPU detected."
    echo "          llama.cpp will run on CPU only."
fi

# --- Step 4: Install llama-cpp-python ---
echo ""
echo "[Step 4/5] Installing llama-cpp-python..."

if [ "$GPU_FOUND" = true ]; then
    echo "[INFO] Attempting GPU build (CUDA 12.1 pre-built wheel)..."
    if $PYTHON -m pip install "llama-cpp-python[server]>=0.3.0" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 2>/dev/null; then
        echo "[OK] GPU (CUDA 12.1) wheel installed."
    else
        echo "[INFO] CUDA 12.1 failed. Trying CUDA 11.8..."
        if $PYTHON -m pip install "llama-cpp-python[server]>=0.3.0" \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118 2>/dev/null; then
            echo "[OK] GPU (CUDA 11.8) wheel installed."
        else
            echo "[INFO] Pre-built GPU wheels failed. Building from source with CUDA..."
            export CMAKE_ARGS="-DGGML_CUDA=on"
            export FORCE_CMAKE=1
            if $PYTHON -m pip install "llama-cpp-python[server]>=0.3.0" --no-binary "llama-cpp-python" 2>/dev/null; then
                echo "[OK] GPU source build successful."
            else
                echo "[WARNING] GPU build failed. Falling back to CPU-only llama-cpp-python..."
                GPU_FOUND=false
            fi
        fi
    fi
fi

if [ "$GPU_FOUND" = false ]; then
    echo "[INFO] Installing CPU-only llama-cpp-python..."
    $PYTHON -m pip install "llama-cpp-python[server]>=0.3.0"
    echo "[OK] CPU-only llama-cpp-python installed."
fi

# --- Step 5: Verify ---
echo ""
echo "[Step 5/5] Verifying installation..."
if $PYTHON -c "from llama_cpp import Llama; print('llama-cpp-python OK')" 2>/dev/null; then
    echo "[OK] llama-cpp-python is ready."
    echo ""
    echo "  Next step: Run download-models.sh to get GGUF files."
    echo "             Then run start-llamacpp.sh to start chatting."
else
    echo "[ERROR] llama-cpp-python installation failed."
    echo "        Check CUDA/build tools and try again."
    exit 1
fi

echo ""
echo "============================================================"
echo " llama-cpp-python installation complete!"
echo " GPU: $GPU_FOUND"
echo "============================================================"
