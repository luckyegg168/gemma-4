#!/usr/bin/env bash
# =============================================================
#  Gemma 4 - Install llama-cpp-python (GPU CUDA)
#  GPU primary with CPU fallback.
#  Auto-detects CPU instruction sets to avoid AVX512 SIGILL
#  on CPUs that only support AVX2 (e.g. Ryzen 5000 series).
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

# --- Step 3: Detect CUDA + CPU capabilities ---
echo ""
echo "[Step 3/5] Detecting CUDA GPU and CPU capabilities..."
GPU_FOUND=false
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] nvidia-smi found."
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    GPU_FOUND=true
elif command -v nvcc &>/dev/null; then
    echo "[OK] CUDA nvcc found: $(nvcc --version | grep release)"
    GPU_FOUND=true
else
    echo "[WARNING] No NVIDIA GPU detected — llama.cpp will run on CPU only."
fi

# Detect CPU instruction set support
CPU_FLAGS=$(grep -m1 "^flags" /proc/cpuinfo 2>/dev/null || true)
HAS_AVX512=false
HAS_AVX2=false
HAS_AVX=false
echo "$CPU_FLAGS" | grep -q "avx512f" && HAS_AVX512=true || true
echo "$CPU_FLAGS" | grep -q " avx2 "  && HAS_AVX2=true  || true
echo "$CPU_FLAGS" | grep -q " avx "   && HAS_AVX=true   || true

echo "[INFO] CPU AVX512=$HAS_AVX512  AVX2=$HAS_AVX2  AVX=$HAS_AVX"

# Build CMAKE_ARGS for source build matching this CPU
CUDA_ARGS="-DGGML_CUDA=on"
if [ "$HAS_AVX512" = false ]; then
    CUDA_ARGS="$CUDA_ARGS -DGGML_AVX512=OFF"
fi
if [ "$HAS_AVX2" = false ]; then
    CUDA_ARGS="$CUDA_ARGS -DGGML_AVX2=OFF"
fi
if [ "$HAS_AVX" = false ]; then
    CUDA_ARGS="$CUDA_ARGS -DGGML_AVX=OFF"
fi

# --- Step 4: Install llama-cpp-python ---
echo ""
echo "[Step 4/5] Installing llama-cpp-python..."

PKG="llama-cpp-python[server]>=0.3.0"
INSTALLED=false

verify_import() {
    # Returns 0 if import works without SIGILL
    $PYTHON -c "from llama_cpp import Llama; print('import OK')" 2>/dev/null
    return $?
}

if [ "$GPU_FOUND" = true ]; then
    echo "[INFO] Trying pre-built CUDA 12.1 wheel..."
    if $PYTHON -m pip install "$PKG" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 2>/dev/null; then
        if verify_import; then
            echo "[OK] GPU (CUDA 12.1) wheel works."
            INSTALLED=true
        else
            echo "[WARNING] CUDA 12.1 wheel installed but crashed (likely AVX512 mismatch)."
            echo "          Rebuilding from source for this CPU (AVX512=$HAS_AVX512 AVX2=$HAS_AVX2)..."
        fi
    fi

    if [ "$INSTALLED" = false ]; then
        echo "[INFO] Trying pre-built CUDA 11.8 wheel..."
        if $PYTHON -m pip install "$PKG" \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118 2>/dev/null; then
            if verify_import; then
                echo "[OK] GPU (CUDA 11.8) wheel works."
                INSTALLED=true
            else
                echo "[WARNING] CUDA 11.8 wheel also crashed — building from source..."
            fi
        fi
    fi

    if [ "$INSTALLED" = false ]; then
        echo "[INFO] Building llama-cpp-python from source with CUDA..."
        echo "[INFO] CMAKE_ARGS: $CUDA_ARGS"
        export CMAKE_ARGS="$CUDA_ARGS"
        export FORCE_CMAKE=1
        if $PYTHON -m pip install "$PKG" --no-binary "llama-cpp-python" --no-cache-dir; then
            if verify_import; then
                echo "[OK] GPU source build works."
                INSTALLED=true
            else
                echo "[WARNING] Source GPU build also crashed — falling back to CPU-only."
                GPU_FOUND=false
            fi
        else
            echo "[WARNING] GPU source build failed — falling back to CPU-only."
            GPU_FOUND=false
        fi
    fi
fi

if [ "$INSTALLED" = false ]; then
    echo "[INFO] Installing CPU-only llama-cpp-python (source build for this CPU)..."
    CPU_ARGS=""
    [ "$HAS_AVX512" = false ] && CPU_ARGS="$CPU_ARGS -DGGML_AVX512=OFF"
    [ "$HAS_AVX2"   = false ] && CPU_ARGS="$CPU_ARGS -DGGML_AVX2=OFF"
    [ "$HAS_AVX"    = false ] && CPU_ARGS="$CPU_ARGS -DGGML_AVX=OFF"
    if [ -n "$CPU_ARGS" ]; then
        export CMAKE_ARGS="$CPU_ARGS"
        $PYTHON -m pip install "$PKG" --no-binary "llama-cpp-python" --no-cache-dir
    else
        $PYTHON -m pip install "$PKG"
    fi
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
    echo "[ERROR] llama-cpp-python import failed."
    echo "        Check CUDA/build tools and try again."
    exit 1
fi

echo ""
echo "============================================================"
echo " llama-cpp-python installation complete!"
echo " GPU: $GPU_FOUND"
echo "============================================================"
