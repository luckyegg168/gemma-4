#!/bin/bash
# install-dep.sh — Install all Python dependencies for Gemma 4
set -e

echo "============================================================"
echo " Gemma 4 - Dependency Installer"
echo "============================================================"
echo ""

# --- Check Python ---
echo "[Step 1/6] Checking Python installation..."
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.9+ first:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python"
    exit 1
fi
python3 --version
echo "[OK] Python found."
echo ""

# --- Upgrade pip ---
echo "[Step 2/6] Upgrading pip..."
python3 -m pip install --upgrade pip
echo "[OK] pip upgraded."
echo ""

# --- Core ML packages ---
echo "[Step 3/6] Installing core packages (transformers, torch, accelerate)..."
python3 -m pip install --upgrade transformers torch accelerate
echo "[OK] Core packages installed."
echo ""

# --- HuggingFace Hub ---
echo "[Step 4/6] Installing HuggingFace Hub CLI..."
python3 -m pip install --upgrade huggingface_hub
echo "[OK] huggingface_hub installed."
echo ""

# --- Multimodal: vision + audio ---
echo "[Step 5/6] Installing multimodal dependencies (Pillow, librosa, soundfile)..."
python3 -m pip install --upgrade Pillow librosa soundfile
echo "[OK] Multimodal dependencies installed."
echo ""

# --- Optional: flash-attn (Linux only) ---
echo "[Step 6/6] Flash Attention (optional, CUDA only)..."
echo "  Flash Attention significantly improves throughput on NVIDIA GPUs."
read -p "  Install flash-attn? This requires CUDA + gcc and takes a while [y/N]: " FA_CHOICE
if [[ "$FA_CHOICE" =~ ^[Yy]$ ]]; then
    python3 -m pip install flash-attn --no-build-isolation
    if [ $? -eq 0 ]; then
        echo "[OK] flash-attn installed."
    else
        echo "[WARNING] flash-attn installation failed. This is optional — continuing without it."
    fi
else
    echo "[SKIP] Skipped flash-attn."
fi
echo ""

# --- Optional: llama-cpp-python for GGUF ---
echo "[Optional] llama-cpp-python (for GGUF/quantized models)..."
read -p "  Install llama-cpp-python? Required for GGUF models [y/N]: " GGUF_CHOICE
if [[ "$GGUF_CHOICE" =~ ^[Yy]$ ]]; then
    echo "  Choose llama.cpp build:"
    echo "  [1] CPU only"
    echo "  [2] CUDA (NVIDIA GPU)"
    echo "  [3] ROCm (AMD GPU)"
    echo "  [4] Metal (macOS)"
    read -p "  Choice [1-4]: " GGUF_BUILD
    if [ "$GGUF_BUILD" = "2" ]; then
        CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install llama-cpp-python --no-cache-dir
    elif [ "$GGUF_BUILD" = "3" ]; then
        CMAKE_ARGS="-DGGML_HIPBLAS=on" python3 -m pip install llama-cpp-python --no-cache-dir
    elif [ "$GGUF_BUILD" = "4" ]; then
        CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install llama-cpp-python --no-cache-dir
    else
        python3 -m pip install llama-cpp-python
    fi
    if [ $? -eq 0 ]; then
        echo "[OK] llama-cpp-python installed."
    else
        echo "[WARNING] llama-cpp-python installation failed."
    fi
else
    echo "[SKIP] Skipped llama-cpp-python."
fi
echo ""

echo "============================================================"
echo " All dependencies installed successfully."
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Run ./download-models.sh to fetch model weights"
echo "  2. Accept license at https://huggingface.co/google/gemma-4-E2B-it"
echo "  3. Run ./start.sh to launch interactive chat"
echo ""
