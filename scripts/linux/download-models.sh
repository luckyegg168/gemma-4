#!/bin/bash
# download-models.sh — Download Gemma 4 model weights from HuggingFace

echo "============================================================"
echo " Gemma 4 - Model Downloader"
echo "============================================================"
echo ""

# --- Check prerequisites ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Run install-dep.sh first."
    exit 1
fi
if ! command -v huggingface-cli &>/dev/null; then
    echo "[ERROR] huggingface-cli not found. Run install-dep.sh first."
    exit 1
fi

# --- HuggingFace login check ---
echo "[INFO] Checking HuggingFace login..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "[WARNING] Not logged in to HuggingFace."
    echo "  Gemma 4 models require a HuggingFace account and license acceptance."
    echo "  Accept the license at: https://huggingface.co/google/gemma-4-E2B-it"
    echo ""
    read -p "  Enter your HuggingFace token (leave blank to skip): " HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli login --token "$HF_TOKEN"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Login failed. Check your token."
            exit 1
        fi
    else
        echo "[WARNING] Skipping login. Downloads may fail for gated models."
    fi
fi
echo ""

# --- Download destination ---
DEFAULT_DIR="$HOME/gemma4-models"
read -p "Download destination [default: $DEFAULT_DIR]: " DEST_DIR
if [ -z "$DEST_DIR" ]; then
    DEST_DIR="$DEFAULT_DIR"
fi
mkdir -p "$DEST_DIR"
echo "[INFO] Models will be downloaded to: $DEST_DIR"
echo ""

# --- Model selection ---
echo "  Select model to download:"
echo ""
echo "  --- Full precision (HuggingFace Transformers) ---"
echo "  [1] E2B   - google/gemma-4-E2B-it    (~5.1B, multimodal text+image+audio)"
echo "  [2] E4B   - google/gemma-4-E4B-it    (~8B,   multimodal text+image+audio)"
echo "  [3] 26B-A4B - google/gemma-4-26B-A4B-it  (MoE, text+image)"
echo "  [4] 31B   - google/gemma-4-31B-it    (~30.7B, text+image)"
echo ""
echo "  --- GGUF quantized (unsloth, llama-cpp-python) ---"
echo "  [5] 26B-A4B Q4_K_M  (16.9 GB, recommended)"
echo "  [6] 26B-A4B IQ4_XS  (13.4 GB, minimum VRAM)"
echo "  [7] 26B-A4B Q8_0    (26.9 GB, near-full quality)"
echo "  [8] All GGUF variants (full collection)"
echo ""
echo "  [9] Custom HuggingFace model ID"
echo "  [0] Back"
echo ""
read -p "Enter choice [0-9]: " DL_CHOICE

do_download() {
    local MODEL_ID="$1"
    local LOCAL_DIR="$2"
    echo ""
    echo "[INFO] Downloading: $MODEL_ID"
    echo "[INFO] Destination: $LOCAL_DIR"
    echo ""
    huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"
    if [ $? -eq 0 ]; then
        echo ""
        echo "[OK] Download complete: $MODEL_ID"
    else
        echo ""
        echo "[ERROR] Download failed for $MODEL_ID"
        echo "  Make sure you have accepted the license at:"
        echo "  https://huggingface.co/google/gemma-4-E2B-it"
    fi
}

do_gguf_download() {
    local PATTERN="$1"
    local DESC="$2"
    local DEST="$DEST_DIR/gemma-4-26B-A4B-GGUF"
    echo ""
    echo "[INFO] Downloading GGUF: $DESC"
    echo "[INFO] Destination: $DEST"
    mkdir -p "$DEST"
    huggingface-cli download "unsloth/gemma-4-26B-A4B-it-GGUF" --include "$PATTERN" --local-dir "$DEST"
    if [ $? -eq 0 ]; then
        echo "[OK] GGUF download complete."
    else
        echo "[ERROR] GGUF download failed."
    fi
}

case "$DL_CHOICE" in
    0) exit 0 ;;
    1) do_download "google/gemma-4-E2B-it" "$DEST_DIR/gemma-4-E2B-it" ;;
    2) do_download "google/gemma-4-E4B-it" "$DEST_DIR/gemma-4-E4B-it" ;;
    3) do_download "google/gemma-4-26B-A4B-it" "$DEST_DIR/gemma-4-26B-A4B-it" ;;
    4) do_download "google/gemma-4-31B-it" "$DEST_DIR/gemma-4-31B-it" ;;
    5) do_gguf_download "*Q4_K_M*" "26B-A4B Q4_K_M (16.9 GB, recommended)" ;;
    6) do_gguf_download "*IQ4_XS*" "26B-A4B IQ4_XS (13.4 GB, minimum VRAM)" ;;
    7) do_gguf_download "*Q8_0*"   "26B-A4B Q8_0 (26.9 GB, near-full quality)" ;;
    8)
        echo ""
        echo "[INFO] Downloading ALL GGUF variants (warning: large download)..."
        DEST_ALL="$DEST_DIR/gemma-4-26B-A4B-GGUF-all"
        mkdir -p "$DEST_ALL"
        huggingface-cli download "unsloth/gemma-4-26B-A4B-it-GGUF" --local-dir "$DEST_ALL"
        [ $? -eq 0 ] && echo "[OK] All GGUF downloaded." || echo "[ERROR] GGUF download failed."
        ;;
    9)
        read -p "Enter HuggingFace model ID (e.g. google/gemma-4-E2B-it): " CUSTOM_ID
        if [ -n "$CUSTOM_ID" ]; then
            SAFE_NAME="${CUSTOM_ID//\//-}"
            do_download "$CUSTOM_ID" "$DEST_DIR/$SAFE_NAME"
        fi
        ;;
    *) echo "[ERROR] Invalid choice." ;;
esac

echo ""
echo "[INFO] You can pass the local path to start.sh for offline inference."
