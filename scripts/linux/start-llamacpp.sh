#!/usr/bin/env bash
# =============================================================
#  Gemma 4 - Interactive Chat via llama.cpp (GGUF)
#  GPU-accelerated GGUF inference with n_gpu_layers offloading
# =============================================================
set -euo pipefail

echo "============================================================"
echo " Gemma 4 - Interactive Chat via llama.cpp (GGUF)"
echo " GPU-accelerated GGUF inference with n_gpu_layers offloading"
echo "============================================================"
echo ""

# --- Check llama_cpp ---
PYTHON=$(command -v python3 || command -v python)
if ! $PYTHON -c "from llama_cpp import Llama" &>/dev/null; then
    echo "[ERROR] llama-cpp-python is not installed."
    echo "        Run install-dep-llamacpp.sh first."
    exit 1
fi

# --- Default model directory ---
DEFAULT_DIR="$HOME/gemma4-models"
read -rp "Models directory [default: $DEFAULT_DIR]: " MODELS_DIR
MODELS_DIR="${MODELS_DIR:-$DEFAULT_DIR}"
echo ""

# --- Model selection ---
echo " Select a GGUF model:"
echo ""
echo " --- E2B (2.3B eff) --- Needs 3-5 GB VRAM ---"
echo " [1]  E2B  Q4_K_M   3.11 GB  Recommended"
echo " [2]  E2B  Q8_0     5.05 GB  High quality"
echo " [3]  E2B  IQ4_XS   2.98 GB  Small option"
echo ""
echo " --- E4B (4.5B eff) --- Needs 5-9 GB VRAM ---"
echo " [4]  E4B  Q4_K_M   4.98 GB  Recommended"
echo " [5]  E4B  Q8_0     8.19 GB  High quality"
echo ""
echo " --- 26B-A4B (MoE) --- Needs 17-28 GB VRAM ---"
echo " [6]  26B-A4B  MXFP4_MOE   16.7 GB  MoE-optimized  Recommended"
echo " [7]  26B-A4B  UD-Q4_K_M   16.9 GB  Standard recommended"
echo " [8]  26B-A4B  Q8_0        26.9 GB  Near full quality"
echo ""
echo " --- 31B (Dense) --- Needs 18-34 GB VRAM ---"
echo " [9]  31B  Q4_K_M   18.3 GB  Recommended"
echo " [10] 31B  IQ4_XS   16.4 GB  Smallest 31B"
echo " [11] 31B  Q8_0     32.6 GB  Near full quality"
echo ""
echo " [12] Custom GGUF file path"
echo " [0]  Exit"
echo ""
read -rp "Enter choice: " MODEL_CHOICE

case "$MODEL_CHOICE" in
    0) exit 0 ;;
    1)  MODEL_FILE="$MODELS_DIR/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf";       GPU_LAYERS=35 ;;
    2)  MODEL_FILE="$MODELS_DIR/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf";         GPU_LAYERS=35 ;;
    3)  MODEL_FILE="$MODELS_DIR/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-IQ4_XS.gguf";       GPU_LAYERS=35 ;;
    4)  MODEL_FILE="$MODELS_DIR/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf";       GPU_LAYERS=42 ;;
    5)  MODEL_FILE="$MODELS_DIR/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf";         GPU_LAYERS=42 ;;
    6)  MODEL_FILE="$MODELS_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-MXFP4_MOE.gguf";  GPU_LAYERS=30 ;;
    7)  MODEL_FILE="$MODELS_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf";  GPU_LAYERS=30 ;;
    8)  MODEL_FILE="$MODELS_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q8_0.gguf";       GPU_LAYERS=30 ;;
    9)  MODEL_FILE="$MODELS_DIR/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf";       GPU_LAYERS=60 ;;
    10) MODEL_FILE="$MODELS_DIR/gemma-4-31B-it-GGUF/gemma-4-31B-it-IQ4_XS.gguf";       GPU_LAYERS=60 ;;
    11) MODEL_FILE="$MODELS_DIR/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q8_0.gguf";         GPU_LAYERS=60 ;;
    12) read -rp "Enter full path to .gguf file: " MODEL_FILE
        read -rp "GPU layers to offload (-1=all, 0=CPU only): " GPU_LAYERS ;;
    *) echo "[ERROR] Invalid choice."; exit 1 ;;
esac

# --- Check model file exists ---
if [ ! -f "$MODEL_FILE" ]; then
    echo "[ERROR] Model file not found: $MODEL_FILE"
    echo "        Run download-models.sh to download GGUF files."
    exit 1
fi

# --- GPU layer configuration ---
echo ""
read -rp "GPU layers to offload [default: $GPU_LAYERS, 0=CPU only, -1=all]: " GPU_OVERRIDE
GPU_LAYERS="${GPU_OVERRIDE:-$GPU_LAYERS}"

# --- Context length ---
CTX_LEN=4096
read -rp "Context length [default: 4096]: " CTX_OVERRIDE
CTX_LEN="${CTX_OVERRIDE:-$CTX_LEN}"

# --- System prompt ---
SYSTEM_PROMPT="You are a helpful assistant."
read -rp "System prompt [default: You are a helpful assistant.]: " SYS_OVERRIDE
SYSTEM_PROMPT="${SYS_OVERRIDE:-$SYSTEM_PROMPT}"

echo ""
echo "[INFO] Starting llama.cpp interactive chat..."
echo "[INFO] Model:      $MODEL_FILE"
echo "[INFO] GPU layers: $GPU_LAYERS"
echo "[INFO] Context:    $CTX_LEN tokens"
echo ""

# --- Run Python chat ---
TEMP_SCRIPT=$(mktemp /tmp/gemma4_llamacpp_XXXXXX.py)
trap "rm -f $TEMP_SCRIPT" EXIT

cat > "$TEMP_SCRIPT" << PYEOF
from llama_cpp import Llama

MODEL_PATH = r"$MODEL_FILE"
N_GPU_LAYERS = $GPU_LAYERS
N_CTX = $CTX_LEN
SYSTEM_PROMPT = r"$SYSTEM_PROMPT"

print(f"Loading: {MODEL_PATH}")
print(f"GPU layers: {N_GPU_LAYERS}  Context: {N_CTX}")
print()

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX,
    chat_format="gemma",
    verbose=False,
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
print("Gemma 4 llama.cpp Chat — type 'quit' to exit, 'reset' to clear history")
print("=" * 60)

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if user_input.lower() == "reset":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("[INFO] Conversation history cleared.")
        continue

    messages.append({"role": "user", "content": user_input})
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=1024,
            stream=True,
        )
        print("Gemma: ", end="", flush=True)
        full_response = ""
        for chunk in response:
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
            full_response += delta
        print()
        messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        print(f"\n[ERROR] {e}")
        messages.pop()
PYEOF

$PYTHON "$TEMP_SCRIPT"

echo ""
echo "[INFO] Chat session ended."
