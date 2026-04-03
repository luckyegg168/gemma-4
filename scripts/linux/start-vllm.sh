#!/usr/bin/env bash
# =============================================================
#  Gemma 4 - vllm Server & Client
#  OpenAI-compatible GPU inference server
#  GPU REQUIRED (CUDA 11.8+)
# =============================================================
set -euo pipefail

VENV_DIR="$HOME/gemma4-env"
if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON=$(command -v python3 || command -v python || true)
fi

echo "============================================================"
echo " Gemma 4 - vllm Server & Client"
echo " OpenAI-compatible GPU inference server"
echo " GPU REQUIRED (CUDA 11.8+)"
echo "============================================================"
echo ""

# --- Check vllm ---
if ! $PYTHON -c "import vllm" &>/dev/null; then
    echo "[ERROR] vllm is not installed."
    echo "        Run install-dep-vllm.sh first."
    exit 1
fi

echo " Select action:"
echo ""
echo " [1] Start vllm Server (foreground)"
echo " [2] Start vllm Server (background daemon)"
echo " [3] Chat via Python client (requires server already running)"
echo " [4] Start server AND open chat client (combined)"
echo " [5] Show server status / test connection"
echo " [0] Exit"
echo ""
read -rp "Enter choice: " ACTION_CHOICE

# ----------------------------------------------------------------
select_model() {
    echo ""
    echo " Select Gemma 4 model:"
    echo ""
    echo " [1] google/gemma-4-E2B-it    (~5 GB,  needs ~8 GB VRAM)"
    echo " [2] google/gemma-4-E4B-it    (~8 GB,  needs ~12 GB VRAM)"
    echo " [3] google/gemma-4-26B-A4B-it (~27 GB, needs ~32 GB VRAM)"
    echo " [4] google/gemma-4-31B-it    (~31 GB, needs ~48 GB VRAM)"
    echo " [5] Custom model ID"
    echo " [0] Back"
    echo ""
    read -rp "Enter choice: " MODEL_CHOICE
    case "$MODEL_CHOICE" in
        0) exit 0 ;;
        1) MODEL_ID="google/gemma-4-E2B-it";      MAX_LEN=8192 ;;
        2) MODEL_ID="google/gemma-4-E4B-it";      MAX_LEN=8192 ;;
        3) MODEL_ID="google/gemma-4-26B-A4B-it";  MAX_LEN=16384 ;;
        4) MODEL_ID="google/gemma-4-31B-it";      MAX_LEN=16384 ;;
        5) read -rp "Enter HuggingFace model ID: " MODEL_ID; MAX_LEN=8192 ;;
        *) echo "[ERROR] Invalid choice."; exit 1 ;;
    esac
}

start_server() {
    select_model

    read -rp "Server port [default: 8000]: " PORT
    PORT="${PORT:-8000}"

    read -rp "dtype [bfloat16 / float16, default: bfloat16]: " DTYPE
    DTYPE="${DTYPE:-bfloat16}"

    read -rp "GPU memory utilization [default: 0.90]: " GPU_UTIL
    GPU_UTIL="${GPU_UTIL:-0.90}"

    echo ""
    echo "[INFO] Starting vllm server..."
    echo "[INFO] Model:    $MODEL_ID"
    echo "[INFO] Port:     $PORT"
    echo "[INFO] dtype:    $DTYPE"
    echo "[INFO] Max len:  $MAX_LEN"
    echo "[INFO] GPU util: $GPU_UTIL"
    echo ""
    echo "[INFO] Server will be available at: http://localhost:$PORT/v1"
    echo "[INFO] Press Ctrl+C to stop the server."
    echo ""

    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --served-model-name "$MODEL_ID"
}

start_client() {
    read -rp "Server port [default: 8000]: " SERVER_PORT
    SERVER_PORT="${SERVER_PORT:-8000}"

    ACTIVE_MODEL=$($PYTHON -c "
import urllib.request, json
try:
    r = urllib.request.urlopen('http://localhost:$SERVER_PORT/v1/models')
    data = json.load(r)
    print(data['data'][0]['id'] if data['data'] else 'unknown')
except:
    print('')
" 2>/dev/null)

    if [ -z "$ACTIVE_MODEL" ]; then
        echo "[ERROR] No running vllm server found at port $SERVER_PORT."
        echo "        Start the server first with option [1] or [2]."
        exit 1
    fi
    echo "[OK] Connected to server. Active model: $ACTIVE_MODEL"
    echo ""

    TEMP_SCRIPT=$(mktemp /tmp/gemma4_vllm_client_XXXXXX.py)
    trap "rm -f $TEMP_SCRIPT" EXIT

    cat > "$TEMP_SCRIPT" << PYEOF
from openai import OpenAI

client = OpenAI(base_url="http://localhost:$SERVER_PORT/v1", api_key="not-needed")
MODEL = "$ACTIVE_MODEL"
messages = [{"role": "system", "content": "You are a helpful assistant."}]

print(f"Gemma 4 vllm Chat (model: {MODEL})")
print(f"Server: http://localhost:$SERVER_PORT/v1")
print("Type 'quit' to exit, 'reset' to clear history")
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
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        print("[INFO] Conversation history cleared.")
        continue

    messages.append({"role": "user", "content": user_input})
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            max_tokens=1024,
            stream=True,
        )
        print("Gemma: ", end="", flush=True)
        full = ""
        for chunk in stream:
            d = chunk.choices[0].delta.content or ""
            print(d, end="", flush=True)
            full += d
        print()
        messages.append({"role": "assistant", "content": full})
    except Exception as e:
        print(f"\n[ERROR] {e}")
        messages.pop()
PYEOF
    $PYTHON "$TEMP_SCRIPT"
}

case "$ACTION_CHOICE" in
    0) exit 0 ;;
    1) start_server ;;
    2)
        select_model
        read -rp "Server port [default: 8000]: " PORT; PORT="${PORT:-8000}"
        LOG_FILE="$HOME/gemma4-vllm.log"
        echo "[INFO] Starting vllm server in background..."
        echo "[INFO] Log: $LOG_FILE"
        nohup $PYTHON -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ID" \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.90 \
            --served-model-name "$MODEL_ID" \
            > "$LOG_FILE" 2>&1 &
        echo "[OK] Server PID: $!  Log: $LOG_FILE"
        echo "     Test: curl http://localhost:$PORT/v1/models"
        ;;
    3) start_client ;;
    4)
        select_model
        read -rp "Server port [default: 8000]: " PORT; PORT="${PORT:-8000}"
        LOG_FILE="$HOME/gemma4-vllm.log"
        echo "[INFO] Starting vllm server in background..."
        nohup $PYTHON -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ID" \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization 0.90 \
            --served-model-name "$MODEL_ID" \
            > "$LOG_FILE" 2>&1 &
        SERVER_PID=$!
        echo "[INFO] Server PID: $SERVER_PID  Log: $LOG_FILE"
        echo "[INFO] Waiting 30 seconds for server to initialize..."
        sleep 30
        ACTIVE_MODEL=$MODEL_ID
        SERVER_PORT=$PORT
        echo "[INFO] Starting chat client..."
        start_client
        ;;
    5)
        read -rp "Server port to test [default: 8000]: " PORT; PORT="${PORT:-8000}"
        echo ""
        $PYTHON -c "
import urllib.request, json
try:
    r = urllib.request.urlopen('http://localhost:$PORT/v1/models')
    data = json.load(r)
    for m in data['data']:
        print(f'  Model: {m[\"id\"]}')
    print('[OK] Server is running.')
except Exception as e:
    print(f'[ERROR] Cannot connect to http://localhost:$PORT  ({e})')
"
        ;;
    *) echo "[ERROR] Invalid choice."; exit 1 ;;
esac

echo ""
echo "[INFO] Done."
