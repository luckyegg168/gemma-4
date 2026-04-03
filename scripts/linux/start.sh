#!/bin/bash
# start.sh — Interactive chat session with a Gemma 4 model

echo "============================================================"
echo " Gemma 4 - Interactive Chat"
echo "============================================================"
echo ""

# --- Check prerequisites ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Run install-dep.sh first."
    exit 1
fi
python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] transformers not installed. Run install-dep.sh first."
    exit 1
fi

# --- Model selection ---
echo "  Select model:"
echo "  [1] E2B   - google/gemma-4-E2B-it    (2.3B eff, multimodal w/ audio)"
echo "  [2] E4B   - google/gemma-4-E4B-it    (4.5B eff, multimodal w/ audio)"
echo "  [3] 26B-A4B - google/gemma-4-26B-A4B-it  (MoE, fast large model)"
echo "  [4] 31B   - google/gemma-4-31B-it    (30.7B, highest quality)"
echo "  [5] Custom model ID or local path"
echo ""
read -p "  Enter choice [1-5]: " MODEL_CHOICE

case "$MODEL_CHOICE" in
    1) MODEL_ID="google/gemma-4-E2B-it" ;;
    2) MODEL_ID="google/gemma-4-E4B-it" ;;
    3) MODEL_ID="google/gemma-4-26B-A4B-it" ;;
    4) MODEL_ID="google/gemma-4-31B-it" ;;
    5)
        read -p "  Enter model ID or local path: " MODEL_ID
        if [ -z "$MODEL_ID" ]; then
            echo "[ERROR] No model specified."
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] Invalid choice."
        exit 1
        ;;
esac

# --- Thinking mode ---
echo ""
echo "  Thinking mode (extended reasoning):"
echo "  [1] Disabled — faster, standard responses"
echo "  [2] Enabled  — deeper reasoning before responding"
echo ""
read -p "  Enter choice [1-2, default 1]: " THINK_CHOICE
if [ "$THINK_CHOICE" = "2" ]; then
    ENABLE_THINKING="True"
    THINK_LABEL="Enabled"
else
    ENABLE_THINKING="False"
    THINK_LABEL="Disabled"
fi

# --- System prompt ---
echo ""
echo "  Enter system prompt (leave blank for default):"
read -p "  > " SYSTEM_PROMPT
if [ -z "$SYSTEM_PROMPT" ]; then
    SYSTEM_PROMPT="You are a helpful AI assistant powered by Google Gemma 4."
fi

# --- Max tokens ---
echo ""
read -p "  Max tokens per response [default: 2048]: " MAX_TOKENS
if [ -z "$MAX_TOKENS" ]; then
    MAX_TOKENS="2048"
fi

# --- Summary ---
echo ""
echo "  ---- Configuration ----"
echo "  Model:      $MODEL_ID"
echo "  Thinking:   $THINK_LABEL"
echo "  Max tokens: $MAX_TOKENS"
echo "  System:     $SYSTEM_PROMPT"
echo ""
echo "[INFO] Loading model... (first run downloads model weights)"
echo "[INFO] Type 'exit' or 'quit' to end the session."
echo "[INFO] Type 'reset' to clear conversation history."
echo "============================================================"
echo ""

# --- Write temp Python script ---
TEMP_SCRIPT="/tmp/gemma4_chat_$$.py"

cat > "$TEMP_SCRIPT" << PYEOF
import sys, os, re

MODEL_ID       = r"""$MODEL_ID"""
ENABLE_THINKING = $ENABLE_THINKING
SYSTEM_PROMPT  = r"""$SYSTEM_PROMPT"""
MAX_NEW_TOKENS = $MAX_TOKENS

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Run install-dep.sh to install required packages.")
    sys.exit(1)

print(f"[INFO] Loading {MODEL_ID} ...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("  Make sure you have accepted the license at:")
    print("  https://huggingface.co/google/gemma-4-E2B-it")
    print("  And run download-models.sh or check your internet connection.")
    sys.exit(1)

device = next(model.parameters()).device
print(f"[OK] Model loaded on device: {device}")
print()

history = []

def clean_for_history(text):
    """Remove thinking blocks from text before storing in history."""
    # Remove <|channel>thought\n...\n<channel|> blocks
    text = re.sub(r'<\|channel\>thought\n.*?\n<channel\|>', '', text, flags=re.DOTALL)
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Session ended.")
        break

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("[INFO] Goodbye!")
        break
    if user_input.lower() == "reset":
        history = []
        print("[INFO] Conversation history cleared.")
        continue

    history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )

    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    try:
        parsed = processor.parse_response(raw_response)
        thinking = parsed.get("thinking", "").strip()
        response = parsed.get("response", "").strip()
    except Exception:
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        thinking = ""

    if ENABLE_THINKING and thinking:
        print(f"\nThinking: {thinking[:300]}{'...' if len(thinking) > 300 else ''}")
        print()

    print(f"Gemma: {response}")
    print()

    # Store only clean response (no thinking traces) in history
    history.append({"role": "model", "content": clean_for_history(response)})
PYEOF

python3 "$TEMP_SCRIPT"
EXIT_CODE=$?
rm -f "$TEMP_SCRIPT"

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[ERROR] Chat session exited with error code $EXIT_CODE"
    echo "  Common causes: missing model, missing dependencies, out of memory."
    echo "  Run install-dep.sh and download-models.sh to prepare your environment."
fi
