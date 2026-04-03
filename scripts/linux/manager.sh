#!/bin/bash
# manager.sh — Menu-driven full management interface for Gemma 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main_menu() {
    clear
    echo "============================================================"
    echo " Gemma 4 - Model Manager"
    echo " Complete management interface for Gemma 4 models"
    echo "============================================================"
    echo ""
    echo "  --- Setup ---"
    echo "  [1] Install Dependencies (transformers baseline)"
    echo "  [2] Install llama.cpp Dependencies (GPU GGUF)"
    echo "  [3] Install vllm Dependencies (GPU API server)"
    echo ""
    echo "  --- Models ---"
    echo "  [4] Download Models"
    echo ""
    echo "  --- Inference ---"
    echo "  [5] Start Chat (transformers)"
    echo "  [6] Start Chat (llama.cpp / GGUF)"
    echo "  [7] Start vllm Server (OpenAI API)"
    echo ""
    echo "  --- Tools ---"
    echo "  [8] Run Benchmark / Quick Test"
    echo "  [9] Show Model Information"
    echo "  [10] Show System Information"
    echo "  [11] Set HuggingFace Token"
    echo "  [12] List Downloaded Models"
    echo ""
    echo "  [0] Exit"
    echo ""
    read -p "Enter choice [0-12]: " MAIN_CHOICE

    case "$MAIN_CHOICE" in
        0) echo ""; echo "Goodbye!"; exit 0 ;;
        1) run_install ;;
        2) run_install_llamacpp ;;
        3) run_install_vllm ;;
        4) run_download ;;
        5) run_chat ;;
        6) run_chat_llamacpp ;;
        7) run_vllm_server ;;
        8) run_benchmark ;;
        9) show_model_info ;;
        10) show_system_info ;;
        11) set_token ;;
        12) list_models ;;
        *) echo "[ERROR] Invalid choice."; sleep 1; main_menu ;;
    esac
}

run_install() {
    clear
    echo "[INFO] Launching dependency installer..."
    bash "$SCRIPT_DIR/install-dep.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_install_llamacpp() {
    clear
    echo "[INFO] Launching llama.cpp dependency installer..."
    bash "$SCRIPT_DIR/install-dep-llamacpp.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_install_vllm() {
    clear
    echo "[INFO] Launching vllm dependency installer..."
    bash "$SCRIPT_DIR/install-dep-vllm.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_download() {
    clear
    echo "[INFO] Launching model downloader..."
    bash "$SCRIPT_DIR/download-models.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_chat() {
    clear
    echo "[INFO] Launching interactive chat (transformers)..."
    bash "$SCRIPT_DIR/start.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_chat_llamacpp() {
    clear
    echo "[INFO] Launching llama.cpp chat..."
    bash "$SCRIPT_DIR/start-llamacpp.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_vllm_server() {
    clear
    echo "[INFO] Launching vllm server..."
    bash "$SCRIPT_DIR/start-vllm.sh"
    read -p "Press Enter to return to menu..."
    main_menu
}

run_benchmark() {
    clear
    echo "============================================================"
    echo " Gemma 4 - Quick Benchmark / Smoke Test"
    echo "============================================================"
    echo ""
    echo "  Select model:"
    echo "  [1] E2B   - google/gemma-4-E2B-it"
    echo "  [2] E4B   - google/gemma-4-E4B-it"
    echo "  [3] 26B-A4B - google/gemma-4-26B-A4B-it"
    echo "  [4] 31B   - google/gemma-4-31B-it"
    echo "  [5] Custom"
    echo "  [0] Back"
    echo ""
    read -p "Enter choice: " BM_CHOICE

    case "$BM_CHOICE" in
        0) main_menu; return ;;
        1) BM_MODEL="google/gemma-4-E2B-it" ;;
        2) BM_MODEL="google/gemma-4-E4B-it" ;;
        3) BM_MODEL="google/gemma-4-26B-A4B-it" ;;
        4) BM_MODEL="google/gemma-4-31B-it" ;;
        5) read -p "Enter model ID: " BM_MODEL ;;
        *) BM_MODEL="" ;;
    esac

    if [ -z "$BM_MODEL" ]; then
        main_menu
        return
    fi

    echo ""
    echo "[INFO] Running benchmark on $BM_MODEL..."
    echo "[INFO] Prompts: Hello world, 2+2=?, capital of France"
    echo ""

    BM_SCRIPT="/tmp/gemma4_bench_$$.py"
    cat > "$BM_SCRIPT" << PYEOF
import time
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

MODEL_ID = r"""$BM_MODEL"""
PROMPTS = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "Write a one-line Python function to reverse a string.",
]

print(f"Loading {MODEL_ID}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
load_time = time.time() - t0
print(f"Load time: {load_time:.1f}s  |  Device: {next(model.parameters()).device}")
print("-" * 60)

total_tokens = 0
total_time = 0.0
for i, prompt in enumerate(PROMPTS, 1):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    t_start = time.time()
    with __import__('torch').inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
    elapsed = time.time() - t_start
    new_tokens = outputs.shape[-1] - input_len
    total_tokens += new_tokens
    total_time += elapsed
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"[{i}] Q: {prompt}")
    print(f"     A: {response[:100]}{'...' if len(response)>100 else ''}")
    print(f"     Tokens: {new_tokens}  Time: {elapsed:.2f}s  Speed: {new_tokens/elapsed:.1f} tok/s")
    print()

print(f"Average speed: {total_tokens/total_time:.1f} tokens/second")
PYEOF

    python3 "$BM_SCRIPT"
    rm -f "$BM_SCRIPT"
    echo ""
    read -p "Press Enter to return to menu..."
    main_menu
}

show_model_info() {
    clear
    echo "============================================================"
    echo " Gemma 4 - Model & GGUF Reference"
    echo "============================================================"
    echo ""
    echo " Full-precision HuggingFace Models:"
    echo "  Model           HF ID                          Params   Context  Audio  VRAM"
    echo "  E2B (Dense)     google/gemma-4-E2B-it          2.3B eff 128K     Yes    ~8 GB"
    echo "  E4B (Dense)     google/gemma-4-E4B-it          4.5B eff 128K     Yes    ~12 GB"
    echo "  26B-A4B (MoE)   google/gemma-4-26B-A4B-it      3.8B act 256K     No     ~32 GB"
    echo "  31B (Dense)     google/gemma-4-31B-it          30.7B    256K     No     ~48 GB"
    echo ""
    echo " GGUF - unsloth/gemma-4-E2B-it-GGUF:"
    echo "  Q4_K_M  3.11 GB  Recommended"
    echo "  Q8_0    5.05 GB  High quality"
    echo "  BF16    9.31 GB  Full precision"
    echo ""
    echo " GGUF - unsloth/gemma-4-E4B-it-GGUF:"
    echo "  Q4_K_M   4.98 GB  Recommended"
    echo "  Q8_0     8.19 GB  High quality"
    echo "  BF16    15.10 GB  Full precision"
    echo ""
    echo " GGUF - unsloth/gemma-4-26B-A4B-it-GGUF (MoE):"
    echo "  MXFP4_MOE  16.7 GB  MoE-optimized (unique quant)  Recommended"
    echo "  UD-Q4_K_M  16.9 GB  Standard recommended"
    echo "  IQ4_XS     13.4 GB  Smallest option"
    echo "  Q8_0       26.9 GB  Near full quality"
    echo "  BF16       50.5 GB  Full precision"
    echo ""
    echo " GGUF - unsloth/gemma-4-31B-it-GGUF (most popular ~84K dl/month):"
    echo "  Q4_K_M  18.3 GB  Recommended"
    echo "  IQ4_XS  16.4 GB  Smallest 31B"
    echo "  Q8_0    32.6 GB  Near full quality"
    echo "  BF16    61.4 GB  Full precision"
    echo ""
    echo " Backend Comparison:"
    echo "  transformers  Full HF weights  GPU+CPU  Multimodal  Easiest setup"
    echo "  llama.cpp     GGUF quant       GPU+CPU  Text only   Best CPU/edge"
    echo "  vllm          Full HF weights  GPU only Text only   Best throughput"
    echo ""
    echo " Recommended sampling: temperature=1.0, top_p=0.95, top_k=64"
    echo ""
    read -p "Press Enter to return to menu..."
    main_menu
}

show_system_info() {
    clear
    echo "============================================================"
    echo " System Information"
    echo "============================================================"
    echo ""

    # Python
    python3 --version 2>&1

    # PyTorch + CUDA
    python3 - <<'EOF' 2>/dev/null
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
n = torch.cuda.device_count()
print(f"CUDA devices: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory // (1024**3)} GB)")
EOF
    if [ $? -ne 0 ]; then
        echo "[WARNING] PyTorch not installed. Run install-dep.sh."
    fi

    echo ""

    # transformers
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null \
        || echo "[WARNING] transformers not installed."

    # huggingface_hub
    python3 -c "import huggingface_hub; print(f'HuggingFace Hub: {huggingface_hub.__version__}')" 2>/dev/null \
        || echo "[WARNING] huggingface_hub not installed."

    echo ""
    echo "  HuggingFace login:"
    huggingface-cli whoami 2>/dev/null \
        || echo "[NOT LOGGED IN] Run option [7] to set your token."

    echo ""
    read -p "Press Enter to return to menu..."
    main_menu
}

set_token() {
    clear
    echo "============================================================"
    echo " Set HuggingFace Access Token"
    echo "============================================================"
    echo ""
    echo "  Gemma 4 models require accepting the license on HuggingFace."
    echo "  1. Visit: https://huggingface.co/google/gemma-4-E2B-it"
    echo "  2. Accept the license agreement"
    echo "  3. Generate a token at: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Paste your HuggingFace token: " HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo "[ERROR] No token entered."
        sleep 1
    else
        huggingface-cli login --token "$HF_TOKEN"
        if [ $? -eq 0 ]; then
            echo "[OK] Login successful."
        else
            echo "[ERROR] Login failed. Check your token."
        fi
    fi
    echo ""
    read -p "Press Enter to return to menu..."
    main_menu
}

list_models() {
    clear
    echo "============================================================"
    echo " Downloaded Models"
    echo "============================================================"
    echo ""
    DEFAULT_DIR="$HOME/gemma4-models"
    read -p "Models directory to scan [default: $DEFAULT_DIR]: " CHECK_DIR
    if [ -z "$CHECK_DIR" ]; then
        CHECK_DIR="$DEFAULT_DIR"
    fi

    echo ""
    if [ ! -d "$CHECK_DIR" ]; then
        echo "  [INFO] Directory does not exist: $CHECK_DIR"
        echo "         No models downloaded yet."
        echo "         Run option [2] to download a model."
    else
        echo "  Contents of $CHECK_DIR:"
        echo ""
        for d in "$CHECK_DIR"/*/; do
            if [ -d "$d" ]; then
                folder_name="$(basename "$d")"
                # Count safetensors / gguf files
                file_count=$(find "$d" -maxdepth 1 \( -name "*.safetensors" -o -name "*.gguf" -o -name "*.bin" \) 2>/dev/null | wc -l)
                echo "    $folder_name  ($file_count weight files)"
            fi
        done
        echo ""
        echo "  Tip: Pass the full path to start.sh for offline inference."
    fi

    echo ""
    read -p "Press Enter to return to menu..."
    main_menu
}

# --- Entry point ---
main_menu
