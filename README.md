# Gemma 4 — Complete Toolkit

A comprehensive toolkit for Google's Gemma 4 model family: knowledge reference, Windows scripts, Linux scripts, and a full knowledge graph.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [File Structure](#3-file-structure)
4. [Quick Start](#4-quick-start)
5. [Model Selection Guide](#5-model-selection-guide)
6. [Script Reference](#6-script-reference)
7. [Python API Examples](#7-python-api-examples)
8. [Thinking Mode](#8-thinking-mode)
9. [Multimodal Usage](#9-multimodal-usage)
10. [GGUF / llama.cpp Usage](#10-gguf--llamacpp-usage)
11. [Knowledge Graph](#11-knowledge-graph)
12. [Troubleshooting](#12-troubleshooting)
13. [License](#13-license)

---

## 1. Project Overview

This toolkit covers the complete **Gemma 4** model family released by Google DeepMind (2025):

| Model | Type | Modalities | Best Use |
|---|---|---|---|
| `gemma-4-E2B-it` | Dense + PLE | Text, Image, **Audio** | Mobile / on-device |
| `gemma-4-E4B-it` | Dense + PLE | Text, Image, **Audio** | Edge GPU |
| `gemma-4-26B-A4B-it` | **MoE** | Text, Image | Fast large-scale inference |
| `gemma-4-31B-it` | Dense | Text, Image | Maximum quality |

All models are released under the **Apache 2.0 license** and require accepting [Google's usage terms](https://huggingface.co/google/gemma-4-E2B-it) on HuggingFace.

---

## 2. Prerequisites

### Required
- **Python 3.9+** (3.11+ recommended)
- **8 GB+ RAM** (E2B minimum), 48 GB+ for 31B in full precision
- **HuggingFace account** with Gemma 4 license accepted

### Recommended
- **NVIDIA GPU** with CUDA 11.8+ (VRAM requirements below)
- **Disk space**: E2B ~10 GB, E4B ~16 GB, 26B-A4B ~50 GB, 31B ~62 GB

### VRAM Requirements

| Model / Backend | Min VRAM | Notes |
|---|---|---|
| E2B full precision | 8 GB | Consumer GPU |
| E4B full precision | 12 GB | RTX 3080 / 4070 |
| **E2B Q4_K_M GGUF** | **4 GB** | llama.cpp — very fast |
| **E4B Q4_K_M GGUF** | **6 GB** | llama.cpp |
| **26B-A4B IQ4_XS GGUF** | **14 GB** | llama.cpp — RTX 3080 12GB |
| **26B-A4B MXFP4_MOE GGUF** | **18 GB** | llama.cpp — MoE optimized |
| **26B-A4B UD-Q4_K_M GGUF** | **18 GB** | llama.cpp — recommended |
| **31B Q4_K_M GGUF** | **20 GB** | llama.cpp — most popular (~84K dl/mo) |
| 26B-A4B vllm | 32 GB | A100 40GB — GPU only |
| 31B vllm | 48 GB+ | A100 80GB — GPU only |

---

## 3. File Structure

```
d:\gemma-4\                         (or ~/gemma-4 on Linux)
├── README.md                       ← This file
├── SKILL.md                        ← Complete knowledge reference (13 sections)
├── knowledge-graph.md              ← Mermaid diagrams + relationship maps
└── scripts/
    ├── windows/
    │   ├── install-dep.bat         ← Install transformers/torch baseline
    │   ├── install-dep-llamacpp.bat ← Install llama-cpp-python (GPU CUDA)
    │   ├── install-dep-vllm.bat    ← Install vllm (GPU server)
    │   ├── download-models.bat     ← Download HF models + GGUF files (16 options)
    │   ├── start.bat               ← Interactive chat (transformers)
    │   ├── start-llamacpp.bat      ← Interactive chat (llama.cpp GGUF, 12 models)
    │   ├── start-vllm.bat          ← vllm server + OpenAI client
    │   └── manager.bat             ← Full management console (13-option menu)
    └── linux/
        ├── install-dep.sh          ← Install transformers/torch baseline
        ├── install-dep-llamacpp.sh ← Install llama-cpp-python (GPU CUDA)
        ├── install-dep-vllm.sh     ← Install vllm (GPU server)
        ├── download-models.sh      ← Download HF models + GGUF files (16 options)
        ├── start.sh                ← Interactive chat (transformers)
        ├── start-llamacpp.sh       ← Interactive chat (llama.cpp GGUF, 12 models)
        ├── start-vllm.sh           ← vllm server + OpenAI client
        └── manager.sh              ← Full management console (13-option menu)
```

---

## 4. Quick Start

### Option A — Transformers (default backend)

```batch
REM Windows
cd d:\gemma-4\scripts\windows
install-dep.bat
download-models.bat        REM choose [1]-[4] for full HF models
start.bat
```

```bash
# Linux
cd ~/gemma-4/scripts/linux && chmod +x *.sh
./install-dep.sh
./download-models.sh       # choose [1]-[4] for full HF models
./start.sh
```

### Option B — llama.cpp / GGUF (GPU primary, CPU fallback)

```batch
REM Windows
cd d:\gemma-4\scripts\windows
install-dep-llamacpp.bat
download-models.bat        REM choose [5]-[15] for GGUF files
start-llamacpp.bat
```

```bash
# Linux
cd ~/gemma-4/scripts/linux && chmod +x *.sh
./install-dep-llamacpp.sh
./download-models.sh       # choose [5]-[15] for GGUF files (e.g. [9] 26B-A4B MXFP4_MOE, [13] 31B Q4_K_M)
./start-llamacpp.sh
```

> **Recommended GGUF choices:** 31B Q4_K_M `[13]` for max quality; 26B-A4B MXFP4_MOE `[9]` for MoE efficiency; E2B Q4_K_M `[5]` for minimum VRAM.

### Option C — vllm (GPU-only OpenAI API server)

> ⚠️ **vllm requires an NVIDIA GPU with CUDA.** CPU mode is NOT supported.

```batch
REM Windows
cd d:\gemma-4\scripts\windows
install-dep-vllm.bat
start-vllm.bat             REM [1] Start server → [2] Chat client (or [3] Combined)
```

```bash
# Linux
cd ~/gemma-4/scripts/linux && chmod +x *.sh
./install-dep-vllm.sh
./start-vllm.sh            # [1] Server foreground  [2] Server background  [3] Client only  [4] Combined
```

The vllm server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

### Option D — Management Console (all-in-one)

```batch
REM Windows
manager.bat
```

```bash
# Linux
./manager.sh
```

### Python (direct — transformers)

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "google/gemma-4-E2B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Hello, Gemma!"}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=True))
```

---

## 5. Model Selection Guide

### By Task

| Task | Recommended Model | Reason |
|---|---|---|
| Voice / audio input | E2B or E4B | Only models with audio encoder |
| On-device / mobile | E2B | Smallest, PLE architecture |
| Consumer GPU (<16 GB) | E4B or 26B-A4B IQ4_XS GGUF | Best quality/VRAM ratio |
| Fast batch inference | 26B-A4B | MoE: 128 experts, runs like 4B |
| Maximum accuracy | 31B | Highest benchmark scores |
| Long documents (256K ctx) | 26B-A4B or 31B | Extended context |

### By VRAM Budget

| VRAM | Best Choice |
|---|---|
| 8-12 GB | E2B (full) or E4B |
| 12-16 GB | E4B or 26B-A4B IQ4_XS GGUF |
| 16-24 GB | 26B-A4B Q4_K_M GGUF (recommended) |
| 24-32 GB | 26B-A4B Q8_0 GGUF or full precision |
| 48 GB+ | 31B full precision |

### Benchmark Summary

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| MMLU Pro | 85.2% | 82.6% | 69.4% | 60.0% |
| AIME 2026 | 89.2% | 88.3% | 42.5% | 37.5% |
| LiveCodeBench | 80.0% | 77.1% | 52.0% | 44.0% |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 43.4% |
| MMMU Pro (vision) | 76.9% | 73.8% | 52.6% | 44.2% |

---

## 6. Script Reference

### `install-dep.bat` / `install-dep.sh` — Transformers Baseline

Installs all Python dependencies (interactive, step-by-step):
1. Checks Python installation
2. Upgrades pip
3. Installs `transformers`, `torch`, `accelerate`
4. Installs `huggingface_hub` (and CLI)
5. Installs `Pillow`, `librosa`, `soundfile` (multimodal)
6. Optionally installs `flash-attn` (Linux CUDA only)
7. Optionally installs `llama-cpp-python` (CPU-only, basic)

---

### `install-dep-llamacpp.bat` / `install-dep-llamacpp.sh` — llama-cpp-python (CUDA)

Installs `llama-cpp-python` with GPU acceleration. Auto-detects CUDA:
1. Detects NVIDIA GPU via `nvidia-smi` / `nvcc`
2. Tries pre-built CUDA 12.1 wheel → CUDA 11.8 wheel
3. Falls back to source build with `CMAKE_ARGS=-DGGML_CUDA=on`
4. Falls back to CPU-only if no GPU available
5. Verifies: `from llama_cpp import Llama`

---

### `install-dep-vllm.bat` / `install-dep-vllm.sh` — vllm (GPU Server)

> ⚠️ Requires NVIDIA GPU with CUDA.

Installs [vllm](https://github.com/vllm-project/vllm) and OpenAI-compatible client:
1. Checks `nvidia-smi` (required)
2. Installs PyTorch with CUDA 12.1 (→ 11.8 fallback)
3. Installs `vllm`
4. Installs `openai` Python SDK
5. Installs `huggingface_hub`, `transformers`, `accelerate`
6. Verifies: `import vllm`

---

### `download-models.bat` / `download-models.sh` — Download Models (16 options)

Downloads model weights from HuggingFace. Menu options:

| Option | Model | Backend | Size |
|---|---|---|---|
| [1] | E2B full precision | transformers | ~10 GB |
| [2] | E4B full precision | transformers | ~16 GB |
| [3] | 26B-A4B full precision | transformers | ~50 GB |
| [4] | 31B full precision | transformers | ~62 GB |
| [5] | E2B Q4_K_M GGUF ⭐ | llama.cpp | 3.11 GB |
| [6] | E2B Q8_0 GGUF | llama.cpp | 5.05 GB |
| [7] | E4B Q4_K_M GGUF ⭐ | llama.cpp | 4.98 GB |
| [8] | E4B Q8_0 GGUF | llama.cpp | 8.19 GB |
| [9] | 26B-A4B MXFP4_MOE GGUF ⭐ (unique!) | llama.cpp | 16.7 GB |
| [10] | 26B-A4B UD-Q4_K_M GGUF ⭐ | llama.cpp | 16.9 GB |
| [11] | 26B-A4B IQ4_XS GGUF | llama.cpp | 13.4 GB |
| [12] | 26B-A4B Q8_0 GGUF | llama.cpp | 26.9 GB |
| [13] | 31B Q4_K_M GGUF ⭐ (most popular) | llama.cpp | 18.3 GB |
| [14] | 31B IQ4_XS GGUF | llama.cpp | 16.4 GB |
| [15] | 31B Q8_0 GGUF | llama.cpp | 32.6 GB |
| [16] | Custom model ID | any | varies |

> ⭐ = recommended choice for the model

---

### `start.bat` / `start.sh` — Interactive Chat (transformers)

Loads models via 🤗 `transformers`. Features:
- Select model from menu (E2B / E4B / 26B-A4B / 31B)
- Automatic `device_map="auto"` (GPU if available)
- Streaming token output
- Commands during chat: `quit`/`exit` to stop, `reset` to clear history

---

### `start-llamacpp.bat` / `start-llamacpp.sh` — Interactive Chat (llama.cpp GGUF)

Loads GGUF files via `llama-cpp-python`. 12-option model menu:

| Menu | Model | File | GPU Layers |
|---|---|---|---|
| [1] | E2B Q4_K_M | `gemma-4-E2B-it-Q4_K_M.gguf` | 35 |
| [2] | E2B Q8_0 | `gemma-4-E2B-it-Q8_0.gguf` | 35 |
| [3] | E2B IQ4_XS | `gemma-4-E2B-it-IQ4_XS.gguf` | 35 |
| [4] | E4B Q4_K_M | `gemma-4-E4B-it-Q4_K_M.gguf` | 42 |
| [5] | E4B Q8_0 | `gemma-4-E4B-it-Q8_0.gguf` | 42 |
| [6] | 26B-A4B MXFP4_MOE ⭐ | `gemma-4-26B-A4B-it-MXFP4_MOE.gguf` | 30 |
| [7] | 26B-A4B UD-Q4_K_M | `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | 30 |
| [8] | 26B-A4B Q8_0 | `gemma-4-26B-A4B-it-Q8_0.gguf` | 30 |
| [9] | 31B Q4_K_M ⭐ | `gemma-4-31B-it-Q4_K_M.gguf` | 60 |
| [10] | 31B IQ4_XS | `gemma-4-31B-it-IQ4_XS.gguf` | 60 |
| [11] | 31B Q8_0 | `gemma-4-31B-it-Q8_0.gguf` | 60 |
| [12] | Custom | user-specified path | user-specified |

- Default context: `4096` tokens (customizable per session)
- GPU primary (`n_gpu_layers=-1`), falls back to CPU if no GPU
- Chat format: `gemma`; sampling: `temperature=1.0`, `top_p=0.95`, `top_k=64`

---

### `start-vllm.bat` — vllm Server + Client (Windows)

> ⚠️ GPU required.

4 modes:
| Mode | Description |
|---|---|
| [1] Start vllm Server | Runs server in foreground (blocks terminal) |
| [2] Chat Client | Connects to existing server at localhost:8000 |
| [3] Start Server + Client | New window for server, waits 30s, launches client |
| [4] Test Connection | Checks if server is running, lists loaded model |

---

### `start-vllm.sh` — vllm Server + Client (Linux)

> ⚠️ GPU required.

5 modes:
| Mode | Description |
|---|---|
| [1] Server (foreground) | Blocks terminal, Ctrl+C to stop |
| [2] Server (background) | `nohup` — survives terminal close, logs to `vllm_server.log` |
| [3] Chat client | Connects to running server |
| [4] Server + client | Background server + 30s wait + client |
| [5] Test / status | Checks server + lists active model |

Server API endpoint: `http://localhost:8000/v1`

---

### `manager.bat` / `manager.sh` — Full Management Console (13-option menu)

| Option | Description |
|---|---|
| [1] Install Dependencies (transformers) | Run `install-dep` |
| [2] Install Dependencies (llama.cpp CUDA) | Run `install-dep-llamacpp` |
| [3] Install Dependencies (vllm GPU) | Run `install-dep-vllm` |
| [4] Download Models | Run `download-models` (16 options) |
| [5] Start Chat (transformers) | Run `start` |
| [6] Start Chat (llama.cpp GGUF) | Run `start-llamacpp` |
| [7] Start vllm Server | Run `start-vllm` |
| [8] Set HuggingFace Token | Save `HF_TOKEN` to `.env` |
| [9] List Downloaded Models | Scan models directory |
| [10] Show Model Info | GGUF tables + benchmark + backend comparison |
| [11] Check GPU Info | `nvidia-smi` output |
| [12] Check Python Environment | Installed packages + versions |
| [0] Exit | — |

Downloaded models are saved to `~/gemma4-models/` by default.

To use a locally downloaded model, pass the folder path as model ID in `start.bat` / `start.sh`.

---

## 7. Python API Examples

### Recommended Sampling Parameters

```python
generation_kwargs = dict(
    max_new_tokens=2048,
    temperature=1.0,   # Required for Gemma 4 — do not lower
    top_p=0.95,
    top_k=64,
    do_sample=True,
)
```

> **Important:** Gemma 4 is trained with `temperature=1.0`. Lowering temperature degrades output quality.

### Multi-turn Conversation

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, re

processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", dtype="auto", device_map="auto")

history = []
system_prompt = "You are a helpful AI assistant."

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": system_prompt}] + history

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024,
                                 temperature=1.0, top_p=0.95, top_k=64, do_sample=True)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"Gemma: {response}\n")
    history.append({"role": "model", "content": response})
```

### Image Input

```python
from PIL import Image
import requests

image = Image.open(requests.get("https://example.com/image.jpg", stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What do you see in this image?"}
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
```

**Image token budget** (pass to `apply_chat_template`):

| Use Case | `image_seq_length` |
|---|---|
| Documents / OCR | 1120 (default) |
| Detailed visual analysis | 256–1120 |
| Fast video frames | 70 |

### Audio Input (E2B / E4B only)

```python
import librosa, soundfile as sf

audio, sr = librosa.load("speech.wav", sr=16000)

messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Transcribe and summarize this audio."}
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, audios=[(audio, sr)], return_tensors="pt").to(model.device)
```

---

## 8. Thinking Mode

Gemma 4 supports **extended reasoning** (thinking mode) where the model generates a chain-of-thought before giving its final answer.

### Enable Thinking

```python
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True   # <-- enable
)
outputs = model.generate(**inputs, max_new_tokens=8192)  # use more tokens for thinking

# Parse thinking vs response
parsed = processor.parse_response(
    processor.decode(outputs[0][input_len:], skip_special_tokens=False)
)
print("Thinking:", parsed["thinking"])
print("Response:", parsed["response"])
```

### Thinking Mode Notes

| Model | Thinking Support | Notes |
|---|---|---|
| E2B | Fully suppressed when disabled | Clean output |
| E4B | Fully suppressed when disabled | Clean output |
| 26B-A4B | Emits empty `<channel>thought` block | Normal behavior |
| 31B | Emits empty `<channel>thought` block | Normal behavior |

**Multi-turn critical rule:** **Never include thinking traces in conversation history.** Only store the final `parsed["response"]` text.

---

## 9. Multimodal Usage

### Modality Support Matrix

| Model | Text | Image | Audio | Video (frames) |
|---|---|---|---|---|
| E2B | ✅ | ✅ | ✅ | ✅ (as frames) |
| E4B | ✅ | ✅ | ✅ | ✅ (as frames) |
| 26B-A4B | ✅ | ✅ | ❌ | ✅ (as frames) |
| 31B | ✅ | ✅ | ❌ | ✅ (as frames) |

### Multimodal Best Practices

1. **Order matters:** Put media before text in the content array
2. **Image budget:** Use `image_seq_length=70` for video, `1120` for documents
3. **Multi-image:** Supported, pass all images in `images=[]` list
4. **Video:** Extract frames manually and pass as multiple images
5. **Audio:** Only E2B and E4B support audio. Sample rate must be 16 kHz.

---

## 10. GGUF / llama.cpp Usage

Use GGUF models from `unsloth/gemma-4-26B-A4B-it-GGUF` for memory-efficient inference with `llama-cpp-python`.

### Install llama-cpp-python

```bash
# CPU only
pip install llama-cpp-python

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# AMD GPU (ROCm)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --no-cache-dir
```

### Inference

```python
from llama_cpp import Llama

llm = Llama(
    model_path="/path/to/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_gpu_layers=-1,   # offload all layers to GPU
    n_ctx=8192,
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain mixture of experts in 2 sentences."},
    ],
    max_tokens=512,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(output["choices"][0]["message"]["content"])
```

### GGUF Quantization Comparison

| Quant | Size | Quality | Use Case |
|---|---|---|---|
| IQ4_XS | 13.4 GB | Good | Minimum VRAM (14 GB GPU) |
| Q4_K_M | 16.9 GB | **Recommended** | Best balance |
| Q5_K_M | 21.2 GB | Very good | Higher quality |
| Q8_0 | 26.9 GB | Near-perfect | Quality priority |
| BF16 | 50.5 GB | Full precision | Research / comparison |

---

## 11. Knowledge Graph

See [knowledge-graph.md](knowledge-graph.md) for:

- **Family hierarchy graph** — model relationships and architecture types
- **Architecture components** — attention, MoE, PLE, vision/audio encoders
- **Modality graph** — which models support which input types
- **Thinking mode state machine** — flow chart of reasoning process
- **Deployment decision tree** — how to choose the right model
- **Benchmark charts** — visual comparison across all models
- **Entity relationship table** — structured data for all entities
- **Software dependency graph** — Python package relationships

---

## 12. Troubleshooting

### `OSError: You need to agree to the license`

You must accept the Gemma 4 license on HuggingFace:
1. Go to https://huggingface.co/google/gemma-4-E2B-it
2. Click "Agree and access repository"
3. Run `huggingface-cli login` or use option [7] in manager script

### `CUDA out of memory`

- Use a smaller model or GGUF quantized version
- Add `torch_dtype=torch.float16` in `from_pretrained`
- Use `device_map="auto"` to spread across GPUs

### `ModuleNotFoundError: No module named 'transformers'`

Run `install-dep.bat` (Windows) or `./install-dep.sh` (Linux).

### Thinking mode content leaking into history

**Never** put `parsed["thinking"]` in history. Only store `parsed["response"]`.

### Audio not working

Audio input is only supported on **E2B** and **E4B** models. The 26B-A4B and 31B models do not have an audio encoder.

### Slow loading on first run

The model is downloaded on first use if not cached. E2B takes ~10 GB, 31B takes ~62 GB. Subsequent loads use the local HuggingFace cache at `~/.cache/huggingface/`.

### Flash attention error on Windows

`flash-attn` is Linux-only. Skip it during install on Windows — it is optional.

---

## 13. License

- **Gemma 4 models:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) with [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Unsloth GGUF:** Apache 2.0
- **Scripts in this repo:** Apache 2.0

---

## Links

| Resource | URL |
|---|---|
| Gemma 4 E2B | https://huggingface.co/google/gemma-4-E2B-it |
| Gemma 4 E4B | https://huggingface.co/google/gemma-4-E4B-it |
| Gemma 4 26B-A4B | https://huggingface.co/google/gemma-4-26B-A4B-it |
| Gemma 4 31B | https://huggingface.co/google/gemma-4-31B-it |
| GGUF (unsloth) | https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF |
| HuggingFace Transformers | https://github.com/huggingface/transformers |
| llama-cpp-python | https://github.com/abetlen/llama-cpp-python |
