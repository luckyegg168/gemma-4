# Gemma 4 — Complete Model Knowledge Skill

> **Domain:** Google DeepMind Gemma 4 open model family  
> **License:** Apache 2.0  
> **Authors:** Google DeepMind  
> **Knowledge Cutoff:** January 2025  
> **Last Updated:** 2025

---

## 1. Family Overview

Gemma 4 is Google DeepMind's fourth generation of open-weight, multimodal language models. The family is designed to cover deployment environments ranging from mobile phones and edge devices to high-end consumer GPUs and data center servers.

**Key Innovations:**
- Configurable **thinking / reasoning mode** (chain-of-thought via special tokens)
- **Hybrid attention** combining local sliding-window and global full-context layers
- **Mixture-of-Experts (MoE)** variant for fast inference at low active parameter cost
- **Per-Layer Embeddings (PLE)** for on-device efficiency in E2B/E4B
- Native **system role** support (unlike Gemma 3)
- Multimodal: **Text + Image + Audio** (small models) or **Text + Image** (large models)
- Function calling / tool use natively supported
- 140+ language pre-training data, 35+ languages supported out-of-the-box
- 262K vocabulary size across all variants

---

## 2. Model Variants and Specifications

### 2.1 Dense Models

| Property | E2B | E4B | 31B |
|---|---|---|---|
| HuggingFace ID | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` | `google/gemma-4-31B-it` |
| Effective Parameters | 2.3B | 4.5B | 30.7B |
| Total Parameters (with embeddings) | ~5.1B | ~8B | ~30.7B |
| Layers | 35 | 42 | 60 |
| Vocabulary Size | 262K | 262K | 262K |
| Sliding Window Size | 512 tokens | 512 tokens | 1024 tokens |
| Context Length | 128K tokens | 128K tokens | 256K tokens |
| Modalities | Text, Image, Audio | Text, Image, Audio | Text, Image |
| Vision Encoder Params | ~150M | ~150M | ~550M |
| Audio Encoder Params | ~300M | ~300M | N/A |
| Architecture Feature | PLE (Per-Layer Embeddings) | PLE | Standard Dense |
| HF Reported Model Size | ~5B | ~8B | ~33B |
| Target Deployment | Mobile / Edge | Mobile / Edge | GPU Workstation / Server |

### 2.2 Mixture-of-Experts (MoE) Model

| Property | 26B-A4B |
|---|---|
| HuggingFace ID | `google/gemma-4-26B-A4B-it` |
| Total Parameters | 25.2B |
| Active Parameters (per forward pass) | 3.8B |
| Layers | 30 |
| Vocabulary Size | 262K |
| Sliding Window Size | 1024 tokens |
| Context Length | 256K tokens |
| Expert Configuration | 8 active / 128 total + 1 shared expert |
| Modalities | Text, Image |
| Vision Encoder Parameters | ~550M |
| HF Reported Model Size | ~27B |
| Effective Compute Speed | Similar to a 4B dense model |
| Target Deployment | Consumer GPU / Laptop / Server |

**Why "26B-A4B":**  
"26B" = total stored weight size; "A4B" = active parameters used per inference pass. Only 3.8B parameters are activated during each forward pass, making this model run almost as fast as a 4B dense model while retaining 26B knowledge capacity.

**Why "E2B" / "E4B":**  
"E" = effective parameters. These models use Per-Layer Embeddings (PLE) where each decoder layer has its own embedding per token. Embedding tables are large in size but only require fast lookups during inference, resulting in lower effective compute parameter counts.

---

## 3. GGUF Quantization (unsloth/gemma-4-26B-A4B-it-GGUF)

Available from [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF).  
Use these quantizations with **llama.cpp** or any GGUF-compatible runtime.

| Bits | Quantization | File Size |
|---|---|---|
| 2-bit | UD-IQ2_XXS | 9.88 GB |
| 2-bit | UD-IQ2_M | 9.97 GB |
| 2-bit | UD-Q2_K_XL | 10.5 GB |
| 3-bit | UD-IQ3_XXS | 11.2 GB |
| 3-bit | UD-IQ3_S | 11.2 GB |
| 3-bit | UD-Q3_K_S | 12.5 GB |
| 3-bit | UD-Q3_K_M | 12.5 GB |
| 3-bit | UD-Q3_K_XL | 12.9 GB |
| 4-bit | UD-IQ4_XS | 13.4 GB |
| 4-bit | UD-IQ4_NL | 13.4 GB |
| 4-bit | UD-Q4_K_S | 16.4 GB |
| 4-bit | MXFP4_MOE | 16.7 GB |
| 4-bit | UD-Q4_K_M | 16.9 GB |
| 4-bit | UD-Q4_K_XL | 17.1 GB |
| 5-bit | UD-Q5_K_S | 18.8 GB |
| 5-bit | UD-Q5_K_M | 21.2 GB |
| 5-bit | UD-Q5_K_XL | 21.3 GB |
| 6-bit | UD-Q6_K | 22.9 GB |
| 6-bit | UD-Q6_K_XL | 23.8 GB |
| 8-bit | Q8_0 | 26.9 GB |
| 8-bit | UD-Q8_K_XL | 27.9 GB |
| 16-bit | BF16 | 50.5 GB |

**Recommended:** `UD-Q4_K_M` (16.9 GB) for balance of quality and memory.  
**Minimum VRAM 8GB:** `UD-IQ4_XS` or `UD-IQ4_NL` (13.4 GB) with CPU offloading.

---

## 4. Architecture Deep Dive

### 4.1 Hybrid Attention Mechanism
All Gemma 4 models interleave two types of attention layers:
- **Local Sliding Window Attention:** Attends only to a local window of neighboring tokens.
  - E2B/E4B: 512-token window
  - 26B-A4B/31B: 1024-token window
- **Global Full Attention:** Attends to the entire sequence.
  - Memory-efficient via **Unified Keys and Values** across global layers
  - Uses **Proportional RoPE (p-RoPE)** for position encoding

**Rule:** The final transformer layer is always a global attention layer.

### 4.2 Per-Layer Embeddings (PLE) — E2B / E4B Only
Instead of a single shared embedding matrix, each decoder layer has its own embedding table per token. This results in:
- Large total parameter count (explains "5.1B total" for a "2.3B effective" model)
- No additional compute overhead — embeddings are lookup operations
- Superior on-device deployment (embedding tables compress well on flash storage)

### 4.3 Mixture of Experts (MoE) — 26B-A4B Only
- 128 expert feed-forward networks + 1 permanent shared expert
- 8 experts selected per token per MoE layer
- Router network dynamically selects experts
- Enables 26B-scale knowledge with 4B-scale compute cost

---

## 5. Capabilities Matrix

| Capability | E2B | E4B | 26B-A4B | 31B |
|---|---|---|---|---|
| Text Generation | ✅ | ✅ | ✅ | ✅ |
| Reasoning / Thinking Mode | ✅ | ✅ | ✅ | ✅ |
| Image Understanding | ✅ | ✅ | ✅ | ✅ |
| Video Understanding (frames) | ✅ | ✅ | ✅ | ✅ |
| Audio Input (ASR/AST) | ✅ | ✅ | ❌ | ❌ |
| Function Calling / Tool Use | ✅ | ✅ | ✅ | ✅ |
| Multilingual | ✅ | ✅ | ✅ | ✅ |
| Code Generation | ✅ | ✅ | ✅ | ✅ |
| Long Context (128K+) | 128K | 128K | 256K | 256K |
| Document / OCR / Chart | ✅ | ✅ | ✅ | ✅ |
| Thinking Fully Suppressible | ✅ | ✅ | ❌* | ❌* |

> *26B-A4B and 31B still output empty `<|channel>thought\n<channel|>` when thinking is disabled.

---

## 6. Benchmark Results

All results are for instruction-tuned (IT) variants.

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| **MMLU Pro** | 85.2% | 82.6% | 69.4% | 60.0% |
| **AIME 2026 (no tools)** | 89.2% | 88.3% | 42.5% | 37.5% |
| **LiveCodeBench v6** | 80.0% | 77.1% | 52.0% | 44.0% |
| **Codeforces ELO** | 2150 | 1718 | 940 | 633 |
| **GPQA Diamond** | 84.3% | 82.3% | 58.6% | 43.4% |
| **BigBench Extra Hard** | 74.4% | 64.8% | 33.1% | 21.9% |
| **MMMLU (multilingual)** | 88.4% | 86.3% | 76.6% | 67.4% |
| **HLE (no tools)** | 19.5% | 8.7% | — | — |
| **HLE (with search)** | 26.5% | 17.2% | — | — |
| **Tau2 (avg 3)** | 76.9% | 68.2% | 42.2% | 24.5% |
| MMMU Pro (Vision) | 76.9% | 73.8% | 52.6% | 44.2% |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 52.4% |
| MedXPertQA MM | 61.3% | 58.1% | 28.7% | 23.5% |
| OmniDocBench 1.5 (↓ better) | 0.131 | 0.149 | 0.181 | 0.290 |
| MRCR v2 8-needle 128K (avg) | 66.4% | 44.1% | 25.4% | 19.1% |
| CoVoST (Audio, ↓ better) | — | — | 35.54 | 33.47 |
| FLEURS (WER, ↓ better) | — | — | 0.08 | 0.09 |

---

## 7. Best Practices for Inference

### 7.1 Sampling Parameters
Always use these standardized parameters for consistent quality:
```
temperature = 1.0
top_p       = 0.95
top_k       = 64
```

### 7.2 Thinking Mode
**Enable thinking:**  
Add `<|think|>` token at the start of the system prompt content.

**Disable thinking:**  
Remove the `<|think|>` token from system prompt; use `enable_thinking=False` in `apply_chat_template`.

**When thinking is active, model output structure:**
```
<|channel>thought\n
[Internal step-by-step reasoning]
<channel|>
[Final answer]
```

**Suppression behavior:**
- E2B / E4B: Full suppression — output contains only the final answer.
- 26B-A4B / 31B: Still outputs empty thought block: `<|channel>thought\n<channel|>[Final answer]`

**Multi-turn rule:** Do NOT include thoughts from previous turns in conversation history. Only include the final parsed answer.

### 7.3 Modality Input Order
For multimodal prompts, always place **image/audio before text** for best performance:
```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "url": "..."},    # image FIRST
        {"type": "text", "text": "Describe this image."}  # text AFTER
    ]}
]
```

### 7.4 Variable Image Resolution (Token Budget)
Control image detail vs. speed using visual token budgets:

| Token Budget | Use Case |
|---|---|
| 70  | Video frame classification, fast captioning |
| 140 | Lightweight image captioning |
| 280 | General image understanding |
| 560 | Fine-grained understanding |
| 1120 | OCR, document parsing, reading small text |

### 7.5 Audio Constraints
- Maximum audio length: **30 seconds**
- Maximum video length: **60 seconds** (at 1 frame per second)
- Audio is only supported on **E2B** and **E4B**

**ASR prompt template:**
```
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.
Follow these specific instructions:
* Only output the transcription, with no newlines.
* Write numbers as digits: write 1.7, not "one point seven"; write 3, not "three".
```

**AST (translation) prompt template:**
```
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
Output the transcription in {SOURCE_LANGUAGE}, then one newline, then "{TARGET_LANGUAGE}: ", then the translation.
```

---

## 8. Code Patterns

### 8.1 Basic Text Inference
```python
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E4B-it"  # replace with desired model

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the Mixture of Experts architecture."},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # set True to enable reasoning
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True,
)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(response)
print(parsed)
```

### 8.2 Enable Thinking Mode
```python
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # enable reasoning
)
outputs = model.generate(**inputs, max_new_tokens=4096)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
parsed = processor.parse_response(response)
# parsed["thinking"]  -- contains the reasoning trace
# parsed["response"] -- contains the final answer
```

### 8.3 Image Input
```python
from PIL import Image

messages = [
    {"role": "user", "content": [
        {"type": "image", "url": "https://example.com/chart.png"},  # or local path
        {"type": "text", "text": "What trend does this chart show?"},
    ]}
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image = Image.open("chart.png")
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
```

### 8.4 Audio Input (E2B / E4B only)
```python
import librosa

audio_path = "speech.wav"
audio, sr = librosa.load(audio_path, sr=16000)

messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Transcribe the following speech segment in English into English text."},
    ]}
]
inputs = processor(text=text, audio=audio, return_tensors="pt").to(model.device)
```

### 8.5 GGUF with llama.cpp
```bash
# Install llama.cpp with CUDA support (Linux/Windows)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Download GGUF file
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
    --include "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
    --local-dir ./models/gemma-4-26B-A4B-it-GGUF/

# Run inference with llama.cpp CLI
./llama-cli -m ./models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
    --chat-template gemma \
    -p "Explain quantum entanglement simply." \
    -n 512 \
    --temp 1.0 --top-p 0.95 --top-k 64
```

---

## 9. Model Selection Guide

| Scenario | Recommended Model | Reason |
|---|---|---|
| Mobile app / wearable | E2B | Lowest compute, audio support, 128K context |
| Edge device / Raspberry Pi | E2B | Fits in constrained memory |
| Laptop without discrete GPU | E4B (GGUF Q4) | Good quality, runs on CPU |
| Consumer GPU (8–16 GB VRAM) | 26B-A4B (GGUF Q4) | Fastest large model, 4B inference cost |
| Consumer GPU (24 GB VRAM) | 31B or 26B-A4B (Q8) | Maximum quality |
| Data center / multi-GPU | 31B (BF16) | Best benchmarks |
| Audio transcription tasks | E2B or E4B | Only models with audio encoder |
| Coding / reasoning benchmark | 31B or 26B-A4B | Highest AIME/LiveCode scores |
| Long document analysis | 26B-A4B or 31B | 256K context |
| Low-latency agentic loop | 26B-A4B | Near-4B speed with 26B capacity |
| Multilingual chatbot | Any | All support 35+ languages |

---

## 10. Hardware Requirements

### HuggingFace Transformers (Full Precision)

| Model | Minimum VRAM | Recommended | Notes |
|---|---|---|---|
| E2B | 8 GB | 12 GB | Multi-GPU or cpu_offload for RAM |
| E4B | 12 GB | 16 GB | |
| 26B-A4B | 32 GB (or two 16 GB) | 40 GB | MoE loads all experts to VRAM |
| 31B | 48 GB | 80 GB | A100/H100 or two A40s |

### GGUF / llama.cpp (Quantized)

| Quantization | Model | VRAM Needed |
|---|---|---|
| Q4_K_M (26B-A4B) | 26B-A4B | ~17 GB |
| IQ4_XS (26B-A4B) | 26B-A4B | ~13.4 GB |
| Q4_K_M (31B) | 31B | ~20 GB (approx) |

---

## 11. Important Notes and Gotchas

1. **Thinking suppression behavior differs by model size**: E2B/E4B fully suppress thoughts; 26B-A4B and 31B emit empty thought blocks.
2. **Multi-turn history**: Never include `<|channel>thought\n...<channel|>` blocks in conversation history.
3. **Modality order matters**: Always put images/audio before text in the content list.
4. **MoE VRAM**: Despite 3.8B active parameters, the 26B-A4B model loads ALL 25.2B weights into VRAM/RAM.
5. **E2B/E4B total size**: The "5.1B" and "8B" total sizes are due to PLE embedding tables, not additional compute layers.
6. **Token budget for images**: Default is 256 tokens; set `image_token_budget` in processor for custom values.
7. **transformers version**: Always use the **latest** `transformers` version (`pip install -U transformers`).
8. **Function calling**: Uses standard OpenAI-style tool definitions; the chat template handles schema injection.

---

## 12. Dependencies

### Python (HuggingFace Transformers)
```
transformers>=4.52.0    # latest version required
torch>=2.0.0
accelerate>=0.30.0
huggingface_hub>=0.23.0

# Optional for audio
librosa>=0.10.0
soundfile>=0.12.0

# Optional for image
Pillow>=10.0.0

# Optional for faster inference
bitsandbytes>=0.43.0    # 4-bit/8-bit quantization
flash-attn>=2.5.0       # Flash Attention 2 (Linux + CUDA)
```

### llama.cpp (GGUF)
```
llama-cpp-python>=0.2.0
# or build llama.cpp from source with CUDA/Metal/Vulkan support
```

---

## 13. Useful Links

| Resource | URL |
|---|---|
| HuggingFace Collection | https://huggingface.co/collections/google/gemma-4 |
| Official Documentation | https://ai.google.dev/gemma/docs/core |
| GitHub | https://github.com/google-gemma |
| Launch Blog | https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ |
| Responsible AI Toolkit | https://ai.google.dev/responsible |
| Apache 2.0 License | https://ai.google.dev/gemma/docs/gemma_4_license |
| Unsloth GGUF | https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF |
| Unsloth Run Guide | https://docs.unsloth.ai/models/gemma-4 |
| E2B IT | https://huggingface.co/google/gemma-4-E2B-it |
| E4B IT | https://huggingface.co/google/gemma-4-E4B-it |
| 26B-A4B IT | https://huggingface.co/google/gemma-4-26B-A4B-it |
| 31B IT | https://huggingface.co/google/gemma-4-31B-it |
