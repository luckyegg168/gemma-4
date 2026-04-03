# Gemma 4 Knowledge Graph

This document provides a structured knowledge graph of the Gemma 4 model family using Mermaid diagrams and tabular relationship maps.

---

## 1. Family Hierarchy

```mermaid
graph TD
    G4["Gemma 4\n(Google DeepMind)"]

    G4 --> DENSE["Dense Arch"]
    G4 --> MOE["MoE Arch"]

    DENSE --> E2B["E2B\ngoogle/gemma-4-E2B-it\n2.3B effective / 5.1B total\n128K ctx | Text+Image+Audio"]
    DENSE --> E4B["E4B\ngoogle/gemma-4-E4B-it\n4.5B effective / 8B total\n128K ctx | Text+Image+Audio"]
    DENSE --> B31["31B\ngoogle/gemma-4-31B-it\n30.7B params\n256K ctx | Text+Image"]

    MOE --> B26["26B-A4B\ngoogle/gemma-4-26B-A4B-it\n25.2B total / 3.8B active\n256K ctx | Text+Image"]

    E2B --> GGUF_E2B["GGUF - unsloth/gemma-4-E2B-it-GGUF\nQ4_K_M 3.11GB | Q8_0 5.05GB | BF16 9.31GB"]
    E4B --> GGUF_E4B["GGUF - unsloth/gemma-4-E4B-it-GGUF\nQ4_K_M 4.98GB | Q8_0 8.19GB | BF16 15.1GB"]
    B26 --> GGUF_26B["GGUF - unsloth/gemma-4-26B-A4B-it-GGUF\nMXFP4_MOE 16.7GB | UD-Q4_K_M 16.9GB\nIQ4_XS 13.4GB | Q8_0 26.9GB | BF16 50.5GB"]
    B31 --> GGUF_31B["GGUF - unsloth/gemma-4-31B-it-GGUF\nQ4_K_M 18.3GB | IQ4_XS 16.4GB\nQ8_0 32.6GB | BF16 61.4GB"]

    GGUF_26B --> MXFP4["MXFP4_MOE\nUnique MoE-optimized quant\nOnly for 26B-A4B"]
```

---

## 1b. Backend Ecosystem

```mermaid
graph LR
    E2B2["E2B"] --> TF["transformers"]
    E4B2["E4B"] --> TF
    B26_2["26B-A4B"] --> TF
    B31_2["31B"] --> TF

    E2B2 --> LLCPP["llama.cpp\n(llama-cpp-python)\nGPU primary, CPU fallback"]
    E4B2 --> LLCPP
    B26_2 --> LLCPP
    B31_2 --> LLCPP

    E2B2 --> VLLM["vllm\nOpenAI-compatible API\nGPU ONLY"]
    E4B2 --> VLLM
    B26_2 --> VLLM
    B31_2 --> VLLM

    TF --> OUT["Text Output"]
    LLCPP --> OUT
    VLLM --> OUT
```

---

## 1c. Toolkit Scripts

```mermaid
graph TD
    MANAGER["manager.bat / manager.sh\n(Management Console)"]

    MANAGER --> INST["install-dep.bat/.sh\n(transformers baseline)"]
    MANAGER --> INST_LLC["install-dep-llamacpp.bat/.sh\n(GPU CUDA GGUF)"]
    MANAGER --> INST_VLLM["install-dep-vllm.bat/.sh\n(GPU vllm server)"]
    MANAGER --> DL["download-models.bat/.sh\n(16-option menu)"]
    MANAGER --> CHAT["start.bat/.sh\n(transformers chat)"]
    MANAGER --> CHAT_LLC["start-llamacpp.bat/.sh\n(GGUF chat, 12 models)"]
    MANAGER --> CHAT_VLLM["start-vllm.bat/.sh\n(server + client)"]

    INST_LLC --> LLCPP2["llama-cpp-python\nCUDA 12.1/11.8 wheel\nor source build"]
    INST_VLLM --> VLLM2["vllm + PyTorch CUDA\nOpenAI SDK"]
    CHAT_LLC --> GGUF_FILES["GGUF files\n(Q4_K_M, Q8_0, etc.)"]
    CHAT_VLLM --> API["http://localhost:8000/v1"]
```

---

## 2. Architecture Components

```mermaid
graph LR
    ATTN["Hybrid Attention"]
    ATTN --> SWA["Local Sliding Window\n(E2B/E4B: 512 tok)\n(26B/31B: 1024 tok)"]
    ATTN --> GLOBAL["Full Global Attention\n(final layer always global)"]
    GLOBAL --> UKVA["Unified KV Arrays"]
    GLOBAL --> PROPE["Proportional RoPE (p-RoPE)"]

    DENSE2["Dense Models\n(E2B, E4B, 31B)"]
    DENSE2 --> PLE["Per-Layer Embeddings (PLE)\nE2B + E4B only\nLarge tables, lookup-only cost"]

    MOE2["MoE Model\n(26B-A4B)"]
    MOE2 --> ROUTER["Expert Router\n8 active / 128 + 1 shared"]
    MOE2 --> EXPERTS["Expert FFN Layers\n3.8B active per token"]
```

---

## 3. Modality Graph

```mermaid
graph TD
    INPUT["Inputs"]
    INPUT --> TEXT["Text\n(all models)"]
    INPUT --> IMAGE["Image\n(all models)\nVariable aspect ratio\nToken budgets: 70–1120"]
    INPUT --> VIDEO["Video as Frames\n(all models)\nMax 60 sec @ 1fps"]
    INPUT --> AUDIO["Audio\n(E2B & E4B only)\nMax 30 sec\nASR + AST"]

    TEXT --> LM["Language Model Core"]
    IMAGE --> VENC["Vision Encoder\nE2B/E4B: ~150M\n26B/31B: ~550M"]
    AUDIO --> AENC["Audio Encoder\nE2B/E4B: ~300M"]
    VENC --> LM
    AENC --> LM
    LM --> OUTPUT["Text Output"]
```

---

## 4. Thinking Mode State Machine

```mermaid
stateDiagram-v2
    [*] --> SystemPrompt
    SystemPrompt --> ThinkingEnabled : add <|think|> token\nor enable_thinking=True
    SystemPrompt --> ThinkingDisabled : no token\nor enable_thinking=False

    ThinkingEnabled --> InternalReasoning : model generates\n<|channel>thought\n[reasoning]<channel|>
    InternalReasoning --> FinalAnswer : parsed by processor.parse_response()

    ThinkingDisabled --> DirectAnswer : E2B / E4B
    ThinkingDisabled --> EmptyBlockAnswer : 26B-A4B / 31B\n(empty thought block)
    EmptyBlockAnswer --> FinalAnswer
    DirectAnswer --> FinalAnswer
```

---

## 5. Deployment Decision Tree

```mermaid
graph TD
    START["Choose Gemma 4 Model"]
    START --> Q1{"Deployment target?"}

    Q1 --> MOBILE["Mobile / Edge Device"]
    Q1 --> LAPTOP["Laptop / No GPU"]
    Q1 --> CGPU["Consumer GPU\n(8–24 GB VRAM)"]
    Q1 --> SERVER["Server / Data Center"]

    MOBILE --> Q2{"Need audio?"}
    Q2 --> YES_AUDIO["Yes → E2B"]
    Q2 --> NO_AUDIO["No → E2B (smaller)\nor E4B (higher quality)"]

    LAPTOP --> GGUF2["E4B or 26B-A4B\nGGUF Q4 via llama.cpp"]

    CGPU --> Q3{"VRAM?"}
    Q3 --> VRAM8["8–16 GB → 26B-A4B Q4\n(IQ4_XS = 13.4 GB)"]
    Q3 --> VRAM24["20–24 GB → 26B-A4B Q8\nor 31B Q4"]

    SERVER --> Q4{"Priorities?"}
    Q4 --> SPEED["Speed → 26B-A4B BF16"]
    Q4 --> QUALITY["Quality → 31B BF16"]
```

---

## 6. Capability Coverage Map

```mermaid
graph LR
    subgraph "Core Language"
        TG["Text Generation"]
        MULTI["Multilingual\n140+ pre-train, 35+ OOB"]
        CODE["Code Gen / Completion"]
        REASON["Reasoning / Thinking"]
    end

    subgraph "Vision"
        IMG["Image Understanding"]
        DOC["Document / OCR / Chart"]
        VID["Video (sequences of frames)"]
        POINT["Object Pointing / Detection"]
    end

    subgraph "Audio (E2B/E4B only)"
        ASR["Speech Recognition (ASR)"]
        AST["Speech Translation (AST)"]
    end

    subgraph "Agentic"
        FC["Function Calling / Tool Use"]
        AGENT["Agentic Workflows"]
    end

    E2B2["E2B"] --> TG & MULTI & CODE & REASON
    E2B2 --> IMG & DOC & VID & POINT
    E2B2 --> ASR & AST
    E2B2 --> FC & AGENT

    E4B2["E4B"] --> TG & MULTI & CODE & REASON
    E4B2 --> IMG & DOC & VID & POINT
    E4B2 --> ASR & AST
    E4B2 --> FC & AGENT

    B26_2["26B-A4B"] --> TG & MULTI & CODE & REASON
    B26_2 --> IMG & DOC & VID & POINT
    B26_2 --> FC & AGENT

    B31_2["31B"] --> TG & MULTI & CODE & REASON
    B31_2 --> IMG & DOC & VID & POINT
    B31_2 --> FC & AGENT
```

---

## 7. Benchmark Relationships

```mermaid
xychart-beta
    title "MMLU Pro Score by Model"
    x-axis ["E2B", "E4B", "26B-A4B", "31B"]
    y-axis "MMLU Pro %" 0 --> 100
    bar [60.0, 69.4, 82.6, 85.2]
```

```mermaid
xychart-beta
    title "AIME 2026 Score by Model"
    x-axis ["E2B", "E4B", "26B-A4B", "31B"]
    y-axis "AIME 2026 %" 0 --> 100
    bar [37.5, 42.5, 88.3, 89.2]
```

---

## 8. Entity Relationship Map (Textual)

| Entity | Relationship | Target Entity |
|---|---|---|
| Gemma 4 | is_family_of | E2B, E4B, 26B-A4B, 31B |
| E2B | has_architecture | Dense + PLE |
| E4B | has_architecture | Dense + PLE |
| 26B-A4B | has_architecture | Mixture-of-Experts |
| 31B | has_architecture | Dense |
| E2B, E4B | supports_modality | Text, Image, Video, Audio |
| 26B-A4B, 31B | supports_modality | Text, Image, Video |
| all_models | uses_mechanism | Hybrid Attention |
| Hybrid Attention | consists_of | Sliding Window + Global |
| Global Attention | uses | Unified KV + p-RoPE |
| E2B, E4B | uses_feature | PLE (Per-Layer Embeddings) |
| 26B-A4B | uses_feature | Expert Router (8/128+1) |
| 26B-A4B | has_gguf | unsloth/gemma-4-26B-A4B-it-GGUF |
| unsloth GGUF | supports_quant | Q2 to BF16 |
| all_models | supports_feature | Thinking Mode |
| all_models | supports_feature | Function Calling |
| all_models | trained_with | transformers + accelerate |
| all_models | license | Apache 2.0 |
| Google DeepMind | created | Gemma 4 |
| E2B_vision_encoder | param_count | ~150M |
| E4B_vision_encoder | param_count | ~150M |
| 26B_vision_encoder | param_count | ~550M |
| 31B_vision_encoder | param_count | ~550M |
| E2B_audio_encoder | param_count | ~300M |
| E4B_audio_encoder | param_count | ~300M |

---

## 9. Software Dependency Graph

```mermaid
graph TD
    PY["Python 3.9+"]
    PY --> TR["transformers (latest)"]
    PY --> TH["torch >= 2.0"]
    PY --> ACC["accelerate >= 0.30"]
    PY --> HH["huggingface_hub"]

    TR --> AP["AutoProcessor"]
    TR --> AM["AutoModelForCausalLM"]

    AP --> GEMMA["Gemma 4 Model"]
    AM --> GEMMA

    OPT["Optional"]
    OPT --> PIL["Pillow (images)"]
    OPT --> LIB["librosa (audio)"]
    OPT --> BNB["bitsandbytes (quantization)"]
    OPT --> FA["flash-attn (Linux+CUDA)"]

    GGUF3["GGUF Path"]
    GGUF3 --> LCPP["llama-cpp-python"]
    GGUF3 --> DL["huggingface-cli download"]

    PIL --> GEMMA
    LIB --> GEMMA
    BNB --> GEMMA
    FA --> GEMMA
    LCPP --> GFILE["*.gguf model file"]
```

---

## 10. Parameter Scale Visualization

```
Model       Effective Params    Total Params      Context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E2B         2.3B  [██░░░░░░░░]  5.1B (total)     128K
E4B         4.5B  [████░░░░░░]  8.0B (total)     128K
26B-A4B     3.8B* [███░░░░░░░]  25.2B (total)    256K  ← MoE: 3.8B active
31B         30.7B [██████████]  30.7B (total)    256K
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
* = active params per forward pass (MoE)
```

---

## Quick Reference Card

| | E2B | E4B | 26B-A4B | 31B |
|---|:---:|:---:|:---:|:---:|
| Params (effective) | 2.3B | 4.5B | 3.8B active | 30.7B |
| Context | 128K | 128K | 256K | 256K |
| Audio | ✅ | ✅ | ❌ | ❌ |
| Speed rank | 1st | 2nd | 3rd | 4th |
| Quality rank | 4th | 3rd | 2nd | 1st |
| VRAM (full) | 8 GB | 12 GB | 32 GB | 48 GB+ |
| VRAM (Q4 GGUF) | ~5 GB | ~8 GB | ~17 GB | ~20 GB |
| On-device | ✅ | ✅ | ❌ | ❌ |
