@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Interactive Chat via llama.cpp (GGUF)
echo  GPU-accelerated GGUF inference with n_gpu_layers offloading
echo ============================================================
echo.

REM --- Check llama_cpp ---
python -c "from llama_cpp import Llama" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] llama-cpp-python is not installed.
    echo         Run install-dep-llamacpp.bat first.
    pause
    exit /b 1
)

REM --- Default model directory ---
set "DEFAULT_DIR=%USERPROFILE%\gemma4-models"
set /p MODELS_DIR="Models directory [default: %DEFAULT_DIR%]: "
if "%MODELS_DIR%"=="" set "MODELS_DIR=%DEFAULT_DIR%"
echo.

REM --- Model selection ---
echo  Select a GGUF model:
echo.
echo  --- E2B (2.3B eff) --- Needs 3-5 GB VRAM ---
echo  [1]  E2B  Q4_K_M   3.11 GB  (unsloth/gemma-4-E2B-it-GGUF)  Recommended
echo  [2]  E2B  Q8_0     5.05 GB  High quality
echo  [3]  E2B  IQ4_XS   2.98 GB  Small option
echo.
echo  --- E4B (4.5B eff) --- Needs 5-9 GB VRAM ---
echo  [4]  E4B  Q4_K_M   4.98 GB  (unsloth/gemma-4-E4B-it-GGUF)  Recommended
echo  [5]  E4B  Q8_0     8.19 GB  High quality
echo.
echo  --- 26B-A4B (MoE) --- Needs 17-28 GB VRAM ---
echo  [6]  26B-A4B  MXFP4_MOE   16.7 GB  MoE-optimized quant  Recommended
echo  [7]  26B-A4B  UD-Q4_K_M  16.9 GB  Standard recommended
echo  [8]  26B-A4B  Q8_0       26.9 GB  Near full quality
echo.
echo  --- 31B (Dense) --- Needs 18-34 GB VRAM ---
echo  [9]  31B  Q4_K_M   18.3 GB  (unsloth/gemma-4-31B-it-GGUF)  Recommended
echo  [10] 31B  IQ4_XS   16.4 GB  Smallest 31B option
echo  [11] 31B  Q8_0     32.6 GB  Near full quality
echo.
echo  [12] Custom GGUF file path
echo  [0]  Exit
echo.
set /p MODEL_CHOICE="Enter choice: "

if "%MODEL_CHOICE%"=="0" exit /b 0

REM E2B
if "%MODEL_CHOICE%"=="1"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf"       & set "GPU_LAYERS=35"
if "%MODEL_CHOICE%"=="2"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q8_0.gguf"         & set "GPU_LAYERS=35"
if "%MODEL_CHOICE%"=="3"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-IQ4_XS.gguf"       & set "GPU_LAYERS=35"
REM E4B
if "%MODEL_CHOICE%"=="4"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"       & set "GPU_LAYERS=42"
if "%MODEL_CHOICE%"=="5"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q8_0.gguf"         & set "GPU_LAYERS=42"
REM 26B-A4B
if "%MODEL_CHOICE%"=="6"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-MXFP4_MOE.gguf"  & set "GPU_LAYERS=30"
if "%MODEL_CHOICE%"=="7"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"  & set "GPU_LAYERS=30"
if "%MODEL_CHOICE%"=="8"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-Q8_0.gguf"       & set "GPU_LAYERS=30"
REM 31B
if "%MODEL_CHOICE%"=="9"  set "MODEL_FILE=%MODELS_DIR%\gemma-4-31B-it-GGUF\gemma-4-31B-it-Q4_K_M.gguf"       & set "GPU_LAYERS=60"
if "%MODEL_CHOICE%"=="10" set "MODEL_FILE=%MODELS_DIR%\gemma-4-31B-it-GGUF\gemma-4-31B-it-IQ4_XS.gguf"       & set "GPU_LAYERS=60"
if "%MODEL_CHOICE%"=="11" set "MODEL_FILE=%MODELS_DIR%\gemma-4-31B-it-GGUF\gemma-4-31B-it-Q8_0.gguf"         & set "GPU_LAYERS=60"
REM Custom
if "%MODEL_CHOICE%"=="12" (
    set /p MODEL_FILE="Enter full path to .gguf file: "
    set /p GPU_LAYERS="GPU layers to offload (-1=all, 0=CPU only): "
)

if not defined MODEL_FILE (
    echo [ERROR] Invalid choice.
    pause
    exit /b 1
)

REM --- Check model file exists ---
if not exist "%MODEL_FILE%" (
    echo [ERROR] Model file not found: %MODEL_FILE%
    echo         Run download-models.bat to download GGUF files.
    pause
    exit /b 1
)

REM --- GPU layer configuration ---
echo.
set /p GPU_LAYERS_OVERRIDE="GPU layers to offload [default: %GPU_LAYERS%, 0=CPU only, -1=all]: "
if not "%GPU_LAYERS_OVERRIDE%"=="" set "GPU_LAYERS=%GPU_LAYERS_OVERRIDE%"

REM --- Context length ---
set "CTX_LEN=4096"
set /p CTX_LEN="Context length [default: 4096, max per model: 128K or 256K]: "
if "%CTX_LEN%"=="" set "CTX_LEN=4096"

REM --- System prompt ---
set "SYSTEM_PROMPT=You are a helpful assistant."
set /p SYSTEM_PROMPT="System prompt [default: You are a helpful assistant.]: "
if "%SYSTEM_PROMPT%"=="" set "SYSTEM_PROMPT=You are a helpful assistant."

echo.
echo [INFO] Starting llama.cpp interactive chat...
echo [INFO] Model:      %MODEL_FILE%
echo [INFO] GPU layers: %GPU_LAYERS%
echo [INFO] Context:    %CTX_LEN% tokens
echo.

REM --- Create and run Python chat script ---
set "TEMP_SCRIPT=%TEMP%\gemma4_llamacpp_%RANDOM%.py"
(
echo # Gemma 4 llama.cpp interactive chat
echo import sys
echo from llama_cpp import Llama
echo.
echo MODEL_PATH = r"%MODEL_FILE%"
echo N_GPU_LAYERS = %GPU_LAYERS%
echo N_CTX = %CTX_LEN%
echo SYSTEM_PROMPT = r"%SYSTEM_PROMPT%"
echo.
echo print(f"Loading: {MODEL_PATH}")
echo print(f"GPU layers: {N_GPU_LAYERS}  Context: {N_CTX}")
echo print()
echo.
echo llm = Llama(
echo     model_path=MODEL_PATH,
echo     n_gpu_layers=N_GPU_LAYERS,
echo     n_ctx=N_CTX,
echo     chat_format="gemma",
echo     verbose=False,
echo )
echo.
echo messages = [{"role": "system", "content": SYSTEM_PROMPT}]
echo print("Gemma 4 llama.cpp Chat — type 'quit' to exit, 'reset' to clear history")
echo print("=" * 60)
echo.
echo while True:
echo     try:
echo         user_input = input("You: ").strip()
echo     except (EOFError, KeyboardInterrupt):
echo         print("\nExiting.")
echo         break
echo     if not user_input:
echo         continue
echo     if user_input.lower() in ("quit", "exit", "q"):
echo         print("Goodbye!")
echo         break
echo     if user_input.lower() == "reset":
echo         messages = [{"role": "system", "content": SYSTEM_PROMPT}]
echo         print("[INFO] Conversation history cleared.")
echo         continue
echo.
echo     messages.append({"role": "user", "content": user_input})
echo.
echo     try:
echo         response = llm.create_chat_completion(
echo             messages=messages,
echo             temperature=1.0,
echo             top_p=0.95,
echo             top_k=64,
echo             max_tokens=1024,
echo             stream=True,
echo         )
echo         print("Gemma: ", end="", flush=True)
echo         full_response = ""
echo         for chunk in response:
echo             delta = chunk["choices"][0]["delta"].get("content", "")
echo             print(delta, end="", flush=True)
echo             full_response += delta
echo         print()
echo         messages.append({"role": "assistant", "content": full_response})
echo     except Exception as e:
echo         print(f"\n[ERROR] {e}")
echo         messages.pop()
) > "%TEMP_SCRIPT%"

python "%TEMP_SCRIPT%"
del "%TEMP_SCRIPT%" >nul 2>&1

echo.
echo [INFO] Chat session ended.
pause
endlocal
