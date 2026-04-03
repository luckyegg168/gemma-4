@echo off
setlocal EnableDelayedExpansion

:MAIN_MENU
cls
echo ============================================================
echo  Gemma 4 - Model Manager
echo  Complete management interface for Gemma 4 models
echo ============================================================
echo.
echo  [1] Install Dependencies
echo  [2] Download Models
echo  [3] Start Interactive Chat
echo  [4] Run Benchmark / Quick Test
echo  [5] Show Model Information
echo  [6] Show System Information
echo  [7] Set HuggingFace Token
echo  [8] List Downloaded Models
echo  [9] Exit
echo.
set /p MAIN_CHOICE="Enter choice [1-9]: "

if "%MAIN_CHOICE%"=="1" goto :RUN_INSTALL
if "%MAIN_CHOICE%"=="2" goto :RUN_DOWNLOAD
if "%MAIN_CHOICE%"=="3" goto :RUN_CHAT
if "%MAIN_CHOICE%"=="4" goto :RUN_BENCHMARK
if "%MAIN_CHOICE%"=="5" goto :SHOW_MODEL_INFO
if "%MAIN_CHOICE%"=="6" goto :SHOW_SYSTEM_INFO
if "%MAIN_CHOICE%"=="7" goto :SET_TOKEN
if "%MAIN_CHOICE%"=="8" goto :LIST_MODELS
if "%MAIN_CHOICE%"=="9" exit /b 0
echo [ERROR] Invalid choice. Press any key to try again.
pause >nul
goto :MAIN_MENU

REM ============================================================
:RUN_INSTALL
cls
echo [INFO] Launching dependency installer...
call "%~dp0install-dep.bat"
goto :MAIN_MENU

REM ============================================================
:RUN_DOWNLOAD
cls
echo [INFO] Launching model downloader...
call "%~dp0download-models.bat"
goto :MAIN_MENU

REM ============================================================
:RUN_CHAT
cls
echo [INFO] Launching interactive chat...
call "%~dp0start.bat"
goto :MAIN_MENU

REM ============================================================
:RUN_BENCHMARK
cls
echo ============================================================
echo  Gemma 4 - Quick Benchmark / Smoke Test
echo ============================================================
echo.
echo  Select model:
echo  [1] E2B   - google/gemma-4-E2B-it
echo  [2] E4B   - google/gemma-4-E4B-it
echo  [3] 26B-A4B - google/gemma-4-26B-A4B-it
echo  [4] 31B   - google/gemma-4-31B-it
echo  [5] Custom
echo  [0] Back
echo.
set /p BM_CHOICE="Enter choice: "
if "%BM_CHOICE%"=="0" goto :MAIN_MENU
if "%BM_CHOICE%"=="1" set "BM_MODEL=google/gemma-4-E2B-it"
if "%BM_CHOICE%"=="2" set "BM_MODEL=google/gemma-4-E4B-it"
if "%BM_CHOICE%"=="3" set "BM_MODEL=google/gemma-4-26B-A4B-it"
if "%BM_CHOICE%"=="4" set "BM_MODEL=google/gemma-4-31B-it"
if "%BM_CHOICE%"=="5" set /p BM_MODEL="Enter model ID: "

if not defined BM_MODEL goto :MAIN_MENU

echo.
echo [INFO] Running benchmark on %BM_MODEL%...
echo [INFO] Prompts: Hello world, 2+2=?, capital of France
echo.

set "TEMP_BM=%TEMP%\gemma4_bench_%RANDOM%.py"
(
echo import time
echo from transformers import AutoProcessor, AutoModelForCausalLM
echo import torch
echo.
echo MODEL_ID = r"%BM_MODEL%"
echo PROMPTS = [
echo     "What is 2 + 2?",
echo     "What is the capital of France?",
echo     "Write a one-line Python function to reverse a string.",
echo ]
echo.
echo print(f"Loading {MODEL_ID}...")
echo t0 = time.time()
echo processor = AutoProcessor.from_pretrained(MODEL_ID)
echo model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
echo load_time = time.time() - t0
echo print(f"Load time: {load_time:.1f}s  |  Device: {next(model.parameters()).device}")
echo print("-" * 60)
echo.
echo total_tokens = 0
echo total_time = 0.0
echo for i, prompt in enumerate(PROMPTS, 1):
echo     messages = [
echo         {"role": "system", "content": "You are a helpful assistant."},
echo         {"role": "user", "content": prompt},
echo     ]
echo     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
echo     inputs = processor(text=text, return_tensors="pt").to(model.device)
echo     input_len = inputs["input_ids"].shape[-1]
echo     t_start = time.time()
echo     with torch.inference_mode():
echo         outputs = model.generate(**inputs, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64, do_sample=True)
echo     elapsed = time.time() - t_start
echo     new_tokens = outputs.shape[-1] - input_len
echo     total_tokens += new_tokens
echo     total_time += elapsed
echo     response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
echo     print(f"[{i}] Q: {prompt}")
echo     print(f"     A: {response[:100]}{'...' if len(response)>100 else ''}")
echo     print(f"     Tokens: {new_tokens}  Time: {elapsed:.2f}s  Speed: {new_tokens/elapsed:.1f} tok/s")
echo     print()
echo.
echo print(f"Average speed: {total_tokens/total_time:.1f} tokens/second")
) > "%TEMP_BM%"

python "%TEMP_BM%"
del "%TEMP_BM%" >nul 2>&1
echo.
pause
goto :MAIN_MENU

REM ============================================================
:SHOW_MODEL_INFO
cls
echo ============================================================
echo  Gemma 4 - Model Comparison
echo ============================================================
echo.
echo  Model              HF ID                           Params   Context  Audio  VRAM(approx)
echo  ---------------    -----------------------------    -------  -------  -----  -----------
echo  E2B (Dense+PLE)    google/gemma-4-E2B-it           2.3B eff 128K     Yes    8 GB
echo  E4B (Dense+PLE)    google/gemma-4-E4B-it           4.5B eff 128K     Yes    12 GB
echo  26B-A4B (MoE)      google/gemma-4-26B-A4B-it       3.8B act 256K     No     32 GB
echo  31B (Dense)        google/gemma-4-31B-it           30.7B    256K     No     48 GB+
echo.
echo  GGUF (26B-A4B)     unsloth/gemma-4-26B-A4B-it-GGUF
echo    IQ4_XS            13.4 GB   - Minimum quality, lowest VRAM
echo    Q4_K_M            16.9 GB   - Recommended balance
echo    Q8_0              26.9 GB   - Near full quality
echo    BF16              50.5 GB   - Full precision
echo.
echo  Architecture Key:
echo    PLE = Per-Layer Embeddings (on-device efficiency)
echo    MoE = Mixture of Experts (128 experts, 8 active per token)
echo    eff = effective compute parameters
echo    act = active parameters per forward pass
echo.
echo  Best for:
echo    Voice/Audio input    E2B or E4B (only models with audio encoder)
echo    Coding/Reasoning     31B ^> 26B-A4B ^> E4B
echo    On-device / Mobile   E2B
echo    Fast large model     26B-A4B (near 4B speed)
echo    Highest quality      31B
echo.
echo  Recommended sampling:  temperature=1.0, top_p=0.95, top_k=64
echo  Document/OCR tasks:    Use image token budget 1120
echo  Video/fast tasks:      Use image token budget 70
echo.
pause
goto :MAIN_MENU

REM ============================================================
:SHOW_SYSTEM_INFO
cls
echo ============================================================
echo  System Information
echo ============================================================
echo.

REM Python info
python --version 2>&1
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); cuda_count=torch.cuda.device_count(); print('CUDA devices:', cuda_count); [print(f'  GPU {i}:', torch.cuda.get_device_name(i), f'({torch.cuda.get_device_properties(i).total_memory // 1073741824} GB)') for i in range(cuda_count)]" 2>nul
if errorlevel 1 (
    echo [WARNING] PyTorch not installed. Run install-dep.bat.
)
echo.

REM transformers version
python -c "import transformers; print('Transformers:', transformers.__version__)" 2>nul
if errorlevel 1 echo [WARNING] transformers not installed.

REM huggingface_hub version
python -c "import huggingface_hub; print('HuggingFace Hub:', huggingface_hub.__version__)" 2>nul
if errorlevel 1 echo [WARNING] huggingface_hub not installed.

REM HF login status
echo.
echo  HuggingFace login:
huggingface-cli whoami 2>nul
if errorlevel 1 echo [NOT LOGGED IN] Run option [7] to set your token.

echo.
pause
goto :MAIN_MENU

REM ============================================================
:SET_TOKEN
cls
echo ============================================================
echo  Set HuggingFace Access Token
echo ============================================================
echo.
echo  Gemma 4 models require accepting the license agreement on HuggingFace.
echo  1. Visit: https://huggingface.co/google/gemma-4-E2B-it
echo  2. Accept the license agreement
echo  3. Generate an access token at: https://huggingface.co/settings/tokens
echo.
set /p HF_TOKEN_INPUT="Paste your HuggingFace token: "
if "%HF_TOKEN_INPUT%"=="" (
    echo [ERROR] No token entered.
    pause
    goto :MAIN_MENU
)
huggingface-cli login --token "%HF_TOKEN_INPUT%"
if errorlevel 1 (
    echo [ERROR] Login failed. Check your token.
) else (
    echo [OK] Login successful.
)
pause
goto :MAIN_MENU

REM ============================================================
:LIST_MODELS
cls
echo ============================================================
echo  Downloaded Models
echo ============================================================
echo.
set "DEFAULT_MODELS_DIR=%USERPROFILE%\gemma4-models"
set /p CHECK_DIR="Models directory to scan [default: %DEFAULT_MODELS_DIR%]: "
if "%CHECK_DIR%"=="" set "CHECK_DIR=%DEFAULT_MODELS_DIR%"

echo.
if not exist "%CHECK_DIR%" (
    echo [INFO] Directory does not exist: %CHECK_DIR%
    echo        No models downloaded yet.
    echo        Run option [2] to download a model.
) else (
    echo  Contents of %CHECK_DIR%:
    echo.
    for /d %%D in ("%CHECK_DIR%\*") do (
        set "FOLDER=%%~nxD"
        set "FOLDER_FULL=%%D"
        REM Calculate approximate size
        set SIZE=0
        for /r "%%D" %%F in (*.safetensors *.gguf *.bin) do (
            set /a SIZE+=%%~zF / 1073741824
        )
        echo    %%~nxD
    )
    echo.
    echo  Tip: Use the full path as MODEL_ID in start.bat for offline inference.
)
echo.
pause
goto :MAIN_MENU
