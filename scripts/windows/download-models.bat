@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Download Models
echo  Download Gemma 4 model weights from HuggingFace Hub
echo ============================================================
echo.

REM --- Check Python / huggingface_hub ---
python -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] huggingface_hub is not installed.
    echo         Run install-dep.bat first.
    pause
    exit /b 1
)

REM --- Default download directory ---
set "DEFAULT_DIR=%USERPROFILE%\gemma4-models"
set /p MODELS_DIR="Download directory [default: %DEFAULT_DIR%]: "
if "%MODELS_DIR%"=="" set "MODELS_DIR=%DEFAULT_DIR%"

echo.
echo  Models directory: %MODELS_DIR%
echo.

REM --- Model selection menu ---
echo  Select a model to download:
echo.
echo  [1] E2B  (google/gemma-4-E2B-it)     ~5 GB   On-device, Text+Image+Audio, 128K ctx
echo  [2] E4B  (google/gemma-4-E4B-it)     ~8 GB   On-device, Text+Image+Audio, 128K ctx
echo  [3] 26B-A4B (google/gemma-4-26B-A4B-it) ~50 GB  MoE, Text+Image, 256K ctx
echo  [4] 31B  (google/gemma-4-31B-it)     ~65 GB  Dense, Text+Image, 256K ctx
echo  [5] 26B-A4B GGUF Q4_K_M (recommended) ~17 GB  Quantized, llama.cpp compatible
echo  [6] 26B-A4B GGUF IQ4_XS (smaller)   ~13.4 GB Quantized, low VRAM
echo  [7] 26B-A4B GGUF Q8_0   (quality)   ~26.9 GB Quantized, near full quality
echo  [8] Download ALL GGUF quantizations  (requires 200+ GB)
echo  [9] Custom model ID
echo  [0] Exit
echo.
set /p CHOICE="Enter your choice [1-9, 0 to exit]: "

if "%CHOICE%"=="0" goto :eof
if "%CHOICE%"=="1" goto :download_e2b
if "%CHOICE%"=="2" goto :download_e4b
if "%CHOICE%"=="3" goto :download_26b
if "%CHOICE%"=="4" goto :download_31b
if "%CHOICE%"=="5" goto :download_gguf_q4km
if "%CHOICE%"=="6" goto :download_gguf_iq4xs
if "%CHOICE%"=="7" goto :download_gguf_q8
if "%CHOICE%"=="8" goto :download_gguf_all
if "%CHOICE%"=="9" goto :download_custom
echo [ERROR] Invalid choice.
pause
exit /b 1

:download_e2b
set "MODEL_ID=google/gemma-4-E2B-it"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-E2B-it"
goto :do_download

:download_e4b
set "MODEL_ID=google/gemma-4-E4B-it"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-E4B-it"
goto :do_download

:download_26b
set "MODEL_ID=google/gemma-4-26B-A4B-it"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-26B-A4B-it"
goto :do_download

:download_31b
set "MODEL_ID=google/gemma-4-31B-it"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-31B-it"
goto :do_download

:download_gguf_q4km
set "MODEL_ID=unsloth/gemma-4-26B-A4B-it-GGUF"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF"
set "INCLUDE_FILTER=*Q4_K_M*"
goto :do_gguf_download

:download_gguf_iq4xs
set "MODEL_ID=unsloth/gemma-4-26B-A4B-it-GGUF"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF"
set "INCLUDE_FILTER=*IQ4_XS*"
goto :do_gguf_download

:download_gguf_q8
set "MODEL_ID=unsloth/gemma-4-26B-A4B-it-GGUF"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF"
set "INCLUDE_FILTER=*Q8_0*"
goto :do_gguf_download

:download_gguf_all
set "MODEL_ID=unsloth/gemma-4-26B-A4B-it-GGUF"
set "LOCAL_DIR=%MODELS_DIR%\gemma-4-26B-A4B-it-GGUF"
echo.
echo [INFO] Downloading ALL GGUF quantizations. This requires 200+ GB of disk space.
set /p CONFIRM="Are you sure? [y/N]: "
if /i not "%CONFIRM%"=="y" goto :eof
goto :do_download

:download_custom
set /p MODEL_ID="Enter HuggingFace model ID (e.g. google/gemma-4-E2B-it): "
set "LOCAL_DIR=%MODELS_DIR%\%MODEL_ID:*/=%"
set "LOCAL_DIR=%LOCAL_DIR:/=\%"
goto :do_download

REM --- HuggingFace token check ---
:do_gguf_download
echo.
echo [INFO] Checking HuggingFace login status...
huggingface-cli whoami >nul 2>&1
if errorlevel 1 (
    echo [INFO] You are not logged in to HuggingFace.
    echo        Gemma 4 models require accepting the license agreement.
    echo        Visit: https://huggingface.co/%MODEL_ID%
    set /p HF_TOKEN="Enter your HuggingFace access token (or press Enter to skip): "
    if not "!HF_TOKEN!"=="" (
        huggingface-cli login --token "!HF_TOKEN!"
    )
)
echo.
echo [INFO] Downloading %MODEL_ID% (filter: %INCLUDE_FILTER%) to %LOCAL_DIR%...
echo.
huggingface-cli download "%MODEL_ID%" --include "%INCLUDE_FILTER%" --local-dir "%LOCAL_DIR%"
if errorlevel 1 (
    echo [ERROR] Download failed. Check your internet connection and HuggingFace token.
    pause
    exit /b 1
)
echo.
echo [OK] Download complete: %LOCAL_DIR%
echo.
echo  To use with llama.cpp:
echo    llama-cli -m "%LOCAL_DIR%\*.gguf" --chat-template gemma -p "Hello"
echo.
pause
exit /b 0

:do_download
echo.
echo [INFO] Checking HuggingFace login status...
huggingface-cli whoami >nul 2>&1
if errorlevel 1 (
    echo [INFO] You are not logged in to HuggingFace.
    echo        Gemma 4 models require accepting the license agreement.
    echo        Visit: https://huggingface.co/%MODEL_ID%
    set /p HF_TOKEN="Enter your HuggingFace access token (or press Enter to skip): "
    if not "!HF_TOKEN!"=="" (
        huggingface-cli login --token "!HF_TOKEN!"
    )
)
echo.
echo [INFO] Downloading %MODEL_ID% to %LOCAL_DIR%...
echo        This may take a long time depending on model size and connection speed.
echo.
huggingface-cli download "%MODEL_ID%" --local-dir "%LOCAL_DIR%"
if errorlevel 1 (
    echo [ERROR] Download failed. Check your internet connection and HuggingFace token.
    pause
    exit /b 1
)
echo.
echo ============================================================
echo  Download complete!
echo  Model saved to: %LOCAL_DIR%
echo.
echo  To use this model:
echo    - Set MODEL_ID="%LOCAL_DIR%" in start.bat
echo    - Or run: python -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained('%LOCAL_DIR%')"
echo ============================================================
pause
endlocal
