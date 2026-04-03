@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Install vllm Dependencies
echo  High-throughput GPU inference with OpenAI-compatible API
echo  GPU REQUIRED (CUDA 11.8+)
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYTHON_VER=%%V
echo [OK] Found Python %PYTHON_VER%
echo.

REM --- Check CUDA GPU ---
echo [INFO] Checking CUDA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] nvidia-smi not found. No NVIDIA GPU detected.
    echo           vllm requires an NVIDIA GPU with CUDA 11.8+.
    echo           CPU inference is NOT supported by vllm.
    echo.
    set /p CONTINUE="Continue installation anyway? [y/N]: "
    if /i not "!CONTINUE!"=="y" (
        echo Installation cancelled.
        pause
        exit /b 1
    )
) else (
    echo [OK] NVIDIA GPU detected.
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)
echo.

REM --- Upgrade pip ---
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM --- Install PyTorch with CUDA (required by vllm) ---
echo [2/5] Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARNING] CUDA 12.1 PyTorch failed. Trying CUDA 11.8...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed. Check your CUDA version.
        pause
        exit /b 1
    )
)
echo.

REM --- Install vllm ---
echo [3/5] Installing vllm...
python -m pip install vllm
if errorlevel 1 (
    echo [ERROR] vllm installation failed.
    echo         Possible issues:
    echo           - CUDA version mismatch (requires CUDA 11.8+)
    echo           - Missing Visual Studio Build Tools
    echo           - Python version incompatibility (requires 3.9-3.12)
    echo         See: https://docs.vllm.ai/en/latest/getting_started/installation.html
    pause
    exit /b 1
)
echo.

REM --- Install OpenAI client SDK ---
echo [4/5] Installing OpenAI client SDK (for vllm API access)...
python -m pip install openai
if errorlevel 1 (
    echo [WARNING] OpenAI SDK installation failed. You can still use curl to query vllm.
)
echo.

REM --- Install HuggingFace Hub ---
echo [5/5] Installing HuggingFace Hub (for model downloads)...
python -m pip install --upgrade huggingface_hub transformers accelerate
if errorlevel 1 (
    echo [ERROR] HuggingFace Hub installation failed.
    pause
    exit /b 1
)
echo.

REM --- Verify vllm ---
echo [INFO] Verifying vllm installation...
python -c "import vllm; print(f'[OK] vllm version: {vllm.__version__}')"
if errorlevel 1 (
    echo [ERROR] vllm import failed. See error above.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  INSTALLATION COMPLETE
echo.
echo  Installed: vllm + PyTorch CUDA + OpenAI SDK
echo.
echo  Next steps:
echo    1. Run start-vllm.bat to start the OpenAI-compatible server
echo    2. Then query it from Python, curl, or any OpenAI SDK client
echo.
echo  Supported Gemma 4 models (HuggingFace full precision):
echo    google/gemma-4-E2B-it     (5B,  needs ~8 GB VRAM)
echo    google/gemma-4-E4B-it     (8B,  needs ~12 GB VRAM)
echo    google/gemma-4-26B-A4B-it (27B, needs ~32 GB VRAM)
echo    google/gemma-4-31B-it     (31B, needs ~48 GB VRAM)
echo.
echo  NOTE: vllm does NOT support GGUF files.
echo        Use start-llamacpp.bat for GGUF inference instead.
echo ============================================================
pause
endlocal
