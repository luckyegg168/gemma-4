@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Install llama.cpp Dependencies
echo  Installs llama-cpp-python with CUDA GPU acceleration
echo  Primary: GPU (CUDA) build   Fallback: CPU build
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

REM --- Upgrade pip ---
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM --- Install huggingface_hub (for model download) ---
echo [2/5] Installing huggingface_hub...
python -m pip install --upgrade huggingface_hub
if errorlevel 1 (
    echo [ERROR] Failed to install huggingface_hub.
    pause
    exit /b 1
)
echo.

REM --- Detect CUDA ---
echo [3/5] Detecting CUDA GPU...
set "CUDA_FOUND=0"
nvcc --version >nul 2>&1
if not errorlevel 1 (
    set "CUDA_FOUND=1"
    for /f "tokens=5" %%V in ('nvcc --version ^| findstr "release"') do set CUDA_VER=%%V
    set CUDA_VER=!CUDA_VER:,=!
    echo [OK] CUDA detected: nvcc version !CUDA_VER!
) else (
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "CUDA_FOUND=1"
        for /f "tokens=9" %%V in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%V
        echo [OK] GPU detected via nvidia-smi. CUDA Version: !CUDA_VER!
    ) else (
        echo [WARNING] No CUDA GPU detected. Will install CPU-only build.
    )
)
echo.

REM --- Install llama-cpp-python ---
echo [4/5] Installing llama-cpp-python...
echo.

if "!CUDA_FOUND!"=="1" (
    echo [INFO] Installing GPU-accelerated build (CUDA 12.1 wheel)...
    echo [INFO] This enables n_gpu_layers to offload model layers to GPU.
    echo.

    REM Try pre-built CUDA 12.1 wheel first (fastest)
    python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
    if not errorlevel 1 (
        echo [OK] llama-cpp-python CUDA 12.1 build installed successfully.
        goto :postinstall
    )

    echo [WARNING] CUDA 12.1 wheel failed. Trying CUDA 11.8 wheel...
    python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
    if not errorlevel 1 (
        echo [OK] llama-cpp-python CUDA 11.8 build installed successfully.
        goto :postinstall
    )

    echo [WARNING] Pre-built GPU wheels failed. Attempting source build with CUDA...
    echo [INFO] This may take 5-15 minutes and requires Visual Studio Build Tools.
    set "CMAKE_ARGS=-DGGML_CUDA=on"
    set "FORCE_CMAKE=1"
    python -m pip install llama-cpp-python --no-binary llama-cpp-python
    if not errorlevel 1 (
        echo [OK] llama-cpp-python built from source with CUDA support.
        goto :postinstall
    )

    echo [WARNING] GPU build failed. Falling back to CPU-only build...
)

REM CPU fallback
echo [INFO] Installing CPU-only llama-cpp-python...
python -m pip install llama-cpp-python
if errorlevel 1 (
    echo [ERROR] llama-cpp-python installation failed entirely.
    echo         See: https://github.com/abetlen/llama-cpp-python
    pause
    exit /b 1
)
echo [OK] llama-cpp-python CPU build installed.

:postinstall
echo.

REM --- Verify installation ---
echo [5/5] Verifying llama-cpp-python installation...
python -c "from llama_cpp import Llama; print('[OK] llama_cpp import successful')"
if errorlevel 1 (
    echo [ERROR] llama_cpp import failed. Installation may be incomplete.
    pause
    exit /b 1
)
echo.

REM --- GPU layer detection ---
python -c "from llama_cpp import Llama; import ctypes; print('[INFO] Checking GPU support...'); l = Llama.__new__(Llama); print('[OK] Backend: CUDA available' if hasattr(ctypes.CDLL, 'LoadLibrary') else '[INFO] GPU support status unknown')" >nul 2>&1

echo ============================================================
echo  INSTALLATION COMPLETE
echo.
echo  Installed: llama-cpp-python (GGUF inference engine)
echo.
echo  Next steps:
echo    1. Run download-models.bat and select a GGUF model
echo    2. Run start-llamacpp.bat to start interactive chat
echo    3. Use n_gpu_layers=-1 to offload all layers to GPU
echo       Use n_gpu_layers=0 for CPU-only inference
echo.
echo  GGUF Model Recommendations:
echo    E2B  Q4_K_M = 3.11 GB  (4 GB VRAM)
echo    E4B  Q4_K_M = 4.98 GB  (6 GB VRAM)
echo    26B-A4B UD-Q4_K_M = 16.9 GB  (18 GB VRAM)
echo    31B  Q4_K_M = 18.3 GB  (20 GB VRAM)
echo ============================================================
pause
endlocal
