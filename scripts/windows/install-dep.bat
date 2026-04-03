@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Gemma 4 - Install Dependencies
echo  Installs all Python packages required to run Gemma 4 models
echo ============================================================
echo.

REM --- Check Python availability ---
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

REM --- Upgrade pip first ---
echo [1/6] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Could not upgrade pip. Continuing anyway.
)
echo.

REM --- Core dependencies ---
echo [2/6] Installing core dependencies (transformers, torch, accelerate)...
python -m pip install --upgrade transformers torch accelerate
if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies.
    pause
    exit /b 1
)
echo.

REM --- HuggingFace Hub CLI ---
echo [3/6] Installing HuggingFace Hub (for model download CLI)...
python -m pip install --upgrade huggingface_hub
if errorlevel 1 (
    echo [ERROR] Failed to install huggingface_hub.
    pause
    exit /b 1
)
echo.

REM --- Image support ---
echo [4/6] Installing image processing support (Pillow)...
python -m pip install --upgrade Pillow
if errorlevel 1 (
    echo [WARNING] Pillow installation failed. Image inputs will not work.
)
echo.

REM --- Audio support ---
echo [5/6] Installing audio processing support (librosa, soundfile)...
python -m pip install --upgrade librosa soundfile
if errorlevel 1 (
    echo [WARNING] Audio library installation failed. Audio inputs will not work.
    echo           This only affects E2B and E4B models.
)
echo.

REM --- Optional: llama-cpp-python for GGUF support ---
echo [6/6] Checking for llama-cpp-python (GGUF / llama.cpp backend)...
python -c "import llama_cpp" >nul 2>&1
if errorlevel 1 (
    echo       llama-cpp-python not found.
    set /p INSTALL_LLAMA="    Install llama-cpp-python for GGUF support? [y/N]: "
    if /i "!INSTALL_LLAMA!"=="y" (
        echo       Installing llama-cpp-python (CPU only)
        python -m pip install llama-cpp-python
        if errorlevel 1 (
            echo [WARNING] llama-cpp-python install failed.
            echo           For GPU support, see: https://github.com/abetlen/llama-cpp-python
        ) else (
            echo [OK] llama-cpp-python installed.
        )
    ) else (
        echo       Skipping llama-cpp-python installation.
    )
) else (
    echo [OK] llama-cpp-python is already installed.
)
echo.

REM --- Summary ---
echo ============================================================
echo  Installation complete!
echo.
echo  Installed packages:
echo    - transformers   (HuggingFace model library)
echo    - torch          (PyTorch deep learning)
echo    - accelerate     (multi-GPU / mixed precision)
echo    - huggingface_hub (model download CLI)
echo    - Pillow         (image processing)
echo    - librosa        (audio processing)
echo    - soundfile      (audio file I/O)
echo.
echo  Next steps:
echo    1. Run download-models.bat to download a Gemma 4 model
echo    2. Run start.bat to start interactive chat
echo    3. Run manager.bat for a full management menu
echo ============================================================
pause
endlocal
