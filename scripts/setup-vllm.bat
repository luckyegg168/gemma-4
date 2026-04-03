@echo off
:: setup-vllm.bat — thin wrapper for setup_vllm.py (Linux/WSL2 only)
cd /d "%~dp0.."
echo [NOTE] vllm is not supported on native Windows.
echo        Run this inside WSL2 (Ubuntu) or use setup-vllm.sh from WSL.
echo.
python setup_vllm.py %*
