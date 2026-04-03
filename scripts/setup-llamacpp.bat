@echo off
:: setup-llamacpp.bat — thin wrapper for setup_llamacpp.py
cd /d "%~dp0.."
python setup_llamacpp.py %*
