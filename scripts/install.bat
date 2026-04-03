@echo off
:: install.bat — thin wrapper for install.py
cd /d "%~dp0.."
python install.py %*
