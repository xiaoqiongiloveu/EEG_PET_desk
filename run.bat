@echo off
setlocal
set "PY=D:\rest-eeg-mvp-v0.3.00\rest-eeg-mvp-v0.3.0\runtime\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0main.py"
if errorlevel 1 pause
