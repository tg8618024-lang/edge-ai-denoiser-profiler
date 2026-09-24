@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   NVIDIA-STYLE REAL-TIME EDGE AI AUDIO DENOISER ^& HARDWARE PROFILER
echo ===============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python was not found on PATH. Please install Python 3.10+ to continue.
        exit /b 1
    )
    echo [INFO] Creating virtual environment at .venv ...
    python -m venv .venv
    set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
    echo [INFO] Installing project requirements ...
    "%PYTHON%" -m pip install --upgrade pip
    "%PYTHON%" -m pip install -r requirements.txt
)

echo [INFO] Starting interactive studio dashboard on http://127.0.0.1:8000 ...
"%PYTHON%" run_dashboard.py %*

endlocal
