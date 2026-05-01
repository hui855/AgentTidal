@echo off
echo === AgentTidal Environment Setup ===
echo.

REM Create virtual environment
echo [1/3] Creating virtual environment...
uv venv
if %errorlevel% neq 0 (
    echo ERROR: uv not found. Install from https://docs.astral.sh/uv/
    exit /b 1
)

REM Install base dependencies
echo [2/3] Installing base dependencies...
uv sync
if %errorlevel% neq 0 (
    echo ERROR: uv sync failed
    exit /b 1
)

REM Install training extras (optional)
echo.
echo [3/3] Training dependencies (torch + CUDA)...
echo.
echo NOTE: Training dependencies are large (~10GB).
echo Install them now? (y/n)
set /p INSTALL_TRAIN=
if /i "%INSTALL_TRAIN%"=="y" (
    echo Installing training dependencies...
    uv sync --extras train
    if %errorlevel% neq 0 (
        echo WARNING: Training install had issues. You may need to install manually.
        echo Try: uv pip install torch --index-url https://download.pytorch.org/whl/cu124
        echo Then: uv sync --extras train
    )
)

echo.
echo === Setup complete! ===
echo.
echo To start the proxy:  python -m src.short_term.collector
echo To simulate data:    python scripts/simulate_conversations.py
echo To run nightly:      python -m src.scheduler.nightly --dry-run
echo To install scheduler: scripts\install_task.bat
echo.
pause
