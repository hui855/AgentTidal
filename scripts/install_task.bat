@echo off
echo === Install AgentTidal Windows Scheduled Task ===
echo.

REM Get absolute path to project root
set PROJECT_DIR=%~dp0..
set PYTHON_PATH=python

REM Convert to Windows path
for %%I in ("%PROJECT_DIR%") do set PROJECT_DIR=%%~fI

echo Project: %PROJECT_DIR%
echo.
echo This will create a scheduled task running daily at 2:00 AM.
echo.

schtasks /create /tn "AgentTidal_Nightly" /tr "cmd /c 'cd /d %PROJECT_DIR% && %PYTHON_PATH% -m src.scheduler.nightly'" /sc daily /st 02:00 /f

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS: Task "AgentTidal_Nightly" created.
    echo It will run daily at 2:00 AM.
) else (
    echo.
    echo ERROR: Failed to create task. Try running as Administrator.
)

pause
