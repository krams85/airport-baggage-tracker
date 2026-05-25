@echo off
title Baggage Tracker v2
chcp 437 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo ==========================================
echo   Airport Baggage Tracker v2
echo ==========================================
echo.

REM -- Find Python: venv first, then system --
set PYTHON=
if exist "%~dp0venv\Scripts\python.exe"  set PYTHON=%~dp0venv\Scripts\python.exe
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe
if "%PYTHON%"=="" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python not found.
        echo Run install.bat first.
        echo.
        pause & exit /b 1
    )
    set PYTHON=python
    echo WARNING: venv not found - using system Python.
    echo Run install.bat for proper setup.
    echo.
)
echo Python: %PYTHON%

REM -- Torch cache outside Cyrillic path --
set TORCH_HOME=%~dp0torch_cache
echo Torch cache: %TORCH_HOME%

REM -- Qt plugins from venv --
set QT_PLUGIN_PATH=
for %%D in (
    "%~dp0venv\Lib\site-packages\PyQt5\Qt5\plugins"
    "%~dp0venv\Lib\site-packages\PyQt5\Qt\plugins"
    "%~dp0.venv\Lib\site-packages\PyQt5\Qt5\plugins"
) do (
    if exist "%%~D\platforms\qwindows.dll" set QT_PLUGIN_PATH=%%~D
)
if not "%QT_PLUGIN_PATH%"=="" (
    echo Qt plugins: %QT_PLUGIN_PATH%
) else (
    echo Qt plugins: system (venv not found or PyQt5 not installed)
)
echo.

REM -- First-run check --
if not exist "%~dp0.installed" (
    echo First run - checking dependencies...
    "%PYTHON%" "%~dp0setup_models.py" --install
    if errorlevel 1 (
        echo.
        echo Setup failed. Run install.bat and try again.
        pause & exit /b 1
    )
    echo. > "%~dp0.installed"
)

REM -- Launch with watchdog (auto-restart on crash, max 5 times) --
echo Starting app...
echo Web dashboard: http://localhost:8765
echo Config: config.yaml
echo Watchdog: enabled (auto-restart on crash, max 5 times)
echo To disable: start.bat --no-watchdog
echo.
echo ---- LOG ----
echo.

cd /d "%~dp0"
set WATCHDOG=1
echo %* | find "--no-watchdog" >nul && set WATCHDOG=0

set CRASH_COUNT=0
:launch
"%PYTHON%" tracker_app.py %*
set EXIT_CODE=%errorlevel%

if "%WATCHDOG%"=="0" goto :done
if "%EXIT_CODE%"=="0" goto :done

set /a CRASH_COUNT+=1
echo.
echo App exited with code %EXIT_CODE% (crash #%CRASH_COUNT%)
if %CRASH_COUNT% GEQ 5 (
    echo Restart limit reached (5). Check logs above.
    goto :done
)
echo Restarting in 5 seconds... (Ctrl+C to cancel)
timeout /t 5 /nobreak >nul
echo.
echo ---- RESTART #%CRASH_COUNT% ----
echo.
goto :launch

:done
echo.
echo ---- CLOSED ----
pause
