@echo off
title Baggage Tracker - Install
chcp 437 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo =====================================================
echo   Airport Baggage Tracker v2 - First-time Install
echo =====================================================
echo.
echo This script will:
echo   1. Create virtual environment (venv)
echo   2. Install PyTorch with CUDA (auto-detected)
echo   3. Install all dependencies
echo   4. Install onnxruntime (GPU or CPU)
echo   5. Verify ReID model
echo.
echo After install, use start.bat to launch the app.
echo.
pause

REM ── 1. Find Python ────────────────────────────────────────────────────────────
echo [1/6] Looking for Python...

set PYTHON=
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON=%~dp0venv\Scripts\python.exe
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python not found in PATH.
        echo Install Python 3.10-3.13 from https://python.org
        echo Make sure to check "Add Python to PATH"
        echo.
        pause & exit /b 1
    )
    set PYTHON=python
)

for /f "usebackq tokens=*" %%v in (`"%PYTHON%" --version 2^>^&1`) do set PY_VER=%%v
echo   Found: %PY_VER%
echo.

REM ── 2. Virtual environment ────────────────────────────────────────────────────
echo [2/6] Virtual environment...

if exist "%~dp0venv\Scripts\python.exe" (
    echo   venv already exists - skipping.
) else (
    echo   Creating venv...
    "%PYTHON%" -m venv "%~dp0venv"
    if errorlevel 1 (
        echo   ERROR creating venv.
        pause & exit /b 1
    )
    echo   venv created.
)

set PYTHON=%~dp0venv\Scripts\python.exe
set PIP=%~dp0venv\Scripts\pip.exe
echo   Python: %PYTHON%
echo.

REM ── 3. PyTorch with correct CUDA ──────────────────────────────────────────────
echo [3/6] Installing PyTorch...

REM Run detection helper - writes KEY=VALUE lines to temp file
"%PYTHON%" "%~dp0_detect_env.py" > "%TEMP%\bt_env.txt" 2>nul

REM Parse KEY=VALUE pairs from temp file
set PY_MINOR=10
set TORCH_CUDA=0
set TORCH_VER=none
set DRIVER_MAJ=0
set HAS_GPU=0
for /f "usebackq tokens=1,2 delims==" %%a in ("%TEMP%\bt_env.txt") do set %%a=%%b

echo   Python minor: 3.%PY_MINOR%
echo   Torch: %TORCH_VER%  CUDA: %TORCH_CUDA%
echo   GPU: %HAS_GPU%  Driver: %DRIVER_MAJ%.x

REM Skip if already good
if "%TORCH_CUDA%"=="1" (
    echo   PyTorch %TORCH_VER% with CUDA already installed - skipping.
    goto :torch_done
)

REM No GPU -> CPU torch
if "%HAS_GPU%"=="0" (
    echo   No NVIDIA GPU - installing CPU-only PyTorch.
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    goto :torch_verify
)

REM Python 3.13+ needs cu124 (first wheel set that supports 3.13)
if %PY_MINOR% GEQ 13 (
    echo   Python 3.13+ detected - installing PyTorch cu124...
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
    goto :torch_verify
)

REM Python 3.10-3.12: pick by driver version
REM  Driver >= 527 -> CUDA 12.x -> cu121
REM  Driver >= 452 -> CUDA 11.x -> cu118
if %DRIVER_MAJ% GEQ 527 (
    echo   Installing PyTorch cu121...
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    goto :torch_verify
)
if %DRIVER_MAJ% GEQ 452 (
    echo   Installing PyTorch cu118...
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
    goto :torch_verify
)

echo   Driver version unclear - trying cu121...
"%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

:torch_verify
if errorlevel 1 (
    echo.
    echo   ERROR installing PyTorch!
    echo   Run manually from this folder:
    echo     venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo.
    pause & exit /b 1
)

:torch_done
"%PYTHON%" "%~dp0_detect_env.py" > "%TEMP%\bt_env2.txt" 2>nul
for /f "usebackq tokens=1,2 delims==" %%a in ("%TEMP%\bt_env2.txt") do set %%a=%%b
echo   Result: torch=%TORCH_VER%  CUDA=%TORCH_CUDA%
echo.

REM ── 4. Requirements ───────────────────────────────────────────────────────────
echo [4/6] Installing requirements.txt...
"%PIP%" install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check
if errorlevel 1 (
    echo   ERROR installing dependencies!
    pause & exit /b 1
)
echo   OK
echo.

REM ── 5. onnxruntime ────────────────────────────────────────────────────────────
echo [5/6] Selecting and installing onnxruntime...

set ORT_PKG=onnxruntime
if "%HAS_GPU%"=="1" (
    echo   NVIDIA GPU detected - installing onnxruntime-gpu
    set ORT_PKG=onnxruntime-gpu
) else (
    echo   No NVIDIA GPU - installing onnxruntime (CPU)
)

"%PIP%" uninstall onnxruntime-gpu onnxruntime -y --quiet >nul 2>&1
"%PIP%" install %ORT_PKG% --quiet --disable-pip-version-check
if errorlevel 1 (
    echo   ERROR installing %ORT_PKG%.
    echo   Try: venv\Scripts\pip install %ORT_PKG%
    pause & exit /b 1
)

"%PYTHON%" -c "import onnxruntime as o; print('  onnxruntime', o.__version__, '- OK')"
echo.

REM ── Qt plugins path ───────────────────────────────────────────────────────────
for %%D in (
    "%~dp0venv\Lib\site-packages\PyQt5\Qt5\plugins"
    "%~dp0venv\Lib\site-packages\PyQt5\Qt\plugins"
) do if exist "%%~D\platforms\qwindows.dll" set QT_PLUGIN_PATH=%%~D

REM ── 6. Models ─────────────────────────────────────────────────────────────────
echo [6/6] Checking models...
set TORCH_HOME=%~dp0torch_cache
"%PYTHON%" "%~dp0setup_models.py" --install
if errorlevel 1 (
    echo   ERROR loading models!
    pause & exit /b 1
)

echo.
echo =====================================================
echo   Install complete!
echo   Run the app:  start.bat
echo   Diagnostics:  check.bat
echo   Settings:     config.yaml
echo =====================================================
echo.
pause
