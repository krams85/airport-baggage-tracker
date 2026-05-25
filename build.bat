@echo off
title Baggage Tracker - Build
chcp 437 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo =====================================================
echo   Airport Baggage Tracker v2 - PyInstaller Build
echo =====================================================
echo.
echo WARNING: build takes 5-15 min, dist will be ~3-5 GB
echo.

REM -- Find Python --
set PYTHON=
if exist "%~dp0venv\Scripts\python.exe"  set PYTHON=%~dp0venv\Scripts\python.exe
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe
if "%PYTHON%"=="" (
    where python >nul 2>&1
    if errorlevel 1 ( echo ERROR: Python not found. & pause & exit /b 1 )
    set PYTHON=python
)
echo Python: %PYTHON%
echo.

REM -- Install PyInstaller if missing --
%PYTHON% -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON% -m pip install pyinstaller --quiet --disable-pip-version-check
    if errorlevel 1 ( echo ERROR: cannot install PyInstaller & pause & exit /b 1 )
)
for /f "tokens=*" %%v in ('%PYTHON% -c "import PyInstaller; print(PyInstaller.__version__)"') do set PYI_VER=%%v
echo PyInstaller: %PYI_VER%
echo.

REM -- Clean previous build --
if exist "%~dp0dist\tracker_app" (
    echo Removing previous dist\tracker_app ...
    rmdir /s /q "%~dp0dist\tracker_app"
)
if exist "%~dp0build" (
    rmdir /s /q "%~dp0build"
)

REM -- Build --
echo Running PyInstaller...
echo.
cd /d "%~dp0"
%PYTHON% -m PyInstaller tracker_app.spec --noconfirm
if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check output above.
    pause & exit /b 1
)

REM -- Copy external files next to exe --
echo.
echo Copying configs and models to dist\tracker_app\ ...
set DIST=%~dp0dist\tracker_app

copy /y "%~dp0config.yaml"               "%DIST%\" >nul
if exist "%~dp0osnet_x1_0_256x128.onnx"  copy /y "%~dp0osnet_x1_0_256x128.onnx"  "%DIST%\" >nul
if exist "%~dp0yolo11n.pt"               copy /y "%~dp0yolo11n.pt"                "%DIST%\" >nul
if exist "%~dp0yolo11s.pt"               copy /y "%~dp0yolo11s.pt"                "%DIST%\" >nul
if exist "%~dp0yolov8n.pt"               copy /y "%~dp0yolov8n.pt"                "%DIST%\" >nul
if exist "%~dp0botsort.yaml"             copy /y "%~dp0botsort.yaml"              "%DIST%\" >nul
if exist "%~dp0bytetrack.yaml"           copy /y "%~dp0bytetrack.yaml"            "%DIST%\" >nul

REM -- Generate start.bat inside dist --
%PYTHON% -c "
import textwrap, pathlib
bat = textwrap.dedent('''
    @echo off
    title Baggage Tracker v2
    cd /d \"%%~dp0\"
    set TORCH_HOME=%%~dp0torch_cache
    set CRASH_COUNT=0
    :launch
    tracker_app.exe %%*
    set EXIT_CODE=%%errorlevel%%
    if \"%%EXIT_CODE%%\"==\"0\" goto :done
    set /a CRASH_COUNT+=1
    if %%CRASH_COUNT%% GEQ 5 (echo Restart limit reached. & goto :done)
    echo Crash #%%CRASH_COUNT%% - restarting in 5s... (Ctrl+C to cancel)
    timeout /t 5 /nobreak >nul
    goto :launch
    :done
    pause
''').lstrip()
pathlib.Path(r'%DIST%\start.bat').write_text(bat, encoding='ascii')
print('  start.bat written.')
"

echo.
echo =====================================================
echo   BUILD COMPLETE
echo   Folder : dist\tracker_app\
echo   Run    : dist\tracker_app\start.bat
echo            dist\tracker_app\tracker_app.exe
echo =====================================================
echo.
pause
