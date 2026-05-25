@echo off
title Baggage Tracker - System Check
chcp 437 >nul 2>&1

set PYTHON=
if exist "%~dp0venv\Scripts\python.exe"  set PYTHON=%~dp0venv\Scripts\python.exe
if exist "%~dp0.venv\Scripts\python.exe" set PYTHON=%~dp0.venv\Scripts\python.exe
if "%PYTHON%"=="" set PYTHON=python

set TORCH_HOME=%~dp0torch_cache
for %%D in (
    "%~dp0venv\Lib\site-packages\PyQt5\Qt5\plugins"
    "%~dp0venv\Lib\site-packages\PyQt5\Qt\plugins"
) do ( if exist "%%~D\platforms\qwindows.dll" set QT_PLUGIN_PATH=%%~D )

cd /d "%~dp0"
"%PYTHON%" check_system.py
echo.
pause
