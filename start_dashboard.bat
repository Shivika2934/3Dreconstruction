@echo off
REM Gradio Dashboard Launcher for Windows
REM =======================================
REM
REM This script will:
REM 1. Check if Python is installed
REM 2. Create a virtual environment (if needed)
REM 3. Install dependencies
REM 4. Start the Gradio dashboard
REM

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   3D Reconstruction Gradio Dashboard - Windows Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python from: https://python.org
    echo And make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version

REM Check if we're already in a virtual environment
if defined VIRTUAL_ENV (
    echo [OK] Virtual environment is active: !VIRTUAL_ENV!
) else (
    echo.
    echo [INFO] Creating virtual environment...
    
    if exist venv (
        echo [OK] Virtual environment already exists
    ) else (
        echo [ACTION] Creating new virtual environment...
        python -m venv venv
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment
            pause
            exit /b 1
        )
        echo [OK] Virtual environment created
    )
    
    echo [ACTION] Activating virtual environment...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo ERROR: Failed to activate virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment activated
)

REM Check if requirements are installed
echo.
echo [ACTION] Checking dependencies...
pip list | findstr gradio >nul 2>&1

if errorlevel 1 (
    echo [WARNING] Missing dependencies detected
    echo.
    echo [ACTION] Installing required packages...
    echo This may take several minutes...
    echo.
    
    pip install -r gradio_requirements.txt
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo.
        echo Try running manually:
        echo   pip install -r gradio_requirements.txt
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo [OK] Dependencies installed successfully
) else (
    echo [OK] All dependencies are installed
)

REM Download models if needed
echo.
echo [ACTION] Checking for required models...

if not exist yolov8n.pt (
    echo [WARNING] YOLO model not found
    echo [ACTION] Downloading YOLO model on first run...
    echo (This will happen when you start the dashboard)
)

echo.
echo ============================================================
echo   Starting Gradio Dashboard
echo ============================================================
echo.
echo Once the server starts, open your browser to:
echo.
echo   http://localhost:7860
echo.
echo Press CTRL+C in this window to stop the server
echo.
echo ============================================================
echo.

REM Start the dashboard
python gradio_dashboard.py

if errorlevel 1 (
    echo.
    echo ERROR: Dashboard failed to start
    echo.
    echo Check the error messages above for details
    echo.
    pause
    exit /b 1
)

pause
