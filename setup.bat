@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   SKIN DISEASE CLASSIFICATION SYSTEM - AUTO SETUP
echo ============================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Python is NOT installed on this computer!
    echo.
    if exist "installers\python-3.10.11-amd64.exe" (
        echo Found Python installer in "installers" folder!
        echo.
        echo Starting Python installation now...
        echo -----------------------------------------------------------
        echo [IMPORTANT]: In the installation window, you MUST check:
        echo    [X] "Add Python 3.10 to PATH"  (at the bottom)
        echo Then click "Install Now".
        echo -----------------------------------------------------------
        echo.
        pause
        start /wait installers\python-3.10.11-amd64.exe
        echo.
        echo If you completed the Python setup, please restart setup.bat!
        pause
        exit /b 0
    ) else (
        echo Please install Python 3.10 from https://www.python.org
        echo Or run: winget install Python.Python.3.10
        echo Be sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

echo [1/4] Found Python:
python --version

:: 2. Create Virtual Environment
if not exist "venv" (
    echo.
    echo [2/4] Creating virtual environment (venv)...
    python -m venv venv
) else (
    echo.
    echo [2/4] Virtual environment (venv) already exists.
)

:: 3. Activate venv
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate

:: 4. Upgrade pip & Install requirements
echo.
echo [4/4] Installing dependencies from requirements.txt...
echo This may take 3-5 minutes depending on internet speed...
python -m pip install --upgrade pip

:: Install PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install python-docx

echo.
echo ============================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo   You can now double-click "run_web.bat" to start the app.
echo ============================================================
pause
