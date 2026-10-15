@echo off
echo ============================================================
echo   STARTING SKIN DISEASE CLASSIFICATION WEB APP
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run "setup.bat" first to install dependencies.
    pause
    exit /b 1
)

echo Activating environment...
call venv\Scripts\activate

echo.
echo Starting Flask Web Application...
echo Open your browser and go to: http://127.0.0.1:5000
echo.
python web\app.py

pause
