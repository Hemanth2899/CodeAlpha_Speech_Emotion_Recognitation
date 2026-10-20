@echo off
setlocal
cd /d "%~dp0"
echo Creating virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create environment. Ensure 64-bit Python 3.11 is installed.
  pause
  exit /b 1
)
echo Installing libraries...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Library installation failed. Read README.md for troubleshooting.
  pause
  exit /b 1
)
echo.
echo Setup complete.
echo To download the dataset, run:
echo .venv\Scripts\python.exe download_dataset.py
pause
