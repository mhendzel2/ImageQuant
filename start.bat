@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at .venv
  echo Run install.bat first.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

python -c "import yaml, napari" >nul 2>&1
if errorlevel 1 (
  echo Required GUI dependencies are missing in .venv. Installing now ...
  python -m pip install --upgrade pip setuptools wheel
  if errorlevel 1 (
    echo [ERROR] Failed to upgrade packaging tools.
    exit /b 1
  )
  python -m pip install -e ".[ui]"
  if errorlevel 1 (
    echo [ERROR] Failed to install required GUI dependencies.
    echo Run install.bat and check for errors.
    exit /b 1
  )
)

echo Launching ProtocolQuant GUI ...
python -m protocolquant.cli gui
exit /b %ERRORLEVEL%
