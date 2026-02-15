@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON_CMD="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3.11 -V >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PYTHON_CMD=py -3.11"
  ) else (
    py -3.10 -V >nul 2>&1
    if %ERRORLEVEL%==0 (
      set "PYTHON_CMD=py -3.10"
    ) else (
      set "PYTHON_CMD=py -3"
    )
  )
) else (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PYTHON_CMD=python"
  )
)

if not defined PYTHON_CMD (
  echo [ERROR] Python was not found. Install Python 3.10+ and try again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

echo Upgrading packaging tools ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip/setuptools/wheel.
  exit /b 1
)

echo Installing ProtocolQuant core + GUI dependencies ...
python -m pip install -e ".[ui]"
if errorlevel 1 (
  echo [ERROR] Required install failed.
  echo         This usually means your Python version is incompatible with one of the pinned wheels.
  echo         Recommended: Python 3.10 or 3.11, then rerun install.bat.
  exit /b 1
)

echo Installing optional Cellpose/Torch segmentation backend ...
python -m pip install -e ".[segmentation-cellpose]"
if errorlevel 1 (
  echo [WARN] Optional segmentation-cellpose install failed.
  echo        The GUI and fallback segmentation still work.
  echo        To retry later: python -m pip install -e ".[segmentation-cellpose]"
)

echo.
echo Install complete (core GUI is ready).
echo Use start.bat to launch the GUI.
exit /b 0
