@echo off
REM
REM Perception AI - Windows Installer
REM
REM Usage: Double-click or run from command prompt
REM

echo ========================================
echo   PERCEPTION AI - INSTALLER
echo   Layer 2 of Drone AI System
echo ========================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo Python found!
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install core dependencies
echo Installing core dependencies...
python -m pip install numpy scipy Pillow
echo.

REM Check if user wants PyTorch
echo.
set /p INSTALL_TORCH="Install PyTorch? (needed for training) [Y/n]: "
if /i "%INSTALL_TORCH%"=="" set INSTALL_TORCH=Y
if /i "%INSTALL_TORCH%"=="Y" (
    echo Installing PyTorch...
    python -m pip install torch torchvision
)
echo.

REM Check if user wants OpenCV
set /p INSTALL_CV="Install OpenCV? (needed for video extraction) [Y/n]: "
if /i "%INSTALL_CV%"=="" set INSTALL_CV=Y
if /i "%INSTALL_CV%"=="Y" (
    echo Installing OpenCV...
    python -m pip install opencv-python
)
echo.

REM Install perception package
echo Installing Perception AI package...
python -m pip install -e .
echo.

REM Install test dependencies
python -m pip install pytest
echo.

REM Create directories
echo Creating directories...
if not exist "data\images" mkdir data\images
if not exist "data\labels" mkdir data\labels
if not exist "models" mkdir models
echo.

REM Verify
echo Verifying installation...
echo.
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
python -c "import PIL; print('PIL:', PIL.__version__)"
python -c "import tkinter; print('tkinter: OK')"
python -c "from perception import PerceptionAI; print('perception: OK')"
echo.

echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo Quick start:
echo.
echo   1. Create sample images:
echo      python tools\image_collector.py sample -n 20
echo.
echo   2. Launch labeling tool:
echo      launch_labeler.bat
echo      or: python tools\labeler.py
echo.
echo   3. Run demo:
echo      python examples\demo.py
echo.

pause
