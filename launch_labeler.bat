@echo off
REM
REM Perception AI - Image Labeler Launcher
REM
REM Quick launcher for the image labeling tool.
REM Double-click to run.
REM

echo ========================================
echo   PERCEPTION AI - IMAGE LABELER
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set default paths
set IMAGES_DIR=data\images
set LABELS_DIR=data\labels

REM Create directories if they don't exist
if not exist "%IMAGES_DIR%" mkdir "%IMAGES_DIR%"
if not exist "%LABELS_DIR%" mkdir "%LABELS_DIR%"

REM Count images (simple check)
dir /b "%IMAGES_DIR%\*.jpg" "%IMAGES_DIR%\*.jpeg" "%IMAGES_DIR%\*.png" 2>nul | find /c /v "" > temp_count.txt
set /p IMAGE_COUNT=<temp_count.txt
del temp_count.txt

if %IMAGE_COUNT%==0 (
    echo No images found in %IMAGES_DIR%
    echo.
    set /p CREATE_SAMPLES="Create sample images to try the labeler? [Y/n]: "
    if /i "%CREATE_SAMPLES%"=="" set CREATE_SAMPLES=Y
    if /i "%CREATE_SAMPLES%"=="Y" (
        echo.
        echo Creating 20 sample images...
        python tools\image_collector.py sample -o %IMAGES_DIR% -n 20
        echo.
    ) else (
        echo.
        echo Please add images to: %IMAGES_DIR%
        echo Then run this launcher again.
        echo.
        pause
        exit /b 0
    )
)

echo Images directory: %IMAGES_DIR%
echo Labels directory: %LABELS_DIR%
echo.
echo Starting labeler...
echo.

REM Launch labeler
python tools\labeler.py --images %IMAGES_DIR% --output %LABELS_DIR%

echo.
echo Labeler closed.
pause
