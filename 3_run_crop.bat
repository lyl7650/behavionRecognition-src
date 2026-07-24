@echo off
REM ============================================================
REM  Step 3: Crop / snapshot (run_crop.py)
REM  Double-click to run. No need to open cmd or activate conda.
REM ============================================================

REM Switch to this .bat's own folder (project root, path-independent)
cd /d "%~dp0"

REM Activate the AR conda environment
call conda activate AR

REM Binary choice: 1 = video clips (default), 2 = frame format
echo.
echo Crop mode:
echo   1) video clips (default)
echo   2) frame format
choice /c 12 /n /m "Choose [1/2]: "
if errorlevel 2 (
    echo Running in FRAME format...
    python -m scripts.pipeline.run_crop --frames
) else (
    echo Running in VIDEO CLIP format ^(default^)...
    python -m scripts.pipeline.run_crop
)

echo.
echo ============================================================
echo  Done. You can close this window now.
echo ============================================================
pause
