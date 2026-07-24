@echo off
REM ============================================================
REM  Step 1: Build directory scaffold (scaffold_dirs.py)
REM  Double-click to run. No need to open cmd or activate conda.
REM ============================================================

REM Switch to this .bat's own folder (project root, path-independent)
cd /d "%~dp0"

REM Activate the AR conda environment
call conda activate AR

REM Run the script
python -m scripts.pipeline.scaffold_dirs

echo.
echo ============================================================
echo  Done. You can close this window now.
echo ============================================================
pause
