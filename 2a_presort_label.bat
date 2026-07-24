@echo off
REM ============================================================
REM  Step 2a: auto_presort LABEL (run_presort_label.py)
REM  Model pre-labels long videos. Needs GPU. Double-click to run.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Running auto_presort LABEL...
python -m scripts.presort.run_presort_label

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Label step failed. Read the messages above.
    echo  Do NOT continue to 2b until this succeeds.
) else (
    echo  Done. Next: double-click 2b_presort_review.bat
)
echo ============================================================
pause
