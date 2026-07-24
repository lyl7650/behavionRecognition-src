@echo off
REM ============================================================
REM  Step 2c: auto_presort APPLY (run_presort_apply.py)
REM  Cuts approved clips into sorted-<date>/<class>/. No GPU needed.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Running auto_presort APPLY...
python -m scripts.presort.run_presort_apply

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Apply step failed. Read the messages above.
) else (
    echo  Done. Check any [WARN] lines above, then double-click 3_run_crop.bat
)
echo ============================================================
pause
