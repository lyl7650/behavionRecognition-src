@echo off
REM ============================================================
REM  Step 2c2: other -> active/resting/eat/drink APPLY
REM  Cuts approved segments into <sorted-date>/<label>/. No GPU needed.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Running other-split APPLY...
python -m scripts.presort.run_presort_apply_other_split

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Apply step failed. Read the messages above.
) else (
    echo  Done. Check any [WARN] lines above.
)
echo ============================================================
pause
