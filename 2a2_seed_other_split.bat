@echo off
REM ============================================================
REM  Step 2a2: seed blank docs for other/ clips
REM  (other -> active/resting/eat/drink re-split workflow)
REM  No GPU needed. Safe to re-run: already-seeded videos are skipped.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

echo Seeding other/ clips for manual re-split...
python -m scripts.presort.auto_presort_seed_other_split

echo.
echo ============================================================
if errorlevel 1 (
    echo  [ERROR] Seed step failed. Read the messages above.
) else (
    echo  Done. Next: double-click 2b2_presort_review_other_split.bat
)
echo ============================================================
pause
