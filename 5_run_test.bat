@echo off
REM ============================================================
REM  Step 5: Evaluate a teacher or distilled student checkpoint.
REM  Results (JSON, CSV, traces, and timeline PNGs) are saved beside
REM  the selected checkpoint in a timestamped test_results folder.
REM ============================================================
cd /d "%~dp0"
call conda activate AR
python -m scripts.pipeline.run_test --interactive
echo.
echo ============================================================
echo  Done. You can close this window now.
echo ============================================================
pause
