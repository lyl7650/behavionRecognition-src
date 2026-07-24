@echo off
REM ============================================================
REM  Pack a review bundle to send to a colleague (Win + Mac).
REM ============================================================
cd /d "%~dp0..\.."
call conda activate AR
echo Building review pack...
python -m scripts.review_tools.pack_review
echo.
if errorlevel 1 (
    echo  [ERROR] Pack failed. Read the messages above.
) else (
    echo  Done. Send the whole review-pack-... folder to your colleague.
)
pause
