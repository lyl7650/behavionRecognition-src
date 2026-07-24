@echo off
REM ============================================================
REM  Pack an other-split review bundle to send to a colleague.
REM  Pick a range of other/ clips; output is self-contained (Win + Mac).
REM ============================================================
cd /d "%~dp0..\.."
call conda activate AR
echo Building other-split review pack...
python -m scripts.review_tools.pack_review_other_split
echo.
if errorlevel 1 (
    echo  [ERROR] Pack failed. Read the messages above.
) else (
    echo  Done. Send the whole review-pack-... folder to your colleague.
)
pause
