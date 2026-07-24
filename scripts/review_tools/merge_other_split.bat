@echo off
REM ============================================================
REM  Merge a colleague's returned other-split annotations back
REM  into the original other/ dirs. No GPU needed.
REM ============================================================
cd /d "%~dp0..\.."
call conda activate AR

set /p PACK=Drag the returned review-pack folder here (or type its path), then press Enter:
REM strip surrounding quotes that Windows adds when you drag a folder in
set PACK=%PACK:"=%

echo Merging annotations from "%PACK%"...
python -m scripts.review_tools.merge_other_split --pack "%PACK%"
echo.
if errorlevel 1 (
    echo  [ERROR] Merge failed. Read the messages above.
) else (
    echo  Done. Next: double-click 2c2_apply_other_split.bat to cut the approved clips.
)
pause
