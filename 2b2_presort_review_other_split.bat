@echo off
REM ============================================================
REM  Step 2b2: other -> active/resting/eat/drink REVIEW
REM  Opens the browser review page for ALL scattered other/ folders
REM  under pet\normal in a single session. No GPU needed.
REM  Run 2a2_seed_other_split.bat first if you haven't.
REM  When finished reviewing, stop the server (Ctrl+C) or just
REM  close this window; then double-click 2c2_apply_other_split.bat.
REM ============================================================

cd /d "%~dp0"
call conda activate AR

REM --- If port 8765 is busy (leftover review server), offer to kill it ---
set "PORT=8765"
set "BUSY_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%PORT% "') do set "BUSY_PID=%%p"
if defined BUSY_PID (
    echo [WARN] Port %PORT% is already in use by process PID %BUSY_PID%.
    echo        This is probably a leftover review server window.
    choice /c YN /m "Kill that process and continue"
    if errorlevel 2 (
        echo Aborted. Close the other review window, then run 2b2 again.
        pause
        exit /b 1
    )
    taskkill /PID %BUSY_PID% /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    echo [OK] Old process killed. Continuing...
    echo.
)

echo ============================================================
echo  Review server is starting. A browser tab will open.
echo  Approve each video, then press Ctrl+C (or close window).
echo  AFTER that, double-click 2c2_apply_other_split.bat
echo ============================================================
echo.
python -m scripts.presort.auto_presort_review_other_split

echo.
if errorlevel 1 (
    echo  [ERROR] Review server did not run. Read the messages above.
    echo  Do NOT continue to 2c2. Fix the problem, then double-click 2b2 again.
) else (
    echo  Server stopped. Next: double-click 2c2_apply_other_split.bat
)
pause
