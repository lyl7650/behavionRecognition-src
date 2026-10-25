@echo off
REM ============================================================
REM  Step 7: VLM review tool - FULL mode (open dir + infer + edit)
REM  Page: choose species, video dir and scene -> "Start infer" (needs GPU).
REM  Editing/approval works with or without inference.
REM  Optional: drag a folder onto this bat to prefill the page's path
REM  (passed through as --dir <path>; species is still chosen on the page).
REM  KEEP THIS CONSOLE WINDOW OPEN - closing it kills the server.
REM ============================================================
title VLM_REVIEW_DO_NOT_CLOSE
cd /d E:\ActionRecognition\behavionRecognition-src
if not "%~1"=="" (
    ".venv-vlm\Scripts\python.exe" -m scripts.vlm_screener.run_all --dir "%~1"
) else (
    ".venv-vlm\Scripts\python.exe" -m scripts.vlm_screener.run_all
)

echo.
echo === review server stopped. Logs: logs\vlm_screener.log ===
pause
