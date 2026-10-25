@echo off
REM ============================================================
REM  Step 8: VLM review tool - VIEW mode (no GPU, no inference).
REM  Same page as 7 (filter + edit + approve), but the inference
REM  button is disabled. Use 7_vlm_screener.bat to run inference.
REM ============================================================
title VLM_REVIEW_VIEW
cd /d E:\ActionRecognition\behavionRecognition-src
set ARGS=
if not "%~1"=="" set ARGS=--dir %~1
".venv-vlm\Scripts\python.exe" -m scripts.vlm_screener.run_all --view-only %ARGS% %*

echo.
echo === review server stopped ===
pause
