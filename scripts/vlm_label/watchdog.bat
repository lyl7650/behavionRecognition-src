@echo off
REM VLM labeling orchestrator (all-ASCII paths only).
REM Safe to run repeatedly: run_overnight holds a PID lock and exits
REM immediately if another instance is alive.
REM KEEP THIS CONSOLE WINDOW OPEN - closing it kills the labeling run.
title VLM_LABELING_DO_NOT_CLOSE
cd /d E:\ActionRecognition\behavionRecognition-src
"E:\ActionRecognition\behavionRecognition-src\.venv-vlm\Scripts\python.exe" -m scripts.vlm_label.run_overnight
echo.
echo === Orchestrator exited. Logs: logs\vlm_overnight*.log + logs\vlm_label_*.log ===
pause
