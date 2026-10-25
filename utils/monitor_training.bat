@echo off
echo ========================================
echo Training Monitor - RepViT M2.3
echo ========================================
echo.

set OUTPUT_DIR=E:\ActionRecognition\behavionRecognition-src\outputs\rabbit\2026-06-24\repvit_m2_3

:loop
cls
echo Training Output Directory: %OUTPUT_DIR%
echo.
echo Latest files:
dir /O-D /T:W "%OUTPUT_DIR%" 2>nul
echo.
echo ========================================
echo Press Ctrl+C to stop monitoring
echo Next update in 30 seconds...
echo ========================================
timeout /t 30 /nobreak >nul
goto loop
