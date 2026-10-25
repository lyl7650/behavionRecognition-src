@echo off
cd /d "%~dp0"
call conda activate AR
python -u train_rabbit_0624_series.py > train_rabbit_0624_series.log 2>&1
pause
