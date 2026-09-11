@echo off
REM ================================================================
REM  Scene Retrain (flat): one-click retrain pipeline
REM  (auto label -> review -> apply+crop -> merge -> train+distill)
REM  No site dimension: each new video batch is merged with the
REM  latest old snapshot directly, then trained.
REM  Double-click to run. No need to open cmd or activate conda.
REM ================================================================
cd /d "%~dp0"

call conda activate AR

python -m scripts.scene_retrain.pipeline

echo.
echo ================================================================
echo  Scene retrain finished. See console output above for artifact
echo  paths. The deployment model is the student's *_ema.pth
echo  (under outputs/), not the teacher checkpoint.
echo ================================================================
pause
