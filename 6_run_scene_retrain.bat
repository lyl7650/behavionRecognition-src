@echo off
REM ================================================================
REM  Scene Retrain (flat): one-click retrain pipeline
REM  (auto label -> review -> apply+crop -> merge -> train+distill)
REM  No site dimension: each new video batch is merged with the
REM  latest old snapshot directly, then trained.
REM  Interactive mode: double-click, then answer prompts.
REM  Metadata-driven mode (no species/model/mode prompts):
REM    6_run_scene_retrain.bat --auto-profile --videos-dir "G:\aws_backup\YYYY-MM-DD\raw" --date YYYY-MM-DD
REM  Each animal_profile group is retrained sequentially, largest first;
REM  every group still pauses for browser review.
REM ================================================================
cd /d "%~dp0"

REM Labeling behavior model override for the dashboard fusion engine.
REM Leave undefined to use the dashboard's built-in encrypted model.
if not defined RABBIT_BEH_MODEL set "RABBIT_BEH_MODEL=E:\ActionRecognition\behavionRecognition-src\outputs\rabbit\2026-07-06\repvit_best_distilled_0706_ema.pth"
REM if not defined PET_BEH_MODEL set "PET_BEH_MODEL=E:\ActionRecognition\behavionRecognition-src\outputs\pet\<date>\<pet_model>.pth"

call conda activate AR

python -m scripts.scene_retrain.pipeline %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ================================================================
echo  Scene retrain finished. See console output above for artifact
echo  paths. The deployment model is the student's *_ema.pth
echo  (under outputs/), not the teacher checkpoint.
echo ================================================================
if /i not "%SCENE_RETRAIN_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
