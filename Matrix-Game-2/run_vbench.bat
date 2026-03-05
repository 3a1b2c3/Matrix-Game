@echo off
:: Matrix-Game-2 - VBench Batch Inference
:: Generates 5 videos per prompt for scenery+indoor types, 81 video frames (21 latent).
:: Usage: run_vbench.bat [output_base] [num_samples] [image_types]

setlocal enabledelayedexpansion

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h"     goto :help
if /i "%~1"=="/?"     goto :help
goto :run

:help
echo.
echo Matrix-Game-2 - VBench Batch Inference
echo.
echo Usage:
echo   run_vbench.bat [output_base] [num_samples] [image_types]
echo.
echo Arguments (positional, all optional):
echo   1  output_base    Base output directory            (default: out\vbench)
echo   2  num_samples    Videos to generate per prompt   (default: 5)
echo   3  image_types    Comma-separated type filter      (default: scenery,indoor)
echo.
echo Notes:
echo   - VBench requires 5 samples per prompt
echo   - num_output_frames=42 -> 165 video frames
echo   - Already-generated videos are skipped automatically
echo   - Outputs: {output_base}\videos\{caption}-{0..N-1}.mp4
echo   - FPS log: {output_base}\videos\fps_log.txt
echo.
exit /b 0

:run
:: ── configurable defaults ──────────────────────────────────────────────────
set CKPT=Matrix-Game-2.0\base_distilled_model\base_distill.safetensors
set CONFIG=configs/inference_yaml/inference_universal.yaml
set PRETRAINED=Matrix-Game-2.0
set NUM_OUTPUT_FRAMES=42
set SEED=42
:: ──────────────────────────────────────────────────────────────────────────

set OUTPUT_BASE=%~1
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=out\vbench
set NUM_SAMPLES=%~2
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=5
set IMAGE_TYPES=%~3
if "%IMAGE_TYPES%"=="" set IMAGE_TYPES=scenery,indoor

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

:: Prepend ROOT only for relative OUTPUT_BASE (skip if already absolute)
if "%OUTPUT_BASE:~1,1%"==":" (
    rem absolute path — keep as-is
) else (
    set OUTPUT_BASE=%ROOT%\%OUTPUT_BASE%
)
set VBENCH_OUTPUT_DIR=%OUTPUT_BASE%\videos

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set _DT=%%a
set LOG_FILE=%OUTPUT_BASE%\vbench_run_%_DT:~0,8%_%_DT:~8,6%.log

if not exist "%ROOT%\%PRETRAINED%" (
    echo ERROR: Pretrained model not found: %ROOT%\%PRETRAINED%
    exit /b 1
)

echo ============================================================
echo Matrix-Game-2  ^|  VBench batch  ^|  Windows
echo ============================================================
echo   output    : %VBENCH_OUTPUT_DIR%
echo   samples   : %NUM_SAMPLES%
echo   types     : %IMAGE_TYPES%
echo   lat frames: %NUM_OUTPUT_FRAMES%  ^(=%NUM_OUTPUT_FRAMES% lat = %NUM_OUTPUT_FRAMES%*4-3 video frames^)
echo   seed      : %SEED%
echo ============================================================

:: Record start time
set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

set PY_ARGS=--config_path "%CONFIG%"
set PY_ARGS=%PY_ARGS% --pretrained_model_path "%PRETRAINED%"
set PY_ARGS=%PY_ARGS% --num_output_frames %NUM_OUTPUT_FRAMES%
set PY_ARGS=%PY_ARGS% --seed %SEED%
set PY_ARGS=%PY_ARGS% --vbench_output_dir "%VBENCH_OUTPUT_DIR%"
set PY_ARGS=%PY_ARGS% --image_types "%IMAGE_TYPES%"
set PY_ARGS=%PY_ARGS% --num_samples %NUM_SAMPLES%
if not "%CKPT%"=="" set PY_ARGS=%PY_ARGS% --checkpoint_path "%ROOT%\%CKPT%"

echo.
echo [MG2-VBench] Generating %NUM_SAMPLES% samples per prompt...
python "%ROOT%\inference_vbench.py" %PY_ARGS% 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"
set EXIT_CODE=%ERRORLEVEL%
echo [MG2-VBench] Done. Exit: %EXIT_CODE%

:: Record end time
set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

echo ============================================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo FPS log: %VBENCH_OUTPUT_DIR%\fps_log.txt
echo ============================================================

exit /b %EXIT_CODE%
