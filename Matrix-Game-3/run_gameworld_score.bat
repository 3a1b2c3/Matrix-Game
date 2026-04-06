@echo off
:: Run GameWorldScore perceptual metrics on Matrix-Game-3 output videos.
:: Usage: run_gameworld_score.bat [videos_path] [output_path]
::   videos_path  default: out\vbench\videos
::   output_path  default: out\gameworld_score

setlocal enabledelayedexpansion

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

set GWS_DIR=C:\workspace\world\matrix3\Matrix-Game-1\GameWorldScore

set VIDEOS_PATH=%~1
set OUTPUT_PATH=%~2

if not defined VIDEOS_PATH set VIDEOS_PATH=%ROOT%\out\vbench\videos
if not defined OUTPUT_PATH set OUTPUT_PATH=%ROOT%\out\gameworld_score

:: Make absolute if relative
if not "%VIDEOS_PATH:~1,1%"==":" set VIDEOS_PATH=%ROOT%\%VIDEOS_PATH%
if not "%OUTPUT_PATH:~1,1%"==":" set OUTPUT_PATH=%ROOT%\%OUTPUT_PATH%

if not exist "%VIDEOS_PATH%" (
    echo ERROR: videos_path not found: %VIDEOS_PATH%
    exit /b 1
)
if not exist "%GWS_DIR%\evaluate.py" (
    echo ERROR: GameWorldScore not found at: %GWS_DIR%
    exit /b 1
)

mkdir "%OUTPUT_PATH%" 2>nul

echo ============================================================
echo GameWorldScore  ^|  Matrix-Game-3
echo ============================================================
echo   videos  : %VIDEOS_PATH%
echo   output  : %OUTPUT_PATH%
echo   GWS dir : %GWS_DIR%
echo ============================================================

pushd "%GWS_DIR%"

python evaluate.py ^
    --videos_path "%VIDEOS_PATH%" ^
    --dimension temporal_consistency aesthetic_quality imaging_quality motion_smoothness ^
    --mode GameWorld_custom ^
    --output_path "%OUTPUT_PATH%"

set EXIT_CODE=%ERRORLEVEL%

:: MG3 action-following (optical flow vs keyboard sidecar)
echo.
echo [MG3] Running action-following evaluation...
python "%GWS_DIR%\GameWorld\mg3_action_control.py" ^
    --videos_path "%VIDEOS_PATH%" ^
    --output_path "%OUTPUT_PATH%"
set ACT_CODE=%ERRORLEVEL%

popd

echo ============================================================
echo Done. Perceptual exit: %EXIT_CODE%  Action-following exit: %ACT_CODE%
echo Results: %OUTPUT_PATH%
echo ============================================================

exit /b %EXIT_CODE%
