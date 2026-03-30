@echo off
:: Matrix-Game-3 - Model Download
:: Downloads the main MG3 checkpoint and optionally the tiny VAE decoder.
::
:: Usage:
::   download_model.bat                          -- MG3 + tiny VAE (default)
::   download_model.bat --no-taesd              -- MG3 checkpoint only
::   download_model.bat --taesd-only             -- tiny VAE only
::   download_model.bat --repo <hf_repo>         -- custom MG3 repo
::   download_model.bat --dir  <local_dir>       -- custom local dir
::
:: The tiny VAE file (taew2_2.safetensors) is downloaded into <local_dir> so that
:: --vae_type taesd resolves it automatically.

setlocal enabledelayedexpansion

set LOCAL_DIR=Matrix-Game-3.0
set HF_REPO=Skywork/Matrix-Game-3.0
set TAESD_REPO=lightx2v/Autoencoders
set TAESD_FILE=taew2_2.safetensors
set DO_MG3=1
set DO_TAESD=1

:parse
if "%~1"=="" goto :run
if /i "%~1"=="--repo"       ( set HF_REPO=%~2    & shift & shift & goto :parse )
if /i "%~1"=="--dir"        ( set LOCAL_DIR=%~2  & shift & shift & goto :parse )
if /i "%~1"=="--no-taesd"   ( set DO_TAESD=0     & shift             & goto :parse )
if /i "%~1"=="--taesd-only" ( set DO_MG3=0 & set DO_TAESD=1 & shift  & goto :parse )
shift & goto :parse

:run
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

if "%LOCAL_DIR:~1,1%"==":" (
    rem absolute — keep as-is
) else (
    set LOCAL_DIR=%ROOT%\%LOCAL_DIR%
)

echo ============================================================
echo Matrix-Game-3  ^|  Model Download
echo ============================================================
if %DO_MG3%==1 (
    echo   MG3 repo  : %HF_REPO%
    echo   local dir : %LOCAL_DIR%
)
if %DO_TAESD%==1 (
    echo   tiny VAE  : %TAESD_REPO% / %TAESD_FILE%
    echo   saved to  : %LOCAL_DIR%\%TAESD_FILE%
)
echo ============================================================

set OVERALL_EXIT=0

if %DO_MG3%==1 (
    echo.
    echo [1/2] Downloading MG3 checkpoint...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='%HF_REPO%', local_dir=r'%LOCAL_DIR%')"
    if !errorlevel! neq 0 (
        echo ERROR: MG3 download failed.
        set OVERALL_EXIT=1
    ) else (
        echo Done: %LOCAL_DIR%
    )
)

if %DO_TAESD%==1 (
    echo.
    echo [2/2] Downloading tiny VAE decoder ^(%TAESD_FILE%^)...
    python -c "from huggingface_hub import hf_hub_download; p=hf_hub_download(repo_id='%TAESD_REPO%', filename='%TAESD_FILE%', local_dir=r'%LOCAL_DIR%'); print('Saved to', p)"
    if !errorlevel! neq 0 (
        echo ERROR: tiny VAE download failed.
        set OVERALL_EXIT=1
    ) else (
        echo Done: %LOCAL_DIR%\%TAESD_FILE%
    )
)

echo.
echo ============================================================
if %OVERALL_EXIT%==0 (
    echo All downloads complete.
) else (
    echo One or more downloads failed. Check errors above.
    echo Make sure huggingface_hub is installed: pip install huggingface_hub
)
echo ============================================================

exit /b %OVERALL_EXIT%
