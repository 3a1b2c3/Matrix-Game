@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set SIZE=704*1280
set CKPT_DIR=Matrix-Game-3.0
set FA_VERSION=2
set NUM_ITERATIONS=12
set NUM_INFERENCE_STEPS=3
set VAE_TYPE=mg_lightvae_v2
set LIGHTVAE_PRUNING_RATE=0.75
set TAESD_PATH=
:: To use the tiny VAE decoder (fastest): set VAE_TYPE=taesd
:: (requires taew2_2.pth in <CKPT_DIR>\ — run: download_model.bat --taesd)
set OUTPUT_DIR=./output

:: Build optional taesd_path arg
set TAESD_ARG=
if defined TAESD_PATH if not "%TAESD_PATH%"=="" set TAESD_ARG=--taesd_path %TAESD_PATH%

echo Output directory: %~dp0output

for /d %%D in (demo_images\*) do (
    set FOLDER=%%~nxD
    set IMAGE=%%D\image.png
    set PROMPT_FILE=%%D\prompt.txt

    if exist "!PROMPT_FILE!" (
        set /p PROMPT=<"!PROMPT_FILE!"
        echo.
        echo === Running !FOLDER! ===
        echo Prompt: !PROMPT!
        python generate.py ^
            --size %SIZE% ^
            --ckpt_dir %CKPT_DIR% ^
            --fa_version %FA_VERSION% ^
            --use_int8 ^
            --num_iterations %NUM_ITERATIONS% ^
            --num_inference_steps %NUM_INFERENCE_STEPS% ^
            --image "!IMAGE!" ^
            --prompt "!PROMPT!" ^
            --save_name !FOLDER! ^
            --seed 42 ^
            --lightvae_pruning_rate %LIGHTVAE_PRUNING_RATE% ^
            --vae_type %VAE_TYPE% ^
            %TAESD_ARG% ^
            --output_dir %OUTPUT_DIR%

        if !errorlevel! neq 0 (
            echo ERROR: !FOLDER! failed with exit code !errorlevel!
        ) else (
            echo Done: !FOLDER!  ->  %~dp0output\!FOLDER!.mp4
        )
    ) else (
        echo SKIP: !FOLDER! has no prompt.txt
    )
)

echo.
echo All demos complete.
endlocal
