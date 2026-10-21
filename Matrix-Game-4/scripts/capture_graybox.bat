@echo off
setlocal

rem Records the EdenGolem taxi game in Matrix-Game gray-box style as an input clip for matrix-game render.
rem Usage: capture_graybox.bat [capture_graybox.py options]     Example: capture_graybox.bat --id taxi-lanes-gb --frames 217
rem Output: captures\<id>.mp4 in this repository.
rem Override the interpreter with PYTHON=<path to python.exe> (needs playwright and imageio-ffmpeg) and the game project with
rem EDENGOLEM_GAME=<folder of the EdenGolem city_racer project>.

if "%PYTHON%"=="" set "PYTHON=C:\workspace\world\EdenLive\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Python not found: %PYTHON%
  echo Set PYTHON to a python.exe that has playwright and imageio-ffmpeg installed.
  exit /b 1
)

"%PYTHON%" "%~dp0capture_graybox.py" %*
endlocal
