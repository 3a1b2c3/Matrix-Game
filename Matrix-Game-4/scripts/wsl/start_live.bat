@echo off
rem Start, stop or inspect live Matrix-Game: starts MediaMTX and the given scene's server on the DGX (if not running), opens one SSH
rem tunnel covering all three scene ports plus the shared MediaMTX ports, and opens the browser (the scene's own page, or the
rem switcher picker when no --scene is given).
rem Usage: scripts\wsl\start_live.bat [start|stop|status|switcher] [--scene racing|castle|ancient-egypt] [--preset ...] [--fps 24]
rem                                   [--prompt-mode fixed|dynamic] [--no-browser]
rem   switcher   just open the local game picker (scripts\wsl\switcher.html); starts nothing
rem   stop       stops the given --scene's server (default: all three) plus MediaMTX -- only what this script itself started --
rem              and closes the tunnel
setlocal enabledelayedexpansion
set "DGX=kschmid@10.74.11.118"
set "DGX_REPO=/localhome/kschmid/Matrix-Game/Matrix-Game-4"
set "PORTS=racing=8025 castle=8026 ancient-egypt=8027"

set "ACTION=start"
set "SCENE="
set "ARGS="
set "BROWSER=1"
set "SKIP=0"
for %%A in (%*) do (
  if !SKIP! == 1 (
    set "SCENE=%%A"
    set "ARGS=!ARGS! --scene %%A"
    set "SKIP=0"
  ) else if "%%A"=="--no-browser" (
    set "BROWSER=0"
  ) else if "%%A"=="--scene" (
    set "SKIP=1"
  ) else if "%%A"=="start" (
    set "ACTION=start"
  ) else if "%%A"=="stop" (
    set "ACTION=stop"
  ) else if "%%A"=="status" (
    set "ACTION=status"
  ) else if "%%A"=="switcher" (
    set "ACTION=switcher"
  ) else (
    set "ARGS=!ARGS! %%A"
  )
)
if not defined SCENE set "SCENE=racing"
if "%SCENE%"=="racing" (set "PORT=8025") else if "%SCENE%"=="castle" (set "PORT=8026") else if "%SCENE%"=="ancient-egypt" (set "PORT=8027") else (
  echo Unknown scene: %SCENE% ^(racing, castle or ancient-egypt^)
  exit /b 2
)
set "URL=http://127.0.0.1:%PORT%/"

if "%ACTION%"=="switcher" (
  start "" "%~dp0switcher.html"
  exit /b 0
)

if "%ACTION%"=="status" (
  ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh status"
  exit /b %ERRORLEVEL%
)

if "%ACTION%"=="stop" (
  ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh stop!ARGS!"
  call :close_tunnel
  exit /b 0
)

ssh -o BatchMode=yes %DGX% "cd %DGX_REPO% && bash scripts/dgx/start_live.sh start!ARGS!"
if errorlevel 1 exit /b 1

call :tunnel_pid
if defined TUNNEL_PID (
  echo tunnel already open ^(pid !TUNNEL_PID!^)
) else (
  start "live tunnel" /min ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 8025:127.0.0.1:8025 -L 8026:127.0.0.1:8026 -L 8027:127.0.0.1:8027 -L 8890:127.0.0.1:8890 -L 8190:127.0.0.1:8190 %DGX%
  timeout /t 3 /nobreak >nul
  echo tunnel opened ^(racing 8025, castle 8026, ancient-egypt 8027^)
)
if "%BROWSER%"=="1" start "" "%URL%"
echo %SCENE% page: %URL%
echo Switcher: scripts\wsl\start_live.bat switcher
exit /b 0

:tunnel_pid
set "TUNNEL_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:"127.0.0.1:8025 .*LISTENING"') do (
  for /f "tokens=1" %%N in ('tasklist /FI "PID eq %%P" /NH ^| findstr /I "ssh.exe"') do set "TUNNEL_PID=%%P"
)
exit /b 0

:close_tunnel
call :tunnel_pid
if defined TUNNEL_PID (
  taskkill /PID !TUNNEL_PID! >nul
  echo closed the tunnel ^(pid !TUNNEL_PID!^)
) else (
  echo no tunnel to close
)
exit /b 0
