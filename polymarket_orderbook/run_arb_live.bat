@echo off
REM Keeps research\venue_jumps\arb_live.py running; exits at once if it already is.
REM PYTHONUTF8: titles with curly quotes or accents crash a cp950 console print, which is the likely
REM reason the first run died silently after 8 minutes.
REM Register: powershell -File register_watch_task.ps1 -Name ArbLiveWatcher -Bat run_arb_live.bat
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONUTF8=1
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? CommandLine -match 'arb_live.py') { exit 1 }"
if errorlevel 1 exit /b 0
python -u research\venue_jumps\arb_live.py >> "logs\arb_live.log" 2>&1
