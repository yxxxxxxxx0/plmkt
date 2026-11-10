@echo off
REM Keeps reward_quoter.py running: exits at once if it already is, so a scheduled tick only restarts a dead one.
REM Register (every 20 min + at logon):  powershell -File register_watch_task.ps1 -Name RewardQuoter -Bat run_reward_quoter.bat
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONUTF8=1
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? CommandLine -match 'reward_quoter.py') { exit 1 }"
if errorlevel 1 exit /b 0
"%~dp0..\.venv\Scripts\python.exe" -u reward_quoter.py >> "logs\reward_quoter.log" 2>&1
