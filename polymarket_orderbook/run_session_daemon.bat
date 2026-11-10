@echo off
REM Windows launcher for the two-venue daemon (run_session.py --daemon).
REM PYTHONUTF8: the console codepage (cp950) cannot encode accented team
REM names, and every child recorder inherits this so their logs don't crash.
REM Uses the venv if present, else python on PATH (needs requirements.txt installed).
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONUTF8=1
set PY=python
if exist "D:\plmkt\.venv\Scripts\python.exe" set PY="D:\plmkt\.venv\Scripts\python.exe"
REM One daemon at a time: skip if one is already running.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? CommandLine -match 'run_session.py --daemon') { exit 1 }"
if errorlevel 1 exit /b 0
%PY% -u run_session.py --daemon %* >> "logs\run_session_daemon.log" 2>&1
