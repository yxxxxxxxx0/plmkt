@echo off
REM Windows launcher for the two-venue daemon (run_session.py --daemon).
REM PYTHONUTF8: the console codepage (cp950) cannot encode accented team
REM names, and every child recorder inherits this so their logs don't crash.
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONUTF8=1
"D:\plmkt\.venv\Scripts\python.exe" -u run_session.py --daemon %* >> "logs\run_session_daemon.log" 2>&1
