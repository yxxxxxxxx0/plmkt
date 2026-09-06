@echo off
REM Build the research panel for every split game. Scheduler-owned because it
REM runs ~70 min and shell-owned jobs on this machine get killed.
cd /d "%~dp0"
if not exist logs mkdir logs
"D:\plmkt\.venv\Scripts\python.exe" -u build_panel.py >> "logs\build_panel.log" 2>&1
