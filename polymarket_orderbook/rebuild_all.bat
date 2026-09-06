@echo off
REM Rebuild every slate into the viewer. Run via Task Scheduler so it is owned
REM by the scheduler service and survives any interactive session ending.
cd /d "%~dp0"
if not exist logs mkdir logs
"D:\plmkt\.venv\Scripts\python.exe" -u rebuild_all_slates.py --skip-uniform >> "logs\rebuild_all.log" 2>&1
