@echo off
REM Unattended slate recording. Registered with Task Scheduler so it is owned
REM by the scheduler and survives the interactive session that created it
REM ending -- recording is the one step in this pipeline that cannot be redone.
REM
REM Usage:  run_slate.bat <hkt-date> <duration-hours>
cd /d "%~dp0"
if not exist logs mkdir logs
"D:\plmkt\.venv\Scripts\python.exe" -u run_slate.py --hkt-date %1 --duration-hours %2 >> "logs\slate_%1_task.log" 2>&1
