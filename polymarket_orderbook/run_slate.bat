@echo off
REM Unattended slate launcher for Task Scheduler.
REM   %1 = HKT date (YYYY-MM-DD)   %2 = duration guard in hours
cd /d "%~dp0"
if not exist logs mkdir logs
"C:\Users\JustinCHENG\AppData\Local\Python\pythoncore-3.14-64\python.exe" run_slate.py --hkt-date %1 --duration-hours %2 >> "logs\scheduler_%1.log" 2>&1
