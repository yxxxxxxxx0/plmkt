@echo off
REM Rebuild every recorded slate into the viewer, keeping the by_game splits.
REM
REM Run via Task Scheduler, not from an interactive shell: this takes hours and
REM a shell-owned job dies with the session that started it.
REM
REM --skip-uniform : the viewer builds from by_game directly, so data/uniform
REM                  would cost ~55 min and ~5 GB per slate for nothing.
REM --keep-cache   : keep data/live/by_game so analysis can re-run without a
REM                  30-45 min re-split per slate. Costs ~145 GB.
cd /d "%~dp0"
if not exist logs mkdir logs
"D:\plmkt\.venv\Scripts\python.exe" -u rebuild_all_slates.py --skip-uniform --keep-cache >> "logs\rebuild_viewer.log" 2>&1
