@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set PY="C:\Users\JustinCHENG\AppData\Local\Python\pythoncore-3.14-64\python.exe"
%PY% -u compress_raw.py data/live/books_2026-08-28.jsonl data/live/books_2026-08-29.jsonl data/live/books_2026-08-30.jsonl >> "logs\compress_raw.log" 2>&1
