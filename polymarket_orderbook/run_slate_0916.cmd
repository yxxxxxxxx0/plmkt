@echo off
cd /d "C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook"
"C:\Users\JustinCHENG\Documents\plmkt\.venv\Scripts\python.exe" -u collect_days.py --hkt-dates 2026-09-16 --duration-hours 14 >> "data\live\collect_2026-09-16.log" 2>&1
