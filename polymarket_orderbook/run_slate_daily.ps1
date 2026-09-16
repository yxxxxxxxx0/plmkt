# Launches the next HKT slate. Fired by the PolymarketSlateDaily task.
#
# Timing: an HKT slate date spans roughly 00:00-14:00 HKT, and its first pitch
# moves a lot -- 2026-09-16 opened at 06:40 HKT but 2026-09-17 opens at 01:10.
# collect_days.py works the real first pitch out from the MLB schedule and
# sleeps until 45 min before it, so the task only has to fire safely *before*
# the earliest pitch any slate can have. 22:30 HKT clears the earliest
# realistic first pitch (~00:05 HKT) with margin.
#
# collect_days.py refuses to start a slate more than an hour late ("recording a
# partial slate is worse than skipping it"), so firing late does not degrade
# the recording -- it loses the whole night. Hence the margin.
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$py   = 'C:\Users\JustinCHENG\Documents\plmkt\.venv\Scripts\python.exe'
Set-Location $base
New-Item -ItemType Directory -Force (Join-Path $base 'logs') | Out-Null

# --days 1 means "tomorrow's HKT date", which at 22:30 is the slate about to start.
$slate = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
$log   = Join-Path $base "logs\collect_$slate.log"
function Note($m) { Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }

# The trigger repeats every 30 min as a reboot watchdog: if Windows restarts
# mid-slate, the next tick restarts the collector. When one is already up the
# tick is a no-op, so the repetition never stacks two recorders on one slate.
$running = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
    Where-Object { $_.CommandLine -match 'collect_days\.py|live_recorder\.py|run_slate\.py' }
if ($running) {
    Note "tick: collector already running (PID $($running.ProcessId -join ',')), nothing to do"
    exit 0
}

Note "tick: starting collect_days --days 1 (slate $slate)"
& $py -u (Join-Path $base 'collect_days.py') --days 1 --duration-hours 14 *>> $log
Note "collect_days exited $LASTEXITCODE"
exit $LASTEXITCODE
