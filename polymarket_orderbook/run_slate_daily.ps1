# Watchdog tick for the overnight recorder. Fired by PolymarketSlateDaily
# every 20 minutes, around the clock.
#
# This script does almost nothing and exits in about two seconds. That is the
# point, and it is a deliberate change from the version that ran until
# 2026-09-22.
#
# What went wrong before, in order:
#
#   The wrapper used to *host* the collector: it ran `& python collect_days.py`
#   and blocked for the whole slate, so the Task Scheduler instance stayed
#   alive for 17+ hours. The task's repetition had StopAtDurationEnd = true, so
#   Task Scheduler terminated that instance every afternoon (event 111 at
#   13:30 on 09-19, 09-20 and 09-21). Terminating the instance killed the
#   PowerShell host but left the Python collector running as an orphan --
#   still asleep, waiting for a first pitch 17 hours away, with its stdout
#   pipe now owned by a dead process.
#
#   The orphan then poisoned the night twice over. Every tick from 22:30 to
#   05:30 saw a live `collect_days.py` process and logged "already running,
#   nothing to do", so the real slate never started. And at 05:50, when the
#   orphan finally woke to record, its first print() hit the dead pipe, raised,
#   and killed it silently. The 2026-09-22 slate was lost with no error
#   anywhere.
#
#   The 06:00 tick did notice the collector was gone -- and then started the
#   wrong slate, because it computed the date as (today + 1 day). At 06:00 the
#   slate that needed rescuing was today's.
#
# So: the collector is now started detached and owns its own log file and
# heartbeat (logs/collector_state.json), the scheduled task no longer stops
# anything at a duration end, liveness is judged by the heartbeat rather than
# by "a pid exists", and the slate is chosen by collect_days.py --auto.

$ErrorActionPreference = 'Stop'
$base  = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$py    = 'C:\Users\JustinCHENG\Documents\plmkt\.venv\Scripts\python.exe'
$logs  = Join-Path $base 'logs'
$log   = Join-Path $logs 'watchdog.log'
$state = Join-Path $logs 'collector_state.json'

# A collector that has not beaten in this long is wedged, not working.
$staleMinutes = 5
# ...but give one that has only just launched time to write its first beat.
$startupGraceMinutes = 3

New-Item -ItemType Directory -Force $logs | Out-Null
function Note($m) {
    # Add-Content only. The old script mixed Add-Content with a `*>>` stream
    # redirect, which PowerShell 5.1 writes as UTF-16 -- so half of every log
    # came out as mojibake and the logs were unreadable exactly when they
    # mattered.
    Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" -Encoding utf8
}
if ((Test-Path $log) -and (Get-Item $log).Length -gt 2MB) {
    Move-Item $log "$log.1" -Force
}

function Get-Collector {
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -match 'collect_days\.py' }
}
function Get-Recorder {
    # run_slate.py counts: after live_recorder.py exits it still has the
    # settle, game-window fetch and compression to do, and starting a second
    # collector on top of that would fight it for the same files.
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -match 'live_recorder\.py|run_slate\.py' }
}

# ---- 1. a recording in progress is never interrupted --------------------
# This also covers a slate started by hand from a terminal, which must not be
# killed just because it is not the collector's child.
$rec = @(Get-Recorder)
if ($rec.Count -gt 0) {
    Note "recording in progress (PID $($rec.ProcessId -join ',')); leaving it alone"
    # A slate that is still compressing when the next one is due would be
    # skipped here, silently. Compression has always finished with ~10 hours to
    # spare, so this stays conservative -- starting a second run_slate would
    # have the two of them rewriting matches.py underneath each other -- but it
    # must not be silent, because a lost slate is unrecoverable.
    $live = @($rec | Where-Object { $_.CommandLine -match 'live_recorder\.py' })
    if ($live.Count -eq 0) {
        Note "  NOTE: no live_recorder, so this is post-processing (settle/windows/compress)."
        Note "  NOTE: if a slate is due within the next hour or two it will be missed."
    }
    exit 0
}

# ---- 2. is a collector waiting for first pitch, and is it actually alive? -
$col = @(Get-Collector)
if ($col.Count -gt 0) {
    $beat = $null
    if (Test-Path $state) {
        try { $beat = Get-Content $state -Raw -Encoding utf8 | ConvertFrom-Json } catch { }
    }
    $age = $null
    if ($beat -and $beat.heartbeat_at) {
        $age = ((Get-Date) - [datetime]::Parse($beat.heartbeat_at)).TotalMinutes
    }
    $youngest = ($col | Measure-Object -Property CreationDate -Minimum).Minimum
    $justStarted = $youngest -and (((Get-Date) - $youngest).TotalMinutes -lt $startupGraceMinutes)

    if ($justStarted) {
        Note "collector PID $($col.ProcessId -join ',') just launched; leaving it alone"
        exit 0
    }
    if ($null -ne $age -and $age -lt $staleMinutes) {
        Note ("collector PID {0} healthy (slate {1}, {2}, beat {3:N1} min ago)" -f `
              ($col.ProcessId -join ','), $beat.slate, $beat.phase, $age)
        exit 0
    }
    $why = if ($null -eq $age) { 'no heartbeat file' } else { "last beat {0:N1} min ago" -f $age }
    Note "collector PID $($col.ProcessId -join ',') is WEDGED ($why); killing it"
    foreach ($p in $col) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
        catch { Note "  could not kill $($p.ProcessId): $($_.Exception.Message)" }
    }
    Start-Sleep -Seconds 2
}

# ---- 3. start one, detached ----------------------------------------------
# Detached on purpose: the collector must outlive this PowerShell process and
# the Task Scheduler instance around it, so that whatever Task Scheduler does
# to the task -- stop, timeout, duration end -- cannot reach a live recording.
$out = Join-Path $logs 'collector_stdout.log'
$err = Join-Path $logs 'collector_stderr.log'
$argv = @('-u', (Join-Path $base 'collect_days.py'), '--auto', '--duration-hours', '14')
try {
    $p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $base `
         -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
} catch {
    # The redirect targets can be locked by a collector that is on its way out.
    Note "redirected launch failed ($($_.Exception.Message)); launching without redirect"
    $p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $base `
         -WindowStyle Hidden -PassThru
}
Note "started collect_days --auto, PID $($p.Id)"
exit 0
