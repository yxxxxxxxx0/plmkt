# Make the nightly recorder run with no human present at all.
#
# History of this task, because the failure mode repeated:
#
#   2026-09-15  registered with LogonType=InteractiveToken. Windows Update
#               rebooted the box at 19:30, nobody logged back in, the 05:50
#               trigger could not fire, the whole slate was lost.
#   2026-09-16  "fixed" by switching to LogonType=S4U, which runs without a
#               session. Registration succeeded and was never tested. S4U also
#               requires the account to hold SeBatchLogonRight; it did not, so
#               the task silently refused to launch and 09-17 was lost too.
#               Task Scheduler reports this as "has not run" rather than as an
#               error, which is why it left no trail.
#
# So this script does both halves and then PROVES the result:
#   1. grant SeBatchLogonRight to the account (this is what S4U was missing)
#   2. re-register the task as S4U
#   3. actually start it, confirm a process spawns, then stop it
#   4. if S4U still cannot execute, fall back to Interactive automatically
#      rather than leaving a task that looks registered but never runs
#
# Needs elevation for step 1. Everything is written to logs\fix_recorder.log.

$ErrorActionPreference = 'Stop'
$base = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$log  = Join-Path $base 'logs\fix_recorder.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Note($m) { "$((Get-Date).ToString('HH:mm:ss'))  $m" | Tee-Object -FilePath $log -Append }

Note "=== starting ==="
$acct = "$env:USERDOMAIN\$env:USERNAME"
$sid  = ([System.Security.Principal.NTAccount]$acct).Translate(
            [System.Security.Principal.SecurityIdentifier]).Value
Note "account $acct  sid $sid"

# ---- 1. grant the batch logon right -------------------------------------
$work = Join-Path $env:TEMP 'plmkt_secpol'
New-Item -ItemType Directory -Force $work | Out-Null
$inf = Join-Path $work 'cur.inf'
$new = Join-Path $work 'new.inf'
$db  = Join-Path $work 'sec.sdb'
& secedit /export /areas USER_RIGHTS /cfg $inf | Out-Null

$lines = Get-Content $inf
$had = $false
$out = foreach ($l in $lines) {
    if ($l -match '^SeBatchLogonRight\s*=') {
        $had = $true
        if ($l -match [regex]::Escape($sid)) { Note "SeBatchLogonRight already includes this account"; $l }
        else { Note "adding account to SeBatchLogonRight"; "$l,*$sid" }
    } else { $l }
}
if (-not $had) {
    Note "SeBatchLogonRight absent, creating it"
    $out = $out -replace '^\[Privilege Rights\]$', "[Privilege Rights]`r`nSeBatchLogonRight = *$sid"
}
$out | Set-Content $new -Encoding Unicode
& secedit /configure /db $db /cfg $new /areas USER_RIGHTS | Out-Null
Note "secedit applied (exit $LASTEXITCODE)"

# ---- 2/3. register and PROVE it runs ------------------------------------
function Register-Recorder([string]$logonType) {
    $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$base\run_slate_daily.ps1`"" `
        -WorkingDirectory $base
    $trg = New-ScheduledTaskTrigger -Daily -At '22:30'
    $trg.Repetition = (New-ScheduledTaskTrigger -Once -At '22:30' `
        -RepetitionInterval (New-TimeSpan -Minutes 30) `
        -RepetitionDuration (New-TimeSpan -Hours 14)).Repetition
    $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 20) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $prn = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType $logonType -RunLevel Limited
    Register-ScheduledTask -TaskName 'PolymarketSlateDaily' -Action $act -Trigger $trg `
        -Settings $set -Principal $prn -Force | Out-Null
}

function Test-Recorder {
    Start-ScheduledTask -TaskName 'PolymarketSlateDaily'
    Start-Sleep -Seconds 15
    $p = Get-CimInstance Win32_Process -Filter "Name like '%python%' or Name='powershell.exe'" |
         Where-Object { $_.CommandLine -match 'run_slate_daily|collect_days' }
    $ran = (Get-ScheduledTaskInfo -TaskName 'PolymarketSlateDaily').LastRunTime.Year -gt 2000
    Stop-ScheduledTask -TaskName 'PolymarketSlateDaily' -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name like '%python%' or Name='powershell.exe'" |
        Where-Object { $_.CommandLine -match 'run_slate_daily|collect_days' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    return ($ran -and $p)
}

Note "registering as S4U"
Register-Recorder 'S4U'
if (Test-Recorder) {
    Note "S4U EXECUTES -- the recorder now runs with nobody logged on"
    $final = 'S4U'
} else {
    Note "S4U still cannot execute; falling back to Interactive"
    Register-Recorder 'Interactive'
    $final = if (Test-Recorder) { 'Interactive (works, but needs a logged-on session)' }
             else { 'NEITHER WORKS -- investigate' }
    Note "fallback result: $final"
}

# ---- 4. turn on the diagnostic log so a future failure leaves a trail ----
try { & wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true; Note "task scheduler operational log enabled" }
catch { Note "could not enable operational log: $($_.Exception.Message)" }

$i = Get-ScheduledTaskInfo -TaskName 'PolymarketSlateDaily'
Note "FINAL: principal=$final  NextRunTime=$($i.NextRunTime)"
Note "=== done ==="
