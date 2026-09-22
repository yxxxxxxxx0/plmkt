<#
Registers PolymarketSlateDaily, the scheduled task that ticks the overnight
recorder's watchdog (run_slate_daily.ps1).

Replaces upgrade_task_s4u.ps1 and fix_recorder_task.ps1, which disagreed with
each other and both reported success they had not earned. fix_recorder_task
logged "S4U EXECUTES -- the recorder now runs with nobody logged on" on
2026-09-21 even though the secedit call two lines earlier had exited 740
(ERROR_ELEVATION_REQUIRED) without granting anything, because its proof that
the task ran was `LastRunTime.Year -gt 2000` -- true of any task that has ever
run, including one that had just refused to start. The task actually found
registered on 2026-09-22 was Interactive, not S4U.

This script proves execution the one way that cannot be faked: it notes the
length of logs\watchdog.log, starts the task, and waits for the watchdog to
append a line. If no line appears the task did not run, whatever Task
Scheduler's own fields say.

Three settings here are load-bearing and were wrong before:

  StopAtDurationEnd = false   With it true, Task Scheduler terminated the task
                              instance at the end of each repetition window --
                              event 111 at 13:30 on 09-19, 09-20 and 09-21.
                              That is what orphaned the collector and cost the
                              2026-09-22 slate. Leaving the repetition
                              open-ended is safe now, because a tick finishes
                              in about two seconds.

  ExecutionTimeLimit = 30 min The old PT20H existed because the wrapper hosted
                              the whole slate. It does not any more -- the
                              collector is detached -- so a tick still alive
                              after 30 minutes is hung and should be killed.
                              Killing it can no longer reach a recording.

  Priority = 4                Scheduled tasks default to priority 7 (below
                              normal), which the collector inherits. A
                              websocket recorder that has to keep up with a
                              full order book should not be scheduled behind
                              everything else on the machine.

Run elevated to get LogonType=S4U, which runs with nobody logged on. Without
elevation the task is registered Interactive, which survives a locked screen
but not a logoff or an unattended reboot -- and the script says so plainly
rather than claiming otherwise.
#>
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$name = 'PolymarketSlateDaily'
$logs = Join-Path $base 'logs'
$wlog = Join-Path $logs 'watchdog.log'
$slog = Join-Path $logs 'register_task.log'
New-Item -ItemType Directory -Force $logs | Out-Null
function Note($m) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    Add-Content -Path $slog -Value $line -Encoding utf8
    Write-Output $line
}

$elevated = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Note "=== registering $name (elevated: $elevated) ==="

# ---- grant SeBatchLogonRight, which is what S4U actually needs ------------
function Grant-BatchLogon {
    $acct = "$env:USERDOMAIN\$env:USERNAME"
    $sid = ([Security.Principal.NTAccount]$acct).Translate(
               [Security.Principal.SecurityIdentifier]).Value
    $work = Join-Path $env:TEMP 'plmkt_secpol'
    New-Item -ItemType Directory -Force $work | Out-Null
    $cur = Join-Path $work 'cur.inf'
    $new = Join-Path $work 'new.inf'
    $db  = Join-Path $work 'sec.sdb'
    & secedit /export /areas USER_RIGHTS /cfg $cur | Out-Null
    if ($LASTEXITCODE -ne 0) { Note "  secedit /export failed (exit $LASTEXITCODE)"; return $false }
    $had = $false
    $out = foreach ($l in (Get-Content $cur)) {
        if ($l -match '^SeBatchLogonRight\s*=') {
            $had = $true
            if ($l -match [regex]::Escape($sid)) { Note "  $acct already holds SeBatchLogonRight"; $l }
            else { Note "  adding $acct to SeBatchLogonRight"; "$l,*$sid" }
        } else { $l }
    }
    if (-not $had) {
        Note "  SeBatchLogonRight absent; creating it"
        $out = $out -replace '^\[Privilege Rights\]$', "[Privilege Rights]`r`nSeBatchLogonRight = *$sid"
    }
    $out | Set-Content $new -Encoding Unicode
    & secedit /configure /db $db /cfg $new /areas USER_RIGHTS | Out-Null
    if ($LASTEXITCODE -ne 0) { Note "  secedit /configure FAILED (exit $LASTEXITCODE)"; return $false }
    Note "  secedit /configure ok"
    return $true
}

function Register-Recorder([string]$logonType) {
    $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$base\run_slate_daily.ps1`"" `
        -WorkingDirectory $base

    # One open-ended repeating trigger rather than a daily window. There is no
    # hour at which it becomes right to stop asking whether the recorder is up:
    # an HKT slate can start as early as 00:20 and runs up to 14 hours.
    # -RepetitionInterval must be passed at construction: without it the
    # trigger has no Repetition object at all to assign to.
    $tick = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
        -RepetitionInterval (New-TimeSpan -Minutes 20) `
        -RepetitionDuration (New-TimeSpan -Days 3650)
    $tick.Repetition.Duration = ''          # empty means indefinitely
    $tick.Repetition.StopAtDurationEnd = $false

    # Cover the two ways the box comes back with nobody doing anything: a
    # reboot (Windows Update did exactly this on 2026-09-15) and a logon.
    # An AtStartup trigger can only be registered by an administrator -- the
    # whole Register-ScheduledTask call fails 0x80070005 without it -- so
    # unelevated we register the logon trigger alone and say what is missing.
    $triggers = @($tick)
    if ($script:elevated) {
        $boot = New-ScheduledTaskTrigger -AtStartup
        $boot.Delay = 'PT2M'
        $triggers += $boot
    }
    $logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $logon.Delay = 'PT2M'
    $triggers += $logon

    $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $set.Priority = 4
    $set.RunOnlyIfIdle = $false
    $set.IdleSettings.StopOnIdleEnd = $false

    $prn = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType $logonType -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $act -Trigger $triggers `
        -Settings $set -Principal $prn -Force | Out-Null
}

function Test-Recorder {
    # Ground truth: did run_slate_daily.ps1 actually append a line?
    $before = if (Test-Path $wlog) { (Get-Item $wlog).Length } else { -1 }
    try { Start-ScheduledTask -TaskName $name }
    catch { Note "  Start-ScheduledTask threw: $($_.Exception.Message)"; return $false }
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        $now = if (Test-Path $wlog) { (Get-Item $wlog).Length } else { -1 }
        if ($now -gt $before) {
            Note "  watchdog wrote: $(Get-Content $wlog -Tail 1 -Encoding utf8)"
            return $true
        }
    }
    $info = Get-ScheduledTaskInfo -TaskName $name
    Note ("  no watchdog output after 60s (LastRunTime {0}, LastTaskResult 0x{1:X8})" -f $info.LastRunTime, $info.LastTaskResult)
    return $false
}

# Always try S4U first, whatever our privilege level: registering it has been
# observed to succeed unelevated, and the real test below -- did the watchdog
# actually append a line? -- is the only thing that settles whether it runs.
if ($elevated) {
    if (-not (Grant-BatchLogon)) { Note "  continuing; S4U may not execute without that right" }
} else {
    Note "not elevated: cannot grant SeBatchLogonRight. Trying S4U anyway."
}

$final = 'BROKEN'
try {
    Note "registering as S4U"
    Register-Recorder 'S4U'
    if (Test-Recorder) { $final = 'S4U' }
} catch {
    Note "  S4U registration failed: $($_.Exception.Message)"
}
if ($final -ne 'S4U') {
    Note "S4U does not run here; falling back to Interactive"
    Register-Recorder 'Interactive'
    if (Test-Recorder) { $final = 'Interactive' }
}

if (-not $script:elevated) {
    Note "MISSING (needs elevation): the at-startup trigger. After an unattended"
    Note "         reboot nothing starts the recorder until someone logs in."
}

try { & wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true } catch { }

$t = Get-ScheduledTask -TaskName $name
$i = Get-ScheduledTaskInfo -TaskName $name
Note "--- result ---"
Note ("  LogonType          : {0}" -f $t.Principal.LogonType)
Note ("  repetition         : every {0}, duration '{1}', StopAtDurationEnd={2}" -f $t.Triggers[0].Repetition.Interval, $t.Triggers[0].Repetition.Duration, $t.Triggers[0].Repetition.StopAtDurationEnd)
Note ("  triggers           : {0}" -f (($t.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task', '' }) -join ', '))
Note ("  ExecutionTimeLimit : {0}   Priority: {1}" -f $t.Settings.ExecutionTimeLimit, $t.Settings.Priority)
Note ("  NextRunTime        : {0}" -f $i.NextRunTime)
if ($final -eq 'S4U') {
    Note "VERIFIED: the watchdog ran, and S4U means it runs with nobody logged on."
} elseif ($final -eq 'Interactive') {
    Note "VERIFIED: the watchdog ran. Interactive survives a LOCKED screen, but NOT"
    Note "          a logoff or an unattended reboot."
    if (-not $elevated) {
        Note "          Re-run this script from an ELEVATED PowerShell to grant"
        Note "          SeBatchLogonRight and retry S4U, which closes that gap."
    } else {
        Note "          S4U was tried with the batch-logon right granted and still"
        Note "          did not execute; this machine's policy is blocking it."
    }
} else {
    Note "FAILED: the task did not execute the watchdog. Investigate before tonight."
}
Note "=== done ==="
