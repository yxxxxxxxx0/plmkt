<#
Registers PolymarketSlateDaily, the scheduled task that ticks the overnight
recorder's watchdog (run_slate_daily.ps1).

S4U DOES NOT WORK ON THIS MACHINE, and this script no longer pretends it might.
The account is AzureAD\JustinCHENG, a cloud-only Entra ID identity (SID prefix
S-1-12-1). Task Scheduler's S4U logon goes through LogonUserS4U, which a
cloud-only account cannot satisfy: on 2026-09-22 at 10:08:39 an elevated
registration produced event 104, "Task Scheduler failed to log on. Failure
occurred in LogonUserS4U", error 0x80070520 (1312, "a specified logon session
does not exist"), and the task then would not run at all. That is also the real
explanation for the 2026-09-16/17 history the old scripts recorded as "S4U
silently refused to launch" and blamed on a missing SeBatchLogonRight -- the
account already held that right on 09-22 and S4U still failed.

So the default is Interactive, which is verified to work. Pass -TryS4U to
attempt S4U anyway; if it fails the script falls straight back and re-verifies.

Two things this script gets right that its predecessors did not:

  Note writes to the host, not the pipeline.
      fix_recorder_task.ps1 and the first version of this script both built
      Note on Write-Output. Inside a function every Note line becomes part of
      that function's return value, so `if (Test-Recorder)` tested a non-empty
      array -- always true. On 2026-09-22 that made this script print
      "VERIFIED: the watchdog ran, and S4U means it runs with nobody logged on"
      about a task that could not log on at all, and leave it broken.
      Write-Host keeps the return value a bare boolean.

  The task is never left in a state that has not been proven to run.
      Everything after registration sits in try/finally, and the finally block
      re-checks and repairs. Interrupting this script with Ctrl+C cannot strand
      a dead task.

Proof of execution is the only thing trusted here: note the length of
logs\watchdog.log, start the task, and wait for the watchdog to append a line.
Task Scheduler's own LastRunTime and LastTaskResult are not evidence -- they
advance for runs that never launched.

Three settings are load-bearing and were wrong before 2026-09-22:

  StopAtDurationEnd = false   With it true, Task Scheduler terminated the task
                              instance at the end of each repetition window
                              (event 111 at 13:30 on 09-19, 09-20 and 09-21),
                              orphaning the collector and costing the 09-22
                              slate. Safe to leave open-ended now: a tick
                              finishes in about two seconds.

  ExecutionTimeLimit = 30 min The old PT20H existed because the wrapper hosted
                              the whole slate. It does not any more, so a tick
                              still alive after 30 minutes is hung and should
                              be killed. Killing it cannot reach a recording.

  Priority = 4                Tasks default to priority 7 (below normal), which
                              the collector inherits. A websocket recorder
                              keeping up with a full order book should not be
                              scheduled behind everything else.

Run elevated to also get the at-startup trigger, which needs administrator
rights to register; without it nothing starts the recorder after an unattended
reboot until someone logs in.
#>
[CmdletBinding()]
param(
    # Attempt LogonType=S4U before falling back. Off by default: see above.
    [switch]$TryS4U
)

$ErrorActionPreference = 'Stop'
$base = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$name = 'PolymarketSlateDaily'
$logs = Join-Path $base 'logs'
$wlog = Join-Path $logs 'watchdog.log'
$slog = Join-Path $logs 'register_task.log'
New-Item -ItemType Directory -Force $logs | Out-Null

function Note($m) {
    # Write-Host, never Write-Output: inside a function Write-Output would be
    # captured into the return value and silently corrupt every boolean test.
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"
    Add-Content -Path $slog -Value $line -Encoding utf8
    Write-Host $line
}

$elevated = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$sid = ([Security.Principal.NTAccount]"$env:USERDOMAIN\$env:USERNAME").Translate(
           [Security.Principal.SecurityIdentifier]).Value
$cloudAccount = $sid.StartsWith('S-1-12-1')   # Entra ID / Azure AD authority

Note "=== registering $name ==="
Note "  account $env:USERDOMAIN\$env:USERNAME  ($sid)"
Note "  elevated=$elevated  cloud-account=$cloudAccount"

function Register-Recorder([string]$logonType) {
    $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$base\run_slate_daily.ps1`"" `
        -WorkingDirectory $base

    # One open-ended repeating trigger rather than a daily window. There is no
    # hour at which it becomes right to stop asking whether the recorder is up:
    # an HKT slate can start as early as 00:20 and runs up to 14 hours.
    # -RepetitionInterval must be passed at construction, or the trigger has no
    # Repetition object to assign to.
    $tick = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
        -RepetitionInterval (New-TimeSpan -Minutes 20) `
        -RepetitionDuration (New-TimeSpan -Days 3650)
    $tick.Repetition.Duration = ''          # empty means indefinitely
    $tick.Repetition.StopAtDurationEnd = $false

    $triggers = @($tick)
    # An AtStartup trigger can only be registered by an administrator: without
    # elevation the whole Register-ScheduledTask call fails 0x80070005.
    if ($elevated) {
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
    # Returns a bare $true/$false. Ground truth is whether run_slate_daily.ps1
    # actually appended a line to its log -- nothing else counts as evidence.
    [OutputType([bool])]
    param()
    $t0 = Get-Date
    $before = if (Test-Path $wlog) { (Get-Item $wlog).Length } else { -1 }
    try { Start-ScheduledTask -TaskName $name }
    catch { Note "  Start-ScheduledTask threw: $($_.Exception.Message)"; return $false }

    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        $now = if (Test-Path $wlog) { (Get-Item $wlog).Length } else { -1 }
        if ($now -gt $before) {
            Note "  watchdog wrote: $(Get-Content $wlog -Tail 1 -Encoding utf8)"
            return $true
        }
        # A logon failure is reported at once; do not sit out the full wait.
        $bad = @(Get-WinEvent -FilterHashtable @{
                    LogName   = 'Microsoft-Windows-TaskScheduler/Operational'
                    Id        = @(101, 103, 104, 323, 331)
                    StartTime = $t0
                 } -ErrorAction SilentlyContinue |
                 Where-Object { $_.Message -match [regex]::Escape($name) })
        if ($bad.Count -gt 0) {
            Note "  Task Scheduler refused it: $(($bad[0].Message -split "`r?`n")[0])"
            return $false
        }
    }
    Note "  no watchdog output after 45s"
    return $false
}

$final = 'BROKEN'
try {
    if ($TryS4U) {
        if ($cloudAccount) {
            Note "WARNING: -TryS4U on a cloud-only account. LogonUserS4U cannot"
            Note "         satisfy an Entra ID identity; this is expected to fail."
        }
        try {
            Note "registering as S4U"
            Register-Recorder 'S4U'
            if (Test-Recorder) { $final = 'S4U' } else { Note "S4U does not execute here" }
        } catch {
            Note "  S4U registration failed: $($_.Exception.Message)"
        }
    } else {
        Note "skipping S4U (cloud-only account cannot use it; -TryS4U forces an attempt)"
    }

    if ($final -ne 'S4U') {
        Note "registering as Interactive"
        Register-Recorder 'Interactive'
        if (Test-Recorder) { $final = 'Interactive' }
    }
} finally {
    # Never leave a task that has not been proven to run -- including when this
    # script is interrupted part-way through an S4U attempt.
    if ($final -eq 'BROKEN') {
        Note "repairing: last known-good configuration is Interactive"
        try {
            Register-Recorder 'Interactive'
            if (Test-Recorder) { $final = 'Interactive (repaired)' }
        } catch { Note "  repair failed: $($_.Exception.Message)" }
    }
}

if ($elevated) { try { & wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true } catch { } }

$t = Get-ScheduledTask -TaskName $name
$i = Get-ScheduledTaskInfo -TaskName $name
Note "--- result ---"
Note ("  LogonType          : {0}" -f $t.Principal.LogonType)
Note ("  repetition         : every {0}, duration '{1}', StopAtDurationEnd={2}" -f $t.Triggers[0].Repetition.Interval, $t.Triggers[0].Repetition.Duration, $t.Triggers[0].Repetition.StopAtDurationEnd)
Note ("  triggers           : {0}" -f (($t.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task', '' }) -join ', '))
Note ("  ExecutionTimeLimit : {0}   Priority: {1}" -f $t.Settings.ExecutionTimeLimit, $t.Settings.Priority)
Note ("  NextRunTime        : {0}" -f $i.NextRunTime)

if ($final -like 'BROKEN*') {
    Note "FAILED: the task does not execute the watchdog. Investigate before tonight."
} else {
    Note "VERIFIED by watchdog.log growing: the task runs. Principal $final."
    if ($final -like 'Interactive*') {
        Note "  Interactive survives a LOCKED screen, but not a logoff."
    }
    if (-not $elevated) {
        Note "  MISSING (needs elevation): the at-startup trigger. After an"
        Note "  unattended reboot nothing starts the recorder until someone logs in."
    }
}
Note "=== done ==="
