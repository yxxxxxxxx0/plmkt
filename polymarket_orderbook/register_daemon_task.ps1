# Registers the TwoVenueRecorderDaemon task: starts run_session_daemon.bat at logon and every 20 min.
# The bat exits at once if a daemon is already running, so the 20-minute tick only restarts a dead one.
# No execution time limit: the default 72 h limit would kill the long-running daemon.
# Interactive logon: survives a locked screen, not a logoff (S4U is unavailable for this Entra ID account; see README).
$bat = Join-Path $PSScriptRoot 'run_session_daemon.bat'
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $PSScriptRoot
$t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Minutes 20)
$set = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$pr = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName 'TwoVenueRecorderDaemon' -Action $action -Trigger $t1, $t2 -Settings $set -Principal $pr `
    -Description 'Polymarket+Kalshi two-venue recorder (run_session.py --daemon)' -Force | Select-Object TaskName, State
