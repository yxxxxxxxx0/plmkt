# Registers a task that runs a keep-alive .bat at logon and every 20 minutes (the .bat exits if its process is up).
# Same settings as register_daemon_task.ps1: no time limit, Interactive logon (S4U is unavailable on this account).
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Bat)
$bat = Join-Path $PSScriptRoot $Bat
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $PSScriptRoot
$t1 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 20)
$set = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$pr = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $t1, $t2 -Settings $set -Principal $pr -Force | Select-Object TaskName, State
