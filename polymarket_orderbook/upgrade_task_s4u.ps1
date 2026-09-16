# Re-registers PolymarketSlateDaily with LogonType=S4U. Needs elevation:
# registering an S4U task requires SeTcbPrivilege, so this must run as admin.
#
# Why S4U matters: the previous task used InteractiveToken, which only runs
# while the user is logged on. On 2026-09-15 Windows Update force-rebooted the
# box at 19:30 and nobody logged back in, so the 05:50 trigger never fired and
# the whole slate was lost. S4U runs whether or not anyone is logged on, and
# stores no password.
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$log  = Join-Path $base 'logs\s4u_upgrade.log'
function Note($m) { Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m"; Write-Output $m }
try {
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
    $prn = New-ScheduledTaskPrincipal -UserId 'JustinCHENG' -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName 'PolymarketSlateDaily' -Action $act -Trigger $trg `
        -Settings $set -Principal $prn -Force | Out-Null
    $lt = (Get-ScheduledTask -TaskName 'PolymarketSlateDaily').Principal.LogonType
    $nr = (Get-ScheduledTaskInfo -TaskName 'PolymarketSlateDaily').NextRunTime
    Note "SUCCESS: LogonType=$lt  NextRunTime=$nr"
} catch {
    Note "FAILED: $($_.Exception.Message)"
    exit 1
}
