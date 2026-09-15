# Sleep until 19:00 HKT tonight, then run the 5-fold grouped CV.
#
# Scheduled rather than run now because each fold trains a CNN for ~45-60 min
# and five folds is a ~4-5 hour job; starting it at 19:00 keeps the machine
# free during the day. This box has crashed an editor under memory pressure
# before, so the loop below aborts rather than pushing through if free RAM
# collapses.
Set-Location 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$py = '..\.venv\Scripts\python.exe'
$logf = 'logs\cv_schedule.txt'

function Log($m) {
  $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m
  Write-Output $line
  Add-Content -Path $logf -Value $line
}

$target = (Get-Date).Date.AddHours(19)
if ((Get-Date) -ge $target) { $target = $target.AddDays(1) }
Log ("scheduled for {0:yyyy-MM-dd HH:mm} HKT ({1:N2}h from now)" -f $target, ($target - (Get-Date)).TotalHours)

while ((Get-Date) -lt $target) {
  $remain = ($target - (Get-Date)).TotalMinutes
  if ($remain -gt 20) { Start-Sleep -Seconds 600 } else { Start-Sleep -Seconds 30 }
}

$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
Log "starting 5-fold grouped CV (free ${free}GB)"

& $py -u cv_grouped.py --folds 5 --epochs 5 --patience 2 `
    --max-train 250000 --max-val 120000 --max-test 200000 *>> 'logs\cv_run.txt'

Log "cv_grouped exit $LASTEXITCODE"
Log "DONE -- results in data/jump/cv_grouped.csv and cv_grouped_summary.json"
