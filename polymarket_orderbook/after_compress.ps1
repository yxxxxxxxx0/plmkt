# Wait for compress_raw.py to finish, then re-trim with the pregame buffer and
# retrain. Chained rather than run in parallel: xz -T0 saturates every core, and
# this machine has already crashed an editor under combined load.
Set-Location 'C:\Users\JustinCHENG\Documents\plmkt\polymarket_orderbook'
$py = '..\.venv\Scripts\python.exe'

function Log($m) {
  $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m
  Write-Output $line
  Add-Content -Path 'logs\after_compress.txt' -Value $line
}

Log "waiting for compression to finish..."
while ($true) {
  $busy = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
          Where-Object { $_.CommandLine -like '*compress_raw*' }
  if (-not $busy) { break }
  Start-Sleep -Seconds 30
}
Log "compression finished"
Start-Sleep -Seconds 20   # let xz's page cache drain before allocating again

# --- re-trim with the 120s pregame buffer, one session at a time -----------
foreach ($s in @('2026-08-28','2026-08-30','2026-09-10','2026-09-11')) {
  $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
  Log "re-trim $s (free ${free}GB)"
  & $py -u trim_sessions.py --sessions $s --force *>> 'logs\after_compress.txt'
  Log "re-trim $s exit $LASTEXITCODE"
}

# --- retrain with the same anti-overfitting settings -----------------------
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)
Log "retraining (free ${free}GB)"
& $py -u taker_signal.py --label-ticks 4 --min-backing 200 `
    --max-train 300000 --max-val 150000 --max-test 250000 `
    --epochs 6 --patience 2 *>> 'logs\retrain2_out.txt'
Log "retrain exit $LASTEXITCODE"

# --- per-game P&L and capital on the new model ----------------------------
& $py -u pnl_by_game.py --model cnn_direction *>> 'logs\after_compress.txt'
Log "pnl_by_game exit $LASTEXITCODE"
& $py -u capital_required.py --model cnn_direction --stake 5 *>> 'logs\after_compress.txt'
Log "capital exit $LASTEXITCODE"
Log "ALL DONE"
