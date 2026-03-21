$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$tradeClockRoot = Join-Path $repoRoot "data\trade_clock"
$pidPath = Join-Path $tradeClockRoot "clock_supervisor.pid"

if (-not (Test-Path $pidPath)) {
    Write-Output "Trade clock PID file not found."
    exit 0
}

$pidValue = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
if (-not $pidValue) {
    Remove-Item $pidPath -ErrorAction SilentlyContinue
    Write-Output "Trade clock PID file was empty."
    exit 0
}

$proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id ([int]$pidValue) -Force
    Write-Output "Stopped trade clock. PID=$pidValue"
} else {
    Write-Output "Trade clock process not running. PID=$pidValue"
}

Remove-Item $pidPath -ErrorAction SilentlyContinue
