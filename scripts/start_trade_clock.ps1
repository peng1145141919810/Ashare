param(
    [string]$Profile = "quick_test",
    [string]$ExecutionMode = "",
    [ValidateSet("default", "on", "off")]
    [string]$PrecisionTrade = "default"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localSettings = Join-Path $repoRoot "quant_research_hub_v6_repacked_clean\quant_research_hub_v6_repacked_clean\hub_v6\local_settings.py"
$serviceScript = Join-Path $repoRoot "trade_clock_service.py"
$tradeClockRoot = Join-Path $repoRoot "data\trade_clock"
$pidPath = Join-Path $tradeClockRoot "clock_supervisor.pid"
$stdoutPath = Join-Path $tradeClockRoot "clock_supervisor.stdout.log"
$stderrPath = Join-Path $tradeClockRoot "clock_supervisor.stderr.log"

New-Item -ItemType Directory -Force -Path $tradeClockRoot | Out-Null

if (Test-Path $pidPath) {
    $existingPid = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid) {
        $proc = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Output "Trade clock already running. PID=$existingPid"
            exit 0
        }
    }
    Remove-Item $pidPath -ErrorAction SilentlyContinue
}

$content = Get-Content $localSettings -Raw -Encoding UTF8
$match = [regex]::Match($content, '(?m)^PYTHON_EXECUTABLE\s*=\s*r?["'']([^"'']+)["'']')
if (-not $match.Success) {
    throw "Unable to resolve PYTHON_EXECUTABLE from local_settings.py"
}
$python = $match.Groups[1].Value.Trim()
if (-not (Test-Path $python)) {
    throw "Resolved Python executable does not exist: $python"
}

$args = @(
    "`"$serviceScript`"",
    "--profile", $Profile,
    "--skip-preflight"
)
if ($ExecutionMode) {
    $args += @("--execution-mode", $ExecutionMode)
}
if ($PrecisionTrade -ne "default") {
    $args += @("--precision-trade", $PrecisionTrade)
}

$proc = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
Set-Content -Path $pidPath -Value $proc.Id -Encoding UTF8

Write-Output "Started trade clock. PID=$($proc.Id) Profile=$Profile"
Write-Output "Stdout: $stdoutPath"
Write-Output "Stderr: $stderrPath"
