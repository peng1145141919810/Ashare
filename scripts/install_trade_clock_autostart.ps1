$ErrorActionPreference = "Stop"

$taskName = "Ashare Trade Clock"
Write-Output "Trade clock daily scheduler no longer supports logon autostart."
Write-Output "Use scripts\\start_trade_clock.ps1 for manual start and scripts\\stop_trade_clock.ps1 for manual stop."
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Output "A legacy scheduled task still exists: $taskName"
    Write-Output "If you want to clean it up, run scripts\\remove_trade_clock_autostart.ps1"
}
