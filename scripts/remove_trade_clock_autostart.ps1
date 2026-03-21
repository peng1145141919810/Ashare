$ErrorActionPreference = "Stop"

$taskName = "Ashare Trade Clock"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "Removed trade clock autostart task: $taskName"
} else {
    Write-Output "Trade clock autostart task not found: $taskName"
}
