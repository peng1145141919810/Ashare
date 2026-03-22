$ErrorActionPreference = "Stop"

$taskName = "Ashare Trade Clock"
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "Removed legacy trade clock autostart task: $taskName"
} else {
    Write-Output "Legacy trade clock autostart task not found: $taskName"
}
