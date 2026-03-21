$ErrorActionPreference = "Stop"

$taskName = "Ashare Trade Clock"
$startScript = Join-Path $PSScriptRoot "start_trade_clock.ps1"
$taskArgument = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startScript`""
$userId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $taskArgument
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null

Write-Output "Installed trade clock autostart task: $taskName"
Write-Output "Command: powershell.exe $taskArgument"
Write-Output "Behavior: logon autostart, hidden launcher, ignore duplicate instances, restart on failure every 1 minute."
