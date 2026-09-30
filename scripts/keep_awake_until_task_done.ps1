param(
    [Parameter(Mandatory = $true)]
    [int]$TaskPid,
    [int]$MaxHours = 14,
    [string]$LogPath = "runs\wind_fault\logs\keep_awake.log"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$LogFullPath = Join-Path $ProjectRoot $LogPath
$LogDir = Split-Path -Parent $LogFullPath
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class TrainingKeepAwake {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"
$KeepAwakeFlags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
$ReleaseFlags = $ES_CONTINUOUS
$Deadline = (Get-Date).AddHours($MaxHours)

function Write-KeepAwakeLog {
    param([string]$Message)
    "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message |
        Tee-Object -FilePath $LogFullPath -Append
}

try {
    Write-KeepAwakeLog "Keep-awake started for PID $TaskPid, max hours $MaxHours."
    while ((Get-Date) -lt $Deadline) {
        $process = Get-Process -Id $TaskPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-KeepAwakeLog "Task PID $TaskPid has exited."
            break
        }
        [TrainingKeepAwake]::SetThreadExecutionState($KeepAwakeFlags) | Out-Null
        Start-Sleep -Seconds 60
    }
}
finally {
    [TrainingKeepAwake]::SetThreadExecutionState($ReleaseFlags) | Out-Null
    Write-KeepAwakeLog "Keep-awake released."
}
