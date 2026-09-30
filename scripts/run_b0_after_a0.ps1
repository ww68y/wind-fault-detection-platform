param(
    [string]$A0RunDir = "",
    [string]$Python = "",
    [int]$PollSeconds = 60,
    [int]$A0ExpectedEpochs = 80,
    [int]$B0Epochs = 50,
    [int]$B0Batch = 16,
    [string]$Device = "0",
    [int]$Workers = 0,
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "b0_after_a0_wait_$Stamp.status.log"
$B0Script = Join-Path $ProjectRoot "scripts\run_b0_yolov8n_416_baseline.ps1"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $StatusLog -Value $line -Encoding UTF8
}

function Get-LatestEpoch {
    param([string]$ResultsCsv)
    if (-not (Test-Path $ResultsCsv)) {
        return 0
    }
    $rows = Import-Csv $ResultsCsv
    if (-not $rows -or $rows.Count -eq 0) {
        return 0
    }
    return [int]$rows[-1].epoch
}

if ([string]::IsNullOrWhiteSpace($A0RunDir)) {
    $A0Root = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\a0_vanilla_yolo_baseline"
    $latest = Get-ChildItem -Path $A0Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "a0_vanilla_yolov8n_640_b16_80e_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No A0 run directory found under $A0Root"
    }
    $A0RunDir = $latest.FullName
}

if (-not (Test-Path $A0RunDir)) {
    throw "A0 run dir not found: $A0RunDir"
}
if (-not (Test-Path $B0Script)) {
    throw "B0 script not found: $B0Script"
}

$A0Results = Join-Path $A0RunDir "results.csv"
$A0Best = Join-Path $A0RunDir "weights\best.pt"

Write-Status "B0-after-A0 watcher started."
Write-Status "A0 run: $A0RunDir"
Write-Status "A0 expected epochs: $A0ExpectedEpochs"
Write-Status "B0 epochs: $B0Epochs"
Write-Status "Poll seconds: $PollSeconds"

while ($true) {
    $epoch = Get-LatestEpoch -ResultsCsv $A0Results
    $hasBest = Test-Path $A0Best
    Write-Status "A0 progress: epoch=$epoch/$A0ExpectedEpochs, best.pt=$hasBest"
    if ($epoch -ge $A0ExpectedEpochs -and $hasBest) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Status "A0 training appears complete. Starting B0."

$b0Args = @{
    Stamp = $Stamp
    Epochs = $B0Epochs
    Batch = $B0Batch
    Device = $Device
    Workers = $Workers
}
if (-not [string]::IsNullOrWhiteSpace($Python)) {
    $b0Args.Python = $Python
}

& $B0Script @b0Args

if ($LASTEXITCODE -ne 0) {
    Write-Status "B0 script failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status "B0-after-A0 watcher finished."
