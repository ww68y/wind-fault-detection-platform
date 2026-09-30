param(
    [string]$B0RunDir = "",
    [string]$Python = "",
    [int]$PollSeconds = 60,
    [int]$B0ExpectedEpochs = 50,
    [int]$C0Epochs = 50,
    [int]$C0Batch = 16,
    [string]$Device = "0",
    [int]$Workers = 0,
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "c0_after_b0_wait_$Stamp.status.log"
$C0Script = Join-Path $ProjectRoot "scripts\run_c0_yolov8n_320_baseline.ps1"

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

if ([string]::IsNullOrWhiteSpace($B0RunDir)) {
    $B0Root = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\b0_yolov8n_416_baseline"
    $latest = Get-ChildItem -Path $B0Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "b0_vanilla_yolov8n_416_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No B0 run directory found under $B0Root"
    }
    $B0RunDir = $latest.FullName
}

if (-not (Test-Path $B0RunDir)) {
    throw "B0 run dir not found: $B0RunDir"
}
if (-not (Test-Path $C0Script)) {
    throw "C0 script not found: $C0Script"
}

$B0Results = Join-Path $B0RunDir "results.csv"
$B0Best = Join-Path $B0RunDir "weights\best.pt"
$B0Analysis = Join-Path $B0RunDir "v6_vs_b0_analysis\analysis.md"

Write-Status "C0-after-B0 watcher started."
Write-Status "B0 run: $B0RunDir"
Write-Status "B0 expected epochs: $B0ExpectedEpochs"
Write-Status "C0 epochs: $C0Epochs"
Write-Status "C0 batch: $C0Batch"
Write-Status "Poll seconds: $PollSeconds"

while ($true) {
    $epoch = Get-LatestEpoch -ResultsCsv $B0Results
    $hasBest = Test-Path $B0Best
    $hasAnalysis = Test-Path $B0Analysis
    Write-Status "B0 progress: epoch=$epoch/$B0ExpectedEpochs, best.pt=$hasBest, analysis=$hasAnalysis"
    if ($epoch -ge $B0ExpectedEpochs -and $hasBest -and $hasAnalysis) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Status "B0 training and analysis appear complete. Starting C0."

$c0Args = @{
    Stamp = $Stamp
    Epochs = $C0Epochs
    Batch = $C0Batch
    Device = $Device
    Workers = $Workers
}
if (-not [string]::IsNullOrWhiteSpace($Python)) {
    $c0Args.Python = $Python
}

& $C0Script @c0Args

if ($LASTEXITCODE -ne 0) {
    Write-Status "C0 script failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status "C0-after-B0 watcher finished."
