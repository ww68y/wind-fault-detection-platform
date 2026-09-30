param(
    [string]$C0RunDir = "",
    [string]$Python = "",
    [int]$PollSeconds = 60,
    [int]$C0ExpectedEpochs = 50,
    [string]$Device = "0",
    [int]$Batch = 8,
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogRoot = Join-Path $ProjectRoot "logs"
$StatusLog = Join-Path $LogRoot "small_eval_after_c0_$Stamp.status.log"
$SmallEvalScript = Join-Path $ProjectRoot "scripts\run_small_object_subset_eval.ps1"

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

if ([string]::IsNullOrWhiteSpace($C0RunDir)) {
    $C0Root = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\c0_yolov8n_320_baseline"
    $latest = Get-ChildItem -Path $C0Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "c0_vanilla_yolov8n_320_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No C0 run directory found under $C0Root"
    }
    $C0RunDir = $latest.FullName
}

if (-not (Test-Path $C0RunDir)) {
    throw "C0 run dir not found: $C0RunDir"
}
if (-not (Test-Path $SmallEvalScript)) {
    throw "Small-object eval script not found: $SmallEvalScript"
}

$C0Results = Join-Path $C0RunDir "results.csv"
$C0Best = Join-Path $C0RunDir "weights\best.pt"
$C0Analysis = Join-Path $C0RunDir "v6_vs_c0_analysis\analysis.md"
$Output = Join-Path $ProjectRoot "paper_assets_msf_yolo\small_object_eval_$Stamp"

Write-Status "Small-object eval watcher started."
Write-Status "C0 run: $C0RunDir"
Write-Status "Expected C0 epochs: $C0ExpectedEpochs"
Write-Status "Output: $Output"

while ($true) {
    $epoch = Get-LatestEpoch -ResultsCsv $C0Results
    $hasBest = Test-Path $C0Best
    $hasAnalysis = Test-Path $C0Analysis
    Write-Status "C0 progress: epoch=$epoch/$C0ExpectedEpochs, best.pt=$hasBest, analysis=$hasAnalysis"
    if ($epoch -ge $C0ExpectedEpochs -and $hasBest -and $hasAnalysis) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Status "C0 complete. Starting small-object subset evaluation."

$evalArgs = @{
    Stamp = $Stamp
    Output = $Output
    Device = $Device
    Batch = $Batch
    IncludeC0 = $true
}
if (-not [string]::IsNullOrWhiteSpace($Python)) {
    $evalArgs.Python = $Python
}

& $SmallEvalScript @evalArgs

if ($LASTEXITCODE -ne 0) {
    Write-Status "Small-object eval failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status "Small-object eval finished."
