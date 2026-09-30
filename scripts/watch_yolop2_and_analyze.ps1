param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"

$candidate = Join-Path $RunDir "weights\best.pt"
$analysisDir = Join-Path $RunDir "analysis"
$watchLog = Join-Path $ProjectRoot "logs\yolop2_post_analysis_watch.log"

function Write-WatchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $watchLog -Value $line -Encoding UTF8
}

Write-WatchLog "Watcher started. pid=$ProcessId run=$RunDir"

while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

Write-WatchLog "Training process exited. Waiting for best.pt: $candidate"

$deadline = (Get-Date).AddMinutes(20)
while (-not (Test-Path $candidate)) {
    if ((Get-Date) -gt $deadline) {
        Write-WatchLog "Timed out waiting for best.pt."
        exit 2
    }
    Start-Sleep -Seconds 15
}

Write-WatchLog "best.pt found. Starting analysis."

$script = Join-Path $ProjectRoot "scripts\analyze_yolop2_model.py"
$baseline = Join-Path $ProjectRoot "models\v04_negative_1024_best.pt"
$data = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"

& $Python $script `
    --candidate $candidate `
    --baseline $baseline `
    --data $data `
    --output $analysisDir `
    --imgsz 1024 `
    --batch 8 `
    --conf 0.45 `
    --device 0

if ($LASTEXITCODE -ne 0) {
    Write-WatchLog "Analysis failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-WatchLog "Analysis finished: $analysisDir\analysis.md"
