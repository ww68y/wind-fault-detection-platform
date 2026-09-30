param(
    [string]$Python = "",
    [string]$Output = "",
    [string]$Device = "0",
    [int]$Batch = 8,
    [double]$Conf = 0.45,
    [double[]]$Thresholds = @(0.01, 0.005),
    [switch]$IncludeC0,
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $ExternalGpuPython = "E:\Desktop\training_bundle_v04_negative\.venv\Scripts\python.exe"
    $ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $ExternalGpuPython) {
        $Python = $ExternalGpuPython
    } elseif (Test-Path $ProjectPython) {
        $Python = $ProjectPython
    } else {
        $Python = "python"
    }
}

$EvalScript = Join-Path $ProjectRoot "scripts\evaluate_small_object_subset.py"
$DataYaml = Join-Path $ProjectRoot "datasets\wind_blade_defect_v05_conservative\data.yaml"
$V6Best = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\msf_yolo_full_innovation\msf_full_p2_shape_hard_finetune_60e_amp_b12_from_warmup_20260606_182744\weights\best.pt"
$A0Best = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\a0_vanilla_yolo_baseline\a0_vanilla_yolov8n_640_b16_80e_20260608_211330\weights\best.pt"
$B0Best = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\b0_yolov8n_416_baseline\b0_vanilla_yolov8n_416_b16_50e_20260609_120034\weights\best.pt"
$C0Best = Join-Path $ProjectRoot "runs\detect\runs\wind_fault\c0_yolov8n_320_baseline\c0_vanilla_yolov8n_320_b16_50e_20260609_125334\weights\best.pt"

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "paper_assets_msf_yolo\small_object_eval_$Stamp"
}

if (-not (Test-Path $EvalScript)) { throw "Eval script not found: $EvalScript" }
if (-not (Test-Path $DataYaml)) { throw "Data yaml not found: $DataYaml" }
if (-not (Test-Path $V6Best)) { throw "V6 best.pt not found: $V6Best" }
if (-not (Test-Path $A0Best)) { throw "A0 best.pt not found: $A0Best" }
if (-not (Test-Path $B0Best)) { throw "B0 best.pt not found: $B0Best" }

$argsList = @(
    $EvalScript,
    "--data", $DataYaml,
    "--output", $Output,
    "--reference", "V6",
    "--batch", "$Batch",
    "--conf", "$Conf",
    "--device", $Device,
    "--splits", "val", "test",
    "--thresholds"
)
foreach ($threshold in $Thresholds) {
    $argsList += "$threshold"
}

$argsList += @(
    "--model", "V6", $V6Best, "1024",
    "--model", "A0", $A0Best, "640",
    "--model", "B0", $B0Best, "416"
)

if ($IncludeC0) {
    if (-not (Test-Path $C0Best)) { throw "C0 best.pt not found: $C0Best" }
    $argsList += @("--model", "C0", $C0Best, "320")
}

& $Python @argsList

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
