$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CondaPython = $null

if ($env:CONDA_PREFIX) {
    $Candidate = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $Candidate) {
        $CondaPython = $Candidate
    }
}

if ($CondaPython) {
    $Python = $CondaPython
    Write-Host "Using active conda Python: $Python"
} else {
    if (!(Test-Path $VenvPython)) {
        Write-Host "未找到虚拟环境，正在创建..."
        & (Join-Path $PSScriptRoot "setup_venv.ps1")
    }
    $Python = $VenvPython
    Write-Host "Using project venv Python: $Python"
}

Set-Location $ProjectRoot
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
