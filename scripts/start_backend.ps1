$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    Write-Host "未找到虚拟环境，正在创建..."
    & (Join-Path $PSScriptRoot "setup_venv.ps1")
}

Set-Location $ProjectRoot
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
