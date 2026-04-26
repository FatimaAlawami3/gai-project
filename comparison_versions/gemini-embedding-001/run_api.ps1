$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$backendDir = Join-Path $projectRoot "backend"
$env:RAG_ENV_FILE = Join-Path $PSScriptRoot "settings.env"

Push-Location $backendDir
try {
    python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
}
finally {
    Pop-Location
}
