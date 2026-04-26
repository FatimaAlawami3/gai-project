$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$backendDir = Join-Path $projectRoot "backend"
$env:RAG_ENV_FILE = Join-Path $PSScriptRoot "settings.env"

Push-Location $backendDir
try {
    python .\build_vector_store.py
}
finally {
    Pop-Location
}
