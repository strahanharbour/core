param([switch]$DryRun=$true)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = (Resolve-Path "$root\src\main").Path
$python = (Resolve-Path "$root\.venv\Scripts\python.exe")
if ($DryRun) { & $python -m engine.entrypoint --dry-run } else { & $python -m engine.entrypoint }

