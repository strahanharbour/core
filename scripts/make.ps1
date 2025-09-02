param(
  [ValidateSet('venv','data','features','backtest','report','test','all')]
  [string]$Target = 'all'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$venvPy = Join-Path $repoRoot '.venv\Scripts\python.exe'

function Ensure-Venv {
  if (-not (Test-Path $venvPy)) {
    Write-Host 'Bootstrapping virtual env via scripts/dev_env.ps1'
    & (Join-Path $scriptDir 'dev_env.ps1')
  }
}

switch ($Target) {
  'venv' {
    & (Join-Path $scriptDir 'dev_env.ps1')
    return
  }
  default { Ensure-Venv }
}

$env:PYTHONPATH = "$repoRoot\src\main"
Write-Host "PYTHONPATH=$($env:PYTHONPATH)"

switch ($Target) {
  'data'      { & $venvPy -m data.ingest_daily_local; break }
  'features'  { & $venvPy -m data.build_features; break }
  'backtest'  { & $venvPy -m research.backtest; break }
  'report'    { & $venvPy -m research.report; break }
  'test'      { & $venvPy -m pytest -q; break }
  'all'       {
    & $venvPy -m data.ingest_daily_local
    & $venvPy -m data.build_features
    & $venvPy -m research.backtest
    & $venvPy -m research.report
    break
  }
}

