<#
Create .venv (optionally recreate), install requirements, and set PYTHONPATH.

Usage:
  powershell -ExecutionPolicy Bypass -File scripts/dev_env.ps1 [-Recreate]
#>

param(
  [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

# Find Python
$pythonCmd = $null
foreach ($cand in @('py','python3','python')) {
  if (Get-Command $cand -ErrorAction SilentlyContinue) { $pythonCmd = $cand; break }
}
if (-not $pythonCmd) { throw 'Python not found. Install Python 3.8+ and ensure it is on PATH.' }

$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if ($Recreate -and (Test-Path $venvPath)) {
  Write-Host "Removing existing venv at $venvPath"
  Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPython)) {
  Write-Host "Creating virtual environment at $venvPath"
  if ($pythonCmd -eq 'py') { & $pythonCmd -3 -m venv $venvPath } else { & $pythonCmd -m venv $venvPath }
}

# Always upgrade pip/setuptools/wheel to regenerate console scripts with correct paths
& $venvPython -m pip install -U pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $repoRoot 'requirements.txt')

# Set PYTHONPATH to src/main relative to current working directory
$env:PYTHONPATH = "$((Get-Location).Path)\src\main"
Write-Host "PYTHONPATH=$($env:PYTHONPATH)"
& $venvPython --version
