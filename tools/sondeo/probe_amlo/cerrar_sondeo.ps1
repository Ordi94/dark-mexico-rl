# Cierra solo la instancia lanzada por lanzar_sondeo.ps1 (PID guardado en salida\pid.txt).
$raiz = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$salida = if ($env:DMPROBE_DIR) { $env:DMPROBE_DIR } else { Join-Path $raiz "tools\sondeo\salida" }
$pidFile = Join-Path $salida "pid.txt"
if (-not (Test-Path $pidFile)) { "sin pid.txt"; exit 0 }
$id = [int](Get-Content $pidFile)
$p = Get-Process -Id $id -ErrorAction SilentlyContinue
if ($p -and $p.ProcessName -eq "MexicoSouls-Win64-Shipping") { Stop-Process -Id $id -Force; "cerrado PID=$id" } else { "PID=$id ya no existe" }
