<#
.SYNOPSIS
  Lanza UNA instancia del juego para sondear el mapa LVL_AMLO y guarda su PID en salida\pid.txt.

.DESCRIPTION
  La carpeta del juego se obtiene con dmbridge.rutas (variable DARKMEXICO_DIR o Steam). La carpeta
  de salida (registros y canal de comandos cmd.lua) es DMPROBE_DIR; por defecto tools\sondeo\salida.
  El mod DarkMexicoProbeAmlo solo actua si se pasa -Activo (variable DMP_AMLO=1).
  -Cpus limita la instancia a un rango de CPUs, por ejemplo "20-27", para no competir con otras
  tareas; sin el parametro no se fija afinidad.

.EXAMPLE
  .\tools\sondeo\probe_amlo\lanzar_sondeo.ps1 -Activo -PasoFijo
#>
param(
    [switch]$Activo,
    [switch]$Ventana,
    [switch]$PasoFijo,
    [string]$Cpus = ""
)
$ErrorActionPreference = "Stop"
$raiz = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
$juego = (& $python -c "from dmbridge.rutas import buscar_carpeta_juego; print(buscar_carpeta_juego())").Trim()
$dir = Join-Path $juego "MexicoSouls\Binaries\Win64"
$exe = Join-Path $dir "MexicoSouls-Win64-Shipping.exe"
if (-not (Test-Path $exe)) { Write-Error "No se encontro el juego en $juego. Define DARKMEXICO_DIR."; exit 1 }

$salida = if ($env:DMPROBE_DIR) { $env:DMPROBE_DIR } else { Join-Path $raiz "tools\sondeo\salida" }
New-Item -ItemType Directory -Force -Path $salida | Out-Null
$env:DMPROBE_DIR = $salida
$pidFile = Join-Path $salida "pid.txt"

Remove-Item Env:DMB_TRAINING -ErrorAction SilentlyContinue
Remove-Item Env:DMB_ID -ErrorAction SilentlyContinue
if ($Activo) { $env:DMP_AMLO = "1" } else { Remove-Item Env:DMP_AMLO -ErrorAction SilentlyContinue }
$env:SteamAppId = "4346760"
$env:SteamGameId = "4346760"

$argv = @("-nosplash", "-unattended", "-nosound")
if ($Ventana) { $argv += @("-windowed", "-ResX=640", "-ResY=360") } else { $argv += @("-nullrhi", "-corelimit=2") }
if ($PasoFijo) { $argv += "-UseFixedTimeStep" }

$p = Start-Process -FilePath $exe -ArgumentList ($argv -join " ") -WorkingDirectory $dir -PassThru -WindowStyle Minimized
Start-Sleep -Milliseconds 300
$p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
if ($Cpus -match "^(\d+)-(\d+)$") {
    $mascara = 0
    foreach ($c in [int]$Matches[1]..[int]$Matches[2]) { $mascara = $mascara -bor (1 -shl $c) }
    $p.ProcessorAffinity = [IntPtr]$mascara
}
Set-Content -Path $pidFile -Value $p.Id
"PID=$($p.Id) args=$($argv -join ' ') DMP_AMLO=$($env:DMP_AMLO) salida=$salida"
