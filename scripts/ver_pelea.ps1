<#
.SYNOPSIS
  Abre una ventana del juego a velocidad normal y deja pelear al modelo (o al bot de reglas).

.DESCRIPTION
  Una sola instancia CON ventana (sin -nullrhi), paso fijo de 1/30 s limitado a 30 fps: la pelea
  transcurre en tiempo real con la misma dinamica del entrenamiento. Cierra el juego al terminar.
  -MaxSegundos limita el tiempo real (para comprobar que abre y cierra).

  Con -Multijefe -Jefe <juice|amlo|chamuco|nahual> el modelo runs\<Nombre>\mejor.zip (por defecto
  multijefe_ppo; con -Imitacion, multijefe_imitacion) pelea contra ese jefe en una ventana.

  Con -Multijefe -Panel se abre ademas un panel web en vivo (http://127.0.0.1:<PanelPuerto>/) que muestra
  las neuronas, la intencion, la confianza y la atencion de la red; solo escucha en 127.0.0.1.
  Ver docs\02-ver-pelear.md.

.EXAMPLE
  .\scripts\ver_pelea.ps1
  .\scripts\ver_pelea.ps1 -Politica bot -MaxSegundos 20
  .\scripts\ver_pelea.ps1 -Imitacion                  # el modelo clonado del bot (runs\imitacion\mejor.zip)
  .\scripts\ver_pelea.ps1 -Multijefe -Jefe amlo
#>
param(
    [ValidateSet("modelo", "bot")][string]$Politica = "modelo",
    [string]$Nombre = "principal",
    [string]$Modelo = "",
    [int]$Episodios = 1,
    [double]$MaxSegundos = 0,
    [switch]$Imitacion,
    [switch]$Multijefe,
    [ValidateSet("juice", "amlo", "chamuco", "nahual")][string]$Jefe = "amlo",
    [switch]$Panel,
    [ValidateRange(1024, 65535)][int]$PanelPuerto = 8765,
    [switch]$SinNavegador
)

$ErrorActionPreference = "Stop"
if ($Multijefe) {
    if (-not $PSBoundParameters.ContainsKey("Nombre")) {
        $Nombre = if ($Imitacion) { "multijefe_imitacion" } else { "multijefe_ppo" }
    }
    if ($Politica -eq "bot") { Write-Error "-Politica bot no esta disponible con -Multijefe."; exit 1 }
} elseif ($Imitacion) { $Nombre = "imitacion" }
if ($Panel -and -not $Multijefe) { Write-Error "-Panel solo esta disponible con -Multijefe (observacion v2)."; exit 1 }
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
if ($Multijefe) {
    $argumentos = @("-m", "dmrl", "ver-jefes", "--jefe", $Jefe, "--nombre", $Nombre, "--episodios", $Episodios)
} else {
    $argumentos = @("-m", "dmrl", "ver", "--politica", $Politica, "--nombre", $Nombre, "--episodios", $Episodios)
}
if ($Panel) {
    if (Get-NetTCPConnection -State Listen -LocalPort $PanelPuerto -ErrorAction SilentlyContinue) {
        Write-Error "El puerto $PanelPuerto ya esta en uso; elige otro con -PanelPuerto."; exit 1
    }
    $argumentos += @("--panel", "--panel-puerto", $PanelPuerto)
    if ($SinNavegador) { $argumentos += "--sin-navegador" }
}
if ($Modelo -ne "") { $argumentos += @("--modelo", $Modelo) }
if ($MaxSegundos -gt 0) { $argumentos += @("--max-segundos", $MaxSegundos) }

Set-Location $raiz
$env:PYTHONUNBUFFERED = "1"
& $python @argumentos
exit $LASTEXITCODE
