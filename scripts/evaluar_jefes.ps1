<#
.SYNOPSIS
  Juega N peleas completas contra un jefe (Sheinbaum, AMLO, Chamuco o Chupacabras) con el bot de reglas.

.DESCRIPTION
  Entorno multi-jefe (observacion v2): cada instancia del juego queda fija al mapa del jefe y reinicia
  la pelea en sitio, sin recargar el mapa (recarga completa cada 150 episodios como respaldo).
  Imprime tasa de victoria, dano, parries, caguamas, duracion y las muertes por ataque, y guarda el
  detalle en runs\multijefe\evaluacion_<jefe>.json. No guarda partida ni logros (mod en modo
  entrenamiento). Prioridad Below Normal y CPUs 0-3 libres.

.EXAMPLE
  .\scripts\evaluar_jefes.ps1 -Jefe amlo -Episodios 50
  .\scripts\evaluar_jefes.ps1 -Jefe amlo -Episodios 20 -FaseInicio 2    # Chamuco por transicion natural
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("juice", "amlo", "chamuco", "nahual")][string]$Jefe,
    [int]$Episodios = 50,
    [int]$Instancias = 4,
    [int]$Semilla = 0,
    [ValidateSet(0, 1, 2)][int]$FaseInicio = 0,
    [double]$EpsilonRuido = 0.0
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
$salida = Join-Path $raiz "runs\multijefe\evaluacion_$Jefe.json"
$argumentos = @("-m", "dmrl", "evaluar-jefes", "--jefe", $Jefe, "--episodios", $Episodios,
    "--instancias", $Instancias, "--semilla", $Semilla, "--epsilon-ruido", $EpsilonRuido, "--salida", $salida)
if ($FaseInicio -ne 0) { $argumentos += @("--fase-inicio", $FaseInicio) }

$entrenando = Get-Content (Join-Path $raiz "runs\entrenamiento.pid") -ErrorAction SilentlyContinue
if ($entrenando -and (Get-Process -Id ([int]$entrenando) -ErrorAction SilentlyContinue)) {
    Write-Warning "Hay un entrenamiento en marcha; la evaluacion compite por CPU con el."
}

Set-Location $raiz
$env:PYTHONUNBUFFERED = "1"
& $python @argumentos
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Detalle: $salida"
