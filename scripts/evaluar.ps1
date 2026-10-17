<#
.SYNOPSIS
  Juega N peleas completas (jefe al 100% de vida) y reporta la tasa de victoria.

.DESCRIPTION
  Con -Politica modelo usa runs\<Nombre>\mejor.zip (o el ultimo checkpoint, o -Modelo). Con
  -Politica bot usa el bot de reglas de referencia. Las peleas se reparten entre -Instancias
  instancias del juego sin ventana. Escribe el detalle en runs\<Nombre>\evaluacion_<politica>.json.

  Con -Multijefe evalua runs\<Nombre>\mejor.zip (por defecto multijefe_ppo; con -Imitacion,
  multijefe_imitacion) contra uno o varios jefes: -Jefe juice, amlo, chamuco, nahual o todos.
  -Episodios son las peleas POR JEFE. -Politica bot juega el bot de reglas con el mismo protocolo.
  Detalle en runs\<Nombre>\evaluacion_multijefe.json.

.EXAMPLE
  .\scripts\evaluar.ps1 -Politica bot -Episodios 50
  .\scripts\evaluar.ps1 -Politica modelo -Episodios 100
  .\scripts\evaluar.ps1 -Imitacion -Episodios 50      # runs\imitacion\mejor.zip (modelo clonado del bot)
  .\scripts\evaluar.ps1 -Multijefe -Jefe todos -Episodios 20
#>
param(
    [ValidateSet("bot", "modelo", "azar")][string]$Politica = "modelo",
    [string]$Nombre = "principal",
    [string]$Modelo = "",
    [int]$Episodios = 50,
    [int]$Instancias = 4,
    [int]$Semilla = 0,
    [switch]$Imitacion,
    [switch]$Multijefe,
    [ValidateSet("todos", "juice", "amlo", "chamuco", "nahual")][string]$Jefe = "todos"
)

$ErrorActionPreference = "Stop"
if ($Multijefe) {
    if (-not $PSBoundParameters.ContainsKey("Nombre")) {
        $Nombre = if ($Imitacion) { "multijefe_imitacion" } else { "multijefe_ppo" }
    }
} elseif ($Imitacion) { $Nombre = "imitacion" }
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
if ($Multijefe) {
    if ($Politica -eq "azar") { Write-Error "-Politica azar no esta disponible con -Multijefe (usa modelo o bot)."; exit 1 }
    $salida = Join-Path $raiz "runs\$Nombre\evaluacion_multijefe.json"
    $jefes = if ($Jefe -eq "todos") { @("juice", "amlo", "chamuco", "nahual") } else { @($Jefe) }
    $argumentos = @("-m", "dmrl", "evaluar-modelo-jefes", "--jefes") + $jefes + @("--episodios", $Episodios,
        "--instancias", $Instancias, "--semilla", $Semilla, "--prefijo", "evm", "--salida", $salida)
    if ($Politica -eq "bot") {
        $argumentos += "--bot"
    } else {
        if ($Modelo -eq "") { $Modelo = Join-Path $raiz "runs\$Nombre\mejor.zip" }
        $argumentos += @("--modelo", $Modelo)
    }
} else {
    $salida = Join-Path $raiz "runs\$Nombre\evaluacion_$Politica.json"
    $argumentos = @("-m", "dmrl", "evaluar", "--politica", $Politica, "--nombre", $Nombre,
        "--episodios", $Episodios, "--instancias", $Instancias, "--semilla", $Semilla, "--salida", $salida)
    if ($Modelo -ne "") { $argumentos += @("--modelo", $Modelo) }
}

$entrenando = Get-Content (Join-Path $raiz "runs\entrenamiento.pid") -ErrorAction SilentlyContinue
if ($entrenando -and (Get-Process -Id ([int]$entrenando) -ErrorAction SilentlyContinue)) {
    Write-Warning "Hay un entrenamiento en marcha; la evaluacion compite por CPU con el."
}

Set-Location $raiz
$env:PYTHONUNBUFFERED = "1"
& $python @argumentos
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Detalle: $salida"
