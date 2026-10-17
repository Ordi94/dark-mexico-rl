<#
.SYNOPSIS
  Clona al bot de reglas en una red MaskablePPO (imitacion con DART y DAgger) y elige el mejor modelo jugando.

.DESCRIPTION
  Ciclo completo y reanudable (todo lo ya hecho en runs\<Nombre> se omite al relanzar):
    1. Ronda 0: el bot pelea (parte de los episodios con ruido de ejecucion, parte iniciando en fase 2)
       y cada estado se etiqueta con la accion del bot.
    2. Ajuste supervisado (entropia cruzada con logits enmascarados, validacion por episodios, parada
       temprana) y evaluacion en juego.
    3. -Rondas rondas de DAgger: la red juega, el bot etiqueta lo que ella visita, se reentrena con todo.
    4. Linea base del bot, eleccion del modelo por desempeno en juego (runs\<Nombre>\mejor.zip) y
       confirmacion con peleas nuevas.
  Usa -Instancias instancias sin ventana (prioridad Below Normal, CPUs 0-3 libres). Ctrl+C o
  detener.ps1 -Nombre <Nombre> piden una parada ordenada. No guarda partida ni logros (mod en modo
  entrenamiento).

.EXAMPLE
  .\scripts\imitar.ps1
  .\scripts\imitar.ps1 -EpisodiosInicial 100 -EpisodiosDagger 50 -Rondas 1 -Nombre prueba
#>
param(
    [string]$Nombre = "imitacion",
    [int]$Instancias = 4,
    [int]$EpisodiosInicial = 600,
    [int]$EpisodiosDagger = 300,
    [int]$Rondas = 3,
    [int]$Semilla = 0,
    [int]$EpisodiosEvaluacion = 50,
    [int]$EpisodiosSecundarios = 20,
    [int]$EpisodiosConfirmacion = 100,
    [int]$Candidatas = 2,
    [ValidateSet("todo", "entrenar")][string]$Accion = "todo",
    [int]$Ronda = 0
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
$runs = Join-Path $raiz "runs"
$pidEntrenamiento = Join-Path $runs "entrenamiento.pid"
New-Item -ItemType Directory -Force -Path (Join-Path $runs $Nombre) | Out-Null

$texto = Get-Content $pidEntrenamiento -ErrorAction SilentlyContinue | Select-Object -First 1
$idProceso = 0
if ([int]::TryParse($texto, [ref]$idProceso) -and (Get-Process -Id $idProceso -ErrorAction SilentlyContinue)) {
    Write-Error "Ya hay un entrenamiento o una imitacion en marcha (PID $idProceso). Usa .\scripts\detener.ps1 antes."
    exit 1
}

$argumentos = @("-m", "dmrl", "imitar", "--nombre", $Nombre, "--instancias", $Instancias,
    "--episodios-inicial", $EpisodiosInicial, "--episodios-dagger", $EpisodiosDagger, "--rondas", $Rondas,
    "--semilla", $Semilla, "--episodios-evaluacion", $EpisodiosEvaluacion,
    "--episodios-secundarios", $EpisodiosSecundarios, "--episodios-confirmacion", $EpisodiosConfirmacion, "--candidatas", $Candidatas,
    "--accion", $Accion, "--ronda", $Ronda)

$env:PYTHONUNBUFFERED = "1"
Set-Location $raiz
# Con "Stop", PowerShell 5.1 trata cualquier linea de stderr del proceso (avisos de torch) como error.
$ErrorActionPreference = "Continue"
try {
    Write-Host "Imitacion '$Nombre' con $Instancias instancias. Ctrl+C o .\scripts\detener.ps1 -Nombre $Nombre para parar."
    & $python @argumentos 2>&1 | Tee-Object -FilePath (Join-Path $runs "$Nombre\imitacion.log") -Append
    $codigo = $LASTEXITCODE
} finally {
    Remove-Item $pidEntrenamiento -ErrorAction SilentlyContinue
    $huerfanos = Get-CimInstance Win32_Process -Filter "Name = 'MexicoSouls-Win64-Shipping.exe'" |
        Where-Object { $_.CommandLine -match "UseFixedTimeStep" }
    foreach ($juego in $huerfanos) {
        Write-Warning "Instancia huerfana del juego (PID $($juego.ProcessId)); se termina."
        Stop-Process -Id $juego.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
exit $codigo
