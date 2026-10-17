<#
.SYNOPSIS
  Paro limpio del entrenamiento: modelo guardado, juegos y Python cerrados, TensorBoard apagado.

.DESCRIPTION
  1. Crea el archivo de parada: el entrenamiento guarda ultimo.zip y cierra cada juego con su orden
     de cierre.
  2. Espera a que el proceso termine; si no lo hace en -EsperaSegundos lo mata con todo su arbol.
  3. Apaga TensorBoard si lo lanzo entrenar.ps1.
  4. Termina cualquier instancia de entrenamiento del juego que haya quedado huerfana (se reconocen
     por -UseFixedTimeStep; el juego normal del usuario no lo lleva y no se toca).

.EXAMPLE
  .\scripts\detener.ps1
  .\scripts\detener.ps1 -Multijefe        # runs\multijefe_ppo (para otra corrida: -Nombre <carpeta de runs>)
#>
param(
    [string]$Nombre = "principal",
    [int]$EsperaSegundos = 90,
    [switch]$Multijefe
)

$ErrorActionPreference = "Stop"
if ($Multijefe -and -not $PSBoundParameters.ContainsKey("Nombre")) { $Nombre = "multijefe_ppo" }
$runs = Join-Path (Split-Path $PSScriptRoot -Parent) "runs"
$pidEntrenamiento = Join-Path $runs "entrenamiento.pid"
$pidTensorBoard = Join-Path $runs "tensorboard.pid"
$archivoParada = Join-Path $runs "$Nombre\PARAR"

function Get-ProcesoVivo([string]$archivo) {
    if (-not (Test-Path $archivo)) { return $null }
    $texto = (Get-Content $archivo -ErrorAction SilentlyContinue | Select-Object -First 1)
    $idProceso = 0
    if (-not [int]::TryParse($texto, [ref]$idProceso)) { return $null }
    return Get-Process -Id $idProceso -ErrorAction SilentlyContinue
}

$entrenamiento = Get-ProcesoVivo $pidEntrenamiento
if (-not ($entrenamiento -and $entrenamiento.ProcessName -like "python*")) {
    # El archivo de PID puede faltar (otra corrida de pruebas lo borra al terminar): se busca el
    # entrenamiento de esta corrida por su linea de comandos para pedirle la parada ordenada en lugar
    # de matar sus juegos por debajo.
    $entrenamiento = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "dmrl (entrenar|entrenar-jefes) " -and
            $_.CommandLine -match "--nombre $Nombre( |$)" } |
        Sort-Object ProcessId | Select-Object -Last 1 |
        ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
}
if ($entrenamiento -and $entrenamiento.ProcessName -like "python*") {
    Write-Host "Pidiendo parada ordenada a PID $($entrenamiento.Id)..."
    New-Item -ItemType Directory -Force -Path (Split-Path $archivoParada) | Out-Null
    New-Item -ItemType File -Force -Path $archivoParada | Out-Null
    $limite = (Get-Date).AddSeconds($EsperaSegundos)
    while ((Get-Process -Id $entrenamiento.Id -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $limite)) {
        Start-Sleep -Seconds 1
    }
    if (Get-Process -Id $entrenamiento.Id -ErrorAction SilentlyContinue) {
        Write-Warning "No termino en $EsperaSegundos s; se mata el arbol de procesos."
        & taskkill.exe /PID $entrenamiento.Id /T /F | Out-Null
    } else {
        Write-Host "Entrenamiento terminado de forma ordenada."
    }
} else {
    Write-Host "No hay entrenamiento en marcha."
}
Remove-Item $archivoParada -ErrorAction SilentlyContinue
Remove-Item $pidEntrenamiento -ErrorAction SilentlyContinue

$tb = Get-ProcesoVivo $pidTensorBoard
if ($tb) {
    Stop-Process -Id $tb.Id -Force -ErrorAction SilentlyContinue
    Write-Host "TensorBoard apagado."
}
Remove-Item $pidTensorBoard -ErrorAction SilentlyContinue

$huerfanos = Get-CimInstance Win32_Process -Filter "Name = 'MexicoSouls-Win64-Shipping.exe'" |
    Where-Object { $_.CommandLine -match "UseFixedTimeStep" }
foreach ($juego in $huerfanos) {
    Write-Warning "Instancia huerfana del juego (PID $($juego.ProcessId)); se termina."
    Stop-Process -Id $juego.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "Listo."
