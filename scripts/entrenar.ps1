<#
.SYNOPSIS
  Inicia o reanuda el entrenamiento y abre TensorBoard.

.DESCRIPTION
  Lanza N instancias del juego sin ventana (prioridad Below Normal, CPUs 0-3 libres) y entrena un
  MaskablePPO. Si existe un checkpoint de la corrida lo reanuda; -DesdeCero lo ignora.
  TensorBoard queda en http://localhost:6006 (si el puerto esta ocupado usa el siguiente libre).
  Ctrl+C o detener.ps1 guardan el modelo y cierran juegos y Python de forma ordenada.

  Con -Multijefe entrena un MaskablePPO unico contra Sheinbaum, AMLO, el Chamuco y el Chupacabras
  (observacion v2) partiendo de runs\multijefe_imitacion\mejor.zip. El nombre por defecto pasa a
  multijefe_ppo y la corrida se reanuda sola desde runs\<Nombre>\checkpoints\ultimo.zip. Cada
  -MinutosEvaluacion minutos evalua el modelo en juego (20 peleas por jefe) y solo reemplaza a
  mejor.zip si mejora la tasa media sin bajar ningun jefe mas de 10 puntos respecto a la imitacion.
  Con -Multijefe no se inicia otro TensorBoard si el puerto ya esta ocupado: se reutiliza el que sirve
  la carpeta runs. Para lanzarlo oculto y que sobreviva a la consola usa
  .\scripts\iniciar_ppo_desacoplado.ps1 (misma corrida, mismos parametros): lanzado desde un PowerShell
  oculto el entrenamiento cae a la mitad de pasos por segundo.

.EXAMPLE
  .\scripts\entrenar.ps1
  .\scripts\entrenar.ps1 -Nombre prueba -Instancias 2 -Pasos 200000 -DesdeCero
  .\scripts\entrenar.ps1 -Multijefe
#>
param(
    [string]$Nombre = "principal",
    [int]$Instancias = 4,
    [long]$Pasos = 50000000,
    [ValidateSet("auto", "cpu", "cuda")][string]$Dispositivo = "auto",
    [int]$PuertoTensorBoard = 6006,
    [switch]$DesdeCero,
    [switch]$SinTensorBoard,
    [switch]$MantenerTensorBoard,
    [switch]$Multijefe,
    [double]$MinutosEvaluacion = 45
)

$ErrorActionPreference = "Stop"
if ($Multijefe -and -not $PSBoundParameters.ContainsKey("Nombre")) { $Nombre = "multijefe_ppo" }
if ($Multijefe -and -not $PSBoundParameters.ContainsKey("Instancias")) { $Instancias = 6 }
if ($Multijefe -and $DesdeCero) {
    Write-Error "Con -Multijefe no hay -DesdeCero: para empezar de nuevo borra runs\$Nombre\checkpoints (se vuelve a partir de la imitacion)."
    exit 1
}
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
$tensorboard = Join-Path $raiz ".venv\Scripts\tensorboard.exe"
$runs = Join-Path $raiz "runs"
$pidEntrenamiento = Join-Path $runs "entrenamiento.pid"
$pidTensorBoard = Join-Path $runs "tensorboard.pid"
New-Item -ItemType Directory -Force -Path (Join-Path $runs $Nombre) | Out-Null

function Get-ProcesoVivo([string]$archivo) {
    if (-not (Test-Path $archivo)) { return $null }
    $texto = (Get-Content $archivo -ErrorAction SilentlyContinue | Select-Object -First 1)
    $idProceso = 0
    if (-not [int]::TryParse($texto, [ref]$idProceso)) { return $null }
    return Get-Process -Id $idProceso -ErrorAction SilentlyContinue
}

function Test-PuertoOcupado([int]$puerto) {
    return [bool](Get-NetTCPConnection -LocalPort $puerto -State Listen -ErrorAction SilentlyContinue)
}

$activo = Get-ProcesoVivo $pidEntrenamiento
if ($activo -and $activo.ProcessName -like "python*") {
    Write-Error "Ya hay un entrenamiento en marcha (PID $($activo.Id)). Usa .\scripts\detener.ps1 antes de iniciar otro."
    exit 1
}

$tbIniciado = $false
if ($Multijefe -and -not $SinTensorBoard -and (Test-PuertoOcupado $PuertoTensorBoard)) {
    Write-Host "TensorBoard ya esta sirviendo en el puerto ${PuertoTensorBoard}: se reutiliza (http://localhost:$PuertoTensorBoard)."
    $SinTensorBoard = $true
}
if (-not $SinTensorBoard) {
    $tb = Get-ProcesoVivo $pidTensorBoard
    if ($tb) {
        Write-Host "TensorBoard ya estaba corriendo (PID $($tb.Id)); se reutiliza."
    } else {
        $puerto = $PuertoTensorBoard
        while ((Test-PuertoOcupado $puerto) -and ($puerto -lt $PuertoTensorBoard + 20)) { $puerto++ }
        if ($puerto -ne $PuertoTensorBoard) {
            Write-Warning "El puerto $PuertoTensorBoard esta ocupado; TensorBoard usara el $puerto."
        }
        $tb = Start-Process -FilePath $tensorboard -PassThru -WindowStyle Hidden `
            -ArgumentList @("--logdir", $runs, "--port", $puerto, "--host", "127.0.0.1")
        $tb.PriorityClass = "BelowNormal"
        Set-Content -Path $pidTensorBoard -Value $tb.Id
        $tbIniciado = $true
        Write-Host "TensorBoard: http://localhost:$puerto"
    }
}

if ($Multijefe) {
    $argumentos = @("-m", "dmrl", "entrenar-jefes", "--nombre", $Nombre, "--instancias", $Instancias,
        "--minutos-evaluacion", $MinutosEvaluacion)
    if ($PSBoundParameters.ContainsKey("Pasos")) { $argumentos += @("--pasos", $Pasos) }
} else {
    $argumentos = @("-m", "dmrl", "entrenar", "--nombre", $Nombre, "--instancias", $Instancias,
        "--pasos", $Pasos, "--dispositivo", $Dispositivo)
    if ($DesdeCero) { $argumentos += "--desde-cero" }
}

$env:PYTHONUNBUFFERED = "1"
Set-Location $raiz
# Con "Stop", PowerShell 5.1 trata cualquier linea de stderr del proceso (avisos de torch) como error.
$ErrorActionPreference = "Continue"
try {
    Write-Host "Entrenando '$Nombre' con $Instancias instancias. Ctrl+C o .\scripts\detener.ps1 para parar."
    if ($Multijefe) {
        # La salida va directa a archivos: con "| Tee-Object" las instancias del juego heredan el tubo de
        # PowerShell y los pasos por segundo caen a la mitad (medido: 256 contra 470).
        $marca = Get-Date -Format "yyyyMMdd_HHmmss"
        $registro = Join-Path $runs "$Nombre\entrenamiento_$marca.log"
        $registroErr = Join-Path $runs "$Nombre\entrenamiento_$marca.err.log"
        Write-Host "Registro: $registro  (seguirlo: Get-Content '$registro' -Wait -Tail 20)"
        Start-Process -FilePath $python -ArgumentList $argumentos -NoNewWindow -Wait `
            -RedirectStandardOutput $registro -RedirectStandardError $registroErr
    } else {
        & $python @argumentos 2>&1 | Tee-Object -FilePath (Join-Path $runs "$Nombre\entrenamiento.log")
    }
} finally {
    if ($tbIniciado -and -not $MantenerTensorBoard) {
        Stop-Process -Id $tb.Id -Force -ErrorAction SilentlyContinue
        Remove-Item $pidTensorBoard -ErrorAction SilentlyContinue
    }
    Remove-Item $pidEntrenamiento -ErrorAction SilentlyContinue
}
