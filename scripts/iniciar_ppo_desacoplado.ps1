<#
.SYNOPSIS
  Lanza el PPO multi-jefe desacoplado de la consola que lo invoca, para que siga vivo al cerrarla.

.DESCRIPTION
  Un proceso hijo de una consola o de una herramienta con "job object" muere cuando esa consola se
  cierra. Win32_Process.Create (WMI) crea el proceso bajo el servicio de WMI, fuera de ese arbol, de
  modo que el entrenamiento sobrevive.

  El entrenamiento se lanza con cmd.exe y no con entrenar.ps1: medido en mi equipo, si el padre es un
  PowerShell oculto el proceso y sus juegos acaban en los nucleos eficientes (CPUs 16-27) y los pasos
  por segundo caen de 450 a 250; con cmd.exe como padre van en los nucleos de rendimiento. Los
  parametros son los de entrenar.ps1 -Multijefe. El registro queda en
  runs\<Nombre>\entrenamiento_<fecha>.log (y .err.log) y se detiene con .\scripts\detener.ps1 -Multijefe;
  se reanuda ejecutando este mismo script otra vez.

.EXAMPLE
  .\scripts\iniciar_ppo_desacoplado.ps1
#>
param(
    [string]$Nombre = "multijefe_ppo",
    [int]$Instancias = 6,
    [double]$MinutosEvaluacion = 45
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
$carpeta = Join-Path $raiz "runs\$Nombre"
New-Item -ItemType Directory -Force -Path $carpeta | Out-Null

$pidArchivo = Join-Path $raiz "runs\entrenamiento.pid"
$texto = Get-Content $pidArchivo -ErrorAction SilentlyContinue | Select-Object -First 1
$idProceso = 0
if ([int]::TryParse($texto, [ref]$idProceso) -and (Get-Process -Id $idProceso -ErrorAction SilentlyContinue)) {
    Write-Error "Ya hay un entrenamiento en marcha (PID $idProceso). Usa .\scripts\detener.ps1 -Multijefe antes de iniciar otro."
    exit 1
}

$marca = Get-Date -Format "yyyyMMdd_HHmmss"
$registro = Join-Path $carpeta "entrenamiento_$marca.log"
$registroErr = Join-Path $carpeta "entrenamiento_$marca.err.log"
$comando = "cmd.exe /c `"set PYTHONUNBUFFERED=1&& `"$python`" -m dmrl entrenar-jefes --nombre $Nombre " +
    "--instancias $Instancias --minutos-evaluacion $MinutosEvaluacion > `"$registro`" 2> `"$registroErr`"`""
$resultado = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $comando; CurrentDirectory = $raiz
}
if ($resultado.ReturnValue -ne 0) { Write-Error "No se pudo crear el proceso (codigo $($resultado.ReturnValue))"; exit 1 }
Write-Host "Entrenamiento lanzado (PID del lanzador $($resultado.ProcessId)). Registro: $registro"
