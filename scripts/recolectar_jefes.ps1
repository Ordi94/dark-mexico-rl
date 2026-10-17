<#
.SYNOPSIS
  Recolecta demostraciones del bot de reglas contra uno o varios jefes (formato v2, reanudable).

.DESCRIPTION
  El bot juega -Episodios peleas contra cada jefe y etiqueta cada estado con su decision (la marca de
  golpe atendido solo se confirma si la accion se ejecuto). Una parte de las peleas usa ruido de
  ejecucion (DART) y otra arranca en la segunda fase en los jefes que la admiten. Escribe lotes .npz
  en runs\<Nombre>\datos\ronda_<Ronda>; al relanzar omite lo ya guardado.

.EXAMPLE
  .\scripts\recolectar_jefes.ps1 -Jefes amlo,nahual -Episodios 5 -Nombre prueba_v2
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Jefes,
    [int]$Episodios = 5,
    [int]$Instancias = 4,
    [string]$Nombre = "imitacion_v2",
    [int]$Ronda = 0,
    [int]$Semilla = 0,
    [double]$MaxMinutos = 0
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path $PSScriptRoot -Parent
$python = Join-Path $raiz ".venv\Scripts\python.exe"
# Con "powershell -File" una lista separada por comas llega como una sola cadena.
$Jefes = @($Jefes | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$argumentos = @("-m", "dmrl", "recolectar-jefes", "--jefes") + $Jefes + @("--episodios", $Episodios,
    "--instancias", $Instancias, "--nombre", $Nombre, "--ronda", $Ronda, "--semilla", $Semilla)
if ($MaxMinutos -gt 0) { $argumentos += @("--max-minutos", $MaxMinutos) }
Set-Location $raiz
$env:PYTHONUNBUFFERED = "1"
& $python @argumentos
exit $LASTEXITCODE
