<#
.SYNOPSIS
  Compara por SHA-256 la carpeta Saved del juego con un respaldo y, si se pide, la restaura.

.DESCRIPTION
  Las instancias de entrenamiento escriben en %LOCALAPPDATA%\MexicoSouls\Saved (un
  CrashReportClient.ini por arranque, la cache de pipelines y GameUserSettings.ini). Ningun archivo
  de partida se crea alli (el mod bloquea el guardado), pero la carpeta cambia mientras el
  entrenamiento corre. Este script informa que archivos faltan, sobran o difieren respecto al
  respaldo; con -Restaurar deja Saved identica al respaldo (espejo) y vuelve a comprobarla.

  Ejecutarlo con el entrenamiento detenido (.\scripts\detener.ps1 -Multijefe): con los juegos abiertos algunos
  archivos pueden estar en uso y volver a cambiar.

.EXAMPLE
  .\scripts\verificar_saved.ps1                          # solo informa, contra el respaldo mas reciente
  .\scripts\verificar_saved.ps1 -Restaurar               # informa, restaura y vuelve a comprobar
  .\scripts\verificar_saved.ps1 -Respaldo backup_savegames_17-10-2026_02-48-05 -Restaurar
#>
param(
    [string]$Respaldo = "",
    [switch]$Restaurar
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path $PSScriptRoot -Parent
$saved = Join-Path $env:LOCALAPPDATA "MexicoSouls\Saved"
if ($Respaldo -eq "") {
    $ultimo = Get-ChildItem $raiz -Directory -Filter "backup_savegames_*" |
        Sort-Object { [datetime]::ParseExact($_.Name.Substring(17), "dd-MM-yyyy_HH-mm-ss", $null) } |
        Select-Object -Last 1
    if (-not $ultimo) { Write-Error "No hay ninguna carpeta backup_savegames_* en $raiz"; exit 1 }
    $Respaldo = $ultimo.FullName
} elseif (-not [System.IO.Path]::IsPathRooted($Respaldo)) {
    $Respaldo = Join-Path $raiz $Respaldo
}
$origen = Join-Path $Respaldo "Saved"
if (-not (Test-Path $origen)) { Write-Error "El respaldo no tiene la carpeta Saved: $origen"; exit 1 }

function Get-Huellas([string]$base) {
    $mapa = @{}
    if (-not (Test-Path $base)) { return $mapa }
    Get-ChildItem $base -Recurse -File | ForEach-Object {
        $relativo = $_.FullName.Substring($base.Length).TrimStart("\")
        $mapa[$relativo] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    }
    return $mapa
}

function Compare-Saved {
    $respaldoHuellas = Get-Huellas $origen
    $actualHuellas = Get-Huellas $saved
    $faltan = @($respaldoHuellas.Keys | Where-Object { -not $actualHuellas.ContainsKey($_) })
    $sobran = @($actualHuellas.Keys | Where-Object { -not $respaldoHuellas.ContainsKey($_) })
    $difieren = @($respaldoHuellas.Keys | Where-Object { $actualHuellas.ContainsKey($_) -and $actualHuellas[$_] -ne $respaldoHuellas[$_] })
    return [pscustomobject]@{ Faltan = $faltan; Sobran = $sobran; Difieren = $difieren; Total = $respaldoHuellas.Count }
}

$juegos = @(Get-CimInstance Win32_Process -Filter "Name = 'MexicoSouls-Win64-Shipping.exe'" |
    Where-Object { $_.CommandLine -match "UseFixedTimeStep" })
if ($juegos.Count -gt 0) {
    Write-Warning "$($juegos.Count) instancias de entrenamiento del juego siguen abiertas; Saved puede seguir cambiando."
}

$resultado = Compare-Saved
Write-Host "Respaldo: $Respaldo ($($resultado.Total) archivos)"
Write-Host "Faltan: $($resultado.Faltan.Count)  Sobran: $($resultado.Sobran.Count)  Difieren: $($resultado.Difieren.Count)"
$resultado.Faltan | ForEach-Object { Write-Host "  falta   $_" }
$resultado.Sobran | ForEach-Object { Write-Host "  sobra   $_" }
$resultado.Difieren | ForEach-Object { Write-Host "  difiere $_" }

$limpio = ($resultado.Faltan.Count + $resultado.Sobran.Count + $resultado.Difieren.Count) -eq 0
if ($limpio) { Write-Host "Saved es identica al respaldo."; exit 0 }
if (-not $Restaurar) { Write-Host "Usa -Restaurar para dejarla identica al respaldo."; exit 2 }

# Espejo: copia lo que falta o cambio y borra lo que sobra. robocopy devuelve 0-7 si no hubo fallos.
& robocopy.exe $origen $saved /MIR /R:3 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Error "robocopy fallo con codigo $LASTEXITCODE"; exit 1 }
$despues = Compare-Saved
Write-Host "Tras restaurar -> Faltan: $($despues.Faltan.Count)  Sobran: $($despues.Sobran.Count)  Difieren: $($despues.Difieren.Count)"
if (($despues.Faltan.Count + $despues.Sobran.Count + $despues.Difieren.Count) -eq 0) {
    Write-Host "Saved es identica al respaldo."
    exit 0
}
Write-Warning "Quedan diferencias (probablemente archivos en uso por el juego); detener el entrenamiento y repetir."
exit 2
