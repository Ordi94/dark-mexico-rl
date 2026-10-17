"""
Localización de la carpeta de instalación del juego.

La ruta se resuelve en este orden:
    1. La variable de entorno DARKMEXICO_DIR, si está definida.
    2. Las bibliotecas de Steam: la ruta de Steam en el registro de Windows y las bibliotecas
       adicionales que lista steamapps/libraryfolders.vdf.
    3. La ruta por defecto de Steam en Program Files (x86), solo para que el mensaje de error
       diga dónde se buscó.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

winreg: ModuleType | None
try:
    import winreg as _winreg
except ImportError:  # pragma: no cover - solo ocurre fuera de Windows
    winreg = None
else:
    winreg = _winreg

VARIABLE_CARPETA_JUEGO = "DARKMEXICO_DIR"
NOMBRE_CARPETA_JUEGO = "DarkMexico"
EJECUTABLE_RELATIVO = Path("MexicoSouls") / "Binaries" / "Win64" / "MexicoSouls-Win64-Shipping.exe"
_CLAVES_STEAM = (
    ("HKEY_CURRENT_USER", r"Software\Valve\Steam", "SteamPath"),
    ("HKEY_LOCAL_MACHINE", r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
)
_PATRON_BIBLIOTECA = re.compile(r'"path"\s+"([^"]+)"')


def raices_steam() -> list[Path]:
    """
    Lee del registro de Windows las carpetas de instalación de Steam.

    Contexto:
        buscar_carpeta_juego. Prueba la clave del usuario y la del equipo; devuelve las que existan.

    Args:
        Ninguno.

    Returns:
        list[Path]: Carpetas de Steam encontradas, sin duplicados y en orden de preferencia.
        Vacía si Steam no está instalado o el sistema no es Windows.

    Raises:
        Ninguna.

    Ejemplo:
        raices = raices_steam()
    """
    if winreg is None:
        return []
    raices: list[Path] = []
    for nombre_hive, subclave, valor in _CLAVES_STEAM:
        try:
            with winreg.OpenKey(getattr(winreg, nombre_hive), subclave) as clave:
                texto = str(winreg.QueryValueEx(clave, valor)[0])
        except OSError:
            continue
        ruta = Path(texto)
        if texto and ruta not in raices:
            raices.append(ruta)
    return raices


def bibliotecas_de_vdf(contenido: str) -> list[Path]:
    """
    Extrae las carpetas de biblioteca de un libraryfolders.vdf.

    Contexto:
        bibliotecas_steam. Steam escribe cada ruta con las barras invertidas duplicadas.

    Args:
        contenido (str): Texto del archivo libraryfolders.vdf.

    Returns:
        list[Path]: Una ruta por cada entrada "path" del archivo.

    Raises:
        Ninguna.

    Ejemplo:
        bibliotecas_de_vdf('"path"  "D:\\\\Juegos"')
    """
    return [Path(coincidencia.replace("\\\\", "\\")) for coincidencia in _PATRON_BIBLIOTECA.findall(contenido)]


def bibliotecas_steam(raices: list[Path]) -> list[Path]:
    """
    Reúne las bibliotecas de Steam: las raíces mismas y las que declara cada libraryfolders.vdf.

    Contexto:
        buscar_carpeta_juego. El juego puede estar en una biblioteca de otro disco.

    Args:
        raices (list[Path]): Carpetas de instalación de Steam.

    Returns:
        list[Path]: Bibliotecas sin duplicados. Un vdf ilegible se ignora.

    Raises:
        Ninguna.

    Ejemplo:
        bibliotecas_steam(raices_steam())
    """
    bibliotecas: list[Path] = []
    for raiz in raices:
        candidatas = [raiz]
        with contextlib.suppress(OSError):
            candidatas += bibliotecas_de_vdf((raiz / "steamapps" / "libraryfolders.vdf").read_text(encoding="utf-8"))
        bibliotecas += [ruta for ruta in dict.fromkeys(candidatas) if ruta not in bibliotecas]
    return bibliotecas


def buscar_carpeta_juego(entorno: Mapping[str, str] | None = None) -> Path:
    """
    Devuelve la carpeta de instalación de Dark México.

    Contexto:
        dmbridge.launcher la evalúa al importarse para fijar GAME_DIR. No falla si el juego no está
        instalado (por ejemplo en integración continua): devuelve la ruta por defecto y es el
        lanzador quien avisa al intentar abrir el juego.

    Args:
        entorno (Mapping[str, str], opcional): Variables de entorno; por defecto las del proceso.

    Returns:
        Path: La carpeta indicada en DARKMEXICO_DIR; si no hay variable, la primera biblioteca de
        Steam donde existe el ejecutable; si no, la ruta por defecto de Steam.

    Raises:
        Ninguna.

    Ejemplo:
        carpeta = buscar_carpeta_juego({"DARKMEXICO_DIR": r"D:\\Juegos\\DarkMexico"})
    """
    variables = os.environ if entorno is None else entorno
    indicada = variables.get(VARIABLE_CARPETA_JUEGO, "").strip()
    if indicada:
        return Path(indicada)
    for biblioteca in bibliotecas_steam(raices_steam()):
        carpeta = biblioteca / "steamapps" / "common" / NOMBRE_CARPETA_JUEGO
        if (carpeta / EJECUTABLE_RELATIVO).exists():
            return carpeta
    archivos_programa = variables.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    return Path(archivos_programa) / "Steam" / "steamapps" / "common" / NOMBRE_CARPETA_JUEGO
