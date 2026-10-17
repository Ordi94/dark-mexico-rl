"""Pruebas de la localización de la carpeta del juego (sin el juego instalado)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dmbridge import rutas
from dmbridge.rutas import (
    EJECUTABLE_RELATIVO,
    bibliotecas_de_vdf,
    bibliotecas_steam,
    buscar_carpeta_juego,
    raices_steam,
)


class _Clave:
    """Contexto mínimo que imita una clave abierta del registro."""

    def __init__(self, ruta: str) -> None:
        self.ruta = ruta

    def __enter__(self) -> _Clave:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _registro_falso(valores: dict[tuple[str, str], str]) -> SimpleNamespace:
    """Registro de Windows falso: devuelve valores[(hive, subclave)] o falla con OSError."""

    def abrir(hive: str, subclave: str) -> _Clave:
        if (hive, subclave) not in valores:
            raise OSError("sin clave")
        return _Clave(valores[(hive, subclave)])

    def consultar(clave: _Clave, _valor: str) -> tuple[str, int]:
        return clave.ruta, 1

    return SimpleNamespace(
        HKEY_CURRENT_USER="HKCU", HKEY_LOCAL_MACHINE="HKLM", OpenKey=abrir, QueryValueEx=consultar,
    )


def _escribir_vdf(raiz: Path, *bibliotecas: Path) -> None:
    """Escribe un libraryfolders.vdf que lista las bibliotecas dadas."""
    (raiz / "steamapps").mkdir(parents=True, exist_ok=True)
    lineas = ['"path" "' + str(ruta).replace("\\", "\\\\") + '"' for ruta in bibliotecas]
    (raiz / "steamapps" / "libraryfolders.vdf").write_text("\n".join(lineas), encoding="utf-8")


def _instalar_juego(biblioteca: Path) -> Path:
    """Crea un ejecutable falso del juego dentro de una biblioteca de Steam."""
    carpeta = biblioteca / "steamapps" / "common" / "DarkMexico"
    (carpeta / EJECUTABLE_RELATIVO).parent.mkdir(parents=True)
    (carpeta / EJECUTABLE_RELATIVO).write_bytes(b"")
    return carpeta


def test_el_vdf_se_lee_con_las_barras_duplicadas() -> None:
    """Steam escribe D:\\\\Juegos en el archivo; la ruta resultante usa una sola barra."""
    texto = '"libraryfolders" { "0" { "path"\t\t"C:\\\\Steam" } "1" { "path" "D:\\\\Juegos" } }'
    assert bibliotecas_de_vdf(texto) == [Path("C:\\Steam"), Path("D:\\Juegos")]


def test_las_raices_salen_del_registro_sin_duplicados(monkeypatch: pytest.MonkeyPatch) -> None:
    """La misma ruta en las dos claves se cuenta una vez; una clave ausente se omite."""
    registro = _registro_falso({
        ("HKCU", r"Software\Valve\Steam"): "C:/Steam",
        ("HKLM", r"SOFTWARE\WOW6432Node\Valve\Steam"): "C:/Steam",
    })
    monkeypatch.setattr(rutas, "winreg", registro)
    assert raices_steam() == [Path("C:/Steam")]
    monkeypatch.setattr(rutas, "winreg", _registro_falso({}))
    assert raices_steam() == []


def test_sin_modulo_de_registro_no_hay_raices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fuera de Windows no existe winreg y la búsqueda sigue con lista vacía."""
    monkeypatch.setattr(rutas, "winreg", None)
    assert raices_steam() == []


def test_las_bibliotecas_incluyen_las_del_vdf_y_toleran_su_ausencia(tmp_path: Path) -> None:
    """Una raíz sin vdf aporta solo a sí misma; con vdf suma las bibliotecas declaradas."""
    sin_vdf = tmp_path / "a"
    sin_vdf.mkdir()
    con_vdf = tmp_path / "b"
    _escribir_vdf(con_vdf, tmp_path / "c", con_vdf)
    assert bibliotecas_steam([sin_vdf, con_vdf, sin_vdf]) == [sin_vdf, con_vdf, tmp_path / "c"]


def test_la_variable_de_entorno_gana_a_steam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DARKMEXICO_DIR se respeta aunque Steam tenga el juego en otra parte."""
    biblioteca = tmp_path / "steam"
    _instalar_juego(biblioteca)
    monkeypatch.setattr(rutas, "winreg", _registro_falso({("HKCU", r"Software\Valve\Steam"): str(biblioteca)}))
    assert buscar_carpeta_juego({"DARKMEXICO_DIR": "  X:/otro  "}) == Path("X:/otro")


def test_el_juego_se_encuentra_en_una_biblioteca_secundaria(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Si el juego está en el disco de otra biblioteca, se localiza por libraryfolders.vdf."""
    principal = tmp_path / "steam"
    secundaria = tmp_path / "discoD"
    carpeta = _instalar_juego(secundaria)
    _escribir_vdf(principal, secundaria)
    monkeypatch.setattr(rutas, "winreg", _registro_falso({("HKCU", r"Software\Valve\Steam"): str(principal)}))
    assert buscar_carpeta_juego({}) == carpeta


def test_sin_juego_instalado_devuelve_la_ruta_por_defecto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin variable ni Steam, la búsqueda no falla: devuelve la ruta donde Steam lo instalaría."""
    monkeypatch.setattr(rutas, "winreg", None)
    esperado = Path("E:/Programas") / "Steam" / "steamapps" / "common" / "DarkMexico"
    assert buscar_carpeta_juego({"ProgramFiles(x86)": "E:/Programas"}) == esperado
    assert buscar_carpeta_juego({}).name == "DarkMexico"


def test_sin_entorno_explicito_se_usa_el_del_proceso(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con entorno=None se leen las variables reales del proceso."""
    monkeypatch.setenv("DARKMEXICO_DIR", "Z:/juego")
    assert buscar_carpeta_juego() == Path("Z:/juego")
