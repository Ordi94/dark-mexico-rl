"""Carga de los datos medidos (tabla de golpes) desde la carpeta data/ del proyecto."""

from __future__ import annotations

import json
from pathlib import Path

from dmrl.dominio.tabla_golpes import TablaGolpes

RAIZ_PROYECTO = Path(__file__).resolve().parents[3]
CARPETA_DATOS = RAIZ_PROYECTO / "data"
ARCHIVO_TABLA_GOLPES = CARPETA_DATOS / "tabla_golpes.json"


def cargar_tabla_golpes(ruta: Path | None = None) -> TablaGolpes:
    """
    Lee la tabla de golpes medida del jefe.

    Contexto:
        Arranque del entorno y del bot de reglas. Sin la tabla el entorno funciona pero pierde
        la informacion de golpes pendientes, asi que su ausencia es un error y no un valor vacio.

    Args:
        ruta (Path, opcional): Archivo JSON; por defecto data/tabla_golpes.json.

    Returns:
        TablaGolpes: Tabla cargada.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON no tiene la forma esperada.

    Ejemplo:
        tabla = cargar_tabla_golpes()
    """
    archivo = ruta or ARCHIVO_TABLA_GOLPES
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError(f"{archivo}: se esperaba un objeto montage -> golpes")
    try:
        return TablaGolpes.desde_dict(datos)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{archivo}: golpe mal formado: {exc!r}") from exc
