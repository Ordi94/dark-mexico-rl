"""Carga de los registros de jefes (data/jefes/<jefe>.json) como un CatalogoJefes."""

from __future__ import annotations

import json
from pathlib import Path

from dmrl.dominio.jefes import CatalogoJefes, PerfilJefe
from dmrl.infraestructura.datos import CARPETA_DATOS

CARPETA_JEFES = CARPETA_DATOS / "jefes"


def cargar_catalogo(carpeta: Path | None = None) -> CatalogoJefes:
    """
    Lee todos los registros de una carpeta y arma el catalogo.

    Contexto:
        Arranque del entorno multi-jefe, de la evaluacion por jefe y de la generacion de la
        especificacion de la observacion.

    Args:
        carpeta (Path, opcional): Carpeta con un .json por jefe; por defecto data/jefes.

    Returns:
        CatalogoJefes: Catalogo con todos los perfiles.

    Raises:
        FileNotFoundError: Si la carpeta no existe o no tiene registros.
        ValueError: Si un registro no es un objeto JSON, esta mal formado o los perfiles no son
            coherentes entre si.

    Ejemplo:
        catalogo = cargar_catalogo()
    """
    base = carpeta or CARPETA_JEFES
    archivos = sorted(base.glob("*.json")) if base.is_dir() else []
    if not archivos:
        raise FileNotFoundError(f"no hay registros de jefes en {base}")
    perfiles = []
    for archivo in archivos:
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        if not isinstance(datos, dict):
            raise ValueError(f"{archivo.name}: se esperaba un objeto JSON")
        perfiles.append(PerfilJefe.desde_dict(datos))
    return CatalogoJefes(perfiles)
