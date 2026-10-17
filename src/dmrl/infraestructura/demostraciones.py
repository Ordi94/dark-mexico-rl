"""
Almacenamiento en disco de las demostraciones y del estado del ciclo de imitacion.

Las demostraciones se guardan en lotes .npz pequenos (uno por tanda de episodios de un trabajador)
escritos de forma atomica: una caida a mitad de escritura nunca deja un lote a medias, y la
recoleccion se puede reanudar contando los episodios ya guardados.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dmrl.dominio.imitacion import Demostraciones, concatenar

SUFIJO_LOTE = ".npz"


def guardar_lote(ruta: Path, demostraciones: Demostraciones, episodios: list[dict[str, Any]]) -> None:
    """
    Escribe un lote de demostraciones de forma atomica.

    Contexto:
        Trabajadores de recoleccion, cada pocos episodios.

    Args:
        ruta (Path): Archivo .npz de destino.
        demostraciones (Demostraciones): Pasos del lote (se validan antes de escribir).
        episodios (list[dict[str, Any]]): Metadatos de cada episodio del lote (plan y resumen).

    Returns:
        None: No retorna valor.

    Raises:
        ValueError: Si las demostraciones no son coherentes.
        OSError: Si no se puede escribir.

    Ejemplo:
        guardar_lote(Path("runs/imitacion/datos/ronda_00/w0_0000.npz"), demostraciones, [])
    """
    demostraciones.validar()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".parcial")
    with temporal.open("wb") as archivo:
        np.savez(
            archivo, obs=demostraciones.obs, mascaras=demostraciones.mascaras,
            etiquetas=demostraciones.etiquetas, ejecutadas=demostraciones.ejecutadas,
            episodios=demostraciones.episodios, recompensas=demostraciones.recompensas,
            meta=np.array(json.dumps(episodios)),
        )
    os.replace(temporal, ruta)


def cargar_lote(ruta: Path) -> tuple[Demostraciones, list[dict[str, Any]]]:
    """
    Lee un lote de demostraciones.

    Contexto:
        Carga del conjunto completo antes de cada ajuste.

    Args:
        ruta (Path): Archivo .npz.

    Returns:
        tuple[Demostraciones, list[dict[str, Any]]]: Pasos y metadatos de los episodios.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el contenido no es coherente.

    Ejemplo:
        demostraciones, meta = cargar_lote(Path("lote.npz"))
    """
    with np.load(ruta, allow_pickle=False) as datos:
        demostraciones = Demostraciones(
            datos["obs"], datos["mascaras"], datos["etiquetas"], datos["ejecutadas"],
            datos["episodios"], datos["recompensas"],
        )
        meta = json.loads(str(datos["meta"]))
    demostraciones.validar()
    return demostraciones, list(meta)


def lotes_de_ronda(carpeta_datos: Path, ronda: int) -> list[Path]:
    """
    Lotes guardados de una ronda.

    Contexto:
        Reanudacion y carga.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos de la corrida.
        ronda (int): Numero de ronda.

    Returns:
        list[Path]: Archivos .npz ordenados por nombre; vacia si la ronda no existe.

    Raises:
        Ninguna.

    Ejemplo:
        lotes_de_ronda(Path("runs/imitacion/datos"), 0)
    """
    carpeta = carpeta_datos / f"ronda_{ronda:02d}"
    return sorted(carpeta.glob(f"*{SUFIJO_LOTE}")) if carpeta.is_dir() else []


def metadatos_de_ronda(carpeta_datos: Path, ronda: int) -> list[dict[str, Any]]:
    """
    Metadatos de todos los episodios guardados de una ronda.

    Contexto:
        Reanudacion (que indices del plan ya estan hechos) y estadisticas de recoleccion.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.
        ronda (int): Numero de ronda.

    Returns:
        list[dict[str, Any]]: Un diccionario por episodio.

    Raises:
        Ninguna.

    Ejemplo:
        len(metadatos_de_ronda(Path("datos"), 0))
    """
    metadatos: list[dict[str, Any]] = []
    for lote in lotes_de_ronda(carpeta_datos, ronda):
        with np.load(lote, allow_pickle=False) as datos:
            metadatos += list(json.loads(str(datos["meta"])))
    return metadatos


def cargar_rondas(carpeta_datos: Path, rondas: list[int]) -> Demostraciones:
    """
    Une las demostraciones de varias rondas.

    Contexto:
        Agregacion de DAgger antes de cada ajuste.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.
        rondas (list[int]): Rondas a cargar.

    Returns:
        Demostraciones: Conjunto unido.

    Raises:
        ValueError: Si las rondas no tienen ningun lote.

    Ejemplo:
        todo = cargar_rondas(Path("datos"), [0, 1])
    """
    partes = [cargar_lote(lote)[0] for ronda in rondas for lote in lotes_de_ronda(carpeta_datos, ronda)]
    if not partes:
        raise ValueError(f"las rondas {rondas} no tienen demostraciones guardadas en {carpeta_datos}")
    return concatenar(partes)


def leer_json(ruta: Path, por_defecto: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Lee un JSON de estado o resultados.

    Contexto:
        Estado reanudable del ciclo y resultados de evaluacion.

    Args:
        ruta (Path): Archivo JSON.
        por_defecto (dict, opcional): Valor si el archivo no existe o esta corrupto.

    Returns:
        dict[str, Any]: Contenido, o por_defecto (vacio si no se da).

    Raises:
        Ninguna: Un archivo ilegible se trata como inexistente.

    Ejemplo:
        estado = leer_json(Path("runs/imitacion/estado.json"))
    """
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(por_defecto or {})
    return datos if isinstance(datos, dict) else dict(por_defecto or {})


def escribir_json(ruta: Path, datos: dict[str, Any]) -> None:
    """
    Escribe un JSON de forma atomica.

    Contexto:
        Estado del ciclo: una interrupcion nunca debe dejar el estado truncado.

    Args:
        ruta (Path): Archivo de destino.
        datos (dict[str, Any]): Contenido serializable.

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede escribir.
        TypeError: Si el contenido no es serializable.

    Ejemplo:
        escribir_json(Path("estado.json"), {"ronda": 1})
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".parcial")
    temporal.write_text(json.dumps(datos, indent=1), encoding="utf-8")
    os.replace(temporal, ruta)
