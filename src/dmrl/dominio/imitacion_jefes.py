"""
Reglas puras de la imitacion multi-jefe: particion estratificada por jefe, metricas por jefe y
arquitecturas candidatas.

Con varios jefes en un mismo conjunto, repartir los episodios de validacion al azar puede dejar a un
jefe con pocos o ninguno (el de menos peleas) y la precision global oculta al jefe que peor se imita.
La particion estratifica por jefe (cada jefe aporta la misma fraccion de sus propios episodios) y las
metricas se calculan por jefe ademas de en conjunto.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmrl.dominio import acciones as acc
from dmrl.dominio.imitacion import MetricaAccion, metricas_por_accion

# Arquitecturas candidatas (neuronas de cada capa oculta, iguales para politica y valor).
ARQUITECTURAS_CANDIDATAS: dict[str, tuple[int, ...]] = {
    "2x256": (256, 256),
    "2x512": (512, 512),
    "3x256": (256, 256, 256),
}


@dataclass(frozen=True)
class MetricasJefe:
    """
    Calidad de la imitacion sobre los pasos de validacion de un jefe.

    Contexto:
        Informe del ajuste multi-jefe (metricas_por_jefe).

    Atributos:
        jefe: identificador del jefe de episodio.
        pasos: pasos de validacion de ese jefe.
        episodios: episodios de validacion de ese jefe.
        precision: fraccion de pasos donde la red coincide con el bot.
        por_accion: recall y precision de cada accion sobre los pasos de ese jefe.
    """

    jefe: str
    pasos: int
    episodios: int
    precision: float
    por_accion: list[MetricaAccion]


def arquitectura_por_nombre(nombre: str) -> tuple[int, ...]:
    """
    Capas ocultas de una arquitectura candidata.

    Contexto:
        CLI imitar-jefes: el usuario elige por nombre y la comparacion en juego usa los mismos nombres.

    Args:
        nombre (str): Una de las claves de ARQUITECTURAS_CANDIDATAS.

    Returns:
        tuple[int, ...]: Neuronas de cada capa oculta.

    Raises:
        ValueError: Si el nombre no es una arquitectura candidata (lista de admitidos).

    Ejemplo:
        arquitectura_por_nombre("2x512")  # (512, 512)
    """
    if nombre not in ARQUITECTURAS_CANDIDATAS:
        raise ValueError(f"arquitectura desconocida {nombre!r}; admitidas: {sorted(ARQUITECTURAS_CANDIDATAS)}")
    return ARQUITECTURAS_CANDIDATAS[nombre]


def particionar_estratificado(episodios: NDArray[np.int64], jefe_de_episodio: Mapping[int, str],
                              fraccion_validacion: float,
                              semilla: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Reparte los pasos entre entrenamiento y validacion por episodios, estratificando por jefe.

    Contexto:
        Antes del ajuste multi-jefe. Igual que particionar_episodios (el sorteo de un episodio depende
        solo de su identificador y la semilla), pero la cuota de validacion se cumple dentro de cada
        jefe: el jefe con menos peleas no puede quedarse sin validacion ni sin entrenamiento.

    Args:
        episodios (NDArray[np.int64]): Identificador de episodio de cada paso.
        jefe_de_episodio (Mapping[int, str]): Jefe de cada identificador de episodio.
        fraccion_validacion (float): Fraccion de episodios de cada jefe que se reserva, en (0, 1).
        semilla (int): Semilla del sorteo.

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: Mascaras de entrenamiento y de validacion;
            complementarias, con al menos un episodio de cada jefe en cada lado (si el jefe tiene dos
            o mas episodios).

    Raises:
        ValueError: Si la fraccion esta fuera de (0, 1) o algun episodio no tiene jefe asignado.

    Ejemplo:
        entrenamiento, validacion = particionar_estratificado(datos.episodios, jefes, 0.15, 0)
    """
    if not 0.0 < fraccion_validacion < 1.0:
        raise ValueError(f"fraccion_validacion fuera de (0, 1): {fraccion_validacion}")
    unicos = np.unique(episodios)
    faltantes = [int(e) for e in unicos if int(e) not in jefe_de_episodio]
    if faltantes:
        raise ValueError(f"episodios sin jefe asignado: {faltantes[:5]}")
    sorteo = {int(e): float(np.random.default_rng((semilla, int(e))).random()) for e in unicos}
    en_validacion: set[int] = set()
    for jefe in sorted({jefe_de_episodio[int(e)] for e in unicos}):
        propios = [int(e) for e in unicos if jefe_de_episodio[int(e)] == jefe]
        elegidos = {e for e in propios if sorteo[e] < fraccion_validacion}
        if len(propios) >= 2:
            if not elegidos:
                elegidos = {min(propios, key=lambda e: sorteo[e])}
            if len(elegidos) == len(propios):
                elegidos.discard(max(propios, key=lambda e: sorteo[e]))
        en_validacion |= elegidos
    validacion = np.isin(episodios, np.asarray(sorted(en_validacion), dtype=np.int64))
    return ~validacion, validacion


def jefe_por_paso(episodios: NDArray[np.int64], jefe_de_episodio: Mapping[int, str]) -> NDArray[np.str_]:
    """
    Jefe de cada paso a partir del jefe de su episodio.

    Contexto:
        Metricas por jefe y reparto de los datos por jefe en los informes.

    Args:
        episodios (NDArray[np.int64]): Identificador de episodio de cada paso.
        jefe_de_episodio (Mapping[int, str]): Jefe de cada identificador de episodio.

    Returns:
        NDArray[np.str_]: Un identificador de jefe por paso.

    Raises:
        KeyError: Si algun episodio no tiene jefe asignado.

    Ejemplo:
        jefe_por_paso(np.array([1, 1, 2]), {1: "amlo", 2: "nahual"})  # ["amlo", "amlo", "nahual"]
    """
    unicos, inverso = np.unique(episodios, return_inverse=True)
    nombres = np.asarray([jefe_de_episodio[int(e)] for e in unicos], dtype=np.str_)
    return nombres[inverso]


def metricas_por_jefe(prediccion: NDArray[np.int64], etiqueta: NDArray[np.int64],
                      jefes: NDArray[np.str_], episodios: NDArray[np.int64]) -> list[MetricasJefe]:
    """
    Precision global y por accion de la red separadas por jefe.

    Contexto:
        Informe de validacion del ajuste multi-jefe: una precision global alta puede ocultar que el
        jefe con menos datos se imita peor.

    Args:
        prediccion (NDArray[np.int64]): Accion elegida por la red en cada paso de validacion.
        etiqueta (NDArray[np.int64]): Accion del bot en cada paso.
        jefes (NDArray[np.str_]): Jefe de cada paso.
        episodios (NDArray[np.int64]): Episodio de cada paso.

    Returns:
        list[MetricasJefe]: Una entrada por jefe presente, ordenada por nombre.

    Raises:
        ValueError: Si las longitudes difieren.

    Ejemplo:
        metricas_por_jefe(pred, etiquetas, jefes, episodios)[0].precision
    """
    if not (prediccion.shape == etiqueta.shape == jefes.shape == episodios.shape):
        raise ValueError("prediccion, etiqueta, jefes y episodios deben tener la misma forma")
    salida: list[MetricasJefe] = []
    for jefe in sorted({str(j) for j in jefes}):
        filas = jefes == jefe
        salida.append(MetricasJefe(
            jefe=jefe, pasos=int(np.count_nonzero(filas)), episodios=int(np.unique(episodios[filas]).size),
            precision=float((prediccion[filas] == etiqueta[filas]).mean()),
            por_accion=metricas_por_accion(prediccion[filas], etiqueta[filas], acc.N_ACCIONES),
        ))
    return salida


def informe_por_jefe(metricas: Sequence[MetricasJefe]) -> dict[str, Any]:
    """
    Serializa las metricas por jefe con nombres de accion legibles.

    Contexto:
        Informe JSON del ajuste (runs/multijefe_imitacion/modelos/<arquitectura>.json).

    Args:
        metricas (Sequence[MetricasJefe]): Salida de metricas_por_jefe.

    Returns:
        dict[str, Any]: jefe -> pasos, episodios, precision y recall/precision de las acciones que
            el bot usa en ese jefe.

    Raises:
        Ninguna.

    Ejemplo:
        informe_por_jefe(metricas)["amlo"]["precision"]
    """
    return {
        m.jefe: {
            "pasos": m.pasos, "episodios": m.episodios, "precision": m.precision,
            "por_accion": [
                {"accion": a.accion, "nombre": acc.NOMBRES_ACCIONES[a.accion], "soporte": a.soporte,
                 "recall": a.recall, "precision": a.precision}
                for a in m.por_accion if a.soporte > 0
            ],
        }
        for m in metricas
    }
