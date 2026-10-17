"""
Agregacion de mediciones de combate en tablas de golpes por jefe.

Una medicion es la lista de golpes recibidos que registra el mod con trace_hits (quien golpeo,
con que montage y en que posicion del montage, a que distancia, si era parable y cuanto quito)
mas los eventos de ataques a distancia. Aqui se agrupan en golpes por montage, se miden los
retardos entre un evento de ataque a distancia y su impacto y se fusionan con el registro del
jefe sin perder lo ya medido que la nueva medicion no cubre.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

# Dos impactos del mismo montage separados menos que esto (segundos de montage) son el mismo golpe.
TOLERANCIA_POSICION = 0.08
# Un golpe nuevo reemplaza al registrado si sus posiciones difieren menos que esto.
TOLERANCIA_FUSION = 0.15
# Observaciones minimas de un golpe para fusionarlo al registro.
MINIMO_OBSERVACIONES = 3
# Un impacto sin montage activo se asocia al ultimo evento a distancia de hasta este tiempo antes.
VENTANA_EVENTO = 6.0
REDONDEO_ALCANCE = 10.0


def percentil(valores: Sequence[float], p: float) -> float:
    """
    Percentil por interpolacion lineal.

    Contexto:
        Alcance de los golpes (percentil 95 de las distancias de impacto).

    Args:
        valores (Sequence[float]): Muestra no vacia.
        p (float): Percentil en [0, 100].

    Returns:
        float: Valor del percentil.

    Raises:
        ValueError: Si la muestra esta vacia o p esta fuera de [0, 100].

    Ejemplo:
        percentil([1.0, 2.0, 3.0], 50.0)  # 2.0
    """
    if not valores:
        raise ValueError("muestra vacia")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentil fuera de [0, 100]: {p}")
    orden = sorted(valores)
    posicion = (len(orden) - 1) * p / 100.0
    bajo, alto = math.floor(posicion), math.ceil(posicion)
    return orden[bajo] + (orden[alto] - orden[bajo]) * (posicion - bajo)


def agrupar_golpes(golpes: Sequence[Mapping[str, Any]], clases: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """
    Agrupa los impactos con montage del atacante en golpes por montage.

    Contexto:
        scripts/medir_jefes.py, sobre los golpes que registra el mod.

    Args:
        golpes (Sequence[Mapping[str, Any]]): Golpes con att, am (montage), ap (posicion), amount,
            parry y d (distancia).
        clases (Sequence[str]): Clases del atacante que interesan (el jefe medido).

    Returns:
        dict[str, list[dict[str, Any]]]: montage -> golpes ordenados por posicion, cada uno con
            pos (mediana), min, max, parry (mayoria), dano (mediana), n y alcance (percentil 95
            de la distancia al impactar, redondeado hacia arriba).

    Raises:
        Ninguna: Los golpes sin montage o de otra clase se ignoran.

    Ejemplo:
        agrupar_golpes([{"att": "B", "am": "M", "ap": 0.9, "amount": 10, "parry": True, "d": 90}], ["B"])
    """
    por_montage: dict[str, list[Mapping[str, Any]]] = {}
    for g in golpes:
        if g.get("att") in clases and g.get("am") is not None and g.get("ap") is not None:
            por_montage.setdefault(str(g["am"]), []).append(g)
    salida: dict[str, list[dict[str, Any]]] = {}
    for montage, filas in por_montage.items():
        filas = sorted(filas, key=lambda f: float(f["ap"]))
        grupos: list[list[Mapping[str, Any]]] = []
        for fila in filas:
            if grupos and float(fila["ap"]) - float(grupos[-1][-1]["ap"]) <= TOLERANCIA_POSICION:
                grupos[-1].append(fila)
            else:
                grupos.append([fila])
        salida[montage] = [_resumir_grupo(g) for g in grupos]
    return salida


def _resumir_grupo(grupo: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Resume los impactos de un mismo golpe.

    Contexto:
        agrupar_golpes.

    Args:
        grupo (Sequence[Mapping[str, Any]]): Impactos del mismo golpe (al menos uno).

    Returns:
        dict[str, Any]: pos, min, max, parry, dano, n y alcance.

    Raises:
        statistics.StatisticsError: Si el grupo esta vacio.

    Ejemplo:
        (uso interno)
    """
    posiciones = [float(g["ap"]) for g in grupo]
    distancias = [float(g["d"]) for g in grupo if g.get("d") is not None]
    resumen: dict[str, Any] = {
        "pos": round(statistics.median(posiciones), 3), "min": round(min(posiciones), 3),
        "max": round(max(posiciones), 3),
        "parry": sum(1 for g in grupo if g.get("parry")) * 2 >= len(grupo),
        "dano": float(statistics.median(float(g["amount"]) for g in grupo)),
        "n": len(grupo),
    }
    if distancias:
        resumen["alcance"] = math.ceil(percentil(distancias, 95.0) / REDONDEO_ALCANCE) * REDONDEO_ALCANCE
    return resumen


def retardos_remotos(eventos: Sequence[Mapping[str, Any]], golpes: Sequence[Mapping[str, Any]],
                     clases: Sequence[str]) -> dict[str, list[dict[str, float]]]:
    """
    Mide el retardo entre cada evento de ataque a distancia y el impacto que provoca.

    Contexto:
        scripts/medir_jefes.py. Solo cuentan los impactos sin montage del atacante activo (el
        golpe ya no pertenece a ninguna animacion) o con posicion mayor que la duracion; se
        asocian al evento mas reciente que ocurrio hasta VENTANA_EVENTO antes.

    Args:
        eventos (Sequence[Mapping[str, Any]]): Eventos con id y t (y px, py, bx, by).
        golpes (Sequence[Mapping[str, Any]]): Golpes con t, d, att y am.
        clases (Sequence[str]): Clases del atacante que interesan.

    Returns:
        dict[str, list[dict[str, float]]]: id del evento -> lista de {retardo, distancia_jugador,
            distancia_jefe_al_lanzar}.

    Raises:
        Ninguna.

    Ejemplo:
        retardos_remotos([{"id": "e", "t": 1.0}], [{"t": 1.9, "att": "B"}], ["B"])
    """
    ordenados = sorted(eventos, key=lambda e: float(e["t"]))
    salida: dict[str, list[dict[str, float]]] = {}
    for g in golpes:
        # El causante de un area ya destruida llega nulo (sin clase): tambien es un impacto remoto.
        if g.get("am") is not None or (g.get("att") is not None and g.get("att") not in clases):
            continue
        tiempo = float(g["t"])
        previos = [e for e in ordenados if 0.0 <= tiempo - float(e["t"]) <= VENTANA_EVENTO]
        if not previos:
            continue
        evento = previos[-1]
        entrada = {"retardo": round(tiempo - float(evento["t"]), 3), "distancia_golpe": float(g.get("d", -1.0))}
        if all(k in evento for k in ("px", "py", "bx", "by")):
            entrada["distancia_lanzamiento"] = round(
                math.hypot(float(evento["px"]) - float(evento["bx"]), float(evento["py"]) - float(evento["by"])), 1)
        salida.setdefault(str(evento["id"]), []).append(entrada)
    return salida


def resumir_retardos(retardos: Mapping[str, Sequence[Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    """
    Estadisticos de los retardos por evento.

    Contexto:
        Informe de scripts/medir_jefes.py.

    Args:
        retardos (Mapping[str, Sequence[Mapping[str, float]]]): Salida de retardos_remotos.

    Returns:
        dict[str, dict[str, float]]: id -> n, minimo, mediana, maximo y desviacion estandar.

    Raises:
        Ninguna.

    Ejemplo:
        resumir_retardos({"e": [{"retardo": 0.9}]})
    """
    salida: dict[str, dict[str, float]] = {}
    for evento, filas in retardos.items():
        valores = [float(f["retardo"]) for f in filas]
        if not valores:
            continue
        salida[evento] = {
            "n": float(len(valores)), "minimo": min(valores), "mediana": statistics.median(valores),
            "maximo": max(valores), "desviacion": statistics.pstdev(valores),
        }
    return salida


def fusionar_golpes(golpes_registro: Mapping[str, Sequence[Mapping[str, Any]]],
                    medidos: Mapping[str, Sequence[Mapping[str, Any]]],
                    agregar: bool = True) -> dict[str, list[dict[str, Any]]]:
    """
    Incorpora golpes medidos a los de un registro.

    Contexto:
        Fusion de una medicion nueva en data/jefes/<jefe>.json. Un golpe medido con al menos
        MINIMO_OBSERVACIONES observaciones reemplaza al registrado que cae a menos de
        TOLERANCIA_FUSION (conservando tipo, post_montage y, si no es melee, alcance); si no hay
        ninguno cercano se agrega como melee. Los golpes del registro que la medicion no cubre se
        conservan y los montages cuyos golpes no son melee no se modifican.

    Args:
        golpes_registro (Mapping[str, Sequence[Mapping[str, Any]]]): montage -> golpes del registro.
        medidos (Mapping[str, Sequence[Mapping[str, Any]]]): Salida de agrupar_golpes.
        agregar (bool): Si es False solo se actualizan los golpes que ya existen (una medicion
            con el jugador teletransportado lejos produce grupos espurios en los saltos).

    Returns:
        dict[str, list[dict[str, Any]]]: Golpes fusionados, ordenados por posicion en cada montage.

    Raises:
        Ninguna.

    Ejemplo:
        fusionar_golpes({"M": [{"pos": 1.0, "parry": True}]}, {"M": [{"pos": 1.02, "n": 5, "parry": True}]})
    """
    fusion: dict[str, list[dict[str, Any]]] = {m: [dict(g) for g in filas] for m, filas in golpes_registro.items()}
    for montage, nuevos in medidos.items():
        existentes = fusion.setdefault(montage, [])
        if existentes and all(g.get("tipo", "melee") != "melee" for g in existentes):
            # Rayo, area o proyectil con tiempos de vuelo: sus golpes se curan a mano porque la
            # posicion medida depende de la distancia y no forma grupos fijos.
            continue
        for nuevo in nuevos:
            if int(nuevo.get("n", 0)) < MINIMO_OBSERVACIONES:
                continue
            cercano = min(existentes, key=lambda g: abs(float(g["pos"]) - float(nuevo["pos"])), default=None)
            if cercano is not None and abs(float(cercano["pos"]) - float(nuevo["pos"])) <= TOLERANCIA_FUSION:
                tipo, post, alcance = cercano.get("tipo", "melee"), cercano.get("post_montage"), cercano.get("alcance")
                cercano.update(nuevo)
                cercano["tipo"] = tipo
                if post is not None:
                    cercano["post_montage"] = post
                if tipo != "melee" and alcance is not None:
                    # En un golpe de area, proyectil o rayo la distancia al impactar no es su alcance.
                    cercano["alcance"] = alcance
            elif agregar:
                existentes.append({**nuevo, "tipo": "melee"})
        existentes.sort(key=lambda g: float(g["pos"]))
    return {m: filas for m, filas in fusion.items() if filas}
