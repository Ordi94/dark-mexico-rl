"""
Observacion generica multi-jefe (version 2).

Misma idea que la observacion v1 (vector float32 por instante, apilado de las ultimas N) pero sin
nada propio de un jefe en su forma:
    * el jugador se codifica igual (es el mismo personaje contra cualquier jefe);
    * el jefe se identifica por un one-hot de identidad y se describe con magnitudes que significan
      lo mismo en todos (vida, fase, aturdimiento, velocidad, progreso del montage) mas un one-hot
      sobre el vocabulario UNION de montages de todos los jefes;
    * la amenaza se expresa con rasgos genericos calculados con la tabla del jefe (tiempo hasta el
      proximo golpe, si se puede parar, dano, golpes que quedan, si el jugador esta al alcance) y,
      aparte, con el ataque a distancia mas cercano (tiempo al impacto, si se esquiva o se para,
      de donde viene).
El orden y significado de cada indice sale de campos_v2; la especificacion se genera con el mismo
codigo (python -m dmrl especificacion-v2).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenaza, Amenazas
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.observacion import (
    VEL_JEFE_MAX,
    Vector,
    _fraccion,
    _one_hot,
    _velocidad_horizontal,
    bloque_jugador,
    campos_jugador,
)
from dmrl.dominio.vocabulario import indice_montage, tamano_codificacion

DISTANCIA_MAX_V2 = 4000.0
GOLPE_HORIZONTE = 1.5
REMOTA_HORIZONTE = 2.5
MONTAGE_RESTANTE_MAX = 4.0
RATE_MAX = 2.0
DANIO_MAX = 50.0
GOLPES_MAX = 5.0
ORIGEN_MAX = 1000.0


def campos_v2(catalogo: CatalogoJefes) -> tuple[tuple[str, str], ...]:
    """
    Lista ordenada (nombre, descripcion) de los indices de la observacion v2.

    Contexto:
        Fuente unica de la especificacion y del tamano; depende del catalogo porque el one-hot de
        identidad y el de montages se derivan de el.

    Args:
        catalogo (CatalogoJefes): Jefes y vocabulario union.

    Returns:
        tuple[tuple[str, str], ...]: Un par por indice.

    Raises:
        Ninguna.

    Ejemplo:
        len(campos_v2(catalogo))
    """
    campos: list[tuple[str, str]] = list(campos_jugador())
    campos += [(f"jefe_id_{p.id}", "one-hot de la identidad del actor jefe (por su clase Blueprint)")
               for p in catalogo.perfiles]
    campos += [
        ("jefe_vida", "vida / vida maxima del actor"),
        ("jefe_fase_avanzada", "1 si el actor es la segunda fase de su pelea (juice_fase2, chamuco)"),
        ("jefe_aturdido", "Is Stunned (tras un parry)"),
        ("jefe_ataca", "Attacking (no es fiable durante algunos combos)"),
        ("jefe_teletransportando", "Is Teleporting (solo AMLO y el Chupacabras; 0 en los demas)"),
        ("jefe_velocidad", "velocidad horizontal / 800 uu/s"),
        ("jefe_montage_tiempo", "posicion / duracion del montage activo del jefe"),
        ("jefe_montage_restante", "segundos de juego que faltan para terminar el montage / 4, tope 1"),
        ("jefe_montage_rate", "play rate del montage / 2"),
    ]
    campos += [(f"jefe_montage_{n}", "one-hot del montage del jefe (vocabulario union)")
               for n in ("ninguno", *catalogo.vocabulario, "otro")]
    campos += [
        ("amenaza_hay", "1 si hay algun impacto pendiente (montage o evento)"),
        ("amenaza_golpe_tiempo", "segundos hasta el proximo impacto / 1.5, tope 1 (1 si no hay)"),
        ("amenaza_golpe_parable", "el proximo impacto se puede parar"),
        ("amenaza_golpe_danio", "dano del proximo impacto / 50, tope 1"),
        ("amenaza_golpes_pendientes", "impactos que quedan en el ataque en curso / 5, tope 1"),
        ("amenaza_en_alcance", "el jugador esta dentro del alcance del proximo impacto"),
        ("amenaza_parry_tiempo", "segundos hasta el proximo impacto parable / 1.5, tope 1 (1 si no hay)"),
        ("amenaza_esquivar_tiempo", "segundos hasta el proximo impacto no parable / 1.5, tope 1 (1 si no hay)"),
        ("remota_activa", "1 si hay un ataque a distancia pendiente (proyectil, area, rayo o evento)"),
        ("remota_tiempo", "segundos al impacto de la amenaza a distancia / 2.5, tope 1 (1 si no hay)"),
        ("remota_esquivable", "se evita con la invulnerabilidad de la maniobra"),
        ("remota_parable", "se puede parar"),
        ("remota_danio", "dano de la amenaza a distancia / 50, tope 1"),
        ("remota_origen_distancia", "distancia al origen del ataque / 1000, tope 1 (0 si cae sobre el jugador)"),
        ("remota_origen_cos", "coseno del angulo del origen en el marco del jugador (0 = de frente)"),
        ("remota_origen_sen", "seno de ese angulo (+ = a su derecha)"),
        ("remota_sobre_jugador", "el ataque cae sobre la posicion del jugador (area, rayo)"),
        ("rel_distancia", "distancia 2D al jefe / 4000, tope 1"),
        ("rel_jefe_cos", "coseno del angulo del jefe en el marco del jugador (0 = de frente)"),
        ("rel_jefe_sen", "seno del angulo del jefe en el marco del jugador (+ = a su derecha)"),
        ("rel_mirada_cos", "coseno del angulo entre la mirada del jefe y la direccion hacia el jugador (1 = lo mira)"),
        ("rel_mirada_sen", "seno de ese angulo"),
    ]
    campos += [(f"accion_previa_{n}", "one-hot de la accion anterior") for n in acc.NOMBRES_ACCIONES]
    campos.append(("tiempo_episodio", "tiempo transcurrido / duracion maxima del episodio"))
    return tuple(campos)


def tamano_simple_v2(catalogo: CatalogoJefes) -> int:
    """
    Cantidad de valores de una observacion simple v2.

    Contexto:
        Espacio de observacion del entorno multi-jefe.

    Args:
        catalogo (CatalogoJefes): Jefes y vocabulario union.

    Returns:
        int: Longitud del vector de un instante.

    Raises:
        Ninguna.

    Ejemplo:
        tamano_simple_v2(catalogo)
    """
    return len(campos_v2(catalogo))


def _tiempo(amenaza: Amenaza | None, horizonte: float) -> float:
    """
    Tiempo hasta un impacto normalizado, 1.0 si no hay.

    Contexto:
        Rasgos de amenaza de la observacion.

    Args:
        amenaza (Amenaza | None): Impacto o None.
        horizonte (float): Segundos que valen 1.0.

    Returns:
        float: Valor en [0, 1].

    Raises:
        Ninguna.

    Ejemplo:
        _tiempo(None, 1.5)  # 1.0
    """
    return _fraccion(amenaza.segundos, horizonte) if amenaza else 1.0


class ConstructorObservacionV2:
    """
    Convierte la observacion cruda y las amenazas en el vector simple de campos_v2.

    Contexto:
        EntornoMultiJefe, en cada reset y cada step.
    """

    def __init__(self, catalogo: CatalogoJefes, config: ConfigEntorno | None = None) -> None:
        """
        Crea el constructor.

        Contexto:
            Construccion del entorno multi-jefe.

        Args:
            catalogo (CatalogoJefes): Jefes y vocabulario union.
            config (ConfigEntorno, opcional): Para la duracion maxima del episodio.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ConstructorObservacionV2(catalogo)
        """
        self.catalogo = catalogo
        self.config = config or ConfigEntorno()
        self.tamano = tamano_simple_v2(catalogo)
        self._vocabulario = catalogo.vocabulario

    def construir(self, obs: dict[str, Any], accion_previa: int, segundos: float, amenazas: Amenazas) -> Vector:
        """
        Vector de observacion del instante actual.

        Contexto:
            EntornoMultiJefe; la salida alimenta a ApiladorObservaciones.

        Args:
            obs (dict[str, Any]): Observacion cruda del puente.
            accion_previa (int): Accion ejecutada en el paso anterior.
            segundos (float): Segundos de juego transcurridos en el episodio.
            amenazas (Amenazas): Impactos pendientes calculados por el seguidor.

        Returns:
            Vector: float32 de longitud tamano; bloques ausentes quedan en cero.

        Raises:
            Ninguna.

        Ejemplo:
            vector = constructor.construir(obs, 0, 12.5, amenazas)
        """
        valores = bloque_jugador(obs.get("player") or {})
        valores += self._jefe(obs.get("boss") or {})
        valores += self._amenaza(amenazas)
        valores += self._remota(obs, amenazas)
        valores += self._relativo(obs)
        valores += _one_hot(accion_previa, acc.N_ACCIONES)
        valores.append(_fraccion(segundos, self.config.segundos_maximos))
        return np.asarray(valores, dtype=np.float32)

    def _jefe(self, b: dict[str, Any]) -> list[float]:
        """
        Bloque del jefe: identidad, estado y montage.

        Contexto:
            construir.

        Args:
            b (dict[str, Any]): Bloque "boss" (vacio si no hay).

        Returns:
            list[float]: Valores en el orden de campos_v2.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        perfil = self.catalogo.por_clase(b.get("cls")) if b else None
        identidad = [0.0] * len(self.catalogo.perfiles)
        if perfil is not None:
            identidad[perfil.indice] = 1.0
        montage = b.get("montage") or {}
        nombre = montage.get("name")
        largo = float(montage.get("len", 0.0))
        pos = float(montage.get("pos", 0.0))
        rate = float(montage.get("rate", 1.0)) or 1.0
        hp_max = float(b.get("hp_max", 0.0))
        restante = max(0.0, (largo - pos) / rate) if largo > 0 else 0.0
        escalares = [
            _fraccion(float(b.get("hp", 0.0)), hp_max) if hp_max > 0 else 0.0,
            float(bool(perfil and perfil.fase_avanzada)),
            float(bool(b.get("stunned"))),
            float(bool(b.get("attacking"))),
            float(bool(b.get("teleporting"))),
            _fraccion(_velocidad_horizontal(b), VEL_JEFE_MAX),
            _fraccion(pos, largo) if largo > 0 else 0.0,
            _fraccion(restante, MONTAGE_RESTANTE_MAX),
            _fraccion(rate if montage else 0.0, RATE_MAX),
        ]
        indice = indice_montage(nombre, self._vocabulario)
        return identidad + escalares + _one_hot(indice, tamano_codificacion(self._vocabulario))

    def _amenaza(self, amenazas: Amenazas) -> list[float]:
        """
        Rasgos genericos de la amenaza (calculados con la tabla del jefe).

        Contexto:
            construir.

        Args:
            amenazas (Amenazas): Impactos pendientes.

        Returns:
            list[float]: Ocho valores en el orden de campos_v2.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        proximo = amenazas.proximo
        parable = amenazas.proximo_parry
        no_parable = next((a for a in amenazas.pendientes if not a.parry), None)
        return [
            float(proximo is not None),
            _tiempo(proximo, GOLPE_HORIZONTE),
            float(bool(proximo and proximo.parry)),
            _fraccion(proximo.danio, DANIO_MAX) if proximo else 0.0,
            _fraccion(float(amenazas.del_ataque), GOLPES_MAX),
            float(bool(proximo and proximo.en_alcance(amenazas.distancia))),
            _tiempo(parable, GOLPE_HORIZONTE),
            _tiempo(no_parable, GOLPE_HORIZONTE),
        ]

    @staticmethod
    def _remota(obs: dict[str, Any], amenazas: Amenazas) -> list[float]:
        """
        Rasgos del ataque a distancia mas cercano.

        Contexto:
            construir. La direccion es la del origen del ataque en el marco del jugador; para un
            area que cae sobre el jugador la distancia es cero y la direccion no aporta.

        Args:
            obs (dict[str, Any]): Observacion cruda (posicion y orientacion del jugador).
            amenazas (Amenazas): Impactos pendientes.

        Returns:
            list[float]: Nueve valores en el orden de campos_v2.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        remota = amenazas.remota
        if remota is None:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sobre = remota.centro == "jugador" or remota.origen_xy is None
        distancia, coseno, seno = 0.0, 0.0, 0.0
        jugador = obs.get("player") or {}
        if not sobre and remota.origen_xy is not None and "pos" in jugador:
            dx = remota.origen_xy[0] - float(jugador["pos"][0])
            dy = remota.origen_xy[1] - float(jugador["pos"][1])
            relativo = math.atan2(dy, dx) - math.radians(float(jugador.get("yaw", 0.0)))
            distancia, coseno, seno = _fraccion(math.hypot(dx, dy), ORIGEN_MAX), math.cos(relativo), math.sin(relativo)
        return [
            1.0, _tiempo(remota, REMOTA_HORIZONTE), float(remota.esquivable), float(remota.parry),
            _fraccion(remota.danio, DANIO_MAX), distancia, coseno, seno, float(sobre),
        ]

    @staticmethod
    def _relativo(obs: dict[str, Any]) -> list[float]:
        """
        Geometria jugador-jefe: distancia, direccion del jefe y hacia donde mira.

        Contexto:
            construir. Igual que la v1 salvo la escala de la distancia (4000 uu).

        Args:
            obs (dict[str, Any]): Observacion cruda completa.

        Returns:
            list[float]: Cinco valores; ceros si falta el jugador o el jefe.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        p, b = obs.get("player"), obs.get("boss")
        if not p or not b or "pos" not in p or "pos" not in b:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        dx, dy = float(b["pos"][0]) - float(p["pos"][0]), float(b["pos"][1]) - float(p["pos"][1])
        hacia_jefe = math.atan2(dy, dx)
        relativo = hacia_jefe - math.radians(float(p.get("yaw", 0.0)))
        mirada = (hacia_jefe + math.pi) - math.radians(float(b.get("yaw", 0.0)))
        return [
            _fraccion(float(obs.get("dist", math.hypot(dx, dy))), DISTANCIA_MAX_V2),
            math.cos(relativo), math.sin(relativo), math.cos(mirada), math.sin(mirada),
        ]


def especificacion_markdown_v2(catalogo: CatalogoJefes, apilado: int = 4) -> str:
    """
    Documento de especificacion de la observacion v2 (cada indice, su nombre y significado).

    Contexto:
        python -m dmrl especificacion-v2 escribe este texto en docs/ESPECIFICACION_OBSERVACION_V2.md.

    Args:
        catalogo (CatalogoJefes): Jefes y vocabulario union.
        apilado (int): Observaciones apiladas, para documentar el tamano total.

    Returns:
        str: Markdown con la tabla de indices.

    Raises:
        Ninguna.

    Ejemplo:
        print(especificacion_markdown_v2(catalogo))
    """
    campos = campos_v2(catalogo)
    tamano = len(campos)
    lineas = [
        "# Especificacion de la observacion v2 (multi-jefe)",
        "",
        f"Observacion simple: {tamano} valores float32 (indices 0..{tamano - 1}). La politica recibe las "
        f"ultimas {apilado} concatenadas (la mas vieja primero): {tamano * apilado} valores.",
        "",
        "Todos los valores estan en [0, 1] salvo los cosenos y senos, que estan en [-1, 1]. Un bloque ausente "
        "(por ejemplo sin jefe durante una carga) se codifica en cero. La observacion v1 "
        "(docs/ESPECIFICACION_OBSERVACION.md) no cambia: los modelos entrenados con ella siguen cargando.",
        "",
        "Bloques: jugador (35, igual que la v1), identidad y estado del jefe, montage del jefe (vocabulario "
        "union), amenaza generica (tabla del jefe), amenaza a distancia, relativo jugador-jefe, accion previa "
        "y tiempo de episodio.",
        "",
        "| Indice | Nombre | Significado |",
        "|---|---|---|",
    ]
    lineas += [f"| {i} | {n} | {d} |" for i, (n, d) in enumerate(campos)]
    lineas += [
        "",
        "Identidades (orden del one-hot): " + ", ".join(f"{p.indice} {p.id}" for p in catalogo.perfiles),
        "",
        f"Vocabulario union de montages del jefe ({tamano_codificacion(catalogo.vocabulario)} posiciones, "
        "nombre exacto del juego): ninguno, " + ", ".join(catalogo.vocabulario) + ", otro",
        "",
    ]
    return "\n".join(lineas)
