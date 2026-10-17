"""
Construccion del vector de observacion del agente a partir de la observacion cruda del puente.

Cada observacion simple es un vector float32 con todos los valores en [0, 1] (o [-1, 1] para
senos y cosenos). El orden y el significado de cada indice estan en CAMPOS, que tambien genera
docs/ESPECIFICACION_OBSERVACION.md (python -m dmrl especificacion). La politica ve las ultimas
`apilado` observaciones concatenadas (ver ApiladorObservaciones).

Los montages se codifican en one-hot sobre un vocabulario fijo: indice 0 = sin montage, 1..n =
montage conocido y n+1 = "otro/desconocido".
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.dominio.vocabulario import VOCAB_JEFE, VOCAB_JUGADOR, indice_montage, tamano_codificacion

VEL_JUGADOR_MAX = 800.0
VEL_JEFE_MAX = 800.0
DISTANCIA_MAX = 2000.0
CAGUAMAS_MAX = 3.0
MONTAGE_RESTANTE_MAX = 4.0
GOLPE_HORIZONTE = 1.5
RATE_MAX = 2.0
PREFIJOS_REACCION = ("Anim_SwordV2_Hit", "Anim_Sword_Hurt")
MONTAGE_BEBIDA = "MM_Idle1_Montage"

Vector = NDArray[np.float32]


def _campos() -> tuple[tuple[str, str], ...]:
    """
    Genera la lista ordenada (nombre, descripcion) de los indices de la observacion simple.

    Contexto:
        Se evalua una vez al importar; es la fuente unica de la especificacion y del tamano.

    Args:
        Ninguno.

    Returns:
        tuple[tuple[str, str], ...]: Un par por indice.

    Raises:
        Ninguna.

    Ejemplo:
        len(_campos())
    """
    campos: list[tuple[str, str]] = [
        ("jugador_vida", "vida / vida maxima"),
        ("jugador_resistencia", "Resistencia (estamina), 0..1"),
        ("jugador_caguamas", "caguamas restantes / 3"),
        ("jugador_poder", "barra de poder del especial, 0..1"),
        ("jugador_bloquea", "IsBlocking"),
        ("jugador_bebiendo", "montage de la bebida activo"),
        ("jugador_progreso_bebida", "posicion / duracion del montage de la bebida (0 si no bebe)"),
        ("jugador_reaccion_golpe", "montage de reaccion a golpe (Hit_* o Hurt_*)"),
        ("jugador_rueda", "IsRolling"),
        ("jugador_ventana_parry", "Is Within Parry Window"),
        ("jugador_ataca", "Is Attacking"),
        ("jugador_puede_atacar", "CanAttack"),
        ("jugador_puede_rodar", "CanRoll"),
        ("jugador_puede_curar", "Can Heal"),
        ("jugador_enfocado", "TargetLock?"),
        ("jugador_velocidad", "velocidad horizontal / 800 uu/s"),
        ("jugador_montage_tiempo", "posicion / duracion del montage activo del jugador"),
    ]
    campos += [(f"jugador_montage_{n}", "one-hot del montage del jugador") for n in _etiquetas(VOCAB_JUGADOR)]
    campos += [
        ("jefe_vida", "vida / vida maxima de su fase (la escala del curriculo ya aplicada)"),
        ("jefe_fase1", "one-hot de fase: fase 1"),
        ("jefe_fase2", "one-hot de fase: fase 2"),
        ("jefe_aturdido", "Is Stunned (tras un parry)"),
        ("jefe_ataca", "Attacking (no es fiable durante algunos combos)"),
        ("jefe_velocidad", "velocidad horizontal / 800 uu/s"),
        ("jefe_montage_tiempo", "posicion / duracion del montage activo del jefe"),
        ("jefe_montage_restante", "segundos de juego que faltan para terminar el montage / 4, tope 1"),
        ("jefe_montage_rate", "play rate del montage / 2"),
        ("jefe_golpe_pendiente", "1 si queda algun golpe por delante en el montage (tabla medida)"),
        ("jefe_golpe_tiempo", "segundos hasta el proximo golpe / 1.5, tope 1 (1 si no hay)"),
        ("jefe_golpe_parable", "el proximo golpe se puede parar"),
        ("jefe_parry_tiempo", "segundos hasta el proximo golpe parable / 1.5, tope 1 (1 si no hay)"),
    ]
    campos += [(f"jefe_montage_{n}", "one-hot del montage del jefe") for n in _etiquetas(VOCAB_JEFE)]
    campos += [
        ("rel_distancia", "distancia 2D al jefe / 2000, tope 1"),
        ("rel_jefe_cos", "coseno del angulo del jefe en el marco del jugador (0 = de frente)"),
        ("rel_jefe_sen", "seno del angulo del jefe en el marco del jugador (+ = a su derecha)"),
        ("rel_mirada_cos", "coseno del angulo entre la mirada del jefe y la direccion hacia el jugador (1 = lo mira)"),
        ("rel_mirada_sen", "seno de ese angulo"),
    ]
    campos += [(f"accion_previa_{n}", "one-hot de la accion anterior") for n in acc.NOMBRES_ACCIONES]
    campos.append(("tiempo_episodio", "tiempo transcurrido / duracion maxima del episodio"))
    return tuple(campos)


def _etiquetas(vocab: tuple[str, ...]) -> tuple[str, ...]:
    """
    Etiquetas de los indices de un one-hot de montages.

    Contexto:
        _campos.

    Args:
        vocab (tuple[str, ...]): Vocabulario de montages.

    Returns:
        tuple[str, ...]: "ninguno", cada montage del vocabulario y "otro".

    Raises:
        Ninguna.

    Ejemplo:
        _etiquetas(("A",))  # ("ninguno", "A", "otro")
    """
    return ("ninguno", *vocab, "otro")


CAMPOS = _campos()
TAMANO_SIMPLE = len(CAMPOS)
TAMANO_JUGADOR = 17 + tamano_codificacion(VOCAB_JUGADOR)


def _fraccion(valor: float, maximo: float) -> float:
    """
    Normaliza valor / maximo y lo acota a [0, 1].

    Contexto:
        Todas las magnitudes escalares de la observacion.

    Args:
        valor (float): Valor medido.
        maximo (float): Escala de normalizacion (> 0).

    Returns:
        float: Valor en [0, 1].

    Raises:
        ZeroDivisionError: Si maximo es cero.

    Ejemplo:
        _fraccion(600.0, 800.0)  # 0.75
    """
    return min(1.0, max(0.0, valor / maximo))


def _one_hot(indice: int, tamano: int) -> list[float]:
    """
    Vector one-hot.

    Contexto:
        Montages y accion previa.

    Args:
        indice (int): Posicion activa.
        tamano (int): Longitud del vector.

    Returns:
        list[float]: Ceros con un 1.0 en indice.

    Raises:
        IndexError: Si el indice esta fuera de rango.

    Ejemplo:
        _one_hot(1, 3)  # [0.0, 1.0, 0.0]
    """
    vector = [0.0] * tamano
    vector[indice] = 1.0
    return vector


def _velocidad_horizontal(entidad: dict[str, Any]) -> float:
    """
    Modulo de la velocidad en el plano.

    Contexto:
        Jugador y jefe.

    Args:
        entidad (dict[str, Any]): Bloque con "vel" [x, y, z].

    Returns:
        float: uu/s; 0 si no hay velocidad.

    Raises:
        Ninguna.

    Ejemplo:
        _velocidad_horizontal({"vel": [3.0, 4.0, 0.0]})  # 5.0
    """
    vel = entidad.get("vel") or (0.0, 0.0, 0.0)
    return math.hypot(float(vel[0]), float(vel[1]))


def bloque_jugador(p: dict[str, Any]) -> list[float]:
    """
    Bloque del jugador de la observacion (compartido por la observacion v1 y la v2).

    Contexto:
        ConstructorObservacion._jugador y el constructor de la observacion multi-jefe: el jugador
        es el mismo personaje contra cualquier jefe, asi que su codificacion no cambia.

    Args:
        p (dict[str, Any]): Bloque "player" de la observacion cruda (vacio si no hay).

    Returns:
        list[float]: 17 escalares y el one-hot del montage del jugador, en el orden de CAMPOS.

    Raises:
        Ninguna.

    Ejemplo:
        len(bloque_jugador({}))  # 35
    """
    montage = p.get("montage") or {}
    nombre = montage.get("name")
    largo = float(montage.get("len", 0.0))
    tiempo = _fraccion(float(montage.get("pos", 0.0)), largo) if largo > 0 else 0.0
    bebiendo = nombre == MONTAGE_BEBIDA
    hp_max = float(p.get("hp_max", 100.0)) or 100.0
    escalares = [
        _fraccion(float(p.get("hp", 0.0)), hp_max),
        _fraccion(float(p.get("stamina", 0.0)), 1.0),
        _fraccion(float(p.get("caguamas", 0)), CAGUAMAS_MAX),
        _fraccion(float(p.get("power", 0.0)), 1.0),
        float(bool(p.get("blocking"))),
        float(bebiendo),
        tiempo if bebiendo else 0.0,
        float(bool(nombre and nombre.startswith(PREFIJOS_REACCION))),
        float(bool(p.get("rolling"))),
        float(bool(p.get("parry_window"))),
        float(bool(p.get("attacking"))),
        float(bool(p.get("can_attack"))),
        float(bool(p.get("can_roll"))),
        float(bool(p.get("can_heal"))),
        float(bool(p.get("target_lock"))),
        _fraccion(_velocidad_horizontal(p), VEL_JUGADOR_MAX),
        tiempo,
    ]
    return escalares + _one_hot(indice_montage(nombre, VOCAB_JUGADOR), tamano_codificacion(VOCAB_JUGADOR))


def campos_jugador() -> tuple[tuple[str, str], ...]:
    """
    Nombre y significado de los indices del bloque del jugador.

    Contexto:
        Especificaciones de las observaciones v1 y v2.

    Args:
        Ninguno.

    Returns:
        tuple[tuple[str, str], ...]: Un par por valor de bloque_jugador.

    Raises:
        Ninguna.

    Ejemplo:
        len(campos_jugador())  # 35
    """
    return CAMPOS[:TAMANO_JUGADOR]


class ConstructorObservacion:
    """
    Convierte la observacion cruda en el vector simple de CAMPOS.

    Contexto:
        DarkMexicoEnv, en cada reset y cada step.
    """

    def __init__(self, tabla: TablaGolpes, config: ConfigEntorno | None = None) -> None:
        """
        Crea el constructor con la tabla de golpes y la configuracion.

        Contexto:
            Construccion del entorno.

        Args:
            tabla (TablaGolpes): Golpes medidos del jefe.
            config (ConfigEntorno, opcional): Para la duracion maxima del episodio.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            constructor = ConstructorObservacion(TablaGolpes())
        """
        self.tabla = tabla
        self.config = config or ConfigEntorno()

    def construir(self, obs: dict[str, Any], accion_previa: int, segundos: float) -> Vector:
        """
        Vector de observacion del instante actual.

        Contexto:
            DarkMexicoEnv; la salida alimenta a ApiladorObservaciones.

        Args:
            obs (dict[str, Any]): Observacion cruda del puente.
            accion_previa (int): Accion ejecutada en el paso anterior.
            segundos (float): Segundos de juego transcurridos en el episodio.

        Returns:
            Vector: float32 de longitud TAMANO_SIMPLE; bloques ausentes quedan en cero.

        Raises:
            Ninguna.

        Ejemplo:
            vector = constructor.construir(obs, 0, 12.5)
        """
        valores: list[float] = []
        valores += self._jugador(obs.get("player") or {})
        valores += self._jefe(obs.get("boss") or {})
        valores += self._relativo(obs)
        valores += _one_hot(accion_previa, acc.N_ACCIONES)
        valores.append(_fraccion(segundos, self.config.segundos_maximos))
        return np.asarray(valores, dtype=np.float32)

    def _jugador(self, p: dict[str, Any]) -> list[float]:
        """
        Bloque del jugador.

        Contexto:
            construir.

        Args:
            p (dict[str, Any]): Bloque "player" (vacio si no hay).

        Returns:
            list[float]: Valores en el orden de CAMPOS.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return bloque_jugador(p)

    def _jefe(self, b: dict[str, Any]) -> list[float]:
        """
        Bloque del jefe, incluido el tiempo al proximo golpe segun la tabla medida.

        Contexto:
            construir.

        Args:
            b (dict[str, Any]): Bloque "boss" (vacio si no hay).

        Returns:
            list[float]: Valores en el orden de CAMPOS.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        montage = b.get("montage") or {}
        nombre = montage.get("name")
        largo = float(montage.get("len", 0.0))
        pos = float(montage.get("pos", 0.0))
        rate = float(montage.get("rate", 1.0)) or 1.0
        hp_max = float(b.get("hp_max", 0.0))
        fase2 = bool(b) and (int(b.get("phase", 0)) >= 1)
        pendiente = self.tabla.proximo_golpe(nombre, pos, rate) if b else None
        pendiente_parry = self.tabla.proximo_golpe(nombre, pos, rate, solo_parry=True) if b else None
        restante = max(0.0, (largo - pos) / rate) if largo > 0 else 0.0
        escalares = [
            _fraccion(float(b.get("hp", 0.0)), hp_max) if hp_max > 0 else 0.0,
            float(bool(b) and not fase2),
            float(fase2),
            float(bool(b.get("stunned"))),
            float(bool(b.get("attacking"))),
            _fraccion(_velocidad_horizontal(b), VEL_JEFE_MAX),
            _fraccion(pos, largo) if largo > 0 else 0.0,
            _fraccion(restante, MONTAGE_RESTANTE_MAX),
            _fraccion(rate if montage else 0.0, RATE_MAX),
            float(pendiente is not None),
            _fraccion(pendiente.segundos, GOLPE_HORIZONTE) if pendiente else 1.0,
            float(bool(pendiente and pendiente.golpe.parry)),
            _fraccion(pendiente_parry.segundos, GOLPE_HORIZONTE) if pendiente_parry else 1.0,
        ]
        return escalares + _one_hot(indice_montage(nombre, VOCAB_JEFE), tamano_codificacion(VOCAB_JEFE))

    @staticmethod
    def _relativo(obs: dict[str, Any]) -> list[float]:
        """
        Geometria jugador-jefe: distancia, direccion del jefe y hacia donde mira.

        Contexto:
            construir. El angulo del jefe se mide respecto a la orientacion del jugador;
            la mirada del jefe respecto a la direccion jefe -> jugador (1 = lo esta mirando).

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
            _fraccion(float(obs.get("dist", math.hypot(dx, dy))), DISTANCIA_MAX),
            math.cos(relativo), math.sin(relativo),
            math.cos(mirada), math.sin(mirada),
        ]


class ApiladorObservaciones:
    """
    Memoria corta: concatena las ultimas N observaciones simples.

    Contexto:
        DarkMexicoEnv. Al empezar un episodio la ventana se llena con la primera observacion
        para que el vector tenga siempre la misma forma.
    """

    def __init__(self, tamano: int, repeticiones: int) -> None:
        """
        Crea el apilador.

        Contexto:
            Construccion del entorno.

        Args:
            tamano (int): Longitud de cada observacion simple.
            repeticiones (int): Observaciones apiladas (>= 1).

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si repeticiones es menor que 1.

        Ejemplo:
            apilador = ApiladorObservaciones(TAMANO_SIMPLE, 4)
        """
        if repeticiones < 1:
            raise ValueError("repeticiones debe ser >= 1")
        self.tamano = tamano
        self._cola: deque[Vector] = deque(maxlen=repeticiones)
        self.repeticiones = repeticiones

    def iniciar(self, primera: Vector) -> Vector:
        """
        Llena la ventana con la primera observacion del episodio.

        Contexto:
            DarkMexicoEnv.reset.

        Args:
            primera (Vector): Observacion simple inicial.

        Returns:
            Vector: Observacion apilada.

        Raises:
            ValueError: Si la longitud no coincide con tamano.

        Ejemplo:
            apilado = apilador.iniciar(vector)
        """
        self._validar(primera)
        self._cola.clear()
        for _ in range(self.repeticiones):
            self._cola.append(primera)
        return self.apilado()

    def agregar(self, nueva: Vector) -> Vector:
        """
        Agrega una observacion y descarta la mas vieja.

        Contexto:
            DarkMexicoEnv.step.

        Args:
            nueva (Vector): Observacion simple del paso.

        Returns:
            Vector: Observacion apilada (la mas vieja primero).

        Raises:
            ValueError: Si la longitud no coincide con tamano.

        Ejemplo:
            apilado = apilador.agregar(vector)
        """
        self._validar(nueva)
        self._cola.append(nueva)
        return self.apilado()

    def apilado(self) -> Vector:
        """
        Concatenacion actual de la ventana.

        Contexto:
            iniciar y agregar.

        Args:
            Ninguno.

        Returns:
            Vector: float32 de longitud tamano * repeticiones.

        Raises:
            Ninguna.

        Ejemplo:
            apilador.apilado().shape
        """
        if not self._cola:
            return np.zeros(self.tamano * self.repeticiones, dtype=np.float32)
        return np.concatenate(list(self._cola)).astype(np.float32, copy=False)

    def _validar(self, vector: Vector) -> None:
        """
        Comprueba la longitud de una observacion simple.

        Contexto:
            iniciar y agregar.

        Args:
            vector (Vector): Observacion a comprobar.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si la longitud no coincide con tamano.

        Ejemplo:
            (uso interno)
        """
        if vector.shape != (self.tamano,):
            raise ValueError(f"se esperaba un vector de {self.tamano} valores, llego {vector.shape}")


def especificacion_markdown(apilado: int = 4) -> str:
    """
    Documento de especificacion de la observacion (cada indice, su nombre y significado).

    Contexto:
        python -m dmrl especificacion escribe este texto en docs/ESPECIFICACION_OBSERVACION.md.

    Args:
        apilado (int): Observaciones apiladas, para documentar el tamano total.

    Returns:
        str: Markdown con la tabla de indices.

    Raises:
        Ninguna.

    Ejemplo:
        print(especificacion_markdown())
    """
    lineas = [
        "# Especificacion de la observacion",
        "",
        f"Observacion simple: {TAMANO_SIMPLE} valores float32 (indices 0..{TAMANO_SIMPLE - 1}). "
        f"La politica recibe las ultimas {apilado} concatenadas (la mas vieja primero): "
        f"{TAMANO_SIMPLE * apilado} valores.",
        "",
        "Todos los valores estan en [0, 1] salvo los cosenos y senos, que estan en [-1, 1]. "
        "Un bloque ausente (por ejemplo sin jefe durante una carga) se codifica en cero.",
        "",
        "| Indice | Nombre | Significado |",
        "|---|---|---|",
    ]
    lineas += [f"| {i} | {n} | {d} |" for i, (n, d) in enumerate(CAMPOS)]
    lineas += [
        "",
        "Vocabularios de montages (nombre exacto del juego):",
        "",
        f"- Jugador ({tamano_codificacion(VOCAB_JUGADOR)} posiciones): "
        + ", ".join(_etiquetas(VOCAB_JUGADOR)),
        f"- Jefe ({tamano_codificacion(VOCAB_JEFE)} posiciones): "
        + ", ".join(_etiquetas(VOCAB_JEFE)),
        "",
    ]
    return "\n".join(lineas)
