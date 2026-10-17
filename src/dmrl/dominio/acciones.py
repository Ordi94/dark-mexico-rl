"""
Espacio de acciones discreto del agente y su mascara de validez.

Indices:
    0        nada
    1-8      moverse en 8 direcciones (0 grados = hacia el jefe, en pasos de 45 grados hacia su
             derecha)
    9-16     maniobra (roll) en las mismas 8 direcciones
    17       ataque
    18       ataque especial
    19       parry: pulsacion corta de LB (pulsa en esta decision, suelta en la siguiente)
    20       bloqueo sostenido: LB mientras se repita la accion
    21       tomar caguama

Correr no esta en la primera version. La mascara solo habilita lo que el juego realmente
acepta (medido en validar_mascara), de modo que el agente no aprende que una accion ignorada
"no sirve" cuando en realidad estaba bloqueada por el estado del personaje.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

N_ACCIONES = 22
ACCION_NADA = 0
ACCION_MOVER_PRIMERA = 1
ACCION_ROLL_PRIMERA = 9
ACCION_ATAQUE = 17
ACCION_ESPECIAL = 18
ACCION_PARRY = 19
ACCION_BLOQUEO = 20
ACCION_CAGUAMA = 21
N_DIRECCIONES = 8
PASO_DIRECCION_GRADOS = 360.0 / N_DIRECCIONES
# El especial exige Power >= 1.0 exacto. La barra sube de 0.1 en 0.1 (diez golpes dan
# 0.9999999999999999, que el juego rechaza) y el JSON redondea a 9 cifras, asi que el mod envia
# el booleano power_ready; POWER_LLENO solo se usa si el bloque no lo trae.
POWER_LLENO = 0.999999999
# Con Resistencia (estamina) en cero el juego ignora el roll, el ataque y casi siempre el especial
# (medido: 3-4% de efecto con Resistencia 0 contra 85-100% con mas de 0.02).
ESTAMINA_MINIMA = 0.02
# Montages del jugador que condicionan lo que el juego acepta (medido en validar_mascara).
MONTAGE_ROLL = "ALS_N_LandRoll_F_Montage"
MONTAGE_ESPECIAL = "Anim_SwordV2_Combo03_05_RM_Montage"
MONTAGE_REACCION_FUERTE = "Anim_Sword_Hurt_Big_F_Root_Montage"
MONTAGES_ATAQUE: tuple[str, ...] = (
    "Anim_SwordV2_Combo01_02_RM_Montage",
    "Anim_Sword_Attack_Slash05_Root_Montage",
    "Anim_SwordV2_Combo02_01_RM_Montage",
    "Anim_SwordV2_Combo03_04_RM_Montage",
)
# Dentro de un montage de ataque el siguiente golpe del combo solo entra pasada esta fraccion
# del montage (antes: 2-11% de efecto; despues: 45-90%).
FRACCION_VENTANA_COMBO = 0.25

NOMBRES_ACCIONES: tuple[str, ...] = (
    ("nada",)
    + tuple(f"mover_{int(i * PASO_DIRECCION_GRADOS)}" for i in range(N_DIRECCIONES))
    + tuple(f"roll_{int(i * PASO_DIRECCION_GRADOS)}" for i in range(N_DIRECCIONES))
    + ("ataque", "especial", "parry", "bloqueo", "caguama")
)


@dataclass(frozen=True)
class ComandoJuego:
    """
    Orden elemental para el juego, independiente del formato del protocolo.

    Contexto:
        Salida de comando_de_accion; la capa de aplicacion la traduce a dmbridge.Action.

    Atributos:
        btn: boton de un disparo ("rb", "y", "b", "x") o None.
        lb: bloqueo mantenido en este paso.
        mx: movimiento a la derecha del jefe en [-1, 1].
        my: movimiento hacia el jefe en [-1, 1].
    """

    btn: str | None = None
    lb: bool = False
    mx: float = 0.0
    my: float = 0.0


def vector_direccion(indice: int) -> tuple[float, float]:
    """
    Vector de movimiento (derecha, hacia el jefe) de una de las 8 direcciones.

    Contexto:
        comando_de_accion y el bot de reglas.

    Args:
        indice (int): Direccion 0..7; 0 es hacia el jefe y crece hacia su derecha.

    Returns:
        tuple[float, float]: (mx, my) con modulo 1.

    Raises:
        ValueError: Si el indice esta fuera de 0..7.

    Ejemplo:
        vector_direccion(2)  # (1.0, 0.0): a la derecha del jefe
    """
    if not 0 <= indice < N_DIRECCIONES:
        raise ValueError(f"direccion fuera de 0..{N_DIRECCIONES - 1}: {indice}")
    angulo = math.radians(indice * PASO_DIRECCION_GRADOS)
    return round(math.sin(angulo), 6), round(math.cos(angulo), 6)


def comando_de_accion(accion: int, accion_previa: int = ACCION_NADA) -> ComandoJuego:
    """
    Traduce una accion discreta a la orden elemental del paso.

    Contexto:
        DarkMexicoEnv.step, justo antes de enviar el paso al juego.

    Args:
        accion (int): Indice de accion 0..21.
        accion_previa (int): Accion del paso anterior; hace que dos parries seguidos sean
            pulsar-soltar-pulsar en lugar de un bloqueo sostenido.

    Returns:
        ComandoJuego: Orden del paso.

    Raises:
        ValueError: Si la accion esta fuera de 0..21.

    Ejemplo:
        comando_de_accion(ACCION_ATAQUE)  # ComandoJuego(btn="rb")
    """
    if not 0 <= accion < N_ACCIONES:
        raise ValueError(f"accion fuera de 0..{N_ACCIONES - 1}: {accion}")
    if accion == ACCION_NADA:
        return ComandoJuego()
    if accion < ACCION_ROLL_PRIMERA:
        mx, my = vector_direccion(accion - ACCION_MOVER_PRIMERA)
        return ComandoJuego(mx=mx, my=my)
    if accion < ACCION_ATAQUE:
        mx, my = vector_direccion(accion - ACCION_ROLL_PRIMERA)
        return ComandoJuego(btn="b", mx=mx, my=my)
    if accion == ACCION_ATAQUE:
        return ComandoJuego(btn="rb")
    if accion == ACCION_ESPECIAL:
        return ComandoJuego(btn="y")
    if accion == ACCION_PARRY:
        return ComandoJuego(lb=accion_previa != ACCION_PARRY)
    if accion == ACCION_BLOQUEO:
        return ComandoJuego(lb=True)
    return ComandoJuego(btn="x")


def mascara_acciones(jugador: dict[str, Any] | None) -> list[bool]:
    """
    Calcula que acciones acepta el juego en el estado actual del jugador.

    Contexto:
        DarkMexicoEnv.action_masks, consumida por MaskablePPO en cada decision.

    Args:
        jugador (dict | None): Bloque "player" de la observacion cruda; None si no hay.

    Returns:
        list[bool]: 22 valores; "nada" siempre es valido, de modo que nunca queda una
            mascara vacia.

    Raises:
        Ninguna: Los campos ausentes se tratan como no disponibles.

    Ejemplo:
        mascara_acciones({"hp": 100, "hp_max": 100, "caguamas": 3})[ACCION_CAGUAMA]  # False
    """
    valida = [False] * N_ACCIONES
    valida[ACCION_NADA] = True
    if not jugador or jugador.get("dead"):
        return valida
    for i in range(ACCION_MOVER_PRIMERA, ACCION_ROLL_PRIMERA):
        valida[i] = True
    if jugador.get("drinking"):
        # Durante la bebida solo se puede caminar; todo lo demas lo ignora el juego.
        return valida
    montage = jugador.get("montage") or {}
    nombre = montage.get("name")
    largo = float(montage.get("len", 0.0))
    fraccion = float(montage.get("pos", 0.0)) / largo if largo > 0 else 1.0
    con_estamina = float(jugador.get("stamina", 0.0)) >= ESTAMINA_MINIMA
    if jugador.get("can_roll") and con_estamina and nombre != MONTAGE_ROLL:
        for i in range(ACCION_ROLL_PRIMERA, ACCION_ATAQUE):
            valida[i] = True
    if jugador.get("can_attack") and con_estamina:
        en_combo_cerrado = nombre in MONTAGES_ATAQUE and fraccion < FRACCION_VENTANA_COMBO
        valida[ACCION_ATAQUE] = not en_combo_cerrado
        valida[ACCION_ESPECIAL] = _barra_llena(jugador) and nombre != MONTAGE_ESPECIAL
    # El bloqueo se acepta casi siempre, incluso repetido sobre su propio montage (reinicia la
    # ventana de parry); no durante la reaccion fuerte ni el especial.
    puede_bloquear = nombre not in (MONTAGE_REACCION_FUERTE, MONTAGE_ESPECIAL)
    valida[ACCION_PARRY] = puede_bloquear
    valida[ACCION_BLOQUEO] = puede_bloquear
    valida[ACCION_CAGUAMA] = _puede_beber(jugador)
    return valida


def _puede_beber(jugador: dict[str, Any]) -> bool:
    """
    Condiciones medidas para que X inicie la bebida.

    Contexto:
        mascara_acciones. El juego exige vida menor al maximo, caguamas disponibles y
        Can Heal (falso al correr o en una reaccion de golpe fuerte).

    Args:
        jugador (dict[str, Any]): Bloque "player" de la observacion.

    Returns:
        bool: True si la bebida arranca.

    Raises:
        Ninguna.

    Ejemplo:
        _puede_beber({"hp": 40, "hp_max": 100, "caguamas": 1, "can_heal": True})
    """
    return (
        int(jugador.get("caguamas", 0)) > 0
        and float(jugador.get("hp", 0.0)) < float(jugador.get("hp_max", 0.0))
        and bool(jugador.get("can_heal"))
    )


def _barra_llena(jugador: dict[str, Any]) -> bool:
    """
    Indica si la barra de poder permite el especial.

    Contexto:
        mascara_acciones. Prefiere power_ready (comparacion exacta hecha dentro del juego).

    Args:
        jugador (dict[str, Any]): Bloque "player" de la observacion.

    Returns:
        bool: True si el juego aceptara el especial por la barra.

    Raises:
        Ninguna.

    Ejemplo:
        _barra_llena({"power": 1.0, "power_ready": False})  # False
    """
    if "power_ready" in jugador:
        return bool(jugador["power_ready"])
    return float(jugador.get("power", 0.0)) >= POWER_LLENO
