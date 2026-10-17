"""
Reglas de victoria, derrota y fase por jefe de episodio.

Cada episodio empieza contra un jefe de episodio (juice, amlo, chamuco, nahual) y se gana al morir
una de sus clases de victoria: la fase 2 de Juice, el Chamuco (tanto en la pelea contra AMLO como en
los episodios directos contra el) o el Chupacabras. El paso de AMLO a su segunda fase es un hito
(premio de fase) pero no gana el episodio.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.recompensa import RESULTADO_MUERTE, RESULTADO_VICTORIA


def resolver_final_de(clases_victoria: tuple[str, ...]) -> Callable[[dict[str, Any]], str | None]:
    """
    Construye la regla de fin de episodio para un jefe.

    Contexto:
        EntornoMultiJefe, al elegir el jefe de cada episodio.

    Args:
        clases_victoria (tuple[str, ...]): Clases cuya muerte gana el episodio.

    Returns:
        Callable[[dict[str, Any]], str | None]: Funcion que recibe la observacion cruda y devuelve
            "victoria", "muerte" o None. Si el jugador muere en el mismo paso en que cae el jefe se
            cuenta la muerte (el juego ya lo reinicia).

    Raises:
        Ninguna.

    Ejemplo:
        resolver_final_de(("BP_Chamuco_C",))(obs)
    """

    def resolver(obs: dict[str, Any]) -> str | None:
        """
        Fin del episodio para la observacion dada.

        Contexto:
            Calculadora de recompensa y entorno.

        Args:
            obs (dict[str, Any]): Observacion cruda del puente.

        Returns:
            str | None: "victoria", "muerte" o None.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ev = obs.get("ev") or {}
        jugador = obs.get("player") or {}
        jefe = obs.get("boss") or {}
        if jugador.get("dead") or ev.get("player_death"):
            return RESULTADO_MUERTE
        if ev.get("victory") or (jefe.get("dead") and jefe.get("cls") in clases_victoria):
            return RESULTADO_VICTORIA
        return None

    return resolver


def resolver_fase_de(catalogo: CatalogoJefes) -> Callable[[dict[str, Any], int], int]:
    """
    Construye la regla de fase de la pelea (1 o 2) a partir de la clase del jefe.

    Contexto:
        EntornoMultiJefe y la calculadora de recompensa.

    Args:
        catalogo (CatalogoJefes): Jefes y su fase.

    Returns:
        Callable[[dict[str, Any], int], int]: Funcion (observacion, fase por defecto) -> fase; es 2
            si el actor jefe es una segunda fase, 1 si no, y la fase por defecto si no hay jefe.

    Raises:
        Ninguna.

    Ejemplo:
        resolver_fase_de(catalogo)(obs, 1)
    """

    def resolver(obs: dict[str, Any], por_defecto: int) -> int:
        """
        Fase de la pelea para la observacion dada.

        Contexto:
            Calculadora de recompensa.

        Args:
            obs (dict[str, Any]): Observacion cruda del puente.
            por_defecto (int): Fase si no hay jefe en la observacion.

        Returns:
            int: 1 o 2.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        jefe = obs.get("boss")
        if not jefe:
            return por_defecto
        perfil = catalogo.por_clase(jefe.get("cls"))
        return 2 if perfil is not None and perfil.fase_avanzada else 1

    return resolver
