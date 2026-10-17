"""
Utilidades de prueba del paquete dmrl: observaciones crudas minimas y un juego sintetico.

El juego sintetico implementa PuertoJuego con una dinamica de pelea muy simple pero con la misma
forma de observacion que el puente real, de modo que el entorno, los callbacks y el entrenamiento
se ejercitan sin necesitar el ejecutable del juego.
"""

from __future__ import annotations

import copy
from typing import Any

from dmbridge import Action, BridgeError


def jugador(**cambios: Any) -> dict[str, Any]:
    """
    Bloque "player" de una observacion cruda en reposo.

    Contexto:
        Base de las observaciones de las pruebas.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        dict[str, Any]: Jugador con vida llena, 3 caguamas y todo disponible.

    Raises:
        Ninguna.

    Ejemplo:
        jugador(hp=40)
    """
    base: dict[str, Any] = {
        "hp": 100.0, "hp_max": 100.0, "dead": False, "blocking": False, "invincible": False,
        "stamina": 1.0, "power": 1.0, "power_ready": True, "caguamas": 3, "attacking": False, "combo": 0,
        "rolling": False,
        "parry_window": False, "can_heal": True, "can_attack": True, "can_roll": True, "target_lock": True,
        "pos": [0.0, 0.0, 0.0], "yaw": 0.0, "vel": [0.0, 0.0, 0.0], "control_yaw": 0.0, "drinking": False,
    }
    base.update(cambios)
    return base


def jefe(**cambios: Any) -> dict[str, Any]:
    """
    Bloque "boss" de una observacion cruda en reposo (fase 1).

    Contexto:
        Base de las observaciones de las pruebas.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        dict[str, Any]: Jefe de fase 1 con vida completa y sin montage.

    Raises:
        Ninguna.

    Ejemplo:
        jefe(stunned=True)
    """
    base: dict[str, Any] = {
        "cls": "BP_Enemy_Juice_C", "hp": 2000.0, "hp_max": 2000.0, "phase": 0, "attacking": False,
        "stunned": False, "dead": False, "parry_count": 0, "pos": [300.0, 0.0, 0.0], "yaw": 180.0,
        "vel": [0.0, 0.0, 0.0],
    }
    base.update(cambios)
    return base


def montage(nombre: str, pos: float = 0.0, rate: float = 1.0, largo: float = 2.0) -> dict[str, Any]:
    """
    Bloque "montage" de una entidad.

    Contexto:
        Observaciones con animaciones activas.

    Args:
        nombre (str): Nombre del montage.
        pos (float): Posicion en segundos.
        rate (float): Play rate.
        largo (float): Duracion del montage.

    Returns:
        dict[str, Any]: Montage con name, pos, rate y len.

    Raises:
        Ninguna.

    Ejemplo:
        montage("Boss_Attack_Uppercut_InP_Montage", 0.5)
    """
    return {"name": nombre, "pos": pos, "rate": rate, "len": largo}


def obs(jug: dict[str, Any] | None = None, jf: dict[str, Any] | None = None, dist: float = 300.0,
        **eventos: Any) -> dict[str, Any]:
    """
    Observacion cruda completa.

    Contexto:
        Entrada de la mayoria de las pruebas del dominio.

    Args:
        jug (dict, opcional): Bloque del jugador (por defecto en reposo).
        jf (dict, opcional): Bloque del jefe (por defecto en reposo).
        dist (float): Distancia 2D.
        **eventos (Any): Campos de "ev" (dmg_dealt, parries, victory, ...).

    Returns:
        dict[str, Any]: Observacion con player, boss, dist y ev.

    Raises:
        Ninguna.

    Ejemplo:
        obs(dmg_dealt=20.0)
    """
    return {"player": jug if jug is not None else jugador(), "boss": jf if jf is not None else jefe(),
            "dist": dist, "ev": dict(eventos), "t": 0.0}


class JuegoSintetico:
    """
    Pelea sintetica con la interfaz de PuertoJuego.

    Contexto:
        Pruebas del entorno, de la evaluacion y del entrenamiento. Cada periodo de pasos el
        jefe golpea (25) salvo que el jugador bloquee; atacar quita 40 al jefe; con vida del
        jefe <= 0 hay victoria y con vida del jugador <= 0 muerte.
    """

    def __init__(self, periodo: int = 8, falla_en: int | None = None, falla_reset: int = 0) -> None:
        """
        Crea el juego.

        Contexto:
            Cada prueba construye el suyo (o lo recibe de una fabrica).

        Args:
            periodo (int): Pasos entre golpes del jefe.
            falla_en (int, opcional): Numero de paso en el que step lanza BridgeError una vez.
            falla_reset (int): Cuantos reset consecutivos lanzan BridgeError antes de funcionar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            juego = JuegoSintetico(periodo=5)
        """
        self.periodo = periodo
        self.falla_en = falla_en
        self.falla_reset = falla_reset
        self.pasos_totales = 0
        self.resets: list[tuple[float | None, int | None]] = []
        self.acciones: list[tuple[Action, int]] = []
        self.cerrado = False
        self._estado: dict[str, Any] = {}

    def reset(self, timeout: float = 0.0, boss_hp_mult: float | None = None,
              start_phase: int | None = None) -> dict[str, Any]:
        """
        Reinicia la pelea sintetica.

        Contexto:
            DarkMexicoEnv.reset.

        Args:
            timeout (float): Ignorado.
            boss_hp_mult (float, opcional): Multiplicador de vida del jefe.
            start_phase (int, opcional): 1 o 2.

        Returns:
            dict[str, Any]: Observacion inicial.

        Raises:
            BridgeError: Mientras queden fallos programados.

        Ejemplo:
            juego.reset(boss_hp_mult=0.3)
        """
        if self.falla_reset > 0:
            self.falla_reset -= 1
            raise BridgeError("reset: fallo programado")
        self.resets.append((boss_hp_mult, start_phase))
        mult = boss_hp_mult or 1.0
        fase2 = start_phase == 2
        self._estado = {"n": 0, "hp": 100.0, "jefe": (1000.0 if fase2 else 2000.0) * mult,
                        "fase2": fase2, "caguamas": 3, "lock": False}
        return self._observacion({})

    def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
        """
        Avanza un paso de la pelea sintetica.

        Contexto:
            DarkMexicoEnv.step.

        Args:
            action (Action): Accion del agente.
            n_frames (int): Frames del paso.

        Returns:
            dict[str, Any]: Observacion con eventos del paso.

        Raises:
            BridgeError: En el paso programado para fallar.

        Ejemplo:
            juego.step(Action(btn="rb"), 2)
        """
        self.pasos_totales += 1
        if self.falla_en is not None and self.pasos_totales == self.falla_en:
            self.falla_en = None
            raise BridgeError("step: fallo programado")
        self.acciones.append((action, n_frames))
        e = self._estado
        e["n"] += 1
        ev: dict[str, Any] = {}
        if action.btn == "rs":
            e["lock"] = True
        if action.btn == "rb":
            e["jefe"] -= 40.0
            ev["dmg_dealt"] = 40.0
        if e["n"] % self.periodo == 0:
            if action.lb:
                ev["parries"] = 1
            else:
                e["hp"] -= 25.0
        if e["jefe"] <= 0:
            ev["victory"] = True
        if e["hp"] <= 0:
            ev["player_death"] = True
        return self._observacion(ev)

    def close(self, quit_game: bool = True) -> None:
        """
        Marca el juego como cerrado.

        Contexto:
            Cierre del entorno.

        Args:
            quit_game (bool): Ignorado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            juego.close()
        """
        self.cerrado = True

    def _observacion(self, ev: dict[str, Any]) -> dict[str, Any]:
        """
        Construye la observacion cruda del estado actual.

        Contexto:
            reset y step.

        Args:
            ev (dict[str, Any]): Eventos del paso.

        Returns:
            dict[str, Any]: Observacion con jugador, jefe, distancia y eventos.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        e = self._estado
        j = jugador(hp=e["hp"], caguamas=e["caguamas"], dead=e["hp"] <= 0, target_lock=e["lock"])
        b = jefe(hp=max(e["jefe"], 0.0), hp_max=1000.0 if e["fase2"] else 2000.0, phase=1 if e["fase2"] else 0,
                 cls="BP_Enemy_JuicePhase2_C" if e["fase2"] else "BP_Enemy_Juice_C", dead=e["jefe"] <= 0)
        return copy.deepcopy({"player": j, "boss": b, "dist": 300.0, "ev": ev, "t": e["n"] * 0.0667})
