"""
Utilidades de prueba del entorno multi-jefe: un catalogo pequeno y un juego sintetico que entiende la
orden reset con la descripcion del jefe.
"""

from __future__ import annotations

import copy
from typing import Any

from dmbridge import Action, BridgeError
from dmrl.dominio.jefes import CatalogoJefes, PerfilJefe


def registro(identificador: str, indice: int, mapa: str = "LVL_AMLO", clases: list[str] | None = None,
             **cambios: Any) -> dict[str, Any]:
    """
    Registro de jefe minimo en el formato de data/jefes/<id>.json.

    Contexto:
        Base de los catalogos de prueba.

    Args:
        identificador (str): Id del jefe.
        indice (int): Indice de identidad.
        mapa (str): Mapa.
        clases (list[str], opcional): Clases Blueprint; por defecto BP_<ID>_C.
        **cambios (Any): Campos a sobrescribir.

    Returns:
        dict[str, Any]: Registro con un golpe parable y uno no parable.

    Raises:
        Ninguna.

    Ejemplo:
        registro("amlo", 2)
    """
    clase = clases or [f"BP_{identificador.upper()}_C"]
    base: dict[str, Any] = {
        "id": identificador, "nombre": identificador, "mapa": mapa, "indice": indice, "clases": clase,
        "vida_max": 1000.0, "fase_avanzada": False, "reacciones": ["Boss_Parried_RM_Montage"],
        "alcance_defecto": 350.0, "vocabulario": [f"{identificador}_corte", f"{identificador}_rafaga"],
        "golpes": {
            f"{identificador}_corte": [{"pos": 1.0, "min": 1.0, "max": 1.0, "parry": True, "dano": 20.0,
                                        "alcance": 200.0, "tipo": "melee"}],
            f"{identificador}_rafaga": [
                {"pos": 0.5, "min": 0.5, "max": 0.5, "parry": False, "dano": 30.0, "alcance": 6000.0,
                 "tipo": "proyectil", "velocidad": 1000.0},
                {"pos": 2.0, "min": 2.0, "max": 2.0, "parry": False, "dano": 40.0, "alcance": 6000.0,
                 "tipo": "rayo", "post_montage": True},
            ],
        },
        "eventos_distancia": {},
        "episodio": {"via": identificador, "start_phase": 1, "vida_fase1": 800.0, "vida_fase2": 0.0,
                     "clases_victoria": clase,
                     "mod": {"id": identificador, "map": mapa, "classes": clase, "purge": clase,
                             "start": {"kind": "spawn_class"}, "hooks": []}},
    }
    base.update(cambios)
    return base


def catalogo_prueba() -> CatalogoJefes:
    """
    Catalogo de cinco identidades que replica la estructura del real con datos minimos.

    Contexto:
        Pruebas de observacion, amenazas, bot y entorno.

    Args:
        Ninguno.

    Returns:
        CatalogoJefes: juice y juice_fase2 en LVL_Boss; amlo, chamuco y nahual en LVL_AMLO.

    Raises:
        Ninguna.

    Ejemplo:
        catalogo = catalogo_prueba()
    """
    juice = registro("juice", 0, "LVL_Boss", ["BP_Enemy_Juice_C"])
    juice["episodio"]["clases_victoria"] = ["BP_Enemy_JuicePhase2_C"]
    juice["episodio"]["vida_fase2"] = 1000.0
    juice["episodio"]["mod"]["start"] = {"kind": "legacy_fog"}
    juice2 = registro("juice_fase2", 1, "LVL_Boss", ["BP_Enemy_JuicePhase2_C"], fase_avanzada=True)
    juice2["episodio"] = {"via": "juice", "start_phase": 2, "vida_fase1": 0.0, "vida_fase2": 1000.0,
                          "clases_victoria": ["BP_Enemy_JuicePhase2_C"]}
    amlo = registro("amlo", 2, clases=["BP_AMLOBOSS_C"])
    amlo["episodio"]["clases_victoria"] = ["BP_Chamuco_C"]
    amlo["episodio"]["vida_fase2"] = 1500.0
    amlo["episodio"]["mod"]["phase2"] = {"class": "BP_Chamuco_C"}
    amlo["eventos_distancia"] = {
        "amlo_throw": {"impactos": [{"retardo": 0.2, "danio": 35.0}], "velocidad": 5000.0, "tipo": "proyectil",
                       "centro": "jefe"},
        "amlo_floor": {"impactos": [{"retardo": 1.9, "danio": 30.0}], "tipo": "area", "centro": "jugador"},
    }
    chamuco = registro("chamuco", 3, clases=["BP_Chamuco_C"], fase_avanzada=True)
    chamuco["episodio"]["start_phase"] = 2
    chamuco["episodio"]["vida_fase1"] = 0.0
    chamuco["episodio"]["vida_fase2"] = 1500.0
    nahual = registro("nahual", 4, clases=["BP_CHUPACABRAS_C"])
    return CatalogoJefes([PerfilJefe.desde_dict(r) for r in (juice, juice2, amlo, chamuco, nahual)])


class JuegoJefesSintetico:
    """
    Juego sintetico con la interfaz de PuertoJuego que entiende reset(boss=..., reload=...).

    Contexto:
        Pruebas del entorno multi-jefe. El jefe de cada episodio es la primera clase de la descripcion;
        el jugador pierde vida con un golpe cada `periodo` pasos salvo que bloquee, atacar quita
        vida al jefe y los eventos de victoria o muerte salen como los del puente real.
    """

    def __init__(self, periodo: int = 8, inicio_malo: int = 0, falla_reset: int = 0, vida_jefe: float = 1000.0,
                 distancia: float = 300.0) -> None:
        """
        Crea el juego.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            periodo (int): Pasos entre golpes del jefe.
            inicio_malo (int): Cuantos reinicios consecutivos devuelven un estado inicial incoherente
                (jugador herido) antes de devolver uno correcto.
            falla_reset (int): Cuantos reset consecutivos lanzan BridgeError antes de funcionar.
            vida_jefe (float): Vida del jefe en cada episodio.
            distancia (float): Distancia 2D jugador-jefe que reporta la observacion.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            JuegoJefesSintetico(periodo=5)
        """
        self.periodo = periodo
        self.inicio_malo = inicio_malo
        self.falla_reset = falla_reset
        self.vida_jefe = vida_jefe
        self.distancia = distancia
        self.resets: list[dict[str, Any]] = []
        self.acciones: list[Action] = []
        self.cerrado = False
        self._e: dict[str, Any] = {}
        self._clase = "BP_AMLOBOSS_C"
        self._multiplicador = 1.0

    def reset(self, timeout: float = 0.0, boss_hp_mult: float | None = None, start_phase: int | None = None,
              boss: dict[str, Any] | None = None, reload: bool = False) -> dict[str, Any]:
        """
        Reinicia la pelea sintetica contra el jefe descrito.

        Contexto:
            EntornoMultiJefe.reset.

        Args:
            timeout (float): Ignorado.
            boss_hp_mult (float, opcional): Multiplicador de vida.
            start_phase (int, opcional): Fase pedida.
            boss (dict, opcional): Descripcion del jefe (seccion mod).
            reload (bool): Recarga completa pedida.

        Returns:
            dict[str, Any]: Observacion inicial.

        Raises:
            BridgeError: Mientras queden fallos programados.

        Ejemplo:
            juego.reset(boss={"classes": ["BP_AMLOBOSS_C"]})
        """
        if self.falla_reset > 0:
            self.falla_reset -= 1
            raise BridgeError("reset: fallo programado")
        self.resets.append({"boss": copy.deepcopy(boss), "reload": reload, "start_phase": start_phase,
                            "mult": boss_hp_mult})
        self._clase = (boss or {"classes": ["BP_AMLOBOSS_C"]})["classes"][0]
        self._multiplicador = boss_hp_mult or 1.0
        herido = self.inicio_malo > 0 and not reload
        if herido:
            self.inicio_malo -= 1
        self._e = {"n": 0, "hp": 60.0 if herido else 100.0, "jefe": self.vida_jefe * self._multiplicador,
                   "lock": False}
        return self._observacion({})

    def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
        """
        Avanza un paso de la pelea sintetica.

        Contexto:
            EntornoMultiJefe.step.

        Args:
            action (Action): Accion del agente.
            n_frames (int): Frames del paso.

        Returns:
            dict[str, Any]: Observacion con eventos del paso.

        Raises:
            Ninguna.

        Ejemplo:
            juego.step(Action(btn="rb"), 2)
        """
        self.acciones.append(action)
        e = self._e
        e["n"] += 1
        ev: dict[str, Any] = {}
        if action.btn == "rs":
            e["lock"] = True
        if action.btn in ("rb", "y"):
            e["jefe"] -= 50.0
            ev["dmg_dealt"] = 50.0
        if e["n"] % self.periodo == 0:
            if action.lb:
                ev["parries"] = 1
            else:
                e["hp"] -= 25.0
                ev["hits"] = [{"t": e["n"] * 0.0667, "amount": 25.0, "parry": True, "att": self._clase,
                               "am": "golpe_sintetico", "ap": 0.5}]
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
        Observacion cruda del estado actual.

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
        from tests.support_dmrl import jefe, jugador

        e = self._e
        vida_max = self.vida_jefe * self._multiplicador
        j = jugador(hp=e["hp"], dead=e["hp"] <= 0, target_lock=e["lock"], caguamas=3)
        b = jefe(cls=self._clase, hp=max(e["jefe"], 0.0), hp_max=vida_max, dead=e["jefe"] <= 0)
        return copy.deepcopy({"player": j, "boss": b, "dist": self.distancia, "ev": ev, "t": e["n"] * 0.0667})
