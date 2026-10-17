"""Entorno multi-jefe sobre un juego sintetico: eleccion de jefe, reinicio, victoria y metricas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from dmbridge import Action, BridgeError
from dmrl.aplicacion.entorno_jefes import EntornoMultiJefe
from dmrl.dominio import acciones as acc
from dmrl.dominio.observacion_v2 import tamano_simple_v2
from dmrl.dominio.seleccion_jefes import ConfigSeleccion
from tests.support_jefes import JuegoJefesSintetico, catalogo_prueba


class JuegoSiempreMalo(JuegoJefesSintetico):
    """Juego cuyo reinicio deja siempre al jugador herido, incluso tras recargar el mapa."""

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        obs = super().reset(*args, **kwargs)
        obs["player"]["hp"] = 10.0
        return obs


class JuegoQueCae(JuegoJefesSintetico):
    """Juego cuyo primer paso falla como si el proceso hubiera caido."""

    def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
        raise BridgeError("juego caido")


class Banco:
    """Fabrica de juegos sinteticos que recuerda cada instancia lanzada."""

    def __init__(self, clase: type[JuegoJefesSintetico] = JuegoJefesSintetico,
                 primera: type[JuegoJefesSintetico] | None = None, **kwargs: Any) -> None:
        self.clase = clase
        self.primera = primera
        self.kwargs = kwargs
        self.juegos: list[JuegoJefesSintetico] = []

    def __call__(self) -> JuegoJefesSintetico:
        usar = self.primera if (self.primera and not self.juegos) else self.clase
        juego = usar(**self.kwargs)
        self.juegos.append(juego)
        return juego


def crear(banco: Banco | None = None, **kwargs: Any) -> tuple[EntornoMultiJefe, Banco]:
    banco = banco or Banco()
    env = EntornoMultiJefe(banco, catalogo_prueba(), kwargs.pop("config", None), 0, **kwargs)
    return env, banco


def test_por_defecto_enfrenta_a_los_jefes_del_mapa_de_amlo_y_fija_el_mapa() -> None:
    env, _ = crear()
    assert env.selector.permitidos == ("amlo", "chamuco", "nahual") and env.mapa == "LVL_AMLO"
    assert env.observation_space.shape == (tamano_simple_v2(catalogo_prueba()) * 4,)
    env.close()


def test_sin_jefes_del_mapa_de_amlo_usa_todos_los_de_episodio() -> None:
    solo_juice = catalogo_prueba()
    env = EntornoMultiJefe(Banco(), solo_juice, jefes=("juice",))
    assert env.mapa == "LVL_Boss"
    env.close()


def test_jefes_invalidos_se_rechazan() -> None:
    with pytest.raises(ValueError, match="no validos"):
        EntornoMultiJefe(Banco(), catalogo_prueba(), jefes=("juice_fase2",))
    with pytest.raises(ValueError, match="no validos"):
        EntornoMultiJefe(Banco(), catalogo_prueba(), jefes=("inexistente",))
    with pytest.raises(ValueError, match="un mapa"):
        EntornoMultiJefe(Banco(), catalogo_prueba(), jefes=("juice", "amlo"))
    with pytest.raises(ValueError, match="no validos"):
        EntornoMultiJefe(Banco(), catalogo_prueba(), jefes=())


def test_reset_envia_la_descripcion_del_jefe_y_la_fase_que_el_mod_entiende() -> None:
    env, banco = crear()
    env.reset(options={"jefe": "amlo", "start_phase": 2})
    env.reset(options={"jefe": "chamuco"})
    env.reset(options={"jefe": "nahual", "boss_hp_mult": 0.5})
    amlo, chamuco, nahual = banco.juegos[0].resets
    assert amlo["boss"]["id"] == "amlo" and amlo["start_phase"] == 2 and amlo["boss"]["trace_hits"] is False
    assert chamuco["boss"]["id"] == "chamuco" and chamuco["start_phase"] == 1
    assert nahual["mult"] == 0.5
    env.close()


def test_chamuco_directo_se_cuenta_como_fase_2_y_no_cobra_el_premio_de_fase() -> None:
    env, _ = crear()
    env.reset(options={"jefe": "chamuco", "boss_hp_mult": 0.05})
    _, recompensa, terminado, _, info = env.step(acc.ACCION_ATAQUE)
    assert terminado and info["episodio"]["fase_inicio"] == 2 and info["episodio"]["fase_max"] == 2
    assert info["episodio"]["victoria"] and 9.0 < recompensa < 11.0
    env.close()


def test_la_victoria_trae_jefe_y_mapa_y_no_tiene_golpe_mortal() -> None:
    env, _ = crear()
    env.reset(options={"jefe": "nahual", "boss_hp_mult": 0.05})
    _, _, terminado, _, info = env.step(acc.ACCION_ATAQUE)
    episodio = info["episodio"]
    assert terminado and episodio["jefe"] == "nahual" and episodio["mapa"] == "LVL_AMLO"
    assert episodio["golpe_mortal"] is None and env.jefe == "nahual"
    assert env.perfil_actual().id == "nahual"
    env.close()


def test_la_muerte_informa_el_ataque_que_la_causo() -> None:
    env, _ = crear(trazar=True)
    env.reset(options={"jefe": "amlo"})
    final: dict[str, Any] = {}
    for _ in range(100):
        _, _, terminado, _, info = env.step(acc.ACCION_NADA)
        if terminado:
            final = info["episodio"]
            break
    assert final["muerte"] and final["golpe_mortal"] == "BP_AMLOBOSS_C:golpe_sintetico"
    env.close()


def test_golpe_mortal_sin_traza_remoto_y_por_evento() -> None:
    env, _ = crear()
    env.reset(options={"jefe": "amlo"})
    assert env._golpe_mortal() == "sin_traza"
    env._golpes.append({"t": 10.0, "amount": 30.0})
    assert env._golpe_mortal() == "remoto"
    env._ultimo_evento = ("amlo_floor", 8.5)
    assert env._golpe_mortal() == "evento:amlo_floor"
    env._ultimo_evento = ("amlo_floor", 1.0)
    assert env._golpe_mortal() == "remoto"
    env.close()


def test_recarga_completa_cada_n_episodios_como_respaldo() -> None:
    env, banco = crear(recarga_cada=2)
    for _ in range(5):
        env.reset(options={"jefe": "amlo"})
    assert [r["reload"] for r in banco.juegos[0].resets] == [True, False, False, True, False]
    assert env.recargas == 2
    env.reset(options={"jefe": "amlo", "reload": True})
    assert banco.juegos[0].resets[-1]["reload"] is True
    env.close()


def test_inicio_incoherente_repite_el_reinicio_recargando_el_mapa() -> None:
    env, banco = crear(banco=Banco(inicio_malo=5), recarga_cada=100)
    env._desde_recarga = 0
    env.reset(options={"jefe": "amlo"})
    assert env.inicios_inconsistentes == 1
    assert [r["reload"] for r in banco.juegos[0].resets] == [False, True]
    env.close()


def test_inicio_que_sigue_mal_tras_recargar_se_trata_como_fallo_del_juego_y_relanza() -> None:
    banco = Banco(clase=JuegoSiempreMalo)
    env, _ = crear(banco=banco)
    with pytest.raises(RuntimeError, match="no consigue reiniciar"):
        env.reset(options={"jefe": "amlo"})
    assert len(banco.juegos) == 4  # el inicial y tres relanzamientos


@pytest.mark.parametrize("malo", ["muerto", "jefe_ajeno", "jefe_muerto", "caguamas", "vida_jefe", "sin_jefe",
                                  "herido"])
def test_inicio_valido_rechaza_estados_incoherentes(malo: str) -> None:
    env, _ = crear()
    env.reset(options={"jefe": "amlo"})
    crudo = {k: (dict(v) if isinstance(v, dict) else v) for k, v in env.observacion_cruda.items()}
    assert env._inicio_valido(crudo, 1.0)
    if malo == "muerto":
        crudo["player"]["dead"] = True
    elif malo == "herido":
        crudo["player"]["hp"] = 50.0
    elif malo == "jefe_ajeno":
        crudo["boss"]["cls"] = "BP_Enemy_Juice_C"
    elif malo == "jefe_muerto":
        crudo["boss"]["dead"] = True
    elif malo == "caguamas":
        crudo["player"]["caguamas"] = 2
    elif malo == "vida_jefe":
        crudo["boss"]["hp"] = 100.0
    else:
        crudo["boss"] = {}
    assert not env._inicio_valido(crudo, 1.0)
    env.close()


def test_opciones_invalidas_de_reset() -> None:
    env, _ = crear()
    with pytest.raises(ValueError, match="fuera de los permitidos"):
        env.reset(options={"jefe": "juice"})
    with pytest.raises(ValueError, match="no admitida"):
        env.reset(options={"jefe": "nahual", "start_phase": 2})
    env.close()


def test_la_fase_2_se_sortea_solo_en_los_jefes_que_la_admiten() -> None:
    env, banco = crear(seleccion=ConfigSeleccion(prob_fase2=1.0))
    env.reset(options={"jefe": "amlo"})
    env.reset(options={"jefe": "nahual"})
    assert [r["start_phase"] for r in banco.juegos[0].resets] == [2, 1]
    env.close()


def test_el_selector_elige_el_jefe_cuando_no_se_pide() -> None:
    env, banco = crear(seleccion=ConfigSeleccion(pesos={"amlo": 0.0, "chamuco": 0.0, "nahual": 1.0}))
    env.reset()
    assert banco.juegos[0].resets[0]["boss"]["id"] == "nahual"
    env.reset(seed=3)
    env.close()


def test_pasos_y_mascara_se_comportan_como_el_entorno_base() -> None:
    env, _ = crear()
    obs, _ = env.reset(options={"jefe": "amlo"})
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert env.mascara_lista()[acc.ACCION_NADA] and len(env.mascara_lista()) == acc.N_ACCIONES
    _, _, _, _, info = env.step(acc.ACCION_BLOQUEO)
    assert env.accion_efectiva(acc.ACCION_BLOQUEO, info) == acc.ACCION_BLOQUEO
    assert env.accion_efectiva(7, {"accion_invalida": True}) == acc.ACCION_NADA
    env.close()


def test_check_env_de_gymnasium() -> None:
    env, _ = crear()
    check_env(env, warn=False, skip_render_check=True)
    env.close()


def test_una_caida_en_pleno_episodio_no_cuenta_en_el_selector() -> None:
    env, banco = crear(banco=Banco(primera=JuegoQueCae))
    env.reset(options={"jefe": "amlo"})
    _, _, terminado, truncado, info = env.step(acc.ACCION_NADA)
    assert not terminado and truncado and info["episodio"]["resultado"] == "error"
    assert env.selector.tasa_victoria("amlo") == 0.5 and len(banco.juegos) == 2
    env.close()


def test_cada_jefe_puede_fijar_su_limite_de_tiempo() -> None:
    env, _ = crear(banco=Banco(periodo=10_000))
    assert env.config.segundos_maximos == 600.0
    env.reset(options={"jefe": "nahual"})
    assert env._limite_frames() == 360.0 * 30
    truncado = False
    for _ in range(360 * 15 + 1):
        _, _, terminado, truncado, _ = env.step(acc.ACCION_NADA)
        if terminado or truncado:
            break
    assert truncado
    env.close()


def test_los_eventos_remotos_del_paso_se_recuerdan_para_la_causa_de_muerte() -> None:
    env, _ = crear()
    env.reset(options={"jefe": "amlo"})
    env._cruda = dict(env._cruda, ev={"remote": [{"id": "amlo_floor", "t": 4.0}],
                                      "hits": [{"t": 5.0, "amount": 30.0, "att": None}]})
    env._al_observar()
    assert env._golpe_mortal() == "evento:amlo_floor"
    env.close()


def test_enfoque_activo_que_perdio_al_jefe_se_suelta_para_volver_a_fijarlo() -> None:
    """Tras la transformacion de AMLO el lock-on puede quedar en un objetivo que ya no existe."""
    env, _ = crear()
    env.reset(options={"jefe": "amlo"})
    base = env.observacion_cruda
    mirando_a_otro_lado = dict(base, dist=300.0)
    mirando_a_otro_lado["player"] = dict(base["player"], target_lock=True, pos=[0.0, 0.0, 0.0], yaw=180.0)
    mirando_a_otro_lado["boss"] = dict(base["boss"], pos=[300.0, 0.0, 0.0])
    env._cruda = mirando_a_otro_lado
    env._pasos_desde_lock = 10
    assert [env._necesita_enfoque() for _ in range(3)] == [False, False, True]
    mirando_bien = dict(mirando_a_otro_lado, player=dict(mirando_a_otro_lado["player"], yaw=0.0))
    env._cruda = mirando_bien
    assert env._necesita_enfoque() is False and env._enfoque_perdido == 0
    lejos = dict(mirando_a_otro_lado, dist=3000.0)
    env._cruda = lejos
    assert env._necesita_enfoque() is False
    sin_enfoque = dict(mirando_a_otro_lado, player=dict(mirando_a_otro_lado["player"], target_lock=False))
    env._cruda = sin_enfoque
    env._pasos_desde_lock = 10
    assert env._necesita_enfoque() is True
    env.close()
