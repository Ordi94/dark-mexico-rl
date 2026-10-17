"""Entorno Gymnasium DarkMexicoEnv sobre un juego sintetico."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from dmbridge import Action, BridgeError
from dmrl.aplicacion.entorno import DarkMexicoEnv, a_action
from dmrl.dominio import acciones as acc
from dmrl.dominio.acciones import ComandoJuego
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from tests.support_dmrl import JuegoSintetico


class Banco:
    """
    Fabrica de juegos sinteticos que recuerda cada instancia lanzada.

    Contexto:
        Permite inspeccionar lo que recibio el juego y simular relanzamientos.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Guarda los parametros de los juegos a crear.

        Contexto:
            Cada prueba construye su banco.

        Args:
            **kwargs (Any): Parametros de JuegoSintetico.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            Banco(periodo=3)
        """
        self.kwargs = kwargs
        self.juegos: list[JuegoSintetico] = []

    def __call__(self) -> JuegoSintetico:
        """
        Lanza un juego sintetico nuevo.

        Contexto:
            Hace de FabricaPuerto.

        Args:
            Ninguno.

        Returns:
            JuegoSintetico: Juego nuevo (los fallos programados solo afectan al primero).

        Raises:
            Ninguna.

        Ejemplo:
            banco()
        """
        kwargs = self.kwargs if not self.juegos else {k: v for k, v in self.kwargs.items() if k == "periodo"}
        juego = JuegoSintetico(**kwargs)
        self.juegos.append(juego)
        return juego


def crear(config: ConfigEntorno | None = None, **kwargs: Any) -> tuple[DarkMexicoEnv, Banco]:
    """
    Crea un entorno sobre un banco de juegos sinteticos.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        config (ConfigEntorno, opcional): Configuracion del entorno.
        **kwargs (Any): Parametros del juego sintetico y del entorno.

    Returns:
        tuple[DarkMexicoEnv, Banco]: Entorno y banco.

    Raises:
        Ninguna.

    Ejemplo:
        env, banco = crear(periodo=3)
    """
    parametros_entorno = {k: kwargs.pop(k) for k in ("multiplicador", "prob_fase2", "semilla") if k in kwargs}
    banco = Banco(**kwargs)
    return DarkMexicoEnv(banco, TablaGolpes(), config, **parametros_entorno), banco


def test_reset_devuelve_la_observacion_apilada_y_pide_el_curriculo_al_juego() -> None:
    """reset envia multiplicador y fase de inicio y devuelve 4 observaciones apiladas."""
    env, banco = crear(multiplicador=0.3)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32 and info == {}
    assert env.observation_space.contains(obs)
    assert banco.juegos[0].resets == [(0.3, 1)]


def test_reset_con_opciones_fuerza_fase_y_multiplicador() -> None:
    """options tiene prioridad sobre el curriculo y la probabilidad de fase 2."""
    env, banco = crear(multiplicador=0.3, prob_fase2=0.0)
    env.reset(options={"start_phase": 2, "boss_hp_mult": 0.7})
    assert banco.juegos[0].resets == [(0.7, 2)]


def test_con_probabilidad_uno_todos_los_episodios_empiezan_en_fase_2() -> None:
    """prob_fase2=1 siempre arranca en la fase 2; con 0 siempre en la 1."""
    env, banco = crear(prob_fase2=1.0, semilla=1)
    env.reset()
    env.fijar_prob_fase2(0.0)
    env.reset()
    assert [f for _, f in banco.juegos[0].resets] == [2, 1]


def test_reset_con_semilla_reinicia_el_azar_propio() -> None:
    """La misma semilla da la misma secuencia de fases de inicio."""
    env, banco = crear(prob_fase2=0.5)
    secuencias = []
    for _ in range(2):
        env.reset(seed=7)
        secuencias.append([banco.juegos[0].resets[-1][1]])
        env.reset()
        secuencias[-1].append(banco.juegos[0].resets[-1][1])
    assert secuencias[0] == secuencias[1]


def test_fijar_multiplicador_y_probabilidad_validan_el_rango() -> None:
    """Valores fuera de rango se rechazan; los validos se usan en el siguiente reset."""
    env, banco = crear()
    with pytest.raises(ValueError):
        env.fijar_multiplicador(0.01)
    with pytest.raises(ValueError):
        env.fijar_multiplicador(1.5)
    with pytest.raises(ValueError):
        env.fijar_prob_fase2(1.5)
    env.fijar_multiplicador(0.5)
    env.reset()
    assert banco.juegos[0].resets[0][0] == 0.5


def test_mascara_antes_del_primer_reset_solo_permite_nada() -> None:
    """Sin observacion todavia, solo la accion nula es valida."""
    env, _ = crear()
    assert env.action_masks().tolist() == [True] + [False] * (acc.N_ACCIONES - 1)
    assert env.observacion_cruda == {}


def test_step_sustituye_una_accion_invalida_por_nada() -> None:
    """La caguama con vida llena no esta en la mascara: se ejecuta 'nada' y se avisa."""
    env, banco = crear()
    env.reset()
    _, _, _, _, info = env.step(acc.ACCION_CAGUAMA)
    assert info["accion_invalida"] is True
    ultima, _ = banco.juegos[0].acciones[-1]
    assert ultima.btn is None


def test_step_rechaza_indices_fuera_de_rango() -> None:
    """Un indice que no existe es un error, no una accion nula."""
    env, _ = crear()
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_step_traduce_la_accion_y_avanza_los_frames_configurados() -> None:
    """Atacar envia el boton RB y 2 frames por defecto."""
    env, banco = crear()
    env.reset()
    env.step(acc.ACCION_ATAQUE)
    accion, frames = banco.juegos[0].acciones[-1]
    assert accion.btn == "rb" and frames == 2


def test_enfoque_se_activa_antes_de_la_primera_accion_si_el_jefe_esta_cerca() -> None:
    """Con el jefe a 300 uu y sin enfoque se pulsa RS en un paso de 1 frame antes de la accion."""
    env, banco = crear()
    env.reset()
    env.step(acc.ACCION_NADA)
    acciones = banco.juegos[0].acciones
    assert acciones[0][0].btn == "rs" and acciones[0][1] == 1
    assert acciones[1][1] == 2
    env.step(acc.ACCION_NADA)
    assert [a.btn for a, _ in banco.juegos[0].acciones].count("rs") == 1


def test_enfoque_no_se_pulsa_si_el_jefe_esta_lejos() -> None:
    """Fuera del alcance (480 uu) RS no hace nada y no se intenta."""
    env, banco = crear()
    env.reset()
    original = banco.juegos[0]._observacion

    def lejos(ev: dict[str, Any]) -> dict[str, Any]:
        """
        Observacion con el jefe a 2000 uu.

        Contexto:
            Sustituye a la del juego sintetico en esta prueba.

        Args:
            ev (dict[str, Any]): Eventos del paso.

        Returns:
            dict[str, Any]: Observacion con distancia grande.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        o = original(ev)
        o["dist"] = 2000.0
        return o

    banco.juegos[0]._observacion = lejos  # type: ignore[method-assign]
    env._cruda = lejos({})
    env.step(acc.ACCION_NADA)
    assert all(a.btn != "rs" for a, _ in banco.juegos[0].acciones)


def test_el_final_durante_el_paso_de_enfoque_termina_el_episodio() -> None:
    """Si el jugador muere en el frame del enfoque no se ejecuta la accion del agente."""

    class MuereAlEnfocar(JuegoSintetico):
        """Juego en el que pulsar RS coincide con la muerte del jugador."""

        def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
            """
            Mata al jugador cuando se pulsa RS.

            Contexto:
                Escenario de la prueba.

            Args:
                action (Action): Accion recibida.
                n_frames (int): Frames del paso.

            Returns:
                dict[str, Any]: Observacion con la muerte del jugador si la accion fue RS.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            observacion = super().step(action, n_frames)
            if action.btn == "rs":
                observacion["player"]["dead"] = True
                observacion["ev"]["player_death"] = True
            return observacion

    juego = MuereAlEnfocar()
    env = DarkMexicoEnv(lambda: juego, TablaGolpes())
    env.reset()
    _, recompensa, terminado, truncado, info = env.step(acc.ACCION_ATAQUE)
    assert terminado and not truncado and info["episodio"]["resultado"] == "muerte"
    assert recompensa < -4.0
    assert len(juego.acciones) == 1


def test_victoria_termina_el_episodio_con_las_metricas_del_resumen() -> None:
    """Con el jefe al 5% de vida tres ataques lo matan: victoria, +10 y progreso completo."""
    env, _ = crear(multiplicador=0.05)
    env.reset(options={"start_phase": 1, "boss_hp_mult": 0.05})
    terminado, info, total = False, {}, 0.0
    while not terminado:
        _, r, terminado, _, info = env.step(acc.ACCION_ATAQUE)
        total += r
    resumen = info["episodio"]
    assert resumen["resultado"] == "victoria" and resumen["victoria"] is True
    assert resumen["progreso_jefe"] == 1.0 and resumen["danio_hecho"] == pytest.approx(120.0)
    assert resumen["fase_inicio"] == 1 and resumen["multiplicador"] == pytest.approx(0.05)
    assert total == pytest.approx(resumen["recompensa"])
    assert total > 10.0


def test_muerte_termina_el_episodio_y_castiga() -> None:
    """Sin defenderse el jefe sintetico mata al jugador: resultado muerte."""
    env, _ = crear(periodo=2)
    env.reset()
    terminado, info = False, {}
    while not terminado:
        _, _, terminado, truncado, info = env.step(acc.ACCION_NADA)
        assert not truncado
    assert info["episodio"]["muerte"] is True and info["episodio"]["danio_recibido"] == pytest.approx(100.0)


def test_truncado_por_tiempo_de_juego() -> None:
    """Con 1 s de limite (30 frames) el episodio se trunca como 'tiempo' en la decision 15.

    El primer paso suma ademas el frame del enfoque: 3 + 2 * 14 = 31 frames >= 30.
    """
    config = replace(ConfigEntorno(), segundos_maximos=1.0)
    env, _ = crear(config, periodo=1000)
    env.reset()
    decisiones = 0
    for _ in range(20):
        decisiones += 1
        _, _, terminado, truncado, info = env.step(acc.ACCION_NADA)
        assert not terminado
        if truncado:
            break
    assert decisiones == 15
    assert info["episodio"]["resultado"] == "tiempo" and info["episodio"]["duracion_s"] == pytest.approx(31 / 30)


def test_caida_durante_un_paso_trunca_el_episodio_y_relanza_la_instancia() -> None:
    """Un error del puente cierra el episodio como 'error' y deja una instancia nueva."""
    env, banco = crear(falla_en=3)
    env.reset()
    env.step(acc.ACCION_NADA)
    _, recompensa, terminado, truncado, info = env.step(acc.ACCION_NADA)
    assert truncado and not terminado and recompensa == 0.0
    assert info["episodio"]["resultado"] == "error" and "fallo programado" in info["episodio"]["error"]
    assert env.reinicios_instancia == 1 and len(banco.juegos) == 2 and banco.juegos[0].cerrado
    env.reset()
    assert banco.juegos[1].resets


def test_reset_fallido_relanza_y_reintenta() -> None:
    """Dos fallos del reset se resuelven relanzando el juego."""
    env, banco = crear(falla_reset=1)
    env.reset()
    assert env.reinicios_instancia == 1 and len(banco.juegos) == 2


def test_reset_que_siempre_falla_termina_con_un_error_claro() -> None:
    """Tras tres relanzamientos seguidos sin exito se abandona."""

    class SiempreFalla(JuegoSintetico):
        """Juego cuyo reset nunca funciona."""

        def reset(self, timeout: float = 0.0, boss_hp_mult: float | None = None,
                  start_phase: int | None = None) -> dict[str, Any]:
            """
            Falla siempre.

            Contexto:
                Escenario de la prueba.

            Args:
                timeout (float): Ignorado.
                boss_hp_mult (float, opcional): Ignorado.
                start_phase (int, opcional): Ignorado.

            Returns:
                dict[str, Any]: Nunca retorna.

            Raises:
                BridgeError: Siempre.

            Ejemplo:
                (uso interno)
            """
            raise BridgeError("reset imposible")

    env = DarkMexicoEnv(SiempreFalla, TablaGolpes())
    with pytest.raises(RuntimeError, match="no consigue reiniciar"):
        env.reset()


def test_close_cierra_el_juego_y_el_entorno_queda_inutilizable() -> None:
    """Despues de close, usar el entorno es un error explicito."""
    env, banco = crear()
    env.reset()
    env.close()
    assert banco.juegos[0].cerrado
    with pytest.raises(RuntimeError):
        env.step(acc.ACCION_NADA)
    env.close()


def test_a_action_conserva_el_marco_del_jefe() -> None:
    """La traduccion al protocolo usa siempre movimiento en el marco del jefe."""
    a = a_action(ComandoJuego(btn="b", lb=True, mx=1.0, my=0.0))
    assert (a.btn, a.lb, a.mx, a.my, a.mf) == ("b", True, 1.0, 0.0, "boss")


def test_check_env_de_stable_baselines_acepta_el_entorno() -> None:
    """El entorno cumple la API de Gymnasium/SB3 (espacios, tipos, reset y step)."""
    env, _ = crear(periodo=50)
    check_env(env, warn=True, skip_render_check=True)
