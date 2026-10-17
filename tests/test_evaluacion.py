"""Evaluacion de politicas y visualizacion sobre un juego sintetico."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion import visualizacion as vis
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from tests.support_dmrl import JuegoSintetico


def crear_env(periodo: int = 6) -> DarkMexicoEnv:
    """
    Entorno sobre un juego sintetico.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        periodo (int): Pasos entre golpes del jefe sintetico.

    Returns:
        DarkMexicoEnv: Entorno listo para reset.

    Raises:
        Ninguna.

    Ejemplo:
        crear_env()
    """
    return DarkMexicoEnv(lambda: JuegoSintetico(periodo=periodo), TablaGolpes())


class PoliticaAtaque:
    """Siempre ataca; sirve para ganar contra el jefe sintetico con poca vida."""

    def reiniciar(self) -> None:
        """
        Sin estado.

        Contexto:
            Cumple la interfaz Politica.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            PoliticaAtaque().reiniciar()
        """

    def decidir(self, env: DarkMexicoEnv, obs: Any) -> int:
        """
        Ataca siempre.

        Contexto:
            Cumple la interfaz Politica.

        Args:
            env (DarkMexicoEnv): Entorno.
            obs (Any): Observacion apilada.

        Returns:
            int: Accion de ataque.

        Raises:
            Ninguna.

        Ejemplo:
            PoliticaAtaque().decidir(env, obs)
        """
        return acc.ACCION_ATAQUE


def test_jugar_episodios_devuelve_un_resumen_por_pelea_y_avisa_al_terminar() -> None:
    """Cada episodio terminado produce su resumen y llama a al_terminar con su numero."""
    env = crear_env()
    avisos: list[int] = []
    resumenes = evl.jugar_episodios(env, PoliticaAtaque(), 3, {"start_phase": 1, "boss_hp_mult": 0.05},
                                    lambda n, r: avisos.append(n))
    assert len(resumenes) == 3 and avisos == [1, 2, 3]
    assert all(r["victoria"] for r in resumenes)


def test_jugar_episodios_usa_por_defecto_la_pelea_completa_al_100_por_ciento() -> None:
    """Sin opciones la evaluacion empieza en fase 1 con el jefe al 100%."""
    juegos: list[JuegoSintetico] = []

    def lanzar() -> JuegoSintetico:
        """
        Lanza un juego sintetico registrado.

        Contexto:
            Fabrica de la prueba.

        Args:
            Ninguno.

        Returns:
            JuegoSintetico: Juego nuevo.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        juegos.append(JuegoSintetico(periodo=2))
        return juegos[-1]

    env = DarkMexicoEnv(lanzar, TablaGolpes(), multiplicador=0.3, prob_fase2=1.0)
    evl.jugar_episodios(env, evl.PoliticaAleatoria(1), 1)
    assert juegos[0].resets == [(1.0, 1)]


def test_resumir_agrega_tasa_de_victoria_dano_y_errores() -> None:
    """Las metricas son promedios del conjunto de episodios; vacio da un diccionario vacio."""
    assert evl.resumir([]) == {}
    r = evl.resumir([
        {"victoria": True, "danio_hecho": 100.0, "fase_max": 2, "resultado": "victoria"},
        {"victoria": False, "danio_hecho": 50.0, "fase_max": 1, "resultado": "error"},
    ])
    assert r["tasa_victoria"] == 0.5 and r["danio_hecho"] == 75.0 and r["tasa_fase2"] == 0.5
    assert r["errores"] == 1.0 and r["parries"] == 0.0


def test_politica_bot_decide_con_la_observacion_cruda() -> None:
    """El bot usa la observacion cruda y la mascara del entorno, no el vector apilado."""
    env = crear_env()
    obs, _ = env.reset()
    politica = evl.PoliticaBot(TablaGolpes())
    politica.reiniciar()
    assert env.action_masks()[politica.decidir(env, obs)]


def test_politica_aleatoria_solo_elige_acciones_validas() -> None:
    """Todas las acciones del azar estan en la mascara."""
    env = crear_env()
    obs, _ = env.reset()
    politica = evl.PoliticaAleatoria(3)
    politica.reiniciar()
    for _ in range(50):
        assert env.action_masks()[politica.decidir(env, obs)]


def test_politica_modelo_pasa_la_mascara_y_devuelve_un_entero() -> None:
    """El modelo recibe la observacion y la mascara; la accion se devuelve como int."""

    class ModeloFalso:
        """Modelo que registra la mascara recibida."""

        recibido: dict[str, Any] = {}  # noqa: RUF012

        def predict(self, obs: Any, action_masks: Any, deterministic: bool) -> tuple[Any, None]:
            """
            Devuelve siempre la accion 4.

            Contexto:
                Sustituye a MaskablePPO.predict.

            Args:
                obs (Any): Observacion.
                action_masks (Any): Mascara de acciones.
                deterministic (bool): Si se pide la accion determinista.

            Returns:
                tuple[Any, None]: Accion 4 y estado nulo.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            self.recibido = {"mascara": action_masks, "det": deterministic}
            return np.array(4), None

    env = crear_env()
    obs, _ = env.reset()
    modelo = ModeloFalso()
    politica = evl.PoliticaModelo(modelo)
    politica.reiniciar()
    assert politica.decidir(env, obs) == 4
    assert modelo.recibido["det"] is True and len(modelo.recibido["mascara"]) == acc.N_ACCIONES


def test_repartir_distribuye_parejo_y_no_deja_trabajadores_vacios() -> None:
    """50 peleas entre 4 instancias son 13, 13, 12, 12; 2 peleas entre 4 son solo 2 trabajadores."""
    assert evl.repartir(50, 4) == [13, 13, 12, 12]
    assert evl.repartir(2, 4) == [1, 1]
    with pytest.raises(ValueError):
        evl.repartir(5, 0)


def test_construir_politica_por_nombre(monkeypatch: pytest.MonkeyPatch) -> None:
    """bot, azar y modelo se construyen; lo desconocido o sin ruta falla."""
    tabla = TablaGolpes()
    assert isinstance(evl.construir_politica("bot", tabla, None, 0), evl.PoliticaBot)
    assert isinstance(evl.construir_politica("azar", tabla, None, 0), evl.PoliticaAleatoria)
    with pytest.raises(ValueError):
        evl.construir_politica("modelo", tabla, None, 0)
    with pytest.raises(ValueError):
        evl.construir_politica("genio", tabla, None, 0)
    import sb3_contrib

    cargado: dict[str, Any] = {}

    def cargar(ruta: str, device: str) -> object:
        """
        Sustituye a MaskablePPO.load registrando los argumentos.

        Contexto:
            Prueba de construir_politica.

        Args:
            ruta (str): Archivo del modelo.
            device (str): Dispositivo pedido.

        Returns:
            object: Modelo falso.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        cargado.update(ruta=ruta, device=device)
        return object()

    monkeypatch.setattr(sb3_contrib.MaskablePPO, "load", cargar)
    assert isinstance(evl.construir_politica("modelo", tabla, "m.zip", 0), evl.PoliticaModelo)
    assert cargado == {"ruta": "m.zip", "device": "cpu"}


def preparar_trabajo(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Sustituye lo que toca el sistema en ejecutar_trabajo.

    Contexto:
        Pruebas de ejecutar_trabajo y evaluar_en_paralelo.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.

    Returns:
        list[str]: Ids de instancia pedidos a la fabrica.

    Raises:
        Ninguna.

    Ejemplo:
        ids = preparar_trabajo(monkeypatch)
    """
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        """
        Fabrica de juegos sinteticos que registra el id.

        Contexto:
            Sustituye a crear_fabrica_puerto.

        Args:
            params (Any): ParametrosInstancia.

        Returns:
            Callable: Constructor de juegos sinteticos.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ids.append(params.id_instancia)
        return lambda: JuegoSintetico(periodo=3)

    monkeypatch.setattr(evl, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(evl, "cargar_tabla_golpes", lambda: TablaGolpes())
    monkeypatch.setattr(evl, "ajustar_proceso_actual", lambda *a, **k: None)
    return ids


def test_ejecutar_trabajo_juega_sus_peleas_y_cierra_el_juego(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un trabajador crea su instancia con id propio, juega y devuelve los resumenes."""
    ids = preparar_trabajo(monkeypatch)
    trabajo = evl.TrabajoEvaluacion(indice=2, episodios=2, politica="azar", prefijo_id="x", config=ConfigEntorno())
    resumenes = evl.ejecutar_trabajo(trabajo)
    assert len(resumenes) == 2 and ids == ["x2"]


def test_evaluar_en_paralelo_reparte_y_junta_los_resumenes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Las peleas se reparten entre trabajadores y los resumenes se concatenan."""
    ids = preparar_trabajo(monkeypatch)
    with ThreadPoolExecutor(max_workers=3) as pool:
        resumenes = evl.evaluar_en_paralelo("azar", 5, 3, ejecutor=pool, opciones={"start_phase": 1})
    assert len(resumenes) == 5 and sorted(ids) == ["ev0", "ev1", "ev2"]


def test_evaluar_en_paralelo_crea_y_cierra_su_propio_ejecutor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin ejecutor dado usa procesos con spawn; aqui se sustituye por hilos para la prueba."""
    preparar_trabajo(monkeypatch)
    creados: list[dict[str, Any]] = []

    class EjecutorHilos(ThreadPoolExecutor):
        """Ejecutor de hilos que acepta los argumentos de ProcessPoolExecutor."""

        def __init__(self, max_workers: int, mp_context: Any) -> None:
            """
            Crea el ejecutor registrando los argumentos.

            Contexto:
                Sustituye a ProcessPoolExecutor en la prueba.

            Args:
                max_workers (int): Trabajadores.
                mp_context (Any): Contexto de multiprocessing pedido.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            creados.append({"n": max_workers, "metodo": mp_context.get_start_method()})
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(evl, "ProcessPoolExecutor", EjecutorHilos)
    from pathlib import Path

    resumenes = evl.evaluar_en_paralelo("azar", 2, 1, ruta_modelo=Path("m.zip"))
    assert len(resumenes) == 2 and creados == [{"n": 1, "metodo": "spawn"}]


def test_ver_pelea_respeta_el_limite_de_tiempo_real() -> None:
    """Con un limite de 0 s no se juega ningun paso y no se devuelve ningun episodio."""
    env = crear_env()
    assert vis.ver_pelea(env, evl.PoliticaAleatoria(1), episodios=3, max_segundos=0.0) == []


def test_ver_pelea_juega_los_episodios_pedidos() -> None:
    """Sin limite de tiempo se juegan todos los episodios."""
    env = crear_env(periodo=2)
    resumenes = vis.ver_pelea(env, evl.PoliticaAleatoria(1), episodios=2)
    assert len(resumenes) == 2


def test_la_ventana_usa_el_perfil_window_con_paso_fijo_y_sin_ajuste_de_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """La instancia con ventana: perfil window, ventana visible, 30 Hz de simulacion, sin afinidad."""
    p = vis.PARAMETROS_VENTANA
    assert p.perfil == "window" and p.ventana_visible and p.sim_hz == 30.0 and p.afinidad == ()
    visto: list[Any] = []
    monkeypatch.setattr(vis, "crear_fabrica_puerto", lambda params: visto.append(params) or JuegoSintetico)
    env = vis.crear_entorno_ventana(TablaGolpes())
    assert isinstance(env, DarkMexicoEnv) and visto == [p]


def test_ver_pelea_espera_entre_decisiones_para_ir_a_velocidad_normal() -> None:
    """Cada decision avanza 0.0667 s de juego: si el reloj real va mas rapido, Python duerme la diferencia."""
    env = crear_env(periodo=1000)
    reloj_falso = {"t": 0.0}
    esperas: list[float] = []

    def dormir(segundos: float) -> None:
        """
        Avanza el reloj falso lo que se duerme.

        Contexto:
            Sustituye a time.sleep en la prueba.

        Args:
            segundos (float): Tiempo a dormir.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        esperas.append(segundos)
        reloj_falso["t"] += segundos

    resumenes = vis.ver_pelea(env, evl.PoliticaAleatoria(1), episodios=1, max_segundos=3.0,
                              reloj=lambda: reloj_falso["t"], dormir=dormir)
    assert resumenes == [] and len(esperas) > 5
    assert all(e > 0 for e in esperas)
    assert sum(esperas) == pytest.approx(3.0, abs=0.2)


def test_ver_pelea_no_duerme_si_el_juego_ya_va_mas_lento_que_el_reloj() -> None:
    """Si el reloj real adelanta al tiempo de juego no se espera nada."""
    env = crear_env(periodo=2)
    avance = {"t": 0.0}

    def reloj() -> float:
        """
        Reloj que avanza 1 s por consulta (mucho mas que el tiempo de juego).

        Contexto:
            Sustituye a time.monotonic en la prueba.

        Args:
            Ninguno.

        Returns:
            float: Segundos falsos.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        avance["t"] += 1.0
        return avance["t"]

    def dormir(segundos: float) -> None:
        """
        Falla si se llama: no debe esperarse.

        Contexto:
            Sustituye a time.sleep en la prueba.

        Args:
            segundos (float): Tiempo a dormir.

        Returns:
            None: Nunca retorna.

        Raises:
            AssertionError: Siempre.

        Ejemplo:
            (uso interno)
        """
        raise AssertionError(f"no debia dormir {segundos}")

    assert len(vis.ver_pelea(env, evl.PoliticaAleatoria(1), episodios=1, reloj=reloj, dormir=dormir)) == 1
