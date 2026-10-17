"""Evaluacion por jefe: bucle del bot, resumen, trabajadores y reparto en paralelo."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion.entorno_jefes import EntornoMultiJefe
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.configuracion import ConfigEntorno
from tests.support_jefes import JuegoJefesSintetico, catalogo_prueba


def preparar(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> list[str]:
    """Sustituye lo que toca el sistema: fabrica de juegos, catalogo y afinidad."""
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        ids.append(params.id_instancia)
        return lambda: JuegoJefesSintetico(**kwargs)

    monkeypatch.setattr(evj, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(evj, "cargar_catalogo", catalogo_prueba)
    monkeypatch.setattr(evj, "ajustar_proceso_actual", lambda *a, **k: None)
    return ids


def test_episodio_con_el_bot_termina_y_confirma_cada_accion(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=4, distancia=100.0)
    catalogo = catalogo_prueba()
    env = evj.crear_entorno_jefes(catalogo, ("nahual",), "t0", ConfigEntorno())
    registro: list[tuple[int, int]] = []
    resumen = evj.jugar_episodio_bot(
        env, BotJefes.desde_catalogo(catalogo), {"jefe": "nahual", "boss_hp_mult": 0.05},
        registrar=lambda obs, mascara, etiqueta, ejecutada, recompensa: registro.append((etiqueta, ejecutada)))
    assert resumen["jefe"] == "nahual" and resumen["victoria"] and registro
    assert all(e == x for e, x in registro)
    env.close()


def test_ruido_dart_cambia_lo_ejecutado_pero_no_la_etiqueta(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=50, distancia=100.0)
    catalogo = catalogo_prueba()
    env = evj.crear_entorno_jefes(catalogo, ("nahual",), "t0", ConfigEntorno())
    registro: list[tuple[int, int]] = []
    evj.jugar_episodio_bot(env, BotJefes.desde_catalogo(catalogo), {"jefe": "nahual", "boss_hp_mult": 0.2}, 1.0,
                           random.Random(0), lambda o, m, e, x, r: registro.append((e, x)))
    assert any(e != x for e, x in registro)
    env.close()


def test_etiqueta_que_la_mascara_rechaza_se_guarda_como_nada(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=50, distancia=100.0)
    catalogo = catalogo_prueba()
    env = evj.crear_entorno_jefes(catalogo, ("nahual",), "t0", ConfigEntorno())

    class BotProhibido(BotJefes):
        def decidir(self, obs: dict[str, Any], mascara: Any, amenazas: Any) -> int:
            return 21  # tomar caguama

    env.mascara_lista = lambda: [True] + [False] * 21  # type: ignore[method-assign]

    etiquetas: list[int] = []
    evj.jugar_episodio_bot(env, BotProhibido(), {"jefe": "nahual", "boss_hp_mult": 0.05}, 0.0, None,
                           lambda o, m, e, x, r: etiquetas.append(e))
    assert set(etiquetas) == {0}
    env.close()


def test_resumen_agrega_muertes_por_golpe_y_reinicios() -> None:
    episodios = [
        {"victoria": True, "muerte": False, "reinicio_s": 0.5, "recargas": 1},
        {"victoria": False, "muerte": True, "golpe_mortal": "B:M", "reinicio_s": 1.5, "recargas": 3,
         "inicios_inconsistentes": 1},
        {"victoria": False, "muerte": True, "golpe_mortal": "B:M"},
        {"victoria": False, "muerte": True, "golpe_mortal": "evento:e"},
    ]
    r = evj.resumir_jefe(episodios)
    assert r["muertes"] == 3 and r["muertes_por_golpe"] == {"B:M": 2, "evento:e": 1}
    assert r["recargas"] == 3 and r["inicios_inconsistentes"] == 1 and r["tasa_victoria"] == 0.25
    assert evj.resumir_jefe([]) == {}


def test_trabajador_juega_sus_episodios_con_su_propia_instancia(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    trabajo = evj.TrabajoEvaluacionJefe(indice=2, jefe="amlo", episodios=3, semilla=5, prefijo_id="z",
                                        opciones={"boss_hp_mult": 0.05})
    resumenes = evj.ejecutar_trabajo_jefe(trabajo)
    assert ids == ["z2"] and len(resumenes) == 3 and all(r["jefe"] == "amlo" for r in resumenes)


def test_evaluar_reparte_entre_instancias(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        resumenes = evj.evaluar_jefe_en_paralelo("chamuco", 5, 2, opciones={"boss_hp_mult": 0.05}, ejecutor=pool)
    assert len(resumenes) == 5 and sorted(ids) == ["ej0", "ej1"]


def test_evaluar_sin_ejecutor_usa_procesos_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=3, distancia=100.0)
    contextos: list[Any] = []

    class EjecutorHilos(ThreadPoolExecutor):
        def __init__(self, max_workers: int, mp_context: Any) -> None:
            super().__init__(max_workers=max_workers)
            contextos.append(mp_context)

    monkeypatch.setattr(evj, "ProcessPoolExecutor", EjecutorHilos)
    assert len(evj.evaluar_jefe_en_paralelo("nahual", 2, 1, opciones={"boss_hp_mult": 0.05})) == 2
    assert len(contextos) == 1


def test_crear_entorno_usa_los_parametros_dados(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = preparar(monkeypatch)
    env = evj.crear_entorno_jefes(catalogo_prueba(), None, "abc", recarga_cada=3, trazar=False)
    assert isinstance(env, EntornoMultiJefe) and ids == ["abc"]
    env.close()
