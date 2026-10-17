"""Recoleccion de demostraciones multi-jefe: planes, reparto por mapa, lotes y reanudacion."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion import recoleccion_jefes as rec
from dmrl.dominio import acciones as acc
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.observacion_v2 import tamano_simple_v2
from dmrl.infraestructura.demostraciones import cargar_lote, lotes_de_ronda, metadatos_de_ronda
from tests.support_jefes import JuegoJefesSintetico, catalogo_prueba


class JuegoQueCae(JuegoJefesSintetico):
    """Juego cuyo primer paso falla como si el proceso hubiera caido."""

    def step(self, action: Any, n_frames: int = 1) -> dict[str, Any]:
        from dmbridge import BridgeError

        raise BridgeError("caido")


def preparar(monkeypatch: pytest.MonkeyPatch, clase: type[JuegoJefesSintetico] = JuegoJefesSintetico,
             **kwargs: Any) -> list[str]:
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        ids.append(params.id_instancia)
        return lambda: clase(**kwargs)

    monkeypatch.setattr(evj, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(rec, "cargar_catalogo", catalogo_prueba)
    monkeypatch.setattr(rec, "ajustar_proceso_actual", lambda *a, **k: None)
    return ids


def test_planificar_jefes_numera_sin_repetir_y_respeta_las_fases_de_cada_jefe() -> None:
    planes = rec.planificar_jefes(catalogo_prueba(), ("amlo", "chamuco", "nahual"), 10, prob_fase2=0.5)
    assert [p.plan.indice for p in planes] == list(range(30))
    por_jefe = {j: [p.plan.fase_inicio for p in planes if p.jefe == j] for j in ("amlo", "chamuco", "nahual")}
    assert sorted(set(por_jefe["amlo"])) == [1, 2] and por_jefe["amlo"].count(2) == 5
    assert set(por_jefe["chamuco"]) == {2} and set(por_jefe["nahual"]) == {1}


def test_el_bot_con_ruido_etiqueta_con_su_decision_y_ejecuta_otra(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=50, distancia=100.0)
    catalogo = catalogo_prueba()
    env = evj.crear_entorno_jefes(catalogo, ("nahual",), "t", trazar=False)
    plan = rec.PlanJefe(rec.PlanEpisodio(7, 1, 1.0, 3), "nahual")
    demostraciones, meta = rec.recolectar_episodio_jefe(env, BotJefes.desde_catalogo(catalogo), plan, 2)
    assert demostraciones is not None and meta["jefe"] == "nahual" and meta["id"] == 2 * 100_000 + 7
    assert (demostraciones.etiquetas != demostraciones.ejecutadas).any()
    demostraciones.validar()
    env.close()


def test_un_episodio_que_termina_en_error_no_produce_datos(monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, clase=JuegoQueCae)
    catalogo = catalogo_prueba()
    env = evj.crear_entorno_jefes(catalogo, ("amlo",), "t", trazar=False)
    demostraciones, meta = rec.recolectar_episodio_jefe(
        env, BotJefes.desde_catalogo(catalogo), rec.PlanJefe(rec.PlanEpisodio(0, 1, 0.0, 0), "amlo"), 0)
    assert demostraciones is None and meta["resultado"] == "error"


def test_repartir_por_mapa_no_mezcla_mapas_y_alterna_dentro_de_cada_uno() -> None:
    catalogo = catalogo_prueba()
    planes = rec.planificar_jefes(catalogo, ("juice", "amlo", "nahual"), 4)
    grupos = rec.repartir_por_mapa(planes, 4, catalogo)
    for g in grupos:
        assert len({catalogo.perfil(p.jefe).mapa for p in g}) == 1
    assert sum(len(g) for g in grupos) == 12 and len(grupos) == 4
    assert rec.repartir_por_mapa([], 4, catalogo) == []
    assert len(rec.repartir_por_mapa(planes, 1, catalogo)) == 2  # al menos una instancia por mapa


def test_ejecutar_recoleccion_guarda_lotes_con_jefe_y_formato_v2(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    planes = tuple(rec.planificar_jefes(catalogo_prueba(), ("amlo", "nahual"), 3))
    trabajo = rec.TrabajoRecoleccionJefes(indice=1, ronda=0, planes=planes, carpeta_datos=str(tmp_path),
                                          prefijo_id="x", lote_episodios=4)
    guardados = rec.ejecutar_recoleccion_jefes(trabajo)
    assert ids == ["x1"] and len(guardados) == 6
    assert len(lotes_de_ronda(tmp_path, 0)) == 2
    datos, meta = cargar_lote(lotes_de_ronda(tmp_path, 0)[0])
    datos.validar()
    assert {m["jefe"] for m in meta} <= {"amlo", "nahual"}
    assert datos.obs.shape[1] == tamano_simple_v2(catalogo_prueba()) * 4
    assert datos.mascaras.shape[1] == acc.N_ACCIONES


def test_parada_pedida_no_juega_y_los_errores_no_se_guardan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preparar(monkeypatch)
    planes = tuple(rec.planificar_jefes(catalogo_prueba(), ("amlo",), 2))
    parada = tmp_path / "PARAR"
    parada.write_text("", encoding="utf-8")
    trabajo = rec.TrabajoRecoleccionJefes(0, 0, planes, str(tmp_path / "d"), archivo_parada=str(parada))
    assert rec.ejecutar_recoleccion_jefes(trabajo) == []
    preparar(monkeypatch, clase=JuegoQueCae)
    assert rec.ejecutar_recoleccion_jefes(rec.TrabajoRecoleccionJefes(0, 0, planes, str(tmp_path / "e"))) == []


def test_recolectar_jefes_es_reanudable_y_reporta_lo_que_falta(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert rec.recolectar_jefes(tmp_path, 0, ("amlo", "nahual"), 2, 2, ejecutor=pool) == 0
        antes = len(ids)
        assert rec.recolectar_jefes(tmp_path, 0, ("amlo", "nahual"), 2, 2, ejecutor=pool) == 0
        assert len(ids) == antes
    assert sorted(m["indice"] for m in metadatos_de_ronda(tmp_path, 0)) == [0, 1, 2, 3]
    resumen = rec.resumen_de_ronda(tmp_path, 0)
    assert set(resumen) == {"amlo", "nahual"} and resumen["amlo"]["episodios"] == 2.0


def test_recolectar_jefes_reintenta_respeta_la_parada_y_crea_su_ejecutor(monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    ids = preparar(monkeypatch, clase=JuegoQueCae)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert rec.recolectar_jefes(tmp_path / "a", 0, ("amlo",), 2, 1, ejecutor=pool) == 2
    assert len(ids) == rec.MAX_PASADAS
    parada = tmp_path / "PARAR"
    parada.write_text("", encoding="utf-8")
    assert rec.recolectar_jefes(tmp_path / "b", 0, ("amlo",), 2, 1, archivo_parada=parada) == 2
    parada.unlink()
    preparar(monkeypatch, periodo=3, distancia=100.0)
    contextos: list[Any] = []

    class EjecutorHilos(ThreadPoolExecutor):
        def __init__(self, max_workers: int, mp_context: Any) -> None:
            super().__init__(max_workers=max_workers)
            contextos.append(mp_context)

    monkeypatch.setattr(rec, "ProcessPoolExecutor", EjecutorHilos)
    assert rec.recolectar_jefes(tmp_path / "c", 0, ("nahual",), 1, 1, archivo_parada=parada) == 0
    assert len(contextos) == 1


def test_resumen_de_ronda_sin_datos_esta_vacio(tmp_path: Path) -> None:
    assert rec.resumen_de_ronda(tmp_path, 0) == {}


def plan_de(jefe: str, indice: int) -> rec.PlanJefe:
    return rec.PlanJefe(rec.PlanEpisodio(indice, 1, 0.0, indice), jefe)


def test_intercalar_por_jefe_pone_uno_de_cada_jefe_antes_del_segundo_de_cualquiera() -> None:
    a1, a2, a3, n1, n2 = (plan_de("amlo", 1), plan_de("amlo", 2), plan_de("amlo", 3), plan_de("nahual", 4),
                          plan_de("nahual", 5))
    assert rec.intercalar_por_jefe([a1, a2, a3, n1, n2]) == [a1, n1, a2, n2, a3]
    assert rec.intercalar_por_jefe([n1, a1, n2]) == [n1, a1, n2]
    assert rec.intercalar_por_jefe([]) == []


def test_un_corte_del_orden_intercalado_deja_a_todos_los_jefes_con_la_misma_cantidad_de_peleas() -> None:
    planes = rec.planificar_jefes(catalogo_prueba(), ("amlo", "chamuco", "nahual"), 6)
    primeros = rec.intercalar_por_jefe(planes)[:9]
    assert {j: sum(1 for p in primeros if p.jefe == j) for j in ("amlo", "chamuco", "nahual")} == {
        "amlo": 3, "chamuco": 3, "nahual": 3}


def test_un_trabajo_con_hora_limite_vencida_no_inicia_episodios(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    ids = preparar(monkeypatch)
    planes = tuple(rec.planificar_jefes(catalogo_prueba(), ("amlo",), 2))
    trabajo = rec.TrabajoRecoleccionJefes(0, 0, planes, str(tmp_path), hora_limite=time.time() - 1.0)
    assert rec.ejecutar_recoleccion_jefes(trabajo) == [] and ids == ["rj0"]
    assert lotes_de_ronda(tmp_path, 0) == []


def test_recolectar_jefes_con_limite_agotado_no_lanza_trabajadores_y_reporta_lo_pendiente(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = preparar(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert rec.recolectar_jefes(tmp_path, 0, ("amlo", "nahual"), 2, 1, ejecutor=pool, limite_segundos=0.0) == 4
    assert ids == []


def test_recolectar_jefes_entrega_a_los_trabajadores_la_hora_limite_y_los_episodios_intercalados(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preparar(monkeypatch)
    recibidos: list[rec.TrabajoRecoleccionJefes] = []

    def espia(trabajo: rec.TrabajoRecoleccionJefes) -> list[dict[str, Any]]:
        recibidos.append(trabajo)
        return []

    monkeypatch.setattr(rec, "ejecutar_recoleccion_jefes", espia)
    antes = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        rec.recolectar_jefes(tmp_path, 0, ("amlo", "nahual"), 3, 1, ejecutor=pool, limite_segundos=600.0)
    assert recibidos and all(t.hora_limite is not None and antes + 599 < t.hora_limite for t in recibidos)
    jefes = [p.jefe for p in recibidos[0].planes]
    assert jefes[:2] == ["amlo", "nahual"] and jefes[:6] == ["amlo", "nahual"] * 3


def test_recolectar_jefes_sin_limite_no_fija_hora_limite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preparar(monkeypatch)
    recibidos: list[rec.TrabajoRecoleccionJefes] = []
    monkeypatch.setattr(rec, "ejecutar_recoleccion_jefes", lambda t: recibidos.append(t) or [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        rec.recolectar_jefes(tmp_path, 0, ("nahual",), 1, 1, ejecutor=pool)
    assert recibidos and all(t.hora_limite is None for t in recibidos)
