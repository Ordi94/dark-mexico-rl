"""Evaluacion en juego de un modelo (o del bot) contra varios jefes: planes, episodios, trabajadores y resumen."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion import evaluacion_modelo_jefes as evm
from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento
from dmrl.aplicacion.imitacion_red import crear_modelo_base, guardar_modelo
from dmrl.dominio import acciones as acc
from dmrl.dominio.observacion_v2 import tamano_simple_v2
from tests.support_jefes import JuegoJefesSintetico, catalogo_prueba


class ModeloFalso:
    """Modelo de mentira con la API de MaskablePPO: elige siempre la misma accion."""

    def __init__(self, accion: int = acc.ACCION_NADA) -> None:
        self.accion = accion
        self.llamadas: list[dict[str, Any]] = []

    def predict(self, obs: Any, action_masks: Any = None, deterministic: bool = False) -> tuple[Any, None]:
        self.llamadas.append({"mascara": np.asarray(action_masks), "deterministic": deterministic})
        return np.asarray(self.accion), None


def preparar(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> list[str]:
    """
    Sustituye el juego real por el sintetico y el catalogo por el de prueba.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        **kwargs (Any): Parametros de JuegoJefesSintetico.

    Returns:
        list[str]: Ids de instancia que se van pidiendo.

    Raises:
        Ninguna.

    Ejemplo:
        ids = preparar(monkeypatch, periodo=3)
    """
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        ids.append(params.id_instancia)
        return lambda: JuegoJefesSintetico(**kwargs)

    monkeypatch.setattr(evj, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(evm, "cargar_catalogo", catalogo_prueba)
    monkeypatch.setattr(evm, "ajustar_proceso_actual", lambda *a, **k: None)
    return ids


def test_planificar_evaluacion_empieza_cada_jefe_en_su_primera_fase_con_indices_unicos() -> None:
    planes = evm.planificar_evaluacion(catalogo_prueba(), ("amlo", "chamuco", "nahual"), 4, semilla=10)
    assert [p.plan.indice for p in planes] == list(range(12))
    fases = {j: {p.plan.fase_inicio for p in planes if p.jefe == j} for j in ("amlo", "chamuco", "nahual")}
    assert fases == {"amlo": {1}, "chamuco": {2}, "nahual": {1}}
    assert all(p.plan.epsilon == 0.0 for p in planes)
    assert len({p.plan.semilla for p in planes}) == 12 and planes[0].plan.semilla == 10


def test_planificar_evaluacion_rechaza_cantidades_negativas_y_jefes_desconocidos() -> None:
    with pytest.raises(ValueError):
        evm.planificar_evaluacion(catalogo_prueba(), ("amlo",), -1)
    with pytest.raises(KeyError):
        evm.planificar_evaluacion(catalogo_prueba(), ("inexistente",), 1)
    assert evm.planificar_evaluacion(catalogo_prueba(), ("amlo",), 0) == []


def test_jugar_episodio_modelo_usa_la_mascara_del_entorno_y_politica_determinista(
        monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=8)
    env = evj.crear_entorno_jefes(catalogo_prueba(), ("amlo",), "t", trazar=False)
    modelo = ModeloFalso()
    resumen = evm.jugar_episodio_modelo(env, modelo, {"jefe": "amlo", "start_phase": 1, "boss_hp_mult": 1.0})
    assert resumen["jefe"] == "amlo" and resumen["muerte"] is True
    assert modelo.llamadas and all(c["deterministic"] for c in modelo.llamadas)
    assert modelo.llamadas[0]["mascara"].shape == (acc.N_ACCIONES,)
    env.close()


def test_el_ruido_de_ejecucion_cambia_lo_que_se_ejecuta_pero_no_la_decision_del_modelo(
        monkeypatch: pytest.MonkeyPatch) -> None:
    juegos: list[JuegoJefesSintetico] = []

    def fabrica_de(params: Any) -> Any:
        def crear() -> JuegoJefesSintetico:
            juegos.append(JuegoJefesSintetico(periodo=6))
            return juegos[-1]

        return crear

    monkeypatch.setattr(evj, "crear_fabrica_puerto", fabrica_de)
    env = evj.crear_entorno_jefes(catalogo_prueba(), ("amlo",), "t", trazar=False)
    modelo = ModeloFalso(acc.ACCION_NADA)
    evm.jugar_episodio_modelo(env, modelo, {"jefe": "amlo"}, epsilon=1.0, azar=random.Random(3))
    assert len(modelo.llamadas) > 0 and {int(np.asarray(c["mascara"]).sum() > 0) for c in modelo.llamadas} == {1}
    assert len({(a.btn, a.lb) for a in juegos[0].acciones}) > 1
    env.close()


def test_ejecutar_trabajo_con_bot_juega_todas_las_peleas_del_trabajador(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    planes = tuple(evm.planificar_evaluacion(catalogo_prueba(), ("amlo", "nahual"), 2))
    resumenes = evm.ejecutar_trabajo_modelo(evm.TrabajoEvaluacionModelo(indice=3, planes=planes, ruta_modelo=None))
    assert ids == ["ev3"] and len(resumenes) == 4
    assert [r["jefe"] for r in resumenes] == ["amlo", "amlo", "nahual", "nahual"]


def test_ejecutar_trabajo_con_un_modelo_guardado_en_disco_usa_su_politica(monkeypatch: pytest.MonkeyPatch,
                                                                          tmp_path: Path) -> None:
    preparar(monkeypatch, periodo=2)
    modelo = crear_modelo_base(ConfigEntrenamiento(nombre="x"), tamano_simple_v2(catalogo_prueba()))
    ruta = tmp_path / "m.zip"
    guardar_modelo(modelo, ruta)
    planes = tuple(evm.planificar_evaluacion(catalogo_prueba(), ("amlo",), 2))
    resumenes = evm.ejecutar_trabajo_modelo(
        evm.TrabajoEvaluacionModelo(indice=0, planes=planes, ruta_modelo=str(ruta), epsilon_ruido=0.1))
    assert len(resumenes) == 2 and all(r["jefe"] == "amlo" for r in resumenes)


def test_evaluar_en_jefes_reparte_por_mapa_y_devuelve_los_episodios_de_todos(
        monkeypatch: pytest.MonkeyPatch) -> None:
    ids = preparar(monkeypatch, periodo=3, distancia=100.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        episodios = evm.evaluar_en_jefes(None, ("juice", "amlo", "nahual"), 2, 3, ejecutor=pool,
                                         catalogo=catalogo_prueba())
    assert len(episodios) == 6 and {e["jefe"] for e in episodios} == {"juice", "amlo", "nahual"}
    assert len(ids) >= 2 and all(i.startswith("ev") for i in ids)


def test_evaluar_en_jefes_crea_su_ejecutor_de_procesos_cuando_no_se_le_da_uno(
        monkeypatch: pytest.MonkeyPatch) -> None:
    preparar(monkeypatch, periodo=3, distancia=100.0)
    creados: list[Any] = []

    class EjecutorHilos(ThreadPoolExecutor):
        def __init__(self, max_workers: int, mp_context: Any) -> None:
            super().__init__(max_workers=max_workers)
            creados.append(mp_context)

    monkeypatch.setattr(evm, "ProcessPoolExecutor", EjecutorHilos)
    episodios = evm.evaluar_en_jefes(None, ("nahual",), 2, 1, config=None, prefijo_id="zz")
    assert len(episodios) == 2 and len(creados) == 1


def test_resumir_por_jefe_calcula_la_tasa_de_cada_jefe_y_las_tasas_las_extraen() -> None:
    episodios = [
        {"jefe": "nahual", "victoria": True, "resultado": "victoria"},
        {"jefe": "nahual", "victoria": False, "resultado": "muerte"},
        {"jefe": "amlo", "victoria": True, "resultado": "victoria"},
        {"jefe": "amlo", "victoria": True, "resultado": "victoria"},
        {"victoria": False, "resultado": "muerte"},
    ]
    resumen = evm.resumir_por_jefe(episodios)
    assert list(resumen) == ["?", "amlo", "nahual"]
    assert resumen["nahual"]["tasa"] == pytest.approx(0.5) and resumen["amlo"]["tasa"] == 1.0
    assert resumen["amlo"]["juego"]["peleas"] == 2
    assert evm.tasas_por_jefe(resumen) == {"?": 0.0, "amlo": 1.0, "nahual": 0.5}


def test_un_jefe_solo_con_errores_tiene_tasa_cero() -> None:
    resumen = evm.resumir_por_jefe([{"jefe": "amlo", "resultado": "error"}])
    assert resumen["amlo"]["tasa"] == 0.0 and resumen["amlo"]["juego"] == {}
