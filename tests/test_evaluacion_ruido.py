"""Politica con ruido de ejecucion para la evaluacion de robustez."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from tests.support_dmrl import JuegoSintetico
from tests.test_evaluacion import PoliticaAtaque, preparar_trabajo


def test_con_epsilon_uno_la_politica_con_ruido_solo_elige_acciones_validas_y_reinicia_la_base() -> None:
    """Con ruido total se desvia del ataque, siempre dentro de la mascara; reiniciar llega a la base."""
    env = DarkMexicoEnv(lambda: JuegoSintetico(periodo=50), TablaGolpes())
    obs, _ = env.reset()
    base = PoliticaAtaque()
    politica = evl.PoliticaConRuido(base, 1.0, 0)
    politica.reiniciar()
    elegidas = {politica.decidir(env, obs) for _ in range(200)}
    assert len(elegidas) > 3 and all(env.action_masks()[a] for a in elegidas)
    sin_ruido = evl.PoliticaConRuido(base, 0.0, 0)
    assert {sin_ruido.decidir(env, obs) for _ in range(20)} == {acc.ACCION_ATAQUE}


def test_epsilon_fuera_de_rango_se_rechaza() -> None:
    """Probabilidad invalida."""
    with pytest.raises(ValueError):
        evl.PoliticaConRuido(PoliticaAtaque(), 1.5)


def test_el_ruido_llega_a_los_trabajadores_de_evaluacion(monkeypatch: pytest.MonkeyPatch) -> None:
    """ejecutar_trabajo envuelve la politica cuando epsilon_ruido es positivo y evaluar_en_paralelo lo reparte."""
    preparar_trabajo(monkeypatch)
    envueltas: list[object] = []
    original = evl.PoliticaConRuido
    monkeypatch.setattr(evl, "PoliticaConRuido", lambda *a, **k: envueltas.append(1) or original(*a, **k))
    trabajo = evl.TrabajoEvaluacion(indice=0, episodios=1, politica="azar", config=ConfigEntorno(), epsilon_ruido=0.3)
    assert len(evl.ejecutar_trabajo(trabajo)) == 1 and envueltas == [1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        resumenes = evl.evaluar_en_paralelo("azar", 2, 2, ejecutor=pool, opciones={"start_phase": 1}, epsilon_ruido=0.2)
    assert len(resumenes) == 2 and len(envueltas) == 3
