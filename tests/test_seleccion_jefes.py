"""Selector de jefes, pesos adaptativos y reparto de instancias entre mapas."""

from __future__ import annotations

import random

import pytest

from dmrl.dominio.seleccion_jefes import (
    MUESTRAS_MINIMAS,
    TASA_SIN_DATOS,
    ConfigSeleccion,
    SelectorJefes,
    repartir_instancias,
)


def test_pesos_por_defecto_son_uniformes() -> None:
    selector = SelectorJefes(("amlo", "chamuco", "nahual"))
    assert selector.probabilidades() == pytest.approx({"amlo": 1 / 3, "chamuco": 1 / 3, "nahual": 1 / 3})


def test_los_pesos_configurados_fijan_la_probabilidad() -> None:
    selector = SelectorJefes(("amlo", "nahual"), ConfigSeleccion(pesos={"amlo": 3.0, "nahual": 1.0}))
    assert selector.probabilidades() == pytest.approx({"amlo": 0.75, "nahual": 0.25})


def test_la_eleccion_sigue_los_pesos() -> None:
    selector = SelectorJefes(("amlo", "nahual"), ConfigSeleccion(pesos={"amlo": 9.0, "nahual": 1.0}), random.Random(0))
    elegidos = [selector.elegir() for _ in range(2000)]
    assert 0.85 < elegidos.count("amlo") / 2000 < 0.95


def test_un_peso_cero_excluye_al_jefe() -> None:
    selector = SelectorJefes(("amlo", "nahual"), ConfigSeleccion(pesos={"nahual": 0.0}), random.Random(1))
    assert {selector.elegir() for _ in range(200)} == {"amlo"}


def test_peso_adaptativo_favorece_al_jefe_que_mas_pierde() -> None:
    selector = SelectorJefes(("amlo", "nahual"), ConfigSeleccion(adaptativo=True, fuerza_adaptativa=2.0))
    for _ in range(MUESTRAS_MINIMAS):
        selector.registrar("amlo", False)
        selector.registrar("nahual", True)
    p = selector.probabilidades()
    assert selector.tasa_victoria("amlo") == 0.0 and selector.tasa_victoria("nahual") == 1.0
    assert p["amlo"] == pytest.approx(3.0 / 4.0) and p["nahual"] == pytest.approx(1.0 / 4.0)


def test_sin_muestras_suficientes_la_tasa_es_neutra() -> None:
    selector = SelectorJefes(("amlo",), ConfigSeleccion(adaptativo=True))
    selector.registrar("amlo", False)
    assert selector.tasa_victoria("amlo") == TASA_SIN_DATOS


def test_la_ventana_solo_recuerda_los_ultimos_episodios() -> None:
    selector = SelectorJefes(("amlo",), ConfigSeleccion(ventana=10))
    for _ in range(10):
        selector.registrar("amlo", False)
    for _ in range(10):
        selector.registrar("amlo", True)
    assert selector.tasa_victoria("amlo") == 1.0


def test_configuracion_invalida_se_rechaza() -> None:
    with pytest.raises(ValueError):
        ConfigSeleccion(pesos={"a": -1.0})
    with pytest.raises(ValueError):
        ConfigSeleccion(fuerza_adaptativa=-1.0)
    with pytest.raises(ValueError):
        ConfigSeleccion(ventana=0)
    with pytest.raises(ValueError):
        ConfigSeleccion(prob_fase2=1.5)
    with pytest.raises(ValueError):
        SelectorJefes(())
    with pytest.raises(ValueError):
        SelectorJefes(("a",), ConfigSeleccion(pesos={"a": 0.0}))


def test_reparto_de_instancias_sigue_el_peso_y_la_velocidad_de_cada_mapa() -> None:
    reparto = repartir_instancias({"LVL_Boss": 1.0, "LVL_AMLO": 3.0}, {"LVL_Boss": 300.0, "LVL_AMLO": 200.0}, 8)
    assert sum(reparto.values()) == 8 and reparto["LVL_AMLO"] > reparto["LVL_Boss"]
    # Mapas igual de rapidos y de igual peso: reparto parejo.
    assert repartir_instancias({"A": 1.0, "B": 1.0}, {"A": 100.0, "B": 100.0}, 4) == {"A": 2, "B": 2}


def test_todo_mapa_con_peso_recibe_al_menos_una_instancia() -> None:
    reparto = repartir_instancias({"A": 100.0, "B": 0.01}, {"A": 100.0, "B": 100.0}, 4)
    assert reparto == {"A": 3, "B": 1}


def test_los_mapas_sin_peso_no_reciben_instancias() -> None:
    assert repartir_instancias({"A": 1.0, "B": 0.0}, {"A": 100.0}, 3) == {"A": 3}


def test_reparto_invalido_se_rechaza() -> None:
    with pytest.raises(ValueError):
        repartir_instancias({"A": 1.0, "B": 1.0}, {"A": 1.0, "B": 1.0}, 1)
    with pytest.raises(ValueError):
        repartir_instancias({"A": 1.0}, {}, 2)


def test_el_minimo_de_una_instancia_por_mapa_se_descuenta_de_los_mapas_grandes() -> None:
    reparto = repartir_instancias({"A": 100.0, "B": 0.01, "C": 0.01}, {"A": 1.0, "B": 1.0, "C": 1.0}, 3)
    assert reparto == {"A": 1, "B": 1, "C": 1}
