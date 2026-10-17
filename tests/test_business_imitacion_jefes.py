"""Reglas de negocio de la imitacion multi-jefe: ningun jefe se queda sin validacion y su calidad se ve por separado."""

from __future__ import annotations

import numpy as np
import pytest

from dmrl.dominio import imitacion_jefes as ij
from tests.test_imitacion_jefes_dominio import episodios_de


@pytest.mark.parametrize("semilla", [0, 1, 2, 3, 4, 5])
def test_la_particion_deja_al_menos_un_episodio_de_cada_jefe_en_validacion_y_en_entrenamiento(
        semilla: int) -> None:
    ids, jefes = episodios_de({"juice": 3, "amlo": 12, "chamuco": 2, "nahual": 25})
    train, val = ij.particionar_estratificado(ids, jefes, 0.15, semilla)
    por_paso = ij.jefe_por_paso(ids, jefes)
    for jefe in set(jefes.values()):
        assert (val & (por_paso == jefe)).any(), f"{jefe} sin validacion"
        assert (train & (por_paso == jefe)).any(), f"{jefe} sin entrenamiento"


def test_un_episodio_nunca_cae_en_entrenamiento_y_validacion_a_la_vez() -> None:
    ids, jefes = episodios_de({"amlo": 9, "nahual": 9})
    train, val = ij.particionar_estratificado(ids, jefes, 0.3, 11)
    assert not (train & val).any() and (train | val).all()
    assert not set(ids[train].tolist()) & set(ids[val].tolist())


def test_cada_jefe_aporta_aproximadamente_la_fraccion_pedida_de_sus_propios_episodios() -> None:
    ids, jefes = episodios_de({"amlo": 200, "nahual": 200})
    _, val = ij.particionar_estratificado(ids, jefes, 0.2, 5)
    por_paso = ij.jefe_por_paso(ids, jefes)
    for jefe in ("amlo", "nahual"):
        fraccion = len(set(ids[val & (por_paso == jefe)].tolist())) / 200
        assert 0.1 < fraccion < 0.3


def test_agregar_episodios_de_otro_jefe_no_mueve_de_lado_a_los_existentes() -> None:
    ids, jefes = episodios_de({"amlo": 30})
    _, val_antes = ij.particionar_estratificado(ids, jefes, 0.2, 3)
    ids2, jefes2 = episodios_de({"amlo": 30, "nahual": 30})
    _, val_despues = ij.particionar_estratificado(ids2, jefes2, 0.2, 3)
    assert np.array_equal(val_antes, val_despues[: len(ids)])


def test_una_precision_global_alta_no_oculta_al_jefe_peor_imitado() -> None:
    prediccion = np.concatenate([np.full(90, 1), np.full(10, 2)]).astype(np.int64)
    etiqueta = np.ones(100, dtype=np.int64)
    jefes = np.asarray(["amlo"] * 90 + ["nahual"] * 10)
    episodios = np.concatenate([np.full(90, 1), np.full(10, 2)]).astype(np.int64)
    metricas = {m.jefe: m for m in ij.metricas_por_jefe(prediccion, etiqueta, jefes, episodios)}
    assert float((prediccion == etiqueta).mean()) == 0.9
    assert metricas["amlo"].precision == 1.0 and metricas["nahual"].precision == 0.0


def test_una_arquitectura_desconocida_se_rechaza_listando_las_admitidas() -> None:
    with pytest.raises(ValueError) as error:
        ij.arquitectura_por_nombre("1x64")
    for nombre in ij.ARQUITECTURAS_CANDIDATAS:
        assert nombre in str(error.value)
