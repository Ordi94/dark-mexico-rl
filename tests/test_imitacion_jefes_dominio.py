"""Particion estratificada, metricas por jefe e informe de la imitacion multi-jefe."""

from __future__ import annotations

import numpy as np
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio import imitacion_jefes as ij


def episodios_de(cuentas: dict[str, int], pasos: int = 3) -> tuple[np.ndarray, dict[int, str]]:
    """
    Identificadores de episodio (pasos contiguos) y el jefe de cada uno.

    Contexto:
        Atajo de las pruebas de la imitacion multi-jefe.

    Args:
        cuentas (dict[str, int]): Jefe -> cantidad de episodios.
        pasos (int): Pasos por episodio.

    Returns:
        tuple: (episodio de cada paso, jefe de cada episodio).

    Raises:
        Ninguna.

    Ejemplo:
        ids, jefes = episodios_de({"amlo": 4, "nahual": 2})
    """
    jefes: dict[int, str] = {}
    siguiente = 100
    for jefe, n in cuentas.items():
        for _ in range(n):
            jefes[siguiente] = jefe
            siguiente += 1
    ids = np.repeat(np.asarray(list(jefes), dtype=np.int64), pasos)
    return ids, jefes


def test_arquitectura_por_nombre_devuelve_las_capas_y_lista_las_admitidas_al_rechazar() -> None:
    assert ij.arquitectura_por_nombre("2x512") == (512, 512)
    assert ij.arquitectura_por_nombre("3x256") == (256, 256, 256)
    with pytest.raises(ValueError, match=r"2x256.*2x512.*3x256"):
        ij.arquitectura_por_nombre("9x9")


def test_la_particion_rechaza_fracciones_fuera_del_intervalo_abierto() -> None:
    ids, jefes = episodios_de({"amlo": 3})
    for fraccion in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraccion_validacion"):
            ij.particionar_estratificado(ids, jefes, fraccion, 0)


def test_la_particion_rechaza_episodios_sin_jefe() -> None:
    ids, jefes = episodios_de({"amlo": 3})
    del jefes[int(ids[0])]
    with pytest.raises(ValueError, match="sin jefe"):
        ij.particionar_estratificado(ids, jefes, 0.2, 0)


def test_la_particion_es_complementaria_y_estable_entre_ejecuciones() -> None:
    ids, jefes = episodios_de({"amlo": 10, "nahual": 6})
    train_a, val_a = ij.particionar_estratificado(ids, jefes, 0.25, 7)
    train_b, val_b = ij.particionar_estratificado(ids, jefes, 0.25, 7)
    assert np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)
    assert np.array_equal(train_a, ~val_a)


def test_un_jefe_con_pocos_episodios_conserva_uno_en_validacion_aunque_el_sorteo_no_lo_elija() -> None:
    ids, jefes = episodios_de({"amlo": 20, "nahual": 2})
    train, val = ij.particionar_estratificado(ids, jefes, 0.01, 1)
    por_paso = ij.jefe_por_paso(ids, jefes)
    for jefe in ("amlo", "nahual"):
        assert (val & (por_paso == jefe)).any() and (train & (por_paso == jefe)).any()


def test_un_jefe_nunca_se_queda_sin_entrenamiento_aunque_el_sorteo_los_elija_a_todos() -> None:
    ids, jefes = episodios_de({"amlo": 2, "nahual": 3})
    train, val = ij.particionar_estratificado(ids, jefes, 0.999, 4)
    por_paso = ij.jefe_por_paso(ids, jefes)
    for jefe in ("amlo", "nahual"):
        assert (val & (por_paso == jefe)).any() and (train & (por_paso == jefe)).any()


def test_un_jefe_con_un_solo_episodio_no_rompe_la_particion() -> None:
    ids, jefes = episodios_de({"amlo": 6, "chamuco": 1})
    train, val = ij.particionar_estratificado(ids, jefes, 0.3, 2)
    assert np.array_equal(train, ~val) and (train | val).all()


def test_jefe_por_paso_asigna_el_jefe_de_cada_episodio_y_falla_si_falta() -> None:
    episodios = np.asarray([1, 1, 2, 3], dtype=np.int64)
    asignado = ij.jefe_por_paso(episodios, {1: "amlo", 2: "nahual", 3: "amlo"})
    assert asignado.tolist() == ["amlo", "amlo", "nahual", "amlo"]
    with pytest.raises(KeyError):
        ij.jefe_por_paso(episodios, {1: "amlo"})


def test_las_metricas_por_jefe_exigen_la_misma_forma() -> None:
    uno = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="misma forma"):
        ij.metricas_por_jefe(uno, uno, np.asarray(["a", "a"]), uno)


def test_las_metricas_por_jefe_se_calculan_por_separado_y_ordenadas() -> None:
    prediccion = np.asarray([1, 1, 2, 2, 3], dtype=np.int64)
    etiqueta = np.asarray([1, 2, 2, 2, 3], dtype=np.int64)
    jefes = np.asarray(["nahual", "nahual", "amlo", "amlo", "amlo"])
    episodios = np.asarray([10, 10, 20, 21, 21], dtype=np.int64)
    metricas = ij.metricas_por_jefe(prediccion, etiqueta, jefes, episodios)
    assert [m.jefe for m in metricas] == ["amlo", "nahual"]
    amlo, nahual = metricas
    assert (amlo.pasos, amlo.episodios, amlo.precision) == (3, 2, 1.0)
    assert (nahual.pasos, nahual.episodios, nahual.precision) == (2, 1, 0.5)
    assert len(amlo.por_accion) == acc.N_ACCIONES


def test_el_informe_por_jefe_solo_incluye_acciones_con_soporte_y_nombres_legibles() -> None:
    prediccion = np.asarray([1, 2], dtype=np.int64)
    etiqueta = np.asarray([1, 1], dtype=np.int64)
    metricas = ij.metricas_por_jefe(prediccion, etiqueta, np.asarray(["amlo", "amlo"]),
                                    np.asarray([1, 1], dtype=np.int64))
    informe = ij.informe_por_jefe(metricas)
    assert set(informe) == {"amlo"} and informe["amlo"]["precision"] == 0.5
    acciones = informe["amlo"]["por_accion"]
    assert [a["accion"] for a in acciones] == [1]
    assert acciones[0]["nombre"] == acc.NOMBRES_ACCIONES[1] and acciones[0]["soporte"] == 2
