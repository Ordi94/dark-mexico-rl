"""Almacenamiento de demostraciones y estado: escritura atomica, reanudacion y lectura tolerante."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmrl.infraestructura import demostraciones as dem
from tests.support_imitacion import demostraciones_sinteticas


def test_un_lote_guardado_se_recupera_identico_sin_dejar_temporales(tmp_path: Path) -> None:
    """Ida y vuelta de pasos y metadatos; el archivo parcial desaparece."""
    datos = demostraciones_sinteticas(episodios=2, pasos=5)
    meta = [{"indice": 0, "pasos": 5}, {"indice": 1, "pasos": 5}]
    ruta = tmp_path / "ronda_00" / "w0_00000.npz"
    dem.guardar_lote(ruta, datos, meta)
    leidos, meta_leida = dem.cargar_lote(ruta)
    assert np.array_equal(leidos.obs, datos.obs) and np.array_equal(leidos.etiquetas, datos.etiquetas)
    assert np.array_equal(leidos.mascaras, datos.mascaras) and meta_leida == meta
    assert [p.name for p in ruta.parent.iterdir()] == ["w0_00000.npz"]


def test_no_se_guarda_un_lote_incoherente(tmp_path: Path) -> None:
    """La validacion corre antes de escribir: no queda archivo."""
    datos = demostraciones_sinteticas(episodios=1, pasos=3)
    datos.mascaras[0, datos.etiquetas[0]] = False
    with pytest.raises(ValueError):
        dem.guardar_lote(tmp_path / "x.npz", datos, [])
    assert not (tmp_path / "x.npz").exists()


def test_lotes_y_metadatos_de_ronda_y_carga_agregada(tmp_path: Path) -> None:
    """Lotes ordenados, metadatos de una ronda y union de varias rondas."""
    assert dem.lotes_de_ronda(tmp_path, 0) == [] and dem.metadatos_de_ronda(tmp_path, 0) == []
    dem.guardar_lote(tmp_path / "ronda_00" / "w1_00005.npz", demostraciones_sinteticas(1, 4, ronda=0),
                     [{"indice": 5}])
    dem.guardar_lote(tmp_path / "ronda_00" / "w0_00000.npz", demostraciones_sinteticas(1, 3, ronda=0, semilla=1),
                     [{"indice": 0}])
    dem.guardar_lote(tmp_path / "ronda_01" / "w0_00000.npz", demostraciones_sinteticas(1, 2, ronda=1, semilla=2),
                     [{"indice": 0}])
    assert [p.name for p in dem.lotes_de_ronda(tmp_path, 0)] == ["w0_00000.npz", "w1_00005.npz"]
    assert sorted(m["indice"] for m in dem.metadatos_de_ronda(tmp_path, 0)) == [0, 5]
    assert len(dem.cargar_rondas(tmp_path, [0, 1])) == 9
    with pytest.raises(ValueError):
        dem.cargar_rondas(tmp_path, [7])


def test_leer_json_tolera_archivos_ausentes_corruptos_o_de_otra_forma(tmp_path: Path) -> None:
    """Un estado ilegible se trata como inexistente, con el valor por defecto."""
    ruta = tmp_path / "estado.json"
    assert dem.leer_json(ruta) == {} and dem.leer_json(ruta, {"a": 1}) == {"a": 1}
    ruta.write_text("{no es json", encoding="utf-8")
    assert dem.leer_json(ruta, {"a": 1}) == {"a": 1}
    ruta.write_text("[1, 2]", encoding="utf-8")
    assert dem.leer_json(ruta) == {}


def test_escribir_json_es_atomico_y_crea_carpetas(tmp_path: Path) -> None:
    """Ida y vuelta sin dejar archivos parciales."""
    ruta = tmp_path / "a" / "b" / "estado.json"
    dem.escribir_json(ruta, {"ronda": 2})
    assert dem.leer_json(ruta) == {"ronda": 2}
    assert [p.name for p in ruta.parent.iterdir()] == ["estado.json"]
