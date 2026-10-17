"""Perfiles de jefe, catalogo y carga de los registros reales."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dmrl.dominio.jefes import ALCANCE_GLOBAL, CatalogoJefes, PerfilJefe
from dmrl.infraestructura.jefes import CARPETA_JEFES, cargar_catalogo
from tests.support_jefes import catalogo_prueba, registro


def test_perfil_se_construye_desde_el_registro() -> None:
    perfil = PerfilJefe.desde_dict(registro("amlo", 2))
    assert perfil.id == "amlo" and perfil.indice == 2 and perfil.mapa == "LVL_AMLO"
    assert perfil.reacciones == frozenset({"Boss_Parried_RM_Montage"})
    corte = perfil.golpes_de("amlo_corte")
    assert corte[0].parry and corte[0].alcance == 200.0 and corte[0].tipo == "melee"
    assert perfil.golpes_de(None) == () and perfil.golpes_de("no_existe") == ()


def test_golpes_se_ordenan_por_posicion_y_leen_velocidad_y_post_montage() -> None:
    datos = registro("nahual", 4)
    datos["vocabulario"].append("x")
    datos["golpes"]["x"] = [
        {"pos": 2.0, "parry": False, "tipo": "rayo", "post_montage": True},
        {"pos": 1.0, "parry": False, "tipo": "proyectil", "velocidad": 3050.0},
    ]
    golpes = PerfilJefe.desde_dict(datos).golpes_de("x")
    assert [g.pos for g in golpes] == [1.0, 2.0]
    assert golpes[0].velocidad == 3050.0 and golpes[1].post_montage
    assert golpes[0].alcance == ALCANCE_GLOBAL


@pytest.mark.parametrize("cambio", [
    {"mapa": "LVL_OTRO"},
    {"vocabulario": ["a", "a"]},
    {"clases": 5},
    {"indice": "x"},
    {"episodio": {"via": "a", "start_phase": 3, "vida_fase1": 1, "vida_fase2": 1}},
    {"episodio": {"via": "a", "start_phase": 1, "vida_fase1": 1, "vida_fase2": 1, "segundos_maximos": 0}},
])
def test_registro_mal_formado_se_rechaza(cambio: dict[str, Any]) -> None:
    datos = registro("amlo", 2)
    datos.update(cambio)
    with pytest.raises(ValueError):
        PerfilJefe.desde_dict(datos)


def test_montage_con_golpes_fuera_del_vocabulario_se_rechaza() -> None:
    datos = registro("amlo", 2)
    datos["golpes"]["suelto"] = [{"pos": 1.0, "parry": True}]
    with pytest.raises(ValueError, match="fuera del vocabulario"):
        PerfilJefe.desde_dict(datos)


def test_registro_sin_campos_obligatorios_se_rechaza() -> None:
    datos = registro("amlo", 2)
    del datos["golpes"]
    with pytest.raises(ValueError):
        PerfilJefe.desde_dict(datos)


@pytest.mark.parametrize("golpe", [
    {"pos": 1.0, "parry": True, "tipo": "magia"},
    {"pos": -1.0, "parry": True},
    {"pos": 1.0, "parry": True, "velocidad": -5.0},
])
def test_golpe_invalido_se_rechaza(golpe: dict[str, Any]) -> None:
    datos = registro("amlo", 2)
    datos["vocabulario"].append("m")
    datos["golpes"]["m"] = [golpe]
    with pytest.raises(ValueError):
        PerfilJefe.desde_dict(datos)


@pytest.mark.parametrize("evento", [
    {"impactos": [{"retardo": 1.0}], "centro": "cielo"},
    {"impactos": [{"retardo": 1.0}], "tipo": "melee"},
    {"impactos": []},
])
def test_evento_invalido_se_rechaza(evento: dict[str, Any]) -> None:
    datos = registro("amlo", 2)
    datos["eventos_distancia"] = {"e": evento}
    with pytest.raises(ValueError):
        PerfilJefe.desde_dict(datos)


def test_catalogo_ordena_por_indice_y_arma_el_vocabulario_union() -> None:
    catalogo = catalogo_prueba()
    assert [p.id for p in catalogo.perfiles] == ["juice", "juice_fase2", "amlo", "chamuco", "nahual"]
    assert catalogo.vocabulario[:2] == ("juice_corte", "juice_rafaga")
    assert len(catalogo.vocabulario) == 10


def test_vocabulario_union_no_repite_un_montage_compartido() -> None:
    a, b = registro("a", 0), registro("b", 1)
    b["vocabulario"] = ["a_corte", "propio"]
    b["golpes"] = {"propio": b["golpes"]["b_corte"]}
    catalogo = CatalogoJefes([PerfilJefe.desde_dict(a), PerfilJefe.desde_dict(b)])
    assert catalogo.vocabulario == ("a_corte", "a_rafaga", "propio")


def test_catalogo_consulta_por_clase_y_por_evento() -> None:
    catalogo = catalogo_prueba()
    assert catalogo.por_clase("BP_Chamuco_C") is catalogo.perfil("chamuco")
    assert catalogo.por_clase("BP_EnemyNPCFRIENDLY_C") is None and catalogo.por_clase(None) is None
    assert catalogo.evento("amlo_throw") is not None and catalogo.evento("no_existe") is None


def test_catalogo_rechaza_incoherencias() -> None:
    with pytest.raises(ValueError):
        CatalogoJefes([])
    a = PerfilJefe.desde_dict(registro("a", 0))
    with pytest.raises(ValueError):
        CatalogoJefes([a, a])
    with pytest.raises(ValueError):
        CatalogoJefes([a, PerfilJefe.desde_dict(registro("b", 5))])
    with pytest.raises(ValueError):
        CatalogoJefes([a, PerfilJefe.desde_dict(registro("b", 1, clases=["BP_A_C"]))])
    otro = registro("b", 1)
    otro["eventos_distancia"] = {"e": {"impactos": [{"retardo": 1.0}]}}
    con_evento = registro("a", 0)
    con_evento["eventos_distancia"] = {"e": {"impactos": [{"retardo": 1.0}]}}
    with pytest.raises(ValueError):
        CatalogoJefes([PerfilJefe.desde_dict(con_evento), PerfilJefe.desde_dict(otro)])


def test_jefes_de_episodio_excluye_la_segunda_fase_de_juice() -> None:
    catalogo = catalogo_prueba()
    assert catalogo.jefes_de_episodio() == ("juice", "amlo", "chamuco", "nahual")
    assert catalogo.jefes_de_episodio("LVL_AMLO") == ("amlo", "chamuco", "nahual")
    assert catalogo.jefes_de_episodio("LVL_Boss") == ("juice",)


def test_fases_de_inicio_y_fase_para_el_mod() -> None:
    catalogo = catalogo_prueba()
    assert catalogo.perfil("juice").fases_inicio == (1, 2)
    assert catalogo.perfil("amlo").fases_inicio == (1, 2)
    assert catalogo.perfil("chamuco").fases_inicio == (2,)
    assert catalogo.perfil("nahual").fases_inicio == (1,)
    assert catalogo.perfil("amlo").fase_para_mod(2) == 2
    assert catalogo.perfil("chamuco").fase_para_mod(2) == 1
    assert catalogo.perfil("nahual").fase_para_mod(1) == 1


def test_vida_efectiva_suma_las_fases_que_quedan_por_jugar() -> None:
    catalogo = catalogo_prueba()
    assert catalogo.vida_efectiva("amlo", 1.0, 1) == 800.0 + 1500.0
    assert catalogo.vida_efectiva("amlo", 0.5, 2) == 750.0
    assert catalogo.vida_efectiva("chamuco", 1.0, 2) == 1500.0


def test_carga_de_los_registros_reales() -> None:
    catalogo = cargar_catalogo()
    assert [p.id for p in catalogo.perfiles] == ["juice", "juice_fase2", "amlo", "chamuco", "nahual"]
    for perfil in catalogo.perfiles:
        assert perfil.golpes and perfil.vocabulario
        for montage in perfil.golpes:
            assert montage in perfil.vocabulario or montage in catalogo.vocabulario
    assert catalogo.jefes_de_episodio("LVL_AMLO") == ("amlo", "chamuco", "nahual")


def test_carga_sin_registros_o_con_registro_que_no_es_objeto(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cargar_catalogo(tmp_path / "no_existe")
    (tmp_path / "x.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_catalogo(tmp_path)
    assert CARPETA_JEFES.name == "jefes"
