"""Modulos Lua puros del mod (seleccion de jefe y validacion de su descripcion), ejecutados con lupa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lupa
import pytest

from dmrl.infraestructura.jefes import CARPETA_JEFES

SCRIPTS = Path(__file__).resolve().parent.parent / "mod" / "DarkMexicoBridge" / "Scripts"


@pytest.fixture(scope="module")
def lua() -> Any:
    runtime = lupa.LuaRuntime(unpack_returned_tuples=True)
    ruta = (SCRIPTS / "?.lua").as_posix()
    runtime.execute(f'package.path = "{ruta};" .. package.path')
    return runtime


def modulo(lua: Any, nombre: str) -> Any:
    """Carga un modulo del mod (require devuelve dos valores la primera vez; aqui solo el modulo)."""
    return lua.eval(f"(require('{nombre}'))")


def tabla(lua: Any, valor: Any) -> Any:
    """Convierte recursivamente listas y diccionarios de Python en tablas Lua."""
    if isinstance(valor, dict):
        return lua.table_from({k: tabla(lua, v) for k, v in valor.items()})
    if isinstance(valor, list):
        return lua.table_from([tabla(lua, v) for v in valor])
    return valor


def candidatos(lua: Any, filas: list[dict[str, Any]]) -> Any:
    return tabla(lua, filas)


def elegir(lua: Any, filas: list[dict[str, Any]], clases: list[str]) -> Any:
    seleccion = modulo(lua, "dmb.selection")
    return seleccion.pick(candidatos(lua, filas), tabla(lua, clases))


def test_el_jefe_se_elige_por_clase_no_por_bandera_is_boss(lua: Any) -> None:
    """Los NPC amistosos tienen IS BOSS? en true y controlador: no deben tomarse por el rival."""
    filas = [
        {"cls": "BP_EnemyNPCFRIENDLY_C", "controlled": True, "dead": False, "is_boss": True},
        {"cls": "BP_Player_C", "controlled": True, "dead": False, "is_boss": True},
        {"cls": "BP_AMLOBOSS_C", "controlled": True, "dead": False, "is_boss": False},
    ]
    assert elegir(lua, filas, ["BP_AMLOBOSS_C"]) == 3
    assert elegir(lua, filas[:2], ["BP_AMLOBOSS_C"]) is None


def test_prefiere_el_vivo_con_controlador_luego_el_controlado_y_al_final_cualquiera(lua: Any) -> None:
    viejo = {"cls": "BP_AMLOBOSS_C", "controlled": False, "dead": False}
    muerto = {"cls": "BP_Chamuco_C", "controlled": True, "dead": True}
    vivo = {"cls": "BP_Chamuco_C", "controlled": True, "dead": False}
    clases = ["BP_AMLOBOSS_C", "BP_Chamuco_C"]
    assert elegir(lua, [viejo, muerto, vivo], clases) == 3
    assert elegir(lua, [viejo, muerto], clases) == 2
    assert elegir(lua, [viejo], clases) == 1
    assert elegir(lua, [], clases) is None


def test_class_set_sin_lista_es_vacio(lua: Any) -> None:
    seleccion = modulo(lua, "dmb.selection")
    assert list(seleccion.class_set(None).items()) == []


def resultado_de(retorno: Any) -> tuple[Any, Any]:
    """Separa (valor, motivo) de lo que devuelve validate: una tabla o nil con el motivo."""
    return retorno if isinstance(retorno, tuple) else (retorno, None)


def validar(lua: Any, descripcion: dict[str, Any]) -> tuple[Any, Any]:
    spec = modulo(lua, "dmb.spec")
    return resultado_de(spec.validate(tabla(lua, descripcion)))


REGISTROS_DE_EPISODIO = [
    p for p in sorted(CARPETA_JEFES.glob("*.json"))
    if json.loads(p.read_text(encoding="utf-8"))["episodio"].get("mod") is not None
]


@pytest.mark.parametrize("registro", REGISTROS_DE_EPISODIO, ids=lambda p: p.stem)
def test_el_mod_acepta_la_descripcion_de_cada_registro_real(lua: Any, registro: Path) -> None:
    """La seccion mod de cada jefe de episodio pasa la validacion del mod tal como se envia."""
    mod = json.loads(registro.read_text(encoding="utf-8"))["episodio"]["mod"]
    json_lua = lua.eval("require('dmb.json').decode")
    descripcion = json_lua(json.dumps({**mod, "trace_hits": True}))
    spec = modulo(lua, "dmb.spec")
    resultado, motivo = resultado_de(spec.validate(descripcion))
    assert resultado is not None, motivo
    assert resultado.id == mod["id"] and resultado.map == mod["map"] and resultado.trace_hits is True


BASE = {
    "id": "amlo", "map": "LVL_AMLO", "classes": ["BP_AMLOBOSS_C"],
    "start": {"kind": "spawn_class", "class_path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C",
              "pos": {"x": 1.0, "y": 2.0, "z": 3.0}, "player_offset": {"x": 450.0, "y": 0.0, "z": -90.0}},
}


def con(**cambios: Any) -> dict[str, Any]:
    descripcion = json.loads(json.dumps(BASE))
    descripcion.update(cambios)
    return descripcion


def test_descripcion_minima_valida_completa_los_valores_por_defecto(lua: Any) -> None:
    resultado, _ = validar(lua, BASE)
    assert resultado.id == "amlo" and resultado.trace_hits is False
    assert list(resultado.purge.values()) == ["BP_AMLOBOSS_C"] and list(resultado.minors.values()) == []


@pytest.mark.parametrize("cambio", [
    {"map": "LVL_OTRO"},
    {"id": "con espacio"},
    {"classes": []},
    {"classes": ["BP_X_C", "con-guion"]},
    {"classes": ["A"] * 40},
    {"purge": "no_es_lista"},
    {"minors": ["mal nombre"]},
    {"start": {"kind": "desconocido"}},
    {"start": {"kind": "spawn_class", "class_path": "/Windows/system32/x.dll",
               "pos": {"x": 0, "y": 0, "z": 0}, "player_offset": {"x": 0, "y": 0, "z": 0}}},
    {"start": {"kind": "spawn_class", "class_path": "/Game/A.A_C", "pos": {"x": "uno", "y": 0, "z": 0},
               "player_offset": {"x": 0, "y": 0, "z": 0}}},
    {"start": {"kind": "spawn_class", "class_path": "/Game/A.A_C", "pos": {"x": 0, "y": 0, "z": 0},
               "player_offset": {"x": 0, "y": 0, "z": 0}, "hp": 10_000_000}},
    {"start": {"kind": "level_event", "level_class": "LVL_AMLO_C", "event": "SPAWN; os.exit()",
               "boss_class": "BP_X_C", "player_offset": {"x": 0, "y": 0, "z": 0}}},
    {"start": {"kind": "level_event", "level_class": "mal nombre", "event": "SPAWN  X", "boss_class": "BP_X_C",
               "player_offset": {"x": 0, "y": 0, "z": 0}}},
    {"start": {"kind": "level_event", "level_class": "L_C", "event": "X", "boss_class": "BP_X_C",
               "wait_seconds": 99, "player_offset": {"x": 0, "y": 0, "z": 0}}},
    {"hooks": [{"path": "no/es/ruta", "event": "victory"}]},
    {"hooks": [{"path": "/Game/A.A_C:death", "event": "ejecutar"}]},
    {"hooks": [{"path": "/Game/A.A_C:death", "event": "remote"}]},
    {"hooks": "no_es_lista"},
    {"phase2": {"class": "mal nombre"}},
    {"phase2": {"class": "BP_X_C", "timeout": 1}},
])
def test_el_mod_rechaza_descripciones_fuera_de_la_lista_de_admitidos(lua: Any, cambio: dict[str, Any]) -> None:
    resultado, motivo = validar(lua, con(**cambio))
    assert resultado is None and motivo


def test_descripcion_que_no_es_objeto_se_rechaza(lua: Any) -> None:
    spec = modulo(lua, "dmb.spec")
    resultado, motivo = resultado_de(spec.validate("texto"))
    assert resultado is None and "objeto" in motivo
    resultado, motivo = resultado_de(spec.validate(tabla(lua, {"id": "x", "map": "LVL_Boss", "classes": ["A"]})))
    assert resultado is None and "start" in motivo


def test_los_ganchos_de_ataques_a_distancia_exigen_id(lua: Any) -> None:
    hook = {"path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C:Ataque Throw Kat", "event": "remote", "id": "amlo_throw_kat"}
    resultado, _ = validar(lua, con(hooks=[hook], phase2={"class": "BP_Chamuco_C", "timeout": 30}))
    assert resultado is not None and resultado.phase2.timeout == 30
