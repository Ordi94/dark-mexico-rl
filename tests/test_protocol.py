"""Pruebas de protocolo: validacion de Action, serializacion y codificacion de mensajes."""

from __future__ import annotations

import json

import pytest

from dmbridge import PROTO_VERSION, Action
from dmbridge.protocol import BUTTONS, MAX_STEP_FRAMES, decode, encode


def test_action_por_defecto_es_neutra() -> None:
    """Una accion sin argumentos no pulsa nada y se serializa vacia."""
    action = Action()
    assert action.to_wire() == {}
    assert action.mf == "boss"


@pytest.mark.parametrize("button", BUTTONS)
def test_action_acepta_cada_boton_admitido(button: str) -> None:
    """Todos los botones de la lista de admitidos se aceptan y viajan en el campo btn."""
    assert Action(btn=button).to_wire() == {"btn": button}


def test_to_wire_incluye_solo_campos_no_neutros() -> None:
    """to_wire omite los valores neutros y redondea el movimiento a 4 decimales."""
    action = Action(btn="y", lb=True, lt=True, mx=0.123456, my=-1.0, mf="cam")
    assert action.to_wire() == {"btn": "y", "lb": True, "lt": True, "mx": 0.1235, "my": -1.0, "mf": "cam"}


def test_action_es_inmutable() -> None:
    """Action es frozen: no se puede alterar tras validarla."""
    action = Action()
    with pytest.raises(AttributeError):
        action.lb = True  # type: ignore[misc]


def test_encode_agrega_version_y_es_compacto() -> None:
    """encode agrega v, usa separadores compactos y no muta el mensaje original."""
    message = {"op": "state"}
    raw = encode(message)
    assert raw == b'{"op":"state","v":1}'
    assert message == {"op": "state"}


def test_encode_sin_op_falla() -> None:
    """Un mensaje sin op no es una orden valida."""
    with pytest.raises(ValueError, match="op"):
        encode({"n": 1})


def test_encode_conserva_unicode_como_utf8() -> None:
    """El codigo Lua con caracteres no ASCII viaja como UTF-8 valido."""
    raw = encode({"op": "exec", "lua": "return 'ninoñ'"})
    assert json.loads(raw.decode("utf-8"))["lua"] == "return 'ninoñ'"


def test_decode_devuelve_el_objeto() -> None:
    """decode devuelve el diccionario cuando la version coincide."""
    assert decode(b'{"op":"state","ok":true,"v":1}') == {"op": "state", "ok": True, "v": 1}


def test_decode_rechaza_json_que_no_es_objeto() -> None:
    """Una lista JSON no es un mensaje del protocolo."""
    with pytest.raises(ValueError, match="objeto"):
        decode(b"[1, 2]")


def test_decode_rechaza_json_invalido() -> None:
    """Un texto que no es JSON levanta ValueError (JSONDecodeError)."""
    with pytest.raises(ValueError):
        decode(b"no es json")


def test_constantes_del_protocolo() -> None:
    """La version y el limite de frames son los que el mod espera."""
    assert PROTO_VERSION == 1
    assert MAX_STEP_FRAMES == 600
