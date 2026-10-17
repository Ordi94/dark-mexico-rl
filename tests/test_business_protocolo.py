"""Reglas de negocio del protocolo: que acciones y versiones se admiten."""

from __future__ import annotations

import math

import pytest

from dmbridge import Action
from dmbridge.protocol import decode


def test_business_action_rechaza_boton_no_admitido() -> None:
    """Solo los botones de la lista de admitidos pueden pulsarse."""
    with pytest.raises(ValueError, match="boton no admitido"):
        Action(btn="lb")


def test_business_action_rechaza_boton_vacio() -> None:
    """Una cadena vacia no es un boton: None es la unica forma de no pulsar."""
    with pytest.raises(ValueError, match="boton no admitido"):
        Action(btn="")


def test_business_action_rechaza_marco_de_movimiento_desconocido() -> None:
    """El movimiento solo se interpreta respecto al jefe o a la camara."""
    with pytest.raises(ValueError, match="marco"):
        Action(mf="world")


@pytest.mark.parametrize(("name", "value"), [("mx", 1.01), ("mx", -1.01), ("my", 2.0), ("my", -5.0)])
def test_business_action_movimiento_fuera_de_rango_se_rechaza(name: str, value: float) -> None:
    """El stick virtual esta acotado a [-1, 1] en ambos ejes."""
    with pytest.raises(ValueError, match="fuera de"):
        Action(**{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_business_action_movimiento_en_los_extremos_es_valido(value: float) -> None:
    """Los extremos del rango cerrado se aceptan."""
    assert Action(mx=value, my=value).mx == value


def test_business_action_movimiento_nan_se_rechaza() -> None:
    """NaN no cumple ninguna comparacion y por eso no puede colarse como movimiento."""
    with pytest.raises(ValueError):
        Action(mx=math.nan)


def test_business_action_marco_cam_se_envia_explicito() -> None:
    """El marco por defecto (boss) no viaja; cam si, porque cambia el significado del stick."""
    assert "mf" not in Action(my=1.0).to_wire()
    assert Action(my=1.0, mf="cam").to_wire()["mf"] == "cam"


def test_business_protocolo_rechaza_version_distinta() -> None:
    """Una respuesta de otra version del protocolo no se interpreta."""
    with pytest.raises(ValueError, match="version"):
        decode(b'{"op":"state","ok":true,"v":2}')


def test_business_protocolo_rechaza_respuesta_sin_version() -> None:
    """Una respuesta sin v tampoco se acepta: no se puede asegurar el formato."""
    with pytest.raises(ValueError, match="version"):
        decode(b'{"op":"state","ok":true}')
