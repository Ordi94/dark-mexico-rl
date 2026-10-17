"""
Protocolo del puente (version 1): JSON de una linea por mensaje en cada sentido.

Ordenes (Python -> juego), todas con "op" y "v":
    hello   {"op":"hello","proto":1}
    reset   {"op":"reset","boss_hp_mult":m,"start_phase":1|2}   reinicia la pelea hasta que el jefe actua
            con "boss" (descripcion del jefe de data/jefes) el mod elige el jefe por clase y reinicia en
            sitio sin recargar el mapa; "reload":true fuerza la recarga completa
    step    {"op":"step","n":N,"a":{...}}          aplica la accion y avanza N frames
    state   {"op":"state"}                         observacion sin avanzar
    config  {"op":"config","dilation":k,"console":[...]}
    exec    {"op":"exec","lua":"..."}               solo con DMB_DEBUG=1
    close   {"op":"close"}                         cierra el juego

Cada orden recibe exactamente una respuesta con el mismo "op", "ok" y, si fallo, "err".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PROTO_VERSION = 1
MAX_STEP_FRAMES = 600
BUTTONS = ("rb", "y", "b", "x", "rs")
MOVE_FRAMES = ("boss", "cam")
MIN_BOSS_HP_MULT = 0.05
MAX_BOSS_HP_MULT = 1.0
START_PHASES = (1, 2)


@dataclass(frozen=True)
class Action:
    """
    Accion del agente para un paso.

    Contexto:
        BridgeClient.step. Equivale a lo que haria un jugador con el mando durante el paso.

    Atributos:
        btn: boton de un disparo pulsado al inicio del paso (rb ataque, y especial,
            b maniobra, x caguama, rs enfoque) o None.
        lb: bloqueo/parry mantenido. El evento de pulsar solo se envia al pasar de False a True.
        lt: correr mantenido.
        mx, my: movimiento en [-1, 1]. Con mf="boss", my>0 avanza hacia el jefe y mx>0 va a
            su derecha; con mf="cam" es el stick relativo a la camara.
        mf: marco del movimiento, "boss" o "cam".
    """

    btn: str | None = None
    lb: bool = False
    lt: bool = False
    mx: float = 0.0
    my: float = 0.0
    mf: str = "boss"

    def __post_init__(self) -> None:
        """
        Valida la accion con lista de admitidos.

        Contexto:
            Construccion de Action; el mod vuelve a validar al recibirla.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si el boton, el marco o el rango del movimiento no son validos.

        Ejemplo:
            Action(btn="rb", my=1.0)
        """
        if self.btn is not None and self.btn not in BUTTONS:
            raise ValueError(f"boton no admitido: {self.btn!r}")
        if self.mf not in MOVE_FRAMES:
            raise ValueError(f"marco de movimiento no admitido: {self.mf!r}")
        for name in ("mx", "my"):
            value = getattr(self, name)
            if not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} fuera de [-1, 1]: {value}")

    def to_wire(self) -> dict[str, Any]:
        """
        Convierte la accion al objeto "a" del mensaje step.

        Contexto:
            BridgeClient.step.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Campos con valor distinto del neutro.

        Raises:
            Ninguna.

        Ejemplo:
            Action(btn="rb").to_wire()  # {"btn": "rb"}
        """
        out: dict[str, Any] = {}
        if self.btn is not None:
            out["btn"] = self.btn
        if self.lb:
            out["lb"] = True
        if self.lt:
            out["lt"] = True
        if self.mx:
            out["mx"] = round(float(self.mx), 4)
        if self.my:
            out["my"] = round(float(self.my), 4)
        if self.mf != "boss":
            out["mf"] = self.mf
        return out


def encode(message: dict[str, Any]) -> bytes:
    """
    Codifica un mensaje al formato de linea del protocolo.

    Contexto:
        BridgeClient.request.

    Args:
        message (dict[str, Any]): Mensaje con al menos "op".

    Returns:
        bytes: JSON compacto en UTF-8 con "v" agregado.

    Raises:
        ValueError: Si falta "op".

    Ejemplo:
        encode({"op": "state"})
    """
    if "op" not in message:
        raise ValueError("el mensaje necesita 'op'")
    body = dict(message)
    body["v"] = PROTO_VERSION
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    """
    Decodifica una linea recibida del juego.

    Contexto:
        BridgeClient.request.

    Args:
        line (bytes): Linea JSON sin salto final.

    Returns:
        dict[str, Any]: Mensaje decodificado.

    Raises:
        ValueError: Si no es un objeto JSON o la version no coincide.

    Ejemplo:
        decode(b'{"op":"state","ok":true,"v":1}')
    """
    msg = json.loads(line)
    if not isinstance(msg, dict):
        raise ValueError("la respuesta no es un objeto JSON")
    if msg.get("v") != PROTO_VERSION:
        raise ValueError(f"version de protocolo inesperada: {msg.get('v')!r}")
    return msg
