"""
Cliente Python del puente de entrenamiento DarkMexicoBridge.

Capas:
    pipes.py     transporte: named pipes de Windows (servidor) con lectura por lineas.
    protocol.py  formato de mensajes, version y validacion de acciones.
    launcher.py  arranque de instancias del juego en modo entrenamiento.
    client.py    sesion con una instancia: hello, reset, step, state, config, close.
"""

from dmbridge.client import BridgeClient, BridgeError, BridgeTimeout
from dmbridge.protocol import PROTO_VERSION, Action

__all__ = ["BridgeClient", "BridgeError", "BridgeTimeout", "PROTO_VERSION", "Action"]
