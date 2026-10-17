"""
Cliente WebSocket minimo para consumir el flujo del panel desde Python.

Sirve a las pruebas, a la validacion con el juego real y, mas adelante, a un grabador que vuelque los
cuadros a un archivo JSONL: cada mensaje del servidor ya es un objeto JSON autocontenido.
"""

from __future__ import annotations

import contextlib
import json
import socket
from typing import Any

from dmrl.infraestructura import websocket_minimo as ws

PLAZO_CONEXION = 5.0


class ClientePanel:
    """
    Conexion de lectura al servidor del panel.

    Contexto:
        Pruebas y scripts de validacion. No es un cliente general: no fragmenta ni negocia extensiones.
    """

    def __init__(self, puerto: int, host: str = "127.0.0.1", ruta: str = "/ws",
                 cabeceras_extra: dict[str, str] | None = None) -> None:
        """
        Abre la conexion y completa el apreton de manos.

        Contexto:
            Inicio de una sesion de lectura.

        Args:
            puerto (int): Puerto del servidor.
            host (str): Direccion del servidor.
            ruta (str): Ruta WebSocket.
            cabeceras_extra (dict[str, str], opcional): Cabeceras adicionales (por ejemplo Origin).

        Returns:
            None: No retorna valor.

        Raises:
            ConnectionError: Si el servidor rechaza la actualizacion o responde algo inesperado.
            OSError: Si no se puede conectar.

        Ejemplo:
            cliente = ClientePanel(8765)
        """
        self._conexion = socket.create_connection((host, puerto), timeout=PLAZO_CONEXION)
        clave = ws.clave_nueva()
        cabeceras = {"Host": f"{host}:{puerto}", "Upgrade": "websocket", "Connection": "Upgrade",
                     "Sec-WebSocket-Key": clave, "Sec-WebSocket-Version": "13"}
        cabeceras.update(cabeceras_extra or {})
        peticion = f"GET {ruta} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in cabeceras.items()) + "\r\n"
        self._conexion.sendall(peticion.encode("ascii"))
        respuesta = b""
        while b"\r\n\r\n" not in respuesta:
            trozo = self._conexion.recv(1)
            if not trozo:
                break
            respuesta += trozo
        primera = respuesta.split(b"\r\n", 1)[0].decode("latin-1")
        if " 101 " not in primera or ws.clave_aceptacion(clave).encode("ascii") not in respuesta:
            self._conexion.close()
            raise ConnectionError(f"el servidor no acepto la conexion: {primera!r}")

    def recibir(self, plazo: float = 2.0) -> dict[str, Any] | None:
        """
        Espera el siguiente mensaje JSON del servidor.

        Contexto:
            Bucle de lectura de pruebas y validaciones. Contesta los ping automaticamente.

        Args:
            plazo (float): Segundos maximos de espera por una trama.

        Returns:
            dict[str, Any] | None: El mensaje, o None si vence el plazo o el servidor cierra.

        Raises:
            ValueError: Si el servidor envia un mensaje que no es JSON valido.

        Ejemplo:
            mensaje = cliente.recibir()
        """
        self._conexion.settimeout(plazo)
        while True:
            try:
                opcode, carga, _ = ws.leer_trama(self._conexion)
            except (TimeoutError, ConnectionError, OSError, ws.ProtocoloError):
                return None
            if opcode == ws.OPCODE_CIERRE:
                return None
            if opcode == ws.OPCODE_PING:
                self._conexion.sendall(ws.codificar_trama(ws.OPCODE_PONG, carga, enmascarar=True))
            elif opcode == ws.OPCODE_TEXTO:
                mensaje = json.loads(carga.decode("utf-8"))
                if not isinstance(mensaje, dict):
                    raise ValueError("el servidor envio un mensaje que no es un objeto JSON")
                return mensaje

    def enviar_trama(self, opcode: int, carga: bytes = b"", enmascarar: bool = True) -> None:
        """
        Envia una trama cruda al servidor.

        Contexto:
            Pruebas de ping, cierre y tramas invalidas.

        Args:
            opcode (int): Tipo de trama.
            carga (bytes): Contenido.
            enmascarar (bool): Si es False envia sin mascara (invalido para un cliente).

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si la conexion falla.

        Ejemplo:
            cliente.enviar_trama(ws.OPCODE_PING, b"x")
        """
        self._conexion.sendall(ws.codificar_trama(opcode, carga, enmascarar=enmascarar))

    def cerrar(self) -> None:
        """
        Cierra la conexion.

        Contexto:
            Fin de la lectura.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores al cerrar se ignoran.

        Ejemplo:
            cliente.cerrar()
        """
        with contextlib.suppress(OSError):
            self._conexion.close()
