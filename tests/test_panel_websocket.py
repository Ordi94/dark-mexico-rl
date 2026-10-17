"""Codec WebSocket minimo y cliente del panel contra servidores guionados."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable

import pytest

from dmrl.infraestructura import websocket_minimo as ws
from dmrl.infraestructura.cliente_panel import ClientePanel


def test_la_clave_de_aceptacion_sigue_el_ejemplo_del_rfc() -> None:
    assert ws.clave_aceptacion("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_validacion_de_claves() -> None:
    assert ws.clave_valida(ws.clave_nueva()) and ws.clave_valida("dGhlIHNhbXBsZSBub25jZQ==")
    assert not ws.clave_valida("corta") and not ws.clave_valida("ñandú") and not ws.clave_valida("")


@pytest.mark.parametrize("largo", [0, 5, 125, 126, 70_000])
@pytest.mark.parametrize("enmascarar", [False, True])
def test_una_trama_codificada_se_lee_igual(largo: int, enmascarar: bool) -> None:
    carga = (bytes(range(256)) * (largo // 256 + 1))[:largo]
    izquierda, derecha = socket.socketpair()
    try:
        threading.Thread(target=lambda: izquierda.sendall(ws.codificar_trama(ws.OPCODE_BINARIO, carga, enmascarar)),
                         daemon=True).start()
        opcode, leida, mascara = ws.leer_trama(derecha)
        assert (opcode, leida, mascara) == (ws.OPCODE_BINARIO, carga, enmascarar)
    finally:
        izquierda.close()
        derecha.close()


def test_codificar_rechaza_cargas_enormes() -> None:
    with pytest.raises(ValueError, match="supera el maximo"):
        ws.codificar_trama(ws.OPCODE_TEXTO, b"x" * (ws.MAXIMO_CARGA + 1))


def leer_con(trama: bytes, maximo: int = ws.MAXIMO_CARGA) -> tuple[int, bytes, bool]:
    """
    Lee una trama construida a mano a traves de un par de sockets.

    Contexto:
        Pruebas de tramas invalidas.

    Args:
        trama (bytes): Bytes a entregar al lector.
        maximo (int): Maximo de carga aceptado.

    Returns:
        tuple[int, bytes, bool]: Resultado de leer_trama.

    Raises:
        ws.ProtocoloError: Si la trama es invalida.
        ConnectionError: Si la trama esta incompleta.

    Ejemplo:
        leer_con(bytes([0x81, 0x00]))
    """
    izquierda, derecha = socket.socketpair()
    try:
        izquierda.sendall(trama)
        izquierda.close()
        return ws.leer_trama(derecha, maximo)
    finally:
        derecha.close()


def test_tramas_fragmentadas_reservadas_o_demasiado_grandes_se_rechazan() -> None:
    with pytest.raises(ws.ProtocoloError, match="fragmentadas"):
        leer_con(bytes([0x01, 0x00]))
    with pytest.raises(ws.ProtocoloError, match="fragmentadas"):
        leer_con(bytes([0xC1, 0x00]))
    with pytest.raises(ws.ProtocoloError, match="supera el maximo"):
        leer_con(ws.codificar_trama(ws.OPCODE_TEXTO, b"x" * 50), maximo=10)


def test_una_trama_incompleta_es_una_conexion_cerrada() -> None:
    with pytest.raises(ConnectionError):
        leer_con(bytes([0x81, 0x05, 0x61]))


def servidor_guionado(guion: Callable[[socket.socket], None]) -> int:
    """
    Servidor de un solo uso que ejecuta un guion sobre la primera conexion.

    Contexto:
        Pruebas del cliente del panel con comportamientos que el servidor real nunca tiene.

    Args:
        guion (Callable): Recibe el socket aceptado.

    Returns:
        int: Puerto en que escucha.

    Raises:
        Ninguna.

    Ejemplo:
        puerto = servidor_guionado(lambda c: c.close())
    """
    escucha = socket.socket()
    escucha.bind(("127.0.0.1", 0))
    escucha.listen(1)

    def atender() -> None:
        """
        Acepta una conexion y ejecuta el guion.

        Contexto:
            Hilo del servidor guionado.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (hilo interno)
        """
        conexion, _ = escucha.accept()
        try:
            guion(conexion)
        finally:
            conexion.close()
            escucha.close()

    threading.Thread(target=atender, daemon=True).start()
    return int(escucha.getsockname()[1])


def aceptar_y(accion: Callable[[socket.socket], None]) -> Callable[[socket.socket], None]:
    """
    Guion que completa el apreton de manos y despues ejecuta una accion.

    Contexto:
        Pruebas del cliente del panel.

    Args:
        accion (Callable): Se ejecuta con el socket ya actualizado a WebSocket.

    Returns:
        Callable: Guion para servidor_guionado.

    Raises:
        Ninguna.

    Ejemplo:
        servidor_guionado(aceptar_y(lambda c: None))
    """
    def guion(conexion: socket.socket) -> None:
        """
        Responde 101 con la clave que corresponde y sigue con la accion.

        Contexto:
            Guion de prueba.

        Args:
            conexion (socket.socket): Conexion aceptada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        peticion = b""
        while b"\r\n\r\n" not in peticion:
            peticion += conexion.recv(1024)
        clave = next(linea.split(b": ")[1] for linea in peticion.split(b"\r\n")
                     if linea.lower().startswith(b"sec-websocket-key")).decode()
        conexion.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Accept: {ws.clave_aceptacion(clave)}\r\n\r\n").encode())
        accion(conexion)

    return guion


def test_el_cliente_falla_si_el_servidor_cierra_durante_el_apreton() -> None:
    puerto = servidor_guionado(lambda c: c.recv(1024))
    with pytest.raises(ConnectionError, match="no acepto"):
        ClientePanel(puerto)


def test_el_cliente_contesta_ping_y_lee_el_texto_siguiente() -> None:
    recibido: list[tuple[int, bytes, bool]] = []

    def accion(conexion: socket.socket) -> None:
        """
        Envia un ping, lee el pong y envia un mensaje.

        Contexto:
            Guion de prueba.

        Args:
            conexion (socket.socket): Conexion ya actualizada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        conexion.sendall(ws.codificar_trama(ws.OPCODE_PING, b"p"))
        recibido.append(ws.leer_trama(conexion))
        conexion.sendall(ws.codificar_trama(ws.OPCODE_TEXTO, json.dumps({"a": 1}).encode()))

    cliente = ClientePanel(servidor_guionado(aceptar_y(accion)))
    assert cliente.recibir() == {"a": 1}
    assert recibido == [(ws.OPCODE_PONG, b"p", True)]
    cliente.cerrar()


def test_el_cliente_rechaza_mensajes_que_no_son_objetos_json_y_ve_el_cierre() -> None:
    def accion(conexion: socket.socket) -> None:
        """
        Envia una lista JSON y despues un cierre.

        Contexto:
            Guion de prueba.

        Args:
            conexion (socket.socket): Conexion ya actualizada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        conexion.sendall(ws.codificar_trama(ws.OPCODE_TEXTO, b"[1]"))
        conexion.sendall(ws.codificar_trama(ws.OPCODE_CIERRE, b""))

    cliente = ClientePanel(servidor_guionado(aceptar_y(accion)))
    with pytest.raises(ValueError, match="no es un objeto"):
        cliente.recibir()
    assert cliente.recibir() is None
    cliente.cerrar()


def test_cerrar_un_cliente_con_el_socket_roto_no_falla() -> None:
    cliente = ClientePanel(servidor_guionado(aceptar_y(lambda c: None)))
    cliente.cerrar()

    class Roto:
        """Socket cuyo cierre falla."""

        def close(self) -> None:
            """
            Falla al cerrar.

            Contexto:
                Prueba de cerrar().

            Args:
                Ninguno.

            Returns:
                None: No retorna valor.

            Raises:
                OSError: Siempre.

            Ejemplo:
                Roto().close()
            """
            raise OSError("roto")

    cliente._conexion = Roto()  # type: ignore[assignment]
    cliente.cerrar()
