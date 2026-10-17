"""
Piezas minimas del protocolo WebSocket (RFC 6455) sobre sockets de la biblioteca estandar.

Lo comparten el servidor del panel y su cliente de pruebas. Solo soporta lo necesario: mensajes
completos (sin fragmentar) de hasta un tamano maximo, ping/pong y cierre. Se escribio aqui en lugar
de agregar una dependencia: el panel corre dentro del proceso que pelea y no debe depender de un
paquete que pueda no estar instalado.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct

GUID_WEBSOCKET = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OPCODE_TEXTO = 0x1
OPCODE_BINARIO = 0x2
OPCODE_CIERRE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA
MAXIMO_CARGA = 1 << 20
LARGO_CLAVE_BYTES = 16


class ProtocoloError(Exception):
    """Trama que el servidor no acepta: fragmentada, demasiado grande o con bits reservados."""


def clave_aceptacion(clave: str) -> str:
    """
    Calcula Sec-WebSocket-Accept a partir de Sec-WebSocket-Key.

    Contexto:
        Apreton de manos del servidor y verificacion del cliente.

    Args:
        clave (str): Valor de Sec-WebSocket-Key enviado por el cliente.

    Returns:
        str: Valor de Sec-WebSocket-Accept.

    Raises:
        Ninguna.

    Ejemplo:
        clave_aceptacion("dGhlIHNhbXBsZSBub25jZQ==")  # "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    """
    resumen = hashlib.sha1((clave + GUID_WEBSOCKET).encode("ascii")).digest()
    return base64.b64encode(resumen).decode("ascii")


def clave_valida(clave: str) -> bool:
    """
    Comprueba que Sec-WebSocket-Key sea base64 de exactamente 16 bytes.

    Contexto:
        Validacion de la peticion de actualizacion en el servidor.

    Args:
        clave (str): Valor recibido.

    Returns:
        bool: True si cumple el formato del RFC.

    Raises:
        Ninguna.

    Ejemplo:
        clave_valida("dGhlIHNhbXBsZSBub25jZQ==")  # True
    """
    try:
        return len(base64.b64decode(clave.encode("ascii"), validate=True)) == LARGO_CLAVE_BYTES
    except (ValueError, UnicodeEncodeError):
        return False


def clave_nueva() -> str:
    """
    Genera una Sec-WebSocket-Key aleatoria.

    Contexto:
        Cliente de pruebas del panel.

    Args:
        Ninguno.

    Returns:
        str: Clave en base64 de 16 bytes aleatorios.

    Raises:
        Ninguna.

    Ejemplo:
        clave_valida(clave_nueva())  # True
    """
    return base64.b64encode(os.urandom(LARGO_CLAVE_BYTES)).decode("ascii")


def codificar_trama(opcode: int, carga: bytes, enmascarar: bool = False) -> bytes:
    """
    Arma una trama completa (FIN activo).

    Contexto:
        El servidor envia tramas sin mascara; el cliente de pruebas las envia enmascaradas, como
        exige el RFC.

    Args:
        opcode (int): Tipo de trama (texto, cierre, ping, pong...).
        carga (bytes): Contenido.
        enmascarar (bool): Si es True aplica una mascara aleatoria de 4 bytes.

    Returns:
        bytes: Trama lista para enviar.

    Raises:
        ValueError: Si la carga supera MAXIMO_CARGA.

    Ejemplo:
        codificar_trama(OPCODE_TEXTO, b"hola")
    """
    largo = len(carga)
    if largo > MAXIMO_CARGA:
        raise ValueError(f"carga de {largo} bytes supera el maximo de {MAXIMO_CARGA}")
    bit_mascara = 0x80 if enmascarar else 0x00
    cabecera = bytes([0x80 | opcode])
    if largo < 126:
        cabecera += bytes([bit_mascara | largo])
    elif largo < 1 << 16:
        cabecera += bytes([bit_mascara | 126]) + struct.pack("!H", largo)
    else:
        cabecera += bytes([bit_mascara | 127]) + struct.pack("!Q", largo)
    if not enmascarar:
        return cabecera + carga
    mascara = os.urandom(4)
    return cabecera + mascara + _aplicar_mascara(carga, mascara)


def _aplicar_mascara(carga: bytes, mascara: bytes) -> bytes:
    """
    Aplica (o quita) la mascara XOR de una trama.

    Contexto:
        codificar_trama y leer_trama.

    Args:
        carga (bytes): Contenido.
        mascara (bytes): Cuatro bytes de mascara.

    Returns:
        bytes: Contenido con XOR aplicado.

    Raises:
        Ninguna.

    Ejemplo:
        _aplicar_mascara(_aplicar_mascara(b"hola", b"abcd"), b"abcd")  # b"hola"
    """
    return bytes(b ^ mascara[i % 4] for i, b in enumerate(carga))


def leer_exacto(conexion: socket.socket, cantidad: int) -> bytes:
    """
    Lee exactamente `cantidad` bytes del socket.

    Contexto:
        leer_trama.

    Args:
        conexion (socket.socket): Socket conectado.
        cantidad (int): Bytes a leer.

    Returns:
        bytes: Los bytes pedidos.

    Raises:
        ConnectionError: Si el otro extremo cierra antes de completar.
        OSError: Si el socket falla o vence su plazo.

    Ejemplo:
        leer_exacto(conexion, 2)
    """
    partes: list[bytes] = []
    faltan = cantidad
    while faltan > 0:
        trozo = conexion.recv(faltan)
        if not trozo:
            raise ConnectionError("el otro extremo cerro la conexion")
        partes.append(trozo)
        faltan -= len(trozo)
    return b"".join(partes)


def leer_trama(conexion: socket.socket, maximo: int = MAXIMO_CARGA) -> tuple[int, bytes, bool]:
    """
    Lee una trama completa.

    Contexto:
        Bucle de lectura del servidor (tramas del navegador) y del cliente de pruebas.

    Args:
        conexion (socket.socket): Socket conectado.
        maximo (int): Tamano maximo de carga aceptado.

    Returns:
        tuple[int, bytes, bool]: (opcode, carga sin mascara, si venia enmascarada).

    Raises:
        ProtocoloError: Si la trama esta fragmentada, usa bits reservados o supera el maximo.
        ConnectionError: Si el otro extremo cierra.
        OSError: Si el socket falla.

    Ejemplo:
        opcode, carga, _ = leer_trama(conexion)
    """
    primero, segundo = leer_exacto(conexion, 2)
    if not primero & 0x80 or primero & 0x70:
        raise ProtocoloError("tramas fragmentadas o con bits reservados no soportadas")
    opcode = primero & 0x0F
    enmascarada = bool(segundo & 0x80)
    largo = segundo & 0x7F
    if largo == 126:
        (largo,) = struct.unpack("!H", leer_exacto(conexion, 2))
    elif largo == 127:
        (largo,) = struct.unpack("!Q", leer_exacto(conexion, 8))
    if largo > maximo:
        raise ProtocoloError(f"carga de {largo} bytes supera el maximo de {maximo}")
    mascara = leer_exacto(conexion, 4) if enmascarada else b""
    carga = leer_exacto(conexion, largo) if largo else b""
    return opcode, _aplicar_mascara(carga, mascara) if enmascarada else carga, enmascarada
