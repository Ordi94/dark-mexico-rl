"""
Transporte por named pipes de Windows para el puente con el juego.

Python actua como servidor: crea dos pipes por instancia, uno por sentido, porque el mod Lua
los abre con io.open (CRT de MSVC) y un mismo FILE* no puede alternar lectura y escritura
sobre un pipe. Ambos se crean duplex porque el mod los abre en modo "r+b".

    \\\\.\\pipe\\darkmexico_<id>_g2c   juego -> Python (lineas JSON de estado/respuesta)
    \\\\.\\pipe\\darkmexico_<id>_c2g   Python -> juego (lineas JSON de ordenes)
"""

from __future__ import annotations

import contextlib
import ctypes
import msvcrt
import os
import time
from ctypes import wintypes
from typing import BinaryIO

_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

_PIPE_ACCESS_DUPLEX = 0x00000003
_PIPE_TYPE_BYTE = 0x00000000
_PIPE_READMODE_BYTE = 0x00000000
_PIPE_WAIT = 0x00000000
_PIPE_NOWAIT = 0x00000001
_INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
_ERROR_PIPE_CONNECTED = 535
_ERROR_PIPE_LISTENING = 536
_ERROR_NO_DATA = 232
_BUFFER_SIZE = 1 << 16

_KERNEL32.CreateNamedPipeW.restype = wintypes.HANDLE
_KERNEL32.CreateNamedPipeW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
]
_KERNEL32.ConnectNamedPipe.restype = wintypes.BOOL
_KERNEL32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
_KERNEL32.SetNamedPipeHandleState.restype = wintypes.BOOL
_KERNEL32.SetNamedPipeHandleState.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID, wintypes.LPVOID,
]
_KERNEL32.CloseHandle.restype = wintypes.BOOL
_KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]


class PipeError(OSError):
    """Error del transporte por pipe (creacion, conexion o pipe roto)."""


def pipe_names(instance_id: str) -> tuple[str, str]:
    """
    Calcula los nombres de los dos pipes de una instancia.

    Contexto:
        Debe coincidir con dmb/channel.lua del mod; lo usan PipeServer y las pruebas.

    Args:
        instance_id (str): Identificador de la instancia (variable DMB_ID del juego).

    Returns:
        tuple[str, str]: (pipe juego->Python, pipe Python->juego).

    Raises:
        ValueError: Si el identificador tiene caracteres fuera de [A-Za-z0-9_-] o mas de 32.

    Ejemplo:
        g2c, c2g = pipe_names("3")
    """
    allowed = all((c.isascii() and c.isalnum()) or c in "_-" for c in instance_id)
    if not instance_id or len(instance_id) > 32 or not allowed:
        raise ValueError(f"identificador de instancia invalido: {instance_id!r}")
    base = rf"\\.\pipe\darkmexico_{instance_id}"
    return base + "_g2c", base + "_c2g"


def _create_pipe(name: str) -> int:
    """
    Crea una instancia de pipe duplex en modo byte y sin espera (para sondear la conexion).

    Contexto:
        PipeServer.__init__. El modo PIPE_NOWAIT permite esperar al cliente con un plazo sin
        hilos ni E/S superpuesta; tras conectar se pasa a modo bloqueante.

    Args:
        name (str): Ruta completa del pipe.

    Returns:
        int: Handle del pipe.

    Raises:
        PipeError: Si Windows no puede crear el pipe (por ejemplo, el nombre ya esta en uso).

    Ejemplo:
        handle = _create_pipe(r"\\\\.\\pipe\\darkmexico_0_g2c")
    """
    handle = _KERNEL32.CreateNamedPipeW(
        name, _PIPE_ACCESS_DUPLEX, _PIPE_TYPE_BYTE | _PIPE_READMODE_BYTE | _PIPE_NOWAIT,
        1, _BUFFER_SIZE, _BUFFER_SIZE, 0, None,
    )
    if handle == _INVALID_HANDLE_VALUE or handle is None:
        raise PipeError(ctypes.get_last_error(), f"no se pudo crear el pipe {name}")
    return int(handle)


def _poll_connected(handle: int) -> bool:
    """
    Comprueba sin bloquear si un cliente ya abrio el pipe.

    Contexto:
        Bucle de espera de PipeServer.wait_connected.

    Args:
        handle (int): Handle de un pipe creado en modo PIPE_NOWAIT.

    Returns:
        bool: True si hay un cliente conectado.

    Raises:
        PipeError: Ante un error distinto de "escuchando" o "cliente que ya se fue".

    Ejemplo:
        listo = _poll_connected(handle)
    """
    if _KERNEL32.ConnectNamedPipe(handle, None):
        return True
    err = ctypes.get_last_error()
    if err == _ERROR_PIPE_CONNECTED:
        return True
    if err in (_ERROR_PIPE_LISTENING, _ERROR_NO_DATA):
        return False
    raise PipeError(err, "error esperando al cliente del pipe")


def _set_blocking(handle: int) -> None:
    """
    Pasa un pipe ya conectado a modo bloqueante en lectura por bytes.

    Contexto:
        PipeServer.wait_connected, una vez que el juego abrio el pipe.

    Args:
        handle (int): Handle del pipe.

    Returns:
        None: No retorna valor.

    Raises:
        PipeError: Si Windows rechaza el cambio de modo.

    Ejemplo:
        _set_blocking(handle)
    """
    mode = wintypes.DWORD(_PIPE_READMODE_BYTE | _PIPE_WAIT)
    if not _KERNEL32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None):
        raise PipeError(ctypes.get_last_error(), "no se pudo pasar el pipe a modo bloqueante")


class PipeServer:
    """
    Par de pipes servidor para una instancia del juego, con envio y recepcion por lineas.

    Contexto:
        Lo crea BridgeClient antes de que el juego intente conectarse (el mod reintenta cada
        pocos frames). Una sola conexion por objeto.
    """

    def __init__(self, instance_id: str) -> None:
        """
        Crea los dos pipes de la instancia.

        Contexto:
            BridgeClient.__init__.

        Args:
            instance_id (str): Identificador de la instancia.

        Returns:
            None: No retorna valor.

        Raises:
            PipeError: Si algun pipe no se puede crear (nombre en uso por otro proceso).
            ValueError: Si el identificador es invalido.

        Ejemplo:
            server = PipeServer("0")
        """
        self.name_g2c, self.name_c2g = pipe_names(instance_id)
        self._h_g2c = _create_pipe(self.name_g2c)
        try:
            self._h_c2g = _create_pipe(self.name_c2g)
        except PipeError:
            _KERNEL32.CloseHandle(self._h_g2c)
            raise
        self._reader: BinaryIO | None = None
        self._fd_out: int | None = None
        self.bytes_in = 0
        self.bytes_out = 0

    def wait_connected(self, timeout: float) -> None:
        """
        Espera a que el juego abra ambos pipes.

        Contexto:
            BridgeClient.connect, tras lanzar la instancia o para reengancharse a una ya viva.

        Args:
            timeout (float): Segundos maximos de espera.

        Returns:
            None: No retorna valor.

        Raises:
            TimeoutError: Si el juego no se conecta en el plazo.
            PipeError: Ante un error de Windows durante la espera.

        Ejemplo:
            server.wait_connected(120.0)
        """
        deadline = time.monotonic() + timeout
        g2c = c2g = False
        while not (g2c and c2g):
            g2c = g2c or _poll_connected(self._h_g2c)
            c2g = c2g or _poll_connected(self._h_c2g)
            if g2c and c2g:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"el juego no se conecto a {self.name_g2c} en {timeout:.0f} s")
            time.sleep(0.01)
        _set_blocking(self._h_g2c)
        _set_blocking(self._h_c2g)
        fd_in = msvcrt.open_osfhandle(self._h_g2c, os.O_RDONLY | os.O_BINARY)
        self._reader = os.fdopen(fd_in, "rb", buffering=_BUFFER_SIZE)
        self._fd_out = msvcrt.open_osfhandle(self._h_c2g, os.O_WRONLY | os.O_BINARY)

    def send_line(self, line: bytes) -> None:
        """
        Envia una linea completa (se agrega el salto de linea).

        Contexto:
            BridgeClient.request.

        Args:
            line (bytes): Contenido sin salto de linea.

        Returns:
            None: No retorna valor.

        Raises:
            PipeError: Si no hay conexion o el juego cerro el pipe.

        Ejemplo:
            server.send_line(b'{"op":"state","v":1}')
        """
        if self._fd_out is None:
            raise PipeError(0, "pipe sin conectar")
        data = line + b"\n"
        try:
            sent = 0
            while sent < len(data):
                sent += os.write(self._fd_out, data[sent:])
        except OSError as exc:
            raise PipeError(exc.errno or 0, "el juego cerro el pipe") from exc
        self.bytes_out += len(data)

    def recv_line(self) -> bytes:
        """
        Lee una linea completa (bloqueante, sin plazo propio).

        Contexto:
            BridgeClient.request. El plazo lo impone el vigilante del cliente, que mata la
            instancia colgada: al morir el juego la lectura termina con fin de archivo.

        Args:
            Ninguno.

        Returns:
            bytes: Linea sin el salto final.

        Raises:
            PipeError: Si no hay conexion o el juego cerro el pipe.

        Ejemplo:
            linea = server.recv_line()
        """
        if self._reader is None:
            raise PipeError(0, "pipe sin conectar")
        try:
            line = self._reader.readline()
        except OSError as exc:
            raise PipeError(exc.errno or 0, "el juego cerro el pipe") from exc
        if not line.endswith(b"\n"):
            raise PipeError(0, "el juego cerro el pipe")
        self.bytes_in += len(line)
        return line[:-1]

    def close(self) -> None:
        """
        Cierra ambos pipes. Idempotente.

        Contexto:
            BridgeClient.close y manejo de errores.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de cierre se ignoran.

        Ejemplo:
            server.close()
        """
        if self._reader is not None:
            with contextlib.suppress(OSError):
                self._reader.close()
            self._reader = None
        elif self._h_g2c:
            _KERNEL32.CloseHandle(self._h_g2c)
        if self._fd_out is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd_out)
            self._fd_out = None
        elif self._h_c2g:
            _KERNEL32.CloseHandle(self._h_c2g)
        self._h_g2c = 0
        self._h_c2g = 0
