"""Pruebas del transporte por named pipes de Windows (pipes reales, sin el juego)."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from types import SimpleNamespace
from typing import IO, Any

import pytest

from dmbridge import pipes
from dmbridge.pipes import PipeError, PipeServer, pipe_names
from tests.support import CLOSE_PIPES, FakeGame, ok_reply


class GameSide:
    """Extremos de cliente (juego) de los dos pipes, abiertos en el hilo de la prueba."""

    def __init__(self, instance_id: str) -> None:
        """
        Abre ambos pipes como lo hace el mod: modo r+b sin buffer.

        Contexto:
            Fixture connected_pair; abrir un pipe ya creado no bloquea.

        Args:
            instance_id (str): Identificador de la instancia.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si los pipes no existen.

        Ejemplo:
            side = GameSide("abc")
        """
        name_g2c, name_c2g = pipe_names(instance_id)
        self.to_python: IO[bytes] = open(name_g2c, "r+b", buffering=0)  # noqa: SIM115
        self.from_python: IO[bytes] = open(name_c2g, "r+b", buffering=0)  # noqa: SIM115

    def close(self) -> None:
        """
        Cierra los dos extremos del juego.

        Contexto:
            Limpieza de la fixture y pruebas de cierre del cliente.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Cerrar dos veces es inofensivo.

        Ejemplo:
            side.close()
        """
        self.to_python.close()
        self.from_python.close()


@pytest.fixture
def connected_pair(instance_id: str) -> Iterator[tuple[PipeServer, GameSide]]:
    """
    Servidor conectado a un juego simulado en el mismo hilo.

    Contexto:
        Pruebas de send_line/recv_line y cierre.

    Args:
        instance_id (str): Identificador unico inyectado.

    Returns:
        Iterator[tuple[PipeServer, GameSide]]: Servidor y extremos del juego.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    server = PipeServer(instance_id)
    side = GameSide(instance_id)
    server.wait_connected(5.0)
    yield server, side
    side.close()
    server.close()


def test_pipe_names_formato() -> None:
    """Los nombres siguen el contrato con el mod: darkmexico_<id>_g2c / _c2g."""
    assert pipe_names("3") == (r"\\.\pipe\darkmexico_3_g2c", r"\\.\pipe\darkmexico_3_c2g")


@pytest.mark.parametrize("bad", ["", "a b", "a/b", "a\\b", "x" * 33, "id.1", "ni\u00f1o"])
def test_pipe_names_invalidos_levantan_value_error(bad: str) -> None:
    """Identificadores vacios, largos o con caracteres fuera de la lista de admitidos se rechazan."""
    with pytest.raises(ValueError, match="identificador"):
        pipe_names(bad)


def test_pipe_names_acepta_guion_y_guion_bajo_y_32_caracteres() -> None:
    """El limite de 32 caracteres y los separadores - y _ se aceptan."""
    pipe_names("a-b_c")
    pipe_names("x" * 32)


def test_ida_y_vuelta_real(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Una linea enviada llega al juego y la respuesta del juego llega al servidor."""
    server, side = connected_pair
    server.send_line(b'{"op":"state"}')
    assert side.from_python.readline() == b'{"op":"state"}\n'
    side.to_python.write(b'{"ok":true}\n')
    assert server.recv_line() == b'{"ok":true}'
    assert server.bytes_out == len(b'{"op":"state"}\n')
    assert server.bytes_in == len(b'{"ok":true}\n')


def test_varias_lineas_consecutivas_conservan_el_orden(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """El lector por lineas separa mensajes que llegan juntos."""
    server, side = connected_pair
    side.to_python.write(b"uno\ndos\ntres\n")
    assert [server.recv_line() for _ in range(3)] == [b"uno", b"dos", b"tres"]


def test_ida_y_vuelta_con_hilo_cliente(instance_id: str) -> None:
    """Un hilo que actua de juego responde a las ordenes del servidor por los pipes reales."""
    server = PipeServer(instance_id)
    game = FakeGame(instance_id, lambda m: ok_reply(m, eco=m["op"]))
    game.start()
    try:
        server.wait_connected(5.0)
        server.send_line(b'{"op":"ping"}')
        assert b'"eco": "ping"' in server.recv_line()
    finally:
        server.close()
        game.stop()


def test_recv_tras_cierre_del_cliente_levanta_pipe_error(instance_id: str) -> None:
    """Cuando el juego cierra, la lectura termina en PipeError en lugar de colgarse."""
    server = PipeServer(instance_id)
    game = FakeGame(instance_id, lambda _m: CLOSE_PIPES)
    game.start()
    try:
        server.wait_connected(5.0)
        server.send_line(b'{"op":"state"}')
        with pytest.raises(PipeError):
            server.recv_line()
    finally:
        server.close()
        game.stop()


def test_recv_con_linea_incompleta_levanta_pipe_error(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Una linea sin salto final al cerrar el pipe se descarta como conexion perdida."""
    server, side = connected_pair
    side.to_python.write(b"parcial")
    side.close()
    with pytest.raises(PipeError):
        server.recv_line()


def test_send_con_el_cliente_cerrado_levanta_pipe_error(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Escribir a un pipe cuyo juego se fue falla con PipeError."""
    server, side = connected_pair
    side.close()
    with pytest.raises(PipeError):
        server.send_line(b"hola")


def test_recv_con_error_de_lectura_levanta_pipe_error(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Un OSError del lector se traduce a PipeError conservando el errno."""
    server, _side = connected_pair

    def broken_readline() -> bytes:
        """
        Simula un fallo de E/S del lector.

        Contexto:
            Sustituye a readline del lector del servidor.

        Args:
            Ninguno.

        Returns:
            bytes: No retorna.

        Raises:
            OSError: Siempre.

        Ejemplo:
            (uso interno)
        """
        raise OSError(5, "fallo simulado")

    original = server._reader
    server._reader = SimpleNamespace(readline=broken_readline, close=original.close)  # type: ignore[assignment,union-attr]
    with pytest.raises(PipeError) as info:
        server.recv_line()
    assert info.value.errno == 5
    server._reader = original


def test_send_y_recv_sin_conectar_levantan_pipe_error(instance_id: str) -> None:
    """Antes de wait_connected no se puede enviar ni recibir."""
    server = PipeServer(instance_id)
    try:
        with pytest.raises(PipeError, match="sin conectar"):
            server.send_line(b"x")
        with pytest.raises(PipeError, match="sin conectar"):
            server.recv_line()
    finally:
        server.close()


def test_wait_connected_sin_cliente_vence_por_timeout(instance_id: str) -> None:
    """Sin juego, la espera termina con TimeoutError y deja el servidor cerrable."""
    server = PipeServer(instance_id)
    try:
        with pytest.raises(TimeoutError, match="no se conecto"):
            server.wait_connected(0.05)
    finally:
        server.close()


def test_wait_connected_con_un_solo_pipe_conectado_vence_por_timeout(instance_id: str) -> None:
    """Si el juego solo abrio uno de los dos pipes la conexion no se da por buena."""
    server = PipeServer(instance_id)
    name_g2c, _ = pipe_names(instance_id)
    with open(name_g2c, "r+b", buffering=0) as partial:
        assert partial is not None
        try:
            with pytest.raises(TimeoutError):
                server.wait_connected(0.05)
        finally:
            server.close()


def test_segundo_servidor_con_el_mismo_id_falla(instance_id: str) -> None:
    """El nombre en uso se informa con PipeError (una instancia por identificador)."""
    first = PipeServer(instance_id)
    try:
        with pytest.raises(PipeError):
            PipeServer(instance_id)
    finally:
        first.close()


def test_fallo_al_crear_el_segundo_pipe_libera_el_primero(instance_id: str) -> None:
    """Si c2g ya esta ocupado, el g2c recien creado se cierra y puede volver a crearse."""
    _, name_c2g = pipe_names(instance_id)
    name_g2c, _ = pipe_names(instance_id)
    blocker = pipes._create_pipe(name_c2g)
    try:
        with pytest.raises(PipeError):
            PipeServer(instance_id)
        again = pipes._create_pipe(name_g2c)
        pipes._KERNEL32.CloseHandle(again)
    finally:
        pipes._KERNEL32.CloseHandle(blocker)


def test_close_es_idempotente_antes_de_conectar(instance_id: str) -> None:
    """Cerrar dos veces un servidor sin conectar no falla y libera el nombre."""
    server = PipeServer(instance_id)
    server.close()
    server.close()
    PipeServer(instance_id).close()


def test_close_es_idempotente_tras_conectar(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Cerrar dos veces un servidor conectado no falla."""
    server, _side = connected_pair
    server.close()
    server.close()


def test_close_ignora_errores_al_cerrar_lector_y_descriptor(connected_pair: tuple[PipeServer, GameSide]) -> None:
    """Los OSError de cierre del lector o del descriptor de salida se ignoran."""
    server, _side = connected_pair

    def failing_close() -> None:
        """
        Simula un fallo al cerrar el lector.

        Contexto:
            Sustituye a close del lector del servidor.

        Args:
            Ninguno.

        Returns:
            None: No retorna.

        Raises:
            OSError: Siempre.

        Ejemplo:
            (uso interno)
        """
        raise OSError(9, "fallo simulado")

    real_reader = server._reader
    assert real_reader is not None
    server._reader = SimpleNamespace(close=failing_close)  # type: ignore[assignment]
    real_fd = server._fd_out
    assert real_fd is not None
    server._fd_out = 1_000_000  # descriptor invalido: os.close lanza OSError
    server.close()
    assert server._reader is None
    assert server._fd_out is None
    real_reader.close()
    os.close(real_fd)


# --- funciones de bajo nivel con Windows simulado ---------------------------------------


def fake_kernel(**results: Any) -> SimpleNamespace:
    """
    Construye un kernel32 falso cuyas funciones devuelven (resultado, codigo de error).

    Contexto:
        Pruebas de las ramas de error de _poll_connected y _set_blocking.

    Args:
        **results (Any): Pares nombre_de_funcion=(valor, error).

    Returns:
        SimpleNamespace: Objeto con las funciones simuladas.

    Raises:
        Ninguna.

    Ejemplo:
        fake_kernel(ConnectNamedPipe=(0, 536))
    """

    def make(value: int, error: int) -> Any:
        """
        Crea una funcion que fija el ultimo error de Windows y devuelve un valor.

        Contexto:
            Cuerpo de fake_kernel.

        Args:
            value (int): Valor devuelto.
            error (int): Codigo para ctypes.get_last_error.

        Returns:
            Any: Funcion simulada.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """

        def call(*_args: Any) -> int:
            """
            Simula la llamada de Windows.

            Contexto:
                Cuerpo de make.

            Args:
                *_args (Any): Ignorados.

            Returns:
                int: Valor configurado.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            ctypes.set_last_error(error)
            return value

        return call

    return SimpleNamespace(**{name: make(*pair) for name, pair in results.items()})


@pytest.mark.parametrize(
    ("result", "error", "expected"),
    [(1, 0, True), (0, 535, True), (0, 536, False), (0, 232, False)],
)
def test_poll_connected_interpreta_los_codigos_de_windows(
    monkeypatch: pytest.MonkeyPatch, result: int, error: int, expected: bool
) -> None:
    """Conectado o ya conectado -> True; escuchando o cliente que se fue -> False."""
    monkeypatch.setattr(pipes, "_KERNEL32", fake_kernel(ConnectNamedPipe=(result, error)))
    assert pipes._poll_connected(1) is expected


def test_poll_connected_error_desconocido_levanta_pipe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un codigo inesperado de Windows se propaga como PipeError."""
    monkeypatch.setattr(pipes, "_KERNEL32", fake_kernel(ConnectNamedPipe=(0, 5)))
    with pytest.raises(PipeError) as info:
        pipes._poll_connected(1)
    assert info.value.errno == 5


def test_set_blocking_falla_si_windows_rechaza_el_cambio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si SetNamedPipeHandleState falla se informa con PipeError."""
    monkeypatch.setattr(pipes, "_KERNEL32", fake_kernel(SetNamedPipeHandleState=(0, 87)))
    with pytest.raises(PipeError, match="bloqueante"):
        pipes._set_blocking(1)


def test_create_pipe_con_nombre_invalido_levanta_pipe_error() -> None:
    """Un nombre que Windows no acepta se informa con PipeError."""
    with pytest.raises(PipeError):
        pipes._create_pipe("sin_prefijo_de_pipe")
