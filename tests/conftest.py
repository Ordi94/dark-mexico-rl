"""Fixtures compartidas: identificadores de pipe unicos y sesion cliente + juego falso."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

from dmbridge.client import BridgeClient
from dmrl.aplicacion.entrenamiento import ARCHIVO_PID
from tests.support import FakeGame, Handler


@pytest.fixture(scope="session", autouse=True)
def conservar_pid_del_entrenamiento_real() -> Iterator[None]:
    """
    Devuelve runs/entrenamiento.pid a su sitio si alguna prueba lo borra.

    Contexto:
        Varias pruebas de los comandos de entrenamiento terminan con ARCHIVO_PID.unlink, como el
        codigo que ejercitan. Si hay un entrenamiento real en marcha, ese archivo es lo que
        detener.ps1, entrenar.ps1 y los scripts de lanzamiento usan para encontrarlo, y correr la
        suite no debe dejarlo huerfano.

    Args:
        Ninguno.

    Returns:
        Iterator[None]: Cede el control a la sesion de pruebas y repone el archivo al terminar.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    contenido = ARCHIVO_PID.read_bytes() if ARCHIVO_PID.exists() else None
    yield
    if contenido is not None and not ARCHIVO_PID.exists():
        ARCHIVO_PID.write_bytes(contenido)


@pytest.fixture
def instance_id() -> str:
    """
    Identificador de instancia unico por prueba.

    Contexto:
        Evita choques de nombre entre pruebas que crean named pipes reales.

    Args:
        Ninguno.

    Returns:
        str: Identificador alfanumerico de 12 caracteres.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    return "t" + uuid.uuid4().hex[:11]


@dataclass
class Session:
    """Cliente y juego falso emparejados para una prueba."""

    client: BridgeClient
    game: FakeGame


@pytest.fixture
def make_session(instance_id: str) -> Iterator[Callable[..., Session]]:
    """
    Fabrica de sesiones cliente + juego falso con limpieza garantizada.

    Contexto:
        Pruebas de BridgeClient. Cierra primero el cliente (libera la lectura del juego) y luego
        detiene el hilo del juego.

    Args:
        instance_id (str): Identificador unico inyectado.

    Returns:
        Iterator[Callable[..., Session]]: Funcion make(handler, reply_timeout, start_game).

    Raises:
        Ninguna.

    Ejemplo:
        session = make_session()
    """
    created: list[Session] = []

    def make(handler: Handler | None = None, reply_timeout: float = 5.0, start_game: bool = True) -> Session:
        """
        Crea el cliente y arranca el juego falso.

        Contexto:
            Cuerpo de la fabrica.

        Args:
            handler (Handler, opcional): Manejador de ordenes del juego falso.
            reply_timeout (float): Plazo de respuesta del cliente.
            start_game (bool): Si es False el juego falso no se conecta.

        Returns:
            Session: Pareja cliente/juego.

        Raises:
            Ninguna.

        Ejemplo:
            make(reply_timeout=0.3)
        """
        client = BridgeClient(instance_id, reply_timeout=reply_timeout)
        game = FakeGame(instance_id, handler)
        session = Session(client, game)
        created.append(session)
        if start_game:
            game.start()
        return session

    yield make
    for session in created:
        session.client.close(quit_game=False)
        session.game.stop()
