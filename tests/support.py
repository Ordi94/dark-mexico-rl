"""
Utilidades compartidas por las pruebas: juego falso por named pipes y proceso falso.

El juego falso habla el protocolo real a traves de los mismos pipes que usa el mod, de modo que
las pruebas ejercitan el transporte de Windows completo sin necesitar el ejecutable del juego.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import threading
from collections.abc import Callable
from typing import IO, Any

from dmbridge.pipes import pipe_names

HANG: Any = object()
CLOSE_PIPES: Any = object()

Handler = Callable[[dict[str, Any]], Any]


def ok_reply(message: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """
    Construye la respuesta exitosa estandar a una orden.

    Contexto:
        Manejador por defecto de FakeGame y base de los manejadores de las pruebas.

    Args:
        message (dict[str, Any]): Orden recibida.
        **extra (Any): Campos adicionales de la respuesta.

    Returns:
        dict[str, Any]: Respuesta con op, ok y la version del protocolo.

    Raises:
        Ninguna.

    Ejemplo:
        ok_reply({"op": "state"}, t=1.0)
    """
    return {"op": message["op"], "ok": True, "v": 1, **extra}


class FakeGame:
    """
    Juego falso: abre los pipes como cliente y responde cada orden con un manejador.

    Contexto:
        Pruebas de BridgeClient y del transporte. El manejador devuelve un dict (respuesta),
        HANG (no responder hasta stop) o CLOSE_PIPES (cerrar sin responder).
    """

    def __init__(self, instance_id: str, handler: Handler | None = None) -> None:
        """
        Prepara el juego falso sin arrancar su hilo.

        Contexto:
            Fixture make_session.

        Args:
            instance_id (str): Identificador de la instancia cuyos pipes se abriran.
            handler (Handler, opcional): Funcion orden -> respuesta; por defecto ok_reply.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si el identificador es invalido.

        Ejemplo:
            game = FakeGame("abc")
        """
        self.name_g2c, self.name_c2g = pipe_names(instance_id)
        self.handler: Handler = handler or ok_reply
        self.received: list[dict[str, Any]] = []
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._out: IO[bytes] | None = None
        self._in: IO[bytes] | None = None
        self._thread = threading.Thread(target=self._run, name=f"fake-game-{instance_id}", daemon=True)

    def start(self) -> None:
        """
        Arranca el hilo que actua como juego.

        Contexto:
            Despues de crear el BridgeClient (los pipes ya existen).

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            RuntimeError: Si el hilo ya estaba arrancado.

        Ejemplo:
            game.start()
        """
        self._thread.start()

    def _run(self) -> None:
        """
        Bucle del hilo: abre los pipes, lee ordenes y escribe respuestas hasta el cierre.

        Contexto:
            Cuerpo del hilo de FakeGame.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de E/S terminan el bucle.

        Ejemplo:
            (uso interno)
        """
        try:
            self._out = open(self.name_g2c, "r+b", buffering=0)  # noqa: SIM115
            self._in = open(self.name_c2g, "r+b", buffering=0)  # noqa: SIM115
            self.connected.set()
            while not self._stop.is_set():
                line = self._in.readline()
                if not line:
                    break
                message = json.loads(line)
                self.received.append(message)
                reply = self.handler(message)
                if reply is HANG:
                    self._stop.wait()
                    break
                if reply is CLOSE_PIPES:
                    break
                self._out.write(json.dumps(reply).encode("utf-8") + b"\n")
        except OSError:
            pass
        finally:
            self._close_files()

    def _close_files(self) -> None:
        """
        Cierra los extremos de cliente de ambos pipes.

        Contexto:
            Final del hilo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de cierre se ignoran.

        Ejemplo:
            (uso interno)
        """
        for stream in (self._out, self._in):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    def stop(self) -> None:
        """
        Pide al hilo que termine y espera su salida con un plazo corto.

        Contexto:
            Cierre de la fixture, despues de cerrar el cliente.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            game.stop()
        """
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class FakeProcess:
    """
    Proceso falso con la interfaz minima de subprocess.Popen que usa el cliente.

    Contexto:
        Pruebas del vigilante y de close/launch sin lanzar ningun ejecutable.
    """

    def __init__(self, on_kill: Callable[[], None] | None = None, wait_timeouts: int = 0) -> None:
        """
        Crea el proceso falso.

        Contexto:
            Pruebas de BridgeClient.

        Args:
            on_kill (Callable, opcional): Funcion llamada al invocar kill().
            wait_timeouts (int): Cuantas llamadas a wait() deben lanzar TimeoutExpired antes
                de devolver.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            proc = FakeProcess(wait_timeouts=1)
        """
        self.killed = 0
        self.wait_calls = 0
        self._on_kill = on_kill
        self._wait_timeouts = wait_timeouts

    def poll(self) -> int | None:
        """
        Estado del proceso.

        Contexto:
            Vigilante del cliente.

        Args:
            Ninguno.

        Returns:
            int | None: None mientras no se haya llamado kill().

        Raises:
            Ninguna.

        Ejemplo:
            proc.poll()
        """
        return -9 if self.killed else None

    def kill(self) -> None:
        """
        Marca el proceso como terminado y ejecuta el gancho on_kill.

        Contexto:
            Vigilante, launch y close.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            proc.kill()
        """
        self.killed += 1
        if self._on_kill is not None:
            self._on_kill()

    def wait(self, timeout: float | None = None) -> int:
        """
        Espera la salida del proceso.

        Contexto:
            close del cliente.

        Args:
            timeout (float, opcional): Plazo solicitado.

        Returns:
            int: Codigo de salida simulado.

        Raises:
            subprocess.TimeoutExpired: Mientras queden esperas configuradas para expirar.

        Ejemplo:
            proc.wait(timeout=1.0)
        """
        self.wait_calls += 1
        if self._wait_timeouts > 0:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired("fake", timeout or 0.0)
        return 0
