"""
Sesion con una instancia del juego a traves de DarkMexicoBridge.

Modo lockstep: cada orden recibe una respuesta y, entre una respuesta y la siguiente orden,
el hilo del juego esta bloqueado esperando. Para que un juego colgado no bloquee a Python para
siempre, un hilo vigilante mata la instancia si una respuesta tarda mas que reply_timeout; al
morir el juego la lectura del pipe termina y la orden falla con BridgeTimeout.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from typing import Any

from dmbridge import protocol
from dmbridge.launcher import LaunchConfig, launch_instance
from dmbridge.pipes import PipeError, PipeServer
from dmbridge.protocol import Action


class BridgeError(RuntimeError):
    """El juego respondio con ok=false o la conexion se perdio."""


class BridgeTimeout(BridgeError):  # noqa: N818 - nombre publico usado por los scripts
    """El juego no respondio a tiempo y la instancia fue terminada."""


class BridgeClient:
    """
    Cliente de una instancia: crea los pipes, opcionalmente lanza el juego y envia ordenes.

    Contexto:
        Entorno de entrenamiento y scripts de prueba (scripts/bridge_smoke.py).
    """

    def __init__(self, instance_id: str = "0", reply_timeout: float = 30.0) -> None:
        """
        Crea los pipes de la instancia (el juego se conecta cuando los encuentra).

        Contexto:
            Antes de lanzar el juego o para reengancharse a una instancia que ya corre.

        Args:
            instance_id (str): Identificador de la instancia (DMB_ID).
            reply_timeout (float): Segundos maximos de espera de cada respuesta.

        Returns:
            None: No retorna valor.

        Raises:
            PipeError: Si los pipes no se pueden crear.

        Ejemplo:
            client = BridgeClient("0")
        """
        self.instance_id = instance_id
        self.reply_timeout = reply_timeout
        self.pipe = PipeServer(instance_id)
        self.process: subprocess.Popen[bytes] | None = None
        self.last_latency = 0.0
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._timed_out = False
        self._stop = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, name=f"dmb-watchdog-{instance_id}", daemon=True)
        self._watchdog.start()

    def _watch(self) -> None:
        """
        Hilo vigilante: termina la instancia si una respuesta excede el plazo.

        Contexto:
            Arrancado en __init__; corre hasta close().

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        while not self._stop.wait(0.25):
            with self._lock:
                expired = self._deadline is not None and time.monotonic() > self._deadline
            if expired and self.process is not None and self.process.poll() is None:
                self._timed_out = True
                self.process.kill()

    def launch(self, config: LaunchConfig | None = None, connect_timeout: float = 180.0) -> dict[str, Any]:
        """
        Lanza una instancia del juego y espera a que se conecte.

        Contexto:
            Punto de entrada habitual del entorno de entrenamiento.

        Args:
            config (LaunchConfig, opcional): Parametros de arranque; su instance_id se fuerza
                al de este cliente.
            connect_timeout (float): Segundos maximos hasta que el mod abre los pipes.

        Returns:
            dict[str, Any]: Respuesta de hello.

        Raises:
            TimeoutError: Si el juego no se conecta a tiempo (la instancia se termina).
            BridgeError: Si el saludo falla.

        Ejemplo:
            info = client.launch(LaunchConfig(profile="nullrhi", sim_hz=30.0))
        """
        config = config or LaunchConfig()
        config.instance_id = self.instance_id
        self.process = launch_instance(config)
        try:
            info = self.connect(connect_timeout)
            dilation = config.dilation()
            if dilation is not None:
                self.config(dilation=dilation)
            return info
        except Exception:
            self.process.kill()
            raise

    def connect(self, timeout: float = 180.0) -> dict[str, Any]:
        """
        Espera la conexion del juego y saluda con la version del protocolo.

        Contexto:
            launch, o para reengancharse a una instancia ya en marcha.

        Args:
            timeout (float): Segundos maximos de espera.

        Returns:
            dict[str, Any]: Respuesta de hello.

        Raises:
            TimeoutError: Si el juego no se conecta.
            BridgeError: Si el juego rechaza la version.

        Ejemplo:
            client.connect(60.0)
        """
        self.pipe.wait_connected(timeout)
        return self.request({"op": "hello", "proto": protocol.PROTO_VERSION})

    def request(self, message: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """
        Envia una orden y espera su respuesta.

        Contexto:
            Todos los metodos de alto nivel.

        Args:
            message (dict[str, Any]): Orden con "op".
            timeout (float, opcional): Plazo propio de esta orden; por defecto reply_timeout.

        Returns:
            dict[str, Any]: Respuesta con ok=true.

        Raises:
            BridgeTimeout: Si el juego no respondio en el plazo (fue terminado).
            BridgeError: Si la respuesta trae ok=false o el pipe se cerro.

        Ejemplo:
            st = client.request({"op": "state"})
        """
        line = protocol.encode(message)
        t0 = time.perf_counter()
        with self._lock:
            self._deadline = time.monotonic() + (timeout or self.reply_timeout)
        try:
            self.pipe.send_line(line)
            raw = self.pipe.recv_line()
        except PipeError as exc:
            if self._timed_out:
                raise BridgeTimeout(f"sin respuesta a {message['op']} en el plazo; instancia terminada") from exc
            raise BridgeError(f"conexion perdida durante {message['op']}: {exc}") from exc
        finally:
            with self._lock:
                self._deadline = None
        self.last_latency = time.perf_counter() - t0
        reply = protocol.decode(raw)
        if not reply.get("ok", False):
            raise BridgeError(f"{reply.get('op')}: {reply.get('err')}")
        return reply

    def reset(self, timeout: float = 120.0, boss_hp_mult: float | None = None,
              start_phase: int | None = None, boss: dict[str, Any] | None = None,
              reload: bool = False) -> dict[str, Any]:
        """
        Reinicia la pelea: devuelve la primera observacion con el jefe ya activo.

        Contexto:
            Inicio de cada episodio. Los parametros opcionales configuran la dificultad del
            episodio (curriculo): el mod los valida y los aplica sin tocar la vida del jugador.

        Args:
            timeout (float): Plazo del reinicio completo (carga del mapa incluida).
            boss_hp_mult (float, opcional): Multiplicador de la vida del jefe, en [0.05, 1.0].
                Se aplica a la fase 1 y tambien al actor de la fase 2.
            start_phase (int, opcional): 1 (defecto) o 2. Con 2 el episodio arranca con el jefe
                de la fase 2 ya activo.
            boss (dict, opcional): Descripcion del jefe (seccion "mod" de data/jefes/<jefe>.json):
                mapa, clases, arranque y eventos. Sin ella el mod usa el flujo original de Juice.
            reload (bool): Con boss, fuerza la recarga completa del mapa en lugar del reinicio
                en sitio (reinicio periodico de respaldo).

        Returns:
            dict[str, Any]: Observacion inicial.

        Raises:
            ValueError: Si boss_hp_mult o start_phase estan fuera de los valores admitidos.
            BridgeError: Si el mod no logro reiniciar.

        Ejemplo:
            obs = client.reset(boss_hp_mult=0.3, start_phase=2)
        """
        message: dict[str, Any] = {"op": "reset"}
        if boss_hp_mult is not None:
            if not protocol.MIN_BOSS_HP_MULT <= float(boss_hp_mult) <= protocol.MAX_BOSS_HP_MULT:
                raise ValueError(f"boss_hp_mult fuera de [{protocol.MIN_BOSS_HP_MULT}, {protocol.MAX_BOSS_HP_MULT}]")
            message["boss_hp_mult"] = round(float(boss_hp_mult), 4)
        if start_phase is not None:
            if start_phase not in protocol.START_PHASES:
                raise ValueError(f"start_phase no admitido: {start_phase!r}")
            message["start_phase"] = start_phase
        if boss is not None:
            message["boss"] = boss
            if reload:
                message["reload"] = True
        return self.request(message, timeout=timeout)

    def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
        """
        Aplica una accion y avanza n_frames frames de simulacion.

        Contexto:
            Bucle del agente.

        Args:
            action (Action): Accion del paso.
            n_frames (int): Frames a simular (1..600).

        Returns:
            dict[str, Any]: Observacion al final del paso, con eventos en "ev".

        Raises:
            ValueError: Si n_frames esta fuera de rango.
            BridgeError: Si el juego rechaza la orden.

        Ejemplo:
            obs = client.step(Action(btn="rb"), 2)
        """
        if not 1 <= n_frames <= protocol.MAX_STEP_FRAMES:
            raise ValueError(f"n_frames fuera de [1, {protocol.MAX_STEP_FRAMES}]")
        return self.request({"op": "step", "n": n_frames, "a": action.to_wire()})

    def state(self) -> dict[str, Any]:
        """
        Observacion actual sin avanzar la simulacion.

        Contexto:
            Diagnostico y comprobaciones de los scripts.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Observacion con resumen de guardas en "guards".

        Raises:
            BridgeError: Si el juego no responde correctamente.

        Ejemplo:
            st = client.state()
        """
        return self.request({"op": "state"})

    def config(self, dilation: float | None = None, console: list[str] | None = None,
               anim_tick_option: int | bool | None = None) -> dict[str, Any]:
        """
        Ajusta la dilatacion de tiempo, el refresco de huesos o ejecuta comandos de consola.

        Contexto:
            Configuracion de rendimiento de una instancia.

        Args:
            dilation (float, opcional): Dilatacion global de tiempo (0.1..20).
            console (list[str], opcional): Comandos de la lista de admitidos del mod.
            anim_tick_option (int | bool, opcional): VisibilityBasedAnimTickOption que se fuerza
                en las mallas del jugador y del jefe (0 refresca huesos sin renderizar);
                False vuelve al valor del juego.

        Returns:
            dict[str, Any]: Respuesta con lo aplicado.

        Raises:
            BridgeError: Si un valor o comando no es admitido.

        Ejemplo:
            client.config(console=["t.MaxFPS 0"])
        """
        msg: dict[str, Any] = {"op": "config"}
        if dilation is not None:
            msg["dilation"] = dilation
        if console is not None:
            msg["console"] = console
        if anim_tick_option is not None:
            msg["anim_tick_option"] = anim_tick_option
        return self.request(msg)

    def exec_lua(self, code: str) -> Any:
        """
        Ejecuta Lua en el hilo del juego (solo instancias con DMB_DEBUG=1).

        Contexto:
            Mediciones de los scripts de viabilidad.

        Args:
            code (str): Codigo Lua; su valor de retorno vuelve en "result".

        Returns:
            Any: Resultado devuelto por el codigo.

        Raises:
            BridgeError: Si exec no esta habilitado o el codigo falla.

        Ejemplo:
            client.exec_lua("return _VERSION")
        """
        return self.request({"op": "exec", "lua": code}).get("result")

    def close(self, quit_game: bool = True) -> None:
        """
        Termina la sesion; por defecto pide al juego que se cierre y espera su salida.

        Contexto:
            Fin del entrenamiento o de un script.

        Args:
            quit_game (bool): Si es True envia close; si el juego no sale en 20 s se termina.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores durante el cierre se ignoran.

        Ejemplo:
            client.close()
        """
        if quit_game:
            with contextlib.suppress(BridgeError, ValueError):
                self.request({"op": "close"}, timeout=10.0)
        self._stop.set()
        self.pipe.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10.0)
