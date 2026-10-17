"""
Acceso a una instancia del juego a traves de dmbridge.

El resto del paquete solo conoce PuertoJuego (lo que necesita del juego); aqui se construye la
implementacion real con BridgeClient y se decide como se lanza cada instancia.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from dmbridge import Action, BridgeClient
from dmbridge.launcher import LaunchConfig

# CPUs 0-3 quedan libres para el usuario; las instancias y los procesos de entrenamiento usan 4-27.
AFINIDAD_ENTRENAMIENTO: tuple[int, ...] = tuple(range(4, 28))


class PuertoJuego(Protocol):
    """Lo que el entorno necesita de una instancia del juego."""

    def reset(self, timeout: float = ..., boss_hp_mult: float | None = ...,
              start_phase: int | None = ..., boss: dict[str, Any] | None = ...,
              reload: bool = ...) -> dict[str, Any]:
        """Reinicia la pelea y devuelve la observacion inicial."""
        ...

    def step(self, action: Action, n_frames: int = ...) -> dict[str, Any]:
        """Aplica la accion, avanza n_frames y devuelve la observacion."""
        ...

    def close(self, quit_game: bool = ...) -> None:
        """Cierra la instancia."""
        ...


FabricaPuerto = Callable[[], PuertoJuego]


@dataclass(frozen=True)
class ParametrosInstancia:
    """
    Como se lanza una instancia del juego.

    Contexto:
        crear_fabrica_puerto. Los valores por defecto son los del perfil de entrenamiento
        medido (sin ventana ni GPU, 30 Hz de simulacion, prioridad Below Normal, CPUs 4-27).

    Atributos:
        id_instancia: nombre de los pipes y del registro de la instancia.
        perfil: perfil de dmbridge.launcher.PROFILE_ARGS.
        sim_hz: frecuencia de simulacion.
        afinidad: CPUs permitidas.
        ventana_visible: ventana normal a velocidad de reloj (modo ver pelea).
        consola: comandos de consola admitidos aplicados tras conectar.
        tiempo_conexion: segundos maximos hasta que el juego abre los pipes.
        plazo_respuesta: segundos maximos de espera de cada respuesta del juego.
    """

    id_instancia: str = "0"
    perfil: str = "training"
    sim_hz: float = 30.0
    afinidad: tuple[int, ...] = AFINIDAD_ENTRENAMIENTO
    ventana_visible: bool = False
    consola: tuple[str, ...] = field(default_factory=tuple)
    tiempo_conexion: float = 180.0
    plazo_respuesta: float = 30.0


def crear_fabrica_puerto(params: ParametrosInstancia) -> FabricaPuerto:
    """
    Devuelve una funcion que lanza una instancia nueva cada vez que se llama.

    Contexto:
        DarkMexicoEnv la invoca al crearse y para relanzar el juego si se cae.

    Args:
        params (ParametrosInstancia): Parametros de lanzamiento.

    Returns:
        FabricaPuerto: Funcion sin argumentos que lanza el juego y devuelve su cliente.

    Raises:
        Ninguna: Los errores de lanzamiento salen al llamar a la funcion devuelta.

    Ejemplo:
        puerto = crear_fabrica_puerto(ParametrosInstancia("0"))()
    """

    def fabrica() -> PuertoJuego:
        """
        Lanza el juego y espera a que el mod se conecte.

        Contexto:
            Cuerpo de la fabrica.

        Args:
            Ninguno.

        Returns:
            PuertoJuego: Cliente conectado.

        Raises:
            TimeoutError: Si el juego no se conecta a tiempo.
            BridgeError: Si el saludo falla.

        Ejemplo:
            (uso interno)
        """
        cliente = BridgeClient(params.id_instancia, reply_timeout=params.plazo_respuesta)
        config = LaunchConfig(
            instance_id=params.id_instancia, profile=params.perfil, sim_hz=params.sim_hz,
            affinity=list(params.afinidad) or None, show_window=params.ventana_visible,
        )
        try:
            cliente.launch(config, connect_timeout=params.tiempo_conexion)
            if params.consola:
                cliente.config(console=list(params.consola))
        except BaseException:
            cerrar_puerto(cliente)
            raise
        return cliente

    return fabrica


def cerrar_puerto(puerto: PuertoJuego | None) -> None:
    """
    Cierra una instancia sin propagar errores (puede estar ya muerta).

    Contexto:
        Apagado del entorno y relanzamiento tras una caida.

    Args:
        puerto (PuertoJuego | None): Instancia a cerrar o None.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna: Cualquier error del cierre se ignora.

    Ejemplo:
        cerrar_puerto(cliente)
    """
    if puerto is None:
        return
    with contextlib.suppress(Exception):
        puerto.close()
