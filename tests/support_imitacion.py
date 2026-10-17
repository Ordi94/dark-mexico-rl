"""
Utilidades de prueba de la imitacion: demostraciones sinteticas con una regla aprendible y un
experto de mentira.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dmrl.dominio import acciones as acc
from dmrl.dominio.imitacion import Demostraciones, identificador_episodio
from dmrl.dominio.observacion import TAMANO_SIMPLE

DIMENSION = TAMANO_SIMPLE * 4


def regla_sintetica(obs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Accion "correcta" de los datos sinteticos: depende solo de dos coordenadas de la observacion.

    Contexto:
        Permite comprobar que la red aprende una regla y la generaliza a episodios nuevos.

    Args:
        obs (np.ndarray): Observaciones (n, DIMENSION).

    Returns:
        np.ndarray: Acciones 1 a 4 (n,).

    Raises:
        Ninguna.

    Ejemplo:
        regla_sintetica(obs)
    """
    return (1 + (obs[:, 0] > 0).astype(np.int64) + 2 * (obs[:, 1] > 0).astype(np.int64)).astype(np.int16)


def demostraciones_sinteticas(episodios: int = 12, pasos: int = 40, semilla: int = 0,
                              ronda: int = 0) -> Demostraciones:
    """
    Demostraciones sinteticas coherentes (la etiqueta siempre es valida segun su mascara).

    Contexto:
        Pruebas del ajuste, la particion y el almacenamiento.

    Args:
        episodios (int): Cantidad de episodios.
        pasos (int): Pasos por episodio.
        semilla (int): Semilla del generador.
        ronda (int): Ronda a la que pertenecen los identificadores.

    Returns:
        Demostraciones: Conjunto con filas contiguas por episodio.

    Raises:
        Ninguna.

    Ejemplo:
        demostraciones_sinteticas(episodios=4, pasos=10)
    """
    azar = np.random.default_rng(semilla)
    n = episodios * pasos
    obs = azar.uniform(-1.0, 1.0, (n, DIMENSION)).astype(np.float32)
    etiquetas = regla_sintetica(obs)
    mascaras = azar.random((n, acc.N_ACCIONES)) > 0.3
    mascaras[np.arange(n), etiquetas] = True
    mascaras[:, acc.ACCION_NADA] = True
    ids = np.repeat([identificador_episodio(ronda, i) for i in range(episodios)], pasos).astype(np.int64)
    return Demostraciones(obs, mascaras, etiquetas, etiquetas.copy(), ids,
                          azar.normal(size=n).astype(np.float32))


class ExpertoFijo:
    """Experto de mentira: ataca siempre que la mascara lo permite."""

    def __init__(self) -> None:
        """
        Crea el experto.

        Contexto:
            Pruebas de la recoleccion.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ExpertoFijo()
        """
        self.reinicios = 0

    def reiniciar(self) -> None:
        """
        Cuenta los reinicios.

        Contexto:
            Cumple la interfaz del bot.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            experto.reiniciar()
        """
        self.reinicios += 1

    def decidir(self, obs: dict[str, Any], mascara: list[bool]) -> int:
        """
        Ataque si es valido; si no, nada.

        Contexto:
            Bucle de recoleccion.

        Args:
            obs (dict[str, Any]): Observacion cruda (no usada).
            mascara (list[bool]): Acciones validas.

        Returns:
            int: Accion elegida.

        Raises:
            Ninguna.

        Ejemplo:
            experto.decidir({}, mascara)
        """
        return acc.ACCION_ATAQUE if mascara[acc.ACCION_ATAQUE] else acc.ACCION_NADA


class EjecutorFijo:
    """Politica ejecutora de mentira que siempre elige la misma accion."""

    def __init__(self, accion: int) -> None:
        """
        Crea el ejecutor.

        Contexto:
            Pruebas de la recoleccion con DAgger.

        Args:
            accion (int): Accion que siempre propone.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EjecutorFijo(1)
        """
        self.accion = accion
        self.reinicios = 0

    def reiniciar(self) -> None:
        """
        Cuenta los reinicios.

        Contexto:
            Cumple la interfaz Politica.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ejecutor.reiniciar()
        """
        self.reinicios += 1

    def decidir(self, env: Any, obs: Any) -> int:
        """
        Propone siempre la misma accion.

        Contexto:
            Cumple la interfaz Politica.

        Args:
            env (Any): Entorno (no usado).
            obs (Any): Observacion (no usada).

        Returns:
            int: La accion fija.

        Raises:
            Ninguna.

        Ejemplo:
            ejecutor.decidir(env, obs)
        """
        return self.accion
