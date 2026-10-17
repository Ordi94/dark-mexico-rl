"""
Utilidades de procesos de Windows para que el entrenamiento no moleste al usuario ni deje
procesos huerfanos: prioridad y afinidad, archivo de PID para el script de paro y cierre de
arboles de procesos.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path

import psutil


def ajustar_proceso_actual(afinidad: Sequence[int] | None, prioridad_baja: bool = True) -> None:
    """
    Pone el proceso actual en prioridad Below Normal y limita las CPUs que puede usar.

    Contexto:
        Arranque del proceso principal de entrenamiento y de cada proceso de entorno: asi
        Python tambien deja libres las CPUs 0-3 al usuario.

    Args:
        afinidad (Sequence[int], opcional): CPUs logicas permitidas; None no la cambia.
        prioridad_baja (bool): Si es True aplica BELOW_NORMAL_PRIORITY_CLASS.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna: Si el sistema rechaza el cambio se deja el valor actual.

    Ejemplo:
        ajustar_proceso_actual(range(4, 28))
    """
    proceso = psutil.Process()
    if prioridad_baja:
        with contextlib.suppress(psutil.Error):
            proceso.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if afinidad:
        with contextlib.suppress(psutil.Error, ValueError):
            proceso.cpu_affinity(list(afinidad))


def escribir_pid(ruta: Path) -> None:
    """
    Guarda el PID del proceso actual para que detener.ps1 lo encuentre.

    Contexto:
        Inicio de un entrenamiento.

    Args:
        ruta (Path): Archivo de destino (se crea la carpeta si falta).

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        escribir_pid(Path("runs/entrenamiento.pid"))
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(str(os.getpid()), encoding="utf-8")


def leer_pid(ruta: Path) -> int | None:
    """
    Lee un PID guardado y comprueba que el proceso siga vivo.

    Contexto:
        Comprobacion de "ya hay un entrenamiento en marcha".

    Args:
        ruta (Path): Archivo de PID.

    Returns:
        int | None: PID vivo, o None si no hay archivo, esta corrupto o el proceso murio.

    Raises:
        Ninguna.

    Ejemplo:
        leer_pid(Path("runs/entrenamiento.pid"))
    """
    try:
        pid = int(ruta.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if psutil.pid_exists(pid) else None


def apagar_arbol(pid: int, espera: float = 10.0) -> int:
    """
    Termina un proceso y todos sus descendientes (juegos incluidos).

    Contexto:
        Paro forzoso cuando el apagado ordenado no termina a tiempo.

    Args:
        pid (int): Proceso raiz.
        espera (float): Segundos que se espera a que mueran tras terminar.

    Returns:
        int: Cantidad de procesos terminados.

    Raises:
        Ninguna: Un proceso que ya no existe se ignora.

    Ejemplo:
        apagar_arbol(1234)
    """
    try:
        raiz = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    procesos = [*raiz.children(recursive=True), raiz]
    for proceso in procesos:
        with contextlib.suppress(psutil.Error):
            proceso.kill()
    psutil.wait_procs(procesos, timeout=espera)
    return len(procesos)


def apagar_descendientes(pid: int, espera: float = 10.0) -> int:
    """
    Termina todos los descendientes de un proceso sin tocar al proceso mismo.

    Contexto:
        Ultima red de seguridad al cerrar el entrenamiento: si algun juego o subproceso de
        entorno no salio con su orden de cierre, no queda vivo.

    Args:
        pid (int): Proceso cuyos hijos se terminan.
        espera (float): Segundos que se espera a que mueran.

    Returns:
        int: Cantidad de procesos terminados.

    Raises:
        Ninguna: Un proceso que ya no existe se ignora.

    Ejemplo:
        apagar_descendientes(os.getpid())
    """
    try:
        hijos = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    for hijo in hijos:
        with contextlib.suppress(psutil.Error):
            hijo.kill()
    psutil.wait_procs(hijos, timeout=espera)
    return len(hijos)
