"""
Arranque de instancias del juego en modo entrenamiento.

Cada instancia es el ejecutable Shipping lanzado directamente (sin el lanzador de Steam) con:
    * variables de entorno DMB_TRAINING=1, DMB_ID, DMB_LOG_DIR que activan DarkMexicoBridge;
    * SteamAppId/SteamGameId para que la API de Steam reconozca el juego sin pasar por Steam;
    * prioridad por debajo de lo normal y afinidad opcional, para que la PC siga usable;
    * un Job Object con KILL_ON_JOB_CLOSE: si el proceso Python muere (incluso de forma
      abrupta), Windows cierra el job y mata las instancias; ningun juego queda colgado
      esperando una orden que ya no llegara.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

from dmbridge.rutas import buscar_carpeta_juego

GAME_DIR = buscar_carpeta_juego()
GAME_EXE = GAME_DIR / "MexicoSouls" / "Binaries" / "Win64" / "MexicoSouls-Win64-Shipping.exe"
STEAM_APP_ID = "4346760"
# Paso de -UseFixedTimeStep en este ejecutable (FApp::FixedDeltaTime por defecto).
FIXED_STEP_HZ = 30.0
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_DIR / "bridge_logs"

# Perfiles de arranque medidos (ver docs/03-como-funciona.md).
PROFILE_ARGS: dict[str, list[str]] = {
    # Perfil recomendado: sin renderizado y con menos hilos de trabajo del motor (misma
    # velocidad por instancia, ~18% menos CPU).
    "training": ["-nullrhi", "-nosplash", "-unattended", "-corelimit=2"],
    "window_small": ["-windowed", "-ResX=320", "-ResY=180", "-nosplash", "-unattended"],
    "nullrhi": ["-nullrhi", "-nosplash", "-unattended"],
    # Ventana normal para ver pelear al agente (sin -nullrhi). Con -UseFixedTimeStep cada frame dura
    # 1/30 s de juego y el motor dibuja al ritmo de la pantalla (~56 fps, ~1.9x): el cliente que
    # quiera tiempo real debe esperar entre decisiones (dmrl.aplicacion.visualizacion).
    "window": ["-windowed", "-ResX=1280", "-ResY=720", "-nosplash", "-unattended"],
    "offscreen": ["-RenderOffscreen", "-ResX=320", "-ResY=180", "-nosplash", "-unattended"],
}

_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_STARTF_USESHOWWINDOW = 0x00000001
_SW_SHOWMINNOACTIVE = 7
_SW_SHOWNORMAL = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
_KERNEL32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
_KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
_KERNEL32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
_KERNEL32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

_JOB_HANDLE: int | None = None


def _kill_on_close_job() -> int:
    """
    Devuelve el Job Object del proceso Python (lo crea la primera vez).

    Contexto:
        launch_instance. Un solo job por proceso Python agrupa todas sus instancias; su handle
        vive hasta que Python termina, y entonces Windows mata a todos sus miembros.

    Args:
        Ninguno.

    Returns:
        int: Handle del job.

    Raises:
        OSError: Si Windows no puede crear o configurar el job.

    Ejemplo:
        job = _kill_on_close_job()
    """
    global _JOB_HANDLE
    if _JOB_HANDLE is None:
        job = _KERNEL32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _KERNEL32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                                 ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        _JOB_HANDLE = int(job)
    return _JOB_HANDLE


@dataclass
class LaunchConfig:
    """
    Parametros de arranque de una instancia.

    Contexto:
        launch_instance y los scripts de medicion.

    Atributos:
        instance_id: nombre de los pipes y del registro de la instancia.
        profile: clave de PROFILE_ARGS (modo de ventana/renderizado).
        sim_hz: frecuencia de simulacion. Con un valor se arranca con -UseFixedTimeStep, que en
            este ejecutable fija el paso en 1/30 s de juego por frame (-FPS=N se ignora); otras
            frecuencias se obtienen con dilatacion de tiempo 30/sim_hz que el cliente aplica al
            conectar y el mod reaplica en cada reinicio. None = paso variable (tiempo real).
        extra_args: argumentos adicionales de linea de comandos.
        debug: habilita la orden exec del mod.
        idle_timeout: segundos sin cliente antes de que el mod cierre el juego.
        affinity: lista de CPUs logicas permitidas o None.
        below_normal: prioridad por debajo de lo normal.
        log_dir: carpeta de bridge_<id>.log.
        sound: si es False se agrega -nosound (el entrenamiento no necesita audio).
        show_window: si es True la ventana del juego se muestra normal (no minimizada ni sin
            activar); solo para el perfil "window" con el que el usuario ve pelear al agente.
        kill_with_parent: si es True la instancia entra al Job Object de este Python y muere
            con el; False solo para probar la salida por inactividad del propio mod.
    """

    instance_id: str = "0"
    profile: str = "training"
    sim_hz: float | None = 30.0
    extra_args: list[str] = field(default_factory=list)
    debug: bool = False
    idle_timeout: float = 120.0
    affinity: list[int] | None = None
    below_normal: bool = True
    log_dir: Path = DEFAULT_LOG_DIR
    sound: bool = False
    show_window: bool = False
    kill_with_parent: bool = True

    def command_line(self) -> list[str]:
        """
        Construye la linea de comandos completa.

        Contexto:
            launch_instance y documentacion del comando exacto.

        Args:
            Ninguno.

        Returns:
            list[str]: Ejecutable y argumentos.

        Raises:
            KeyError: Si el perfil no existe.

        Ejemplo:
            LaunchConfig(sim_hz=30.0).command_line()
        """
        args = [str(GAME_EXE), *PROFILE_ARGS[self.profile]]
        if not self.sound:
            args.append("-nosound")
        if self.sim_hz is not None:
            args.append("-UseFixedTimeStep")
        return args + list(self.extra_args)

    def dilation(self) -> float | None:
        """
        Dilatacion de tiempo necesaria para la frecuencia pedida sobre el paso fijo de 1/30 s.

        Contexto:
            BridgeClient.launch, tras el saludo.

        Args:
            Ninguno.

        Returns:
            float | None: 30/sim_hz, o None si no hace falta (30 Hz o paso variable).

        Raises:
            ValueError: Si sim_hz no es positivo.

        Ejemplo:
            LaunchConfig(sim_hz=60.0).dilation()  # 0.5
        """
        if self.sim_hz is None or self.sim_hz == FIXED_STEP_HZ:
            return None
        if self.sim_hz <= 0:
            raise ValueError("sim_hz debe ser positivo")
        return FIXED_STEP_HZ / self.sim_hz

    def environment(self) -> dict[str, str]:
        """
        Construye el entorno del proceso del juego.

        Contexto:
            launch_instance.

        Args:
            Ninguno.

        Returns:
            dict[str, str]: Copia del entorno actual con las variables del puente y de Steam.

        Raises:
            Ninguna.

        Ejemplo:
            env = LaunchConfig().environment()
        """
        env = dict(os.environ)
        env.update({
            "DMB_TRAINING": "1",
            "DMB_ID": self.instance_id,
            "DMB_LOG_DIR": str(self.log_dir),
            "DMB_IDLE_TIMEOUT": str(int(self.idle_timeout)),
            "SteamAppId": STEAM_APP_ID,
            "SteamGameId": STEAM_APP_ID,
        })
        if self.debug:
            env["DMB_DEBUG"] = "1"
        else:
            env.pop("DMB_DEBUG", None)
        return env


def launch_instance(config: LaunchConfig) -> subprocess.Popen[bytes]:
    """
    Lanza una instancia del juego en modo entrenamiento.

    Contexto:
        BridgeClient.launch y scripts de medicion. La ventana (si el perfil la tiene) se pide
        minimizada y sin activar para no robar el foco al usuario.

    Args:
        config (LaunchConfig): Parametros de arranque.

    Returns:
        subprocess.Popen[bytes]: Proceso del juego (miembro del job de este Python).

    Raises:
        FileNotFoundError: Si no existe el ejecutable del juego.
        OSError: Si no se puede crear el proceso o asignarlo al job.

    Ejemplo:
        proc = launch_instance(LaunchConfig(instance_id="0"))
    """
    if not GAME_EXE.exists():
        raise FileNotFoundError(
            f"No se encontro el juego en {GAME_EXE}. Define DARKMEXICO_DIR con la carpeta de instalacion."
        )
    config.log_dir.mkdir(parents=True, exist_ok=True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= _STARTF_USESHOWWINDOW
    startup.wShowWindow = _SW_SHOWNORMAL if config.show_window else _SW_SHOWMINNOACTIVE
    flags = _BELOW_NORMAL_PRIORITY_CLASS if config.below_normal else 0
    proc = subprocess.Popen(
        config.command_line(), cwd=str(GAME_EXE.parent), env=config.environment(),
        startupinfo=startup, creationflags=flags,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if config.kill_with_parent:
        job = _kill_on_close_job()
        if not _KERNEL32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            err = ctypes.get_last_error()
            proc.kill()
            raise ctypes.WinError(err)
    if config.affinity:
        import psutil

        psutil.Process(proc.pid).cpu_affinity(config.affinity)
    return proc
