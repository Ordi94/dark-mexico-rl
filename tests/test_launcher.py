"""Pruebas del lanzador: linea de comandos, entorno, dilatacion y lanzamiento con Job Object."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from collections.abc import Iterator
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from dmbridge import launcher
from dmbridge.launcher import PROFILE_ARGS, LaunchConfig, launch_instance

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.mark.parametrize("profile", sorted(PROFILE_ARGS))
def test_command_line_incluye_los_argumentos_del_perfil(profile: str) -> None:
    """La linea de comandos empieza con el ejecutable y los argumentos del perfil elegido."""
    args = LaunchConfig(profile=profile).command_line()
    assert args[0] == str(launcher.GAME_EXE)
    assert args[1 : 1 + len(PROFILE_ARGS[profile])] == PROFILE_ARGS[profile]


def test_command_line_perfil_inexistente_falla() -> None:
    """Un perfil desconocido levanta KeyError en lugar de lanzar el juego con argumentos vacios."""
    with pytest.raises(KeyError):
        LaunchConfig(profile="no_existe").command_line()


def test_command_line_sim_hz_con_valor_usa_paso_fijo() -> None:
    """Cualquier frecuencia numerica arranca con -UseFixedTimeStep."""
    assert "-UseFixedTimeStep" in LaunchConfig(sim_hz=30.0).command_line()
    assert "-UseFixedTimeStep" in LaunchConfig(sim_hz=60.0).command_line()


def test_command_line_sim_hz_none_no_usa_paso_fijo() -> None:
    """Con sim_hz None el juego corre en tiempo real (paso variable)."""
    assert "-UseFixedTimeStep" not in LaunchConfig(sim_hz=None).command_line()


def test_command_line_sonido_desactivado_por_defecto_y_activable() -> None:
    """-nosound se agrega salvo que se pida sonido."""
    assert "-nosound" in LaunchConfig().command_line()
    assert "-nosound" not in LaunchConfig(sound=True).command_line()


def test_command_line_extra_args_van_al_final() -> None:
    """Los argumentos extra se anexan despues de los del perfil y del paso fijo."""
    assert LaunchConfig(extra_args=["-foo", "-bar=1"]).command_line()[-2:] == ["-foo", "-bar=1"]


def test_dilation_sin_dilatacion_para_30hz_y_paso_variable() -> None:
    """A 30 Hz o sin paso fijo no hace falta dilatar el tiempo."""
    assert LaunchConfig(sim_hz=30.0).dilation() is None
    assert LaunchConfig(sim_hz=None).dilation() is None


def test_dilation_otras_frecuencias_son_30_sobre_hz() -> None:
    """Para frecuencias distintas de 30 Hz la dilatacion es 30/sim_hz."""
    assert LaunchConfig(sim_hz=60.0).dilation() == 0.5
    assert LaunchConfig(sim_hz=15.0).dilation() == 2.0


@pytest.mark.parametrize("hz", [0.0, -30.0])
def test_dilation_frecuencia_no_positiva_falla(hz: float) -> None:
    """Una frecuencia no positiva no tiene sentido y se rechaza."""
    with pytest.raises(ValueError, match="positivo"):
        LaunchConfig(sim_hz=hz).dilation()


def test_environment_define_variables_del_puente_y_de_steam(tmp_path: Path) -> None:
    """El entorno lleva DMB_* con valores de texto y los identificadores de Steam."""
    env = LaunchConfig(instance_id="7", log_dir=tmp_path, idle_timeout=45.9).environment()
    assert env["DMB_TRAINING"] == "1"
    assert env["DMB_ID"] == "7"
    assert env["DMB_LOG_DIR"] == str(tmp_path)
    assert env["DMB_IDLE_TIMEOUT"] == "45"
    assert env["SteamAppId"] == env["SteamGameId"] == launcher.STEAM_APP_ID


def test_environment_dmb_debug_solo_con_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """DMB_DEBUG se define con debug=True y se elimina aunque el entorno padre lo tenga."""
    monkeypatch.setenv("DMB_DEBUG", "1")
    assert "DMB_DEBUG" not in LaunchConfig(debug=False).environment()
    assert LaunchConfig(debug=True).environment()["DMB_DEBUG"] == "1"


def test_environment_es_una_copia_del_entorno_actual(monkeypatch: pytest.MonkeyPatch) -> None:
    """El entorno hereda las variables actuales sin modificar os.environ."""
    monkeypatch.setenv("VARIABLE_DE_PRUEBA", "valor")
    env = LaunchConfig().environment()
    assert env["VARIABLE_DE_PRUEBA"] == "valor"
    env["DMB_ID"] = "cambiado"
    assert "DMB_ID" not in os.environ


# --- launch_instance ---------------------------------------------------------------------


class PopenRecorder:
    """Sustituto de subprocess.Popen que registra la llamada y lanza un proceso inocuo."""

    def __init__(self) -> None:
        """
        Prepara el registro de llamadas.

        Contexto:
            Fixture launch_env.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            recorder = PopenRecorder()
        """
        self.args: list[str] = []
        self.kwargs: dict[str, Any] = {}
        self.children: list[subprocess.Popen[bytes]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        """
        Registra los argumentos y lanza un proceso que solo duerme.

        Contexto:
            Sustituye a subprocess.Popen dentro del modulo launcher.

        Args:
            args (list[str]): Linea de comandos que se habria lanzado.
            **kwargs (Any): Opciones de Popen que se habrian usado.

        Returns:
            subprocess.Popen[bytes]: Proceso real e inocuo (no es el juego).

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.args = args
        self.kwargs = kwargs
        child = subprocess.Popen(
            SLEEPER, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self.children.append(child)
        return child


@pytest.fixture
def launch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[PopenRecorder]:
    """
    Aisla launch_instance del juego real: GAME_EXE apunta a un archivo existente y Popen se registra.

    Contexto:
        Pruebas de launch_instance. Garantiza que ningun proceso hijo sobreviva a la prueba.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        tmp_path (Path): Carpeta temporal.

    Returns:
        Iterator[PopenRecorder]: Registro de la llamada a Popen.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    fake_exe = tmp_path / "juego" / "falso.exe"
    fake_exe.parent.mkdir()
    fake_exe.write_bytes(b"")
    recorder = PopenRecorder()
    monkeypatch.setattr(launcher, "GAME_EXE", fake_exe)
    monkeypatch.setattr(
        launcher,
        "subprocess",
        SimpleNamespace(Popen=recorder, STARTUPINFO=subprocess.STARTUPINFO, DEVNULL=subprocess.DEVNULL),
    )
    yield recorder
    for child in recorder.children:
        child.kill()
        child.wait(timeout=10)


def in_job(child: subprocess.Popen[bytes]) -> bool:
    """
    Indica si un proceso pertenece al Job Object del lanzador.

    Contexto:
        Verificacion de launch_instance con kill_with_parent.

    Args:
        child (subprocess.Popen[bytes]): Proceso a consultar.

    Returns:
        bool: True si IsProcessInJob lo reporta dentro del job.

    Raises:
        OSError: Si la consulta a Windows falla.

    Ejemplo:
        in_job(proc)
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    result = wintypes.BOOL()
    job = launcher._kill_on_close_job()
    if not kernel32.IsProcessInJob(int(child._handle), job, ctypes.byref(result)):  # type: ignore[attr-defined]
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(result.value)


def test_launch_instance_sin_ejecutable_falla(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Si GAME_EXE no existe se informa con FileNotFoundError antes de crear nada."""
    monkeypatch.setattr(launcher, "GAME_EXE", tmp_path / "no_existe.exe")
    with pytest.raises(FileNotFoundError):
        launch_instance(LaunchConfig(log_dir=tmp_path / "logs"))
    assert not (tmp_path / "logs").exists()


def test_launch_instance_pasa_linea_entorno_y_opciones_a_popen(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Popen recibe la linea de comandos, el entorno, el cwd del ejecutable y la ventana minimizada."""
    config = LaunchConfig(instance_id="9", log_dir=tmp_path / "logs", kill_with_parent=False)
    launch_instance(config)
    assert launch_env.args == config.command_line()
    kwargs = launch_env.kwargs
    assert kwargs["cwd"] == str(launcher.GAME_EXE.parent)
    assert kwargs["env"]["DMB_ID"] == "9"
    assert kwargs["creationflags"] == 0x4000
    assert kwargs["startupinfo"].dwFlags & 0x1
    assert kwargs["startupinfo"].wShowWindow == 7
    assert (tmp_path / "logs").is_dir()


def test_launch_instance_prioridad_normal_si_below_normal_es_false(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Sin below_normal no se pide prioridad reducida."""
    launch_instance(LaunchConfig(log_dir=tmp_path, below_normal=False, kill_with_parent=False))
    assert launch_env.kwargs["creationflags"] == 0


def test_launch_instance_asigna_la_instancia_al_job_object(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Con kill_with_parent la instancia entra al Job Object (muere con Python)."""
    proc = launch_instance(LaunchConfig(log_dir=tmp_path))
    assert in_job(proc)


def test_launch_instance_sin_kill_with_parent_no_entra_al_job(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Con kill_with_parent=False la instancia queda fuera del job."""
    proc = launch_instance(LaunchConfig(log_dir=tmp_path, kill_with_parent=False))
    assert not in_job(proc)


def test_launch_instance_aplica_afinidad(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Si se pide afinidad, la instancia queda restringida a esas CPUs."""
    proc = launch_instance(LaunchConfig(log_dir=tmp_path, kill_with_parent=False, affinity=[0]))
    assert psutil.Process(proc.pid).cpu_affinity() == [0]


class FakeKernel:
    """Sustituto de kernel32 con resultados configurables para las funciones de Job Object."""

    def __init__(self, create: int = 1, set_info: int = 1, assign: int = 1) -> None:
        """
        Configura los resultados de cada llamada.

        Contexto:
            Pruebas de rutas de error del Job Object.

        Args:
            create (int): Handle devuelto por CreateJobObjectW (0 = fallo).
            set_info (int): Resultado de SetInformationJobObject (0 = fallo).
            assign (int): Resultado de AssignProcessToJobObject (0 = fallo).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            FakeKernel(assign=0)
        """
        self._create, self._set_info, self._assign = create, set_info, assign

    def CreateJobObjectW(self, *_args: Any) -> int:  # noqa: N802
        """
        Simula la creacion del job.

        Contexto:
            Llamada desde _kill_on_close_job.

        Args:
            *_args (Any): Ignorados.

        Returns:
            int: Handle configurado.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ctypes.set_last_error(5)
        return self._create

    def SetInformationJobObject(self, *_args: Any) -> int:  # noqa: N802
        """
        Simula la configuracion del job.

        Contexto:
            Llamada desde _kill_on_close_job.

        Args:
            *_args (Any): Ignorados.

        Returns:
            int: Resultado configurado.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ctypes.set_last_error(6)
        return self._set_info

    def AssignProcessToJobObject(self, *_args: Any) -> int:  # noqa: N802
        """
        Simula la asignacion del proceso al job.

        Contexto:
            Llamada desde launch_instance.

        Args:
            *_args (Any): Ignorados.

        Returns:
            int: Resultado configurado.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ctypes.set_last_error(7)
        return self._assign


def test_kill_on_close_job_reutiliza_el_mismo_handle() -> None:
    """El job se crea una sola vez por proceso Python."""
    assert launcher._kill_on_close_job() == launcher._kill_on_close_job()


def test_kill_on_close_job_falla_si_no_se_puede_crear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si CreateJobObjectW falla se propaga un OSError con el codigo de Windows."""
    monkeypatch.setattr(launcher, "_KERNEL32", FakeKernel(create=0))
    monkeypatch.setattr(launcher, "_JOB_HANDLE", None)
    with pytest.raises(OSError) as info:
        launcher._kill_on_close_job()
    assert info.value.winerror == 5


def test_kill_on_close_job_falla_si_no_se_puede_configurar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si SetInformationJobObject falla no se deja un job a medio configurar como valido."""
    monkeypatch.setattr(launcher, "_KERNEL32", FakeKernel(set_info=0))
    monkeypatch.setattr(launcher, "_JOB_HANDLE", None)
    with pytest.raises(OSError) as info:
        launcher._kill_on_close_job()
    assert info.value.winerror == 6
    assert launcher._JOB_HANDLE is None


def test_launch_instance_mata_el_proceso_si_falla_la_asignacion_al_job(
    launch_env: PopenRecorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Una instancia que no pudo entrar al job se termina: no puede quedar huerfana."""
    monkeypatch.setattr(launcher, "_KERNEL32", FakeKernel(assign=0))
    monkeypatch.setattr(launcher, "_JOB_HANDLE", None)
    with pytest.raises(OSError) as info:
        launch_instance(LaunchConfig(log_dir=tmp_path))
    assert info.value.winerror == 7
    assert launch_env.children[0].wait(timeout=10) is not None


def test_launch_instance_muestra_la_ventana_normal_con_show_window(launch_env: PopenRecorder, tmp_path: Path) -> None:
    """Con show_window la ventana se pide normal (SW_SHOWNORMAL) en lugar de minimizada sin activar."""
    launch_instance(LaunchConfig(log_dir=tmp_path, profile="window", show_window=True, kill_with_parent=False))
    assert launch_env.kwargs["startupinfo"].wShowWindow == 1
    assert "-nullrhi" not in launch_env.args
    assert "-windowed" in launch_env.args
