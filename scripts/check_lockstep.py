"""
Robustez del modo lockstep.

Casos:
    block     bloqueos largos del hilo del juego (Python tarda 5, 15 y 40 s en responder) con
              ventana y sonido activos: el proceso debe seguir vivo, el siguiente paso debe
              avanzar exactamente un frame de juego (sin salto de tiempo) y el combate debe
              continuar. Registra si Windows marco la ventana como colgada.
    orphan    Python muere de forma abrupta (os._exit) con el juego bloqueado esperando orden:
              con Job Object el juego muere con Python; sin Job Object el mod ve el pipe roto,
              deja de bloquear y cierra el juego tras DMB_IDLE_TIMEOUT.
    hang      el juego se cuelga (bucle infinito en el hilo del juego): el vigilante del
              cliente lo termina al vencer el plazo de respuesta.

Uso:
    .venv\\Scripts\\python.exe scripts\\check_lockstep.py block|orphan|hang
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from dmbridge import BridgeClient, BridgeTimeout  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmbridge.reference_bots import ParryBot  # noqa: E402

RESULTS = ROOT / "bench_results"
_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def hung_windows(pid: int) -> list[bool]:
    """
    Estado "no responde" de las ventanas visibles de un proceso.

    Contexto:
        Caso block: comprobar si Windows marca la ventana como colgada durante el bloqueo.

    Args:
        pid (int): Proceso del juego.

    Returns:
        list[bool]: IsHungAppWindow de cada ventana visible del proceso.

    Raises:
        Ninguna.

    Ejemplo:
        hung_windows(proc.pid)
    """
    found: list[bool] = []

    def cb(hwnd: int, _: int) -> bool:
        owner = wintypes.DWORD()
        _USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and _USER32.IsWindowVisible(hwnd):
            found.append(bool(_USER32.IsHungAppWindow(hwnd)))
        return True

    _USER32.EnumWindows(_ENUM_PROC(cb), 0)
    return found


def case_block() -> dict[str, Any]:
    """
    Bloqueos largos del hilo del juego entre dos pasos.

    Contexto:
        main, caso block.

    Args:
        Ninguno.

    Returns:
        dict[str, Any]: Medidas por duracion de bloqueo.

    Raises:
        BridgeError: Si la instancia muere o deja de responder.

    Ejemplo:
        case_block()
    """
    client = BridgeClient("lockblock", reply_timeout=120)
    out: dict[str, Any] = {"blocks": []}
    try:
        client.launch(LaunchConfig(profile="window_small", sim_hz=30.0, sound=True, debug=True))
        assert client.process is not None
        obs = client.reset()
        bot = ParryBot()
        for _ in range(60):
            obs = client.step(bot.act(obs), 1)
        for secs in (5, 15, 40):
            t_before, f_before = obs["t"], obs["frame"]
            time.sleep(secs / 2)
            mid_hung = hung_windows(client.process.pid)
            time.sleep(secs / 2)
            end_hung = hung_windows(client.process.pid)
            obs = client.step(bot.act(obs), 1)
            lat = client.last_latency
            alive = client.process.poll() is None
            for _ in range(150):
                obs = client.step(bot.act(obs), 1)
            out["blocks"].append({
                "seconds": secs, "alive": alive, "hung_mid": mid_hung, "hung_end": end_hung,
                "first_step_ms": round(lat * 1e3, 2), "game_dt_first_step": round(obs["dt"], 5),
                "frames_advanced_150": obs["frame"] - f_before - 1,
                "game_time_advanced": round(obs["t"] - t_before, 4),
                "hung_after": hung_windows(client.process.pid),
            })
            print(out["blocks"][-1])
    finally:
        client.close()
    return out


CHILD = r"""
import os, sys, time
sys.path.insert(0, {root!r})
from dmbridge import BridgeClient, Action
from dmbridge.launcher import LaunchConfig
c = BridgeClient({iid!r})
c.launch(LaunchConfig(profile="nullrhi", sim_hz=30.0, idle_timeout=15, kill_with_parent={job}))
c.reset()
for _ in range(30):
    c.step(Action(my=1.0), 1)
print(c.process.pid, flush=True)
os._exit(3)
"""


def case_orphan(job: bool) -> dict[str, Any]:
    """
    Muerte abrupta de Python con el juego bloqueado esperando la siguiente orden.

    Contexto:
        main, caso orphan, con y sin Job Object.

    Args:
        job (bool): Si la instancia pertenece al Job Object del Python que muere.

    Returns:
        dict[str, Any]: Segundos hasta que el proceso del juego termino.

    Raises:
        RuntimeError: Si el proceso hijo no informa el pid del juego.

    Ejemplo:
        case_orphan(job=False)
    """
    code = CHILD.format(root=str(ROOT), iid=f"orphan{int(job)}", job=job)
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    t_exit = time.monotonic()
    lines = child.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"el hijo no informo pid: {child.stderr[-500:]}")
    pid = int(lines[-1])
    gone_after = None
    while time.monotonic() - t_exit < 90:
        try:
            alive = psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            gone_after = round(time.monotonic() - t_exit, 2)
            break
        time.sleep(0.2)
    if gone_after is None:
        with contextlib.suppress(psutil.NoSuchProcess):
            psutil.Process(pid).kill()
    res = {"job": job, "child_exit_code": child.returncode, "game_gone_after_s": gone_after}
    print(res)
    return res


def case_hang() -> dict[str, Any]:
    """
    El juego deja de responder: el vigilante del cliente lo termina.

    Contexto:
        main, caso hang.

    Args:
        Ninguno.

    Returns:
        dict[str, Any]: Si se levanto BridgeTimeout, en cuanto tiempo y si el proceso murio.

    Raises:
        Ninguna.

    Ejemplo:
        case_hang()
    """
    client = BridgeClient("hang", reply_timeout=5)
    client.launch(LaunchConfig(profile="nullrhi", sim_hz=30.0, debug=True))
    assert client.process is not None
    client.reset()
    t0 = time.monotonic()
    raised = False
    try:
        client.exec_lua("while true do end")
    except BridgeTimeout:
        raised = True
    elapsed = round(time.monotonic() - t0, 2)
    time.sleep(1)
    res = {"timeout_raised": raised, "after_s": elapsed, "process_dead": client.process.poll() is not None}
    client.close(quit_game=False)
    print(res)
    return res


def main() -> None:
    """
    Ejecuta el caso pedido y guarda el resultado en bench_results/lockstep_<caso>.json.

    Contexto:
        Linea de comandos.

    Args:
        Ninguno (lee sys.argv).

    Returns:
        None: No retorna valor.

    Raises:
        SystemExit: Si el caso no existe.

    Ejemplo:
        python scripts/check_lockstep.py orphan
    """
    case = sys.argv[1] if len(sys.argv) > 1 else "block"
    if case == "block":
        res = case_block()
    elif case == "orphan":
        res = {"with_job": case_orphan(True), "without_job": case_orphan(False)}
    elif case == "hang":
        res = case_hang()
    else:
        raise SystemExit(f"caso desconocido: {case}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"lockstep_{case}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("solo Windows")
    main()
