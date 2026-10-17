"""
Verificacion de las protecciones del modo entrenamiento en una instancia real.

Fuerza una victoria rapida (vida del jefe reducida por exec, solo en esta prueba) y comprueba:
    * SaveGameToSlot se llama (victoria y traje desbloqueado) y el mod anula el guardado:
      no aparece MexicoSouls\\Saved\\SaveGames\\savegame.sav ni cambia si ya existia;
    * la proteccion de logros esta activa (los intentos que ocurran se registran con progreso
      0; el juego no siempre intenta escribir un logro en una victoria forzada);
    * un menu de pausa abierto con el evento Start del mando se cierra solo.

Uso:
    .venv\\Scripts\\python.exe scripts\\check_guards.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmbridge.reference_bots import ParryBot  # noqa: E402

SAVE_DIR = Path(os.environ["LOCALAPPDATA"]) / "MexicoSouls" / "Saved" / "SaveGames"
BOSS_HP = "local b = BRIDGE.G.boss_actor(); b.BPC_DamageSystem.Health = {hp}; return BRIDGE.G.class_name(b)"
PLAYER_FULL = "local p = BRIDGE.G.pawn(); p.BPC_DamageSystem.Health = 100.0; return true"
PRESS_START = ("local p = BRIDGE.G.pawn(); "
               "p['InpActEvt_Gamepad_Special_Right_K2Node_InputKeyEvent_24']({KeyName = FName('Gamepad_Special_Right')}); "
               "local n = 0; for _, w in ipairs(FindAllOf('WP_PauseGame_C') or {}) do if w:IsInViewport() then n = n + 1 end end; return n")
COUNT_PAUSE = ("local n = 0; for _, w in ipairs(FindAllOf('WP_PauseGame_C') or {}) do "
               "if w:IsValid() and w:IsInViewport() then n = n + 1 end end; return n")


def save_snapshot() -> dict[str, str]:
    """
    Huella (sha256) de cada archivo de SaveGames.

    Contexto:
        Antes y despues de la victoria forzada.

    Args:
        Ninguno.

    Returns:
        dict[str, str]: Nombre de archivo -> hash; vacio si la carpeta no existe.

    Raises:
        OSError: Si un archivo existe pero no se puede leer.

    Ejemplo:
        antes = save_snapshot()
    """
    if not SAVE_DIR.exists():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SAVE_DIR.iterdir() if p.is_file()}


def run_until(client: BridgeClient, bot: ParryBot, obs: dict[str, Any], key: str, limit: float) -> tuple[dict[str, Any], bool]:
    """
    Pelea con el bot hasta que aparezca un evento o se agote el tiempo de juego.

    Contexto:
        Fases de la victoria forzada.

    Args:
        client (BridgeClient): Sesion con la instancia.
        bot (ParryBot): Politica de referencia.
        obs (dict[str, Any]): Observacion actual.
        key (str): Evento esperado en obs["ev"] (phase2, victory).
        limit (float): Segundos de juego maximos.

    Returns:
        tuple[dict[str, Any], bool]: Ultima observacion y si el evento ocurrio.

    Raises:
        BridgeError: Si la instancia no responde.

    Ejemplo:
        obs, ok = run_until(client, bot, obs, "victory", 60.0)
    """
    t0 = obs["t"]
    while obs["t"] - t0 < limit:
        obs = client.step(bot.act(obs), 1)
        if obs.get("ev", {}).get(key):
            return obs, True
        if 0 < (obs.get("player") or {}).get("hp", 100) < 60:
            client.exec_lua(PLAYER_FULL)
    return obs, False


def main() -> None:
    """
    Ejecuta la verificacion y escribe el resultado en bench_results/guards.json.

    Contexto:
        Linea de comandos.

    Args:
        Ninguno.

    Returns:
        None: No retorna valor.

    Raises:
        SystemExit: Con codigo 1 si alguna proteccion falla.

    Ejemplo:
        python scripts/check_guards.py
    """
    before = save_snapshot()
    client = BridgeClient("guards", reply_timeout=60)
    result: dict[str, Any] = {"save_files_before": before}
    try:
        client.launch(LaunchConfig(profile="nullrhi", sim_hz=30.0, debug=True))
        obs = client.reset()
        bot = ParryBot()
        result["pause_open_after_start"] = client.exec_lua(PRESS_START)
        for _ in range(10):
            obs = client.step(Action(), 1)
        result["pause_open_after_10_frames"] = client.exec_lua(COUNT_PAUSE)
        result["boss_phase1"] = client.exec_lua(BOSS_HP.format(hp=290.0))
        obs, result["phase2_event"] = run_until(client, bot, obs, "phase2", 90.0)
        for _ in range(30 * 8):
            obs = client.step(bot.act(obs), 1)
        result["boss_phase2"] = client.exec_lua(BOSS_HP.format(hp=5.0))
        obs, result["victory_event"] = run_until(client, bot, obs, "victory", 90.0)
        # La segunda escritura (traje desbloqueado) llega ~10 s despues de la victoria.
        for _ in range(30 * 25):
            obs = client.step(Action(), 1)
        result["guards"] = client.state()["guards"]
    finally:
        client.close()
    after = save_snapshot()
    result["save_files_after"] = after
    result["save_untouched"] = before == after
    g = result.get("guards") or {}
    ok = (result["save_untouched"] and g.get("saves_blocked", 0) >= 1 and g.get("saves_unblocked", 1) == 0
          and g.get("essential") is True and str(result.get("pause_open_after_10_frames")) == "0")
    result["ok"] = ok
    out = Path(__file__).resolve().parent.parent / "bench_results" / "guards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
