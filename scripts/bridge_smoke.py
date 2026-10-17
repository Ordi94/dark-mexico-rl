"""
Prueba de humo del puente: una instancia, acciones aleatorias, 2 minutos de juego.

Lanza una instancia de entrenamiento, hace reset y ejecuta acciones aleatorias (pasos de
--frames frames) hasta acumular --game-seconds segundos de juego, reiniciando cuando el
jugador muere o gana. Reporta pasos por segundo, latencia de ida y vuelta, aceleracion y
verifica que el estado cambie de forma coherente:
    * cada paso avanza exactamente n frames y n*dt segundos de juego;
    * la vida esta acotada por su maximo y las caguamas nunca suben dentro de un episodio;
    * posiciones finitas y dentro del escenario;
    * a lo largo de la prueba el jugador se desplaza, el jefe cambia de montage y hay dano.
Sale con codigo 1 si alguna comprobacion falla.

Uso:
    .venv\\Scripts\\python.exe scripts\\bridge_smoke.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402

WORLD_LIMIT = 100_000.0


def random_action(rng: random.Random) -> Action:
    """
    Accion aleatoria sobre todo el espacio del protocolo.

    Contexto:
        Bucle principal de la prueba de humo.

    Args:
        rng (random.Random): Generador con semilla.

    Returns:
        Action: Accion aleatoria.

    Raises:
        Ninguna.

    Ejemplo:
        a = random_action(random.Random(1))
    """
    return Action(
        btn=rng.choice([None, None, None, "rb", "b", "y", "x", "rs"]),
        lb=rng.random() < 0.15, lt=rng.random() < 0.1,
        mx=round(rng.uniform(-1, 1), 3), my=round(rng.uniform(-1, 1), 3),
        mf=rng.choice(["boss", "boss", "cam"]),
    )


class Checker:
    """
    Comprobaciones de coherencia entre observaciones consecutivas.

    Contexto:
        main; acumula violaciones y hechos observados durante toda la prueba.
    """

    def __init__(self) -> None:
        """
        Inicializa contadores.

        Contexto:
            Inicio de la prueba.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            chk = Checker()
        """
        self.violations: list[str] = []
        self.player_travel = 0.0
        self.boss_montages: set[str] = set()
        self.player_damage = 0.0
        self.boss_damage = 0.0
        self.events = {"parries": 0, "hits_taken": 0, "hits_dealt": 0, "phase2": 0, "victory": 0}

    def fail(self, msg: str) -> None:
        """
        Registra una violacion (se guardan como maximo 20).

        Contexto:
            Metodos de comprobacion.

        Args:
            msg (str): Descripcion.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            chk.fail("frame no avanzo")
        """
        if len(self.violations) < 20:
            self.violations.append(msg)

    def check_entity(self, ent: dict[str, Any], who: str) -> None:
        """
        Comprueba rangos de una entidad (jugador o jefe).

        Contexto:
            check_step.

        Args:
            ent (dict[str, Any]): Bloque player o boss de la observacion.
            who (str): Nombre para los mensajes.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            chk.check_entity(obs["player"], "jugador")
        """
        if ent["hp"] > ent["hp_max"] + 1e-6:
            self.fail(f"{who}: vida {ent['hp']} mayor que el maximo {ent['hp_max']}")
        for v in ent["pos"] + ent["vel"]:
            if not math.isfinite(v) or abs(v) > WORLD_LIMIT:
                self.fail(f"{who}: posicion/velocidad fuera de rango {ent['pos']} {ent['vel']}")
                break

    def check_step(self, prev: dict[str, Any], obs: dict[str, Any], n: int) -> None:
        """
        Compara dos observaciones consecutivas del mismo episodio.

        Contexto:
            Despues de cada step.

        Args:
            prev (dict[str, Any]): Observacion anterior.
            obs (dict[str, Any]): Observacion nueva.
            n (int): Frames pedidos en el paso.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            chk.check_step(prev, obs, 2)
        """
        if obs["frame"] - prev["frame"] != n:
            self.fail(f"el paso avanzo {obs['frame'] - prev['frame']} frames en lugar de {n}")
        expected = n * obs["dt"]
        if abs((obs["t"] - prev["t"]) - expected) > 1e-3 + 1e-3 * n:
            self.fail(f"tiempo de juego avanzo {obs['t'] - prev['t']:.4f} s, se esperaba {expected:.4f}")
        p0, p1 = prev.get("player"), obs.get("player")
        b0, b1 = prev.get("boss"), obs.get("boss")
        if p1:
            self.check_entity(p1, "jugador")
        if b1:
            self.check_entity(b1, "jefe")
        if p0 and p1:
            if p1["caguamas"] > p0["caguamas"]:
                self.fail(f"caguamas subieron de {p0['caguamas']} a {p1['caguamas']} dentro del episodio")
            self.player_travel += math.dist(p0["pos"][:2], p1["pos"][:2])
            self.player_damage += max(0.0, p0["hp"] - p1["hp"])
        if b0 and b1 and b0["cls"] == b1["cls"]:
            self.boss_damage += max(0.0, b0["hp"] - b1["hp"])
        if b1 and b1.get("montage"):
            self.boss_montages.add(b1["montage"]["name"])
        ev = obs.get("ev", {})
        for k in ("parries", "hits_taken", "hits_dealt"):
            self.events[k] += int(ev.get(k, 0))
        self.events["phase2"] += int(bool(ev.get("phase2")))
        self.events["victory"] += int(bool(ev.get("victory")))

    def check_totals(self) -> None:
        """
        Comprobaciones globales al final de la prueba.

        Contexto:
            Fin de main.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            chk.check_totals()
        """
        if self.player_travel < 100:
            self.fail(f"el jugador casi no se movio ({self.player_travel:.0f} uu)")
        if len(self.boss_montages) < 2:
            self.fail(f"el jefe no cambio de montage ({self.boss_montages})")
        if self.player_damage + self.boss_damage <= 0:
            self.fail("no hubo dano en toda la prueba")


def main() -> None:
    """
    Ejecuta la prueba de humo y escribe el resumen en bench_results/smoke.json.

    Contexto:
        Linea de comandos.

    Args:
        Ninguno (lee sys.argv).

    Returns:
        None: No retorna valor.

    Raises:
        SystemExit: Con codigo 1 si hubo violaciones de coherencia.

    Ejemplo:
        python scripts/bridge_smoke.py --game-seconds 120
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="training")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--game-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    chk = Checker()
    client = BridgeClient("smoke", reply_timeout=30)
    lat: list[float] = []
    steps = episodes = deaths = 0
    game = 0.0
    try:
        w = time.perf_counter()
        hello = client.launch(LaunchConfig(profile=a.profile, sim_hz=a.hz))
        startup = time.perf_counter() - w
        w = time.perf_counter()
        obs = client.reset()
        first_reset = time.perf_counter() - w
        guards = obs.get("guards", {})
        if not guards.get("essential"):
            chk.fail(f"protecciones inactivas: {guards}")
        reset_times: list[float] = []
        w0 = time.perf_counter()
        while game < a.game_seconds:
            prev = obs
            obs = client.step(random_action(rng), a.frames)
            lat.append(client.last_latency)
            steps += 1
            game += obs["t"] - prev["t"]
            chk.check_step(prev, obs, a.frames)
            player = obs.get("player") or {}
            ev = obs.get("ev", {})
            if player.get("dead") or ev.get("player_death") or ev.get("victory"):
                deaths += int(bool(player.get("dead") or ev.get("player_death")))
                episodes += 1
                r0 = time.perf_counter()
                obs = client.reset()
                reset_times.append(time.perf_counter() - r0)
                p = obs.get("player") or {}
                if p.get("hp") != p.get("hp_max") or p.get("caguamas") != 3:
                    chk.fail(f"reset no dejo el estado inicial: hp={p.get('hp')} caguamas={p.get('caguamas')}")
        wall = time.perf_counter() - w0
        final_guards = client.state().get("guards")
    finally:
        client.close()
    chk.check_totals()
    lat.sort()
    summary = {
        "startup_s": round(startup, 2), "first_reset_s": round(first_reset, 2), "hello_map": hello.get("map"),
        "steps": steps, "frames_per_step": a.frames, "game_s": round(game, 1), "wall_s": round(wall, 2),
        "steps_per_s": round(steps / wall, 1), "game_s_per_real_s": round(game / wall, 2),
        "latency_ms": {"p50": round(lat[len(lat) // 2] * 1e3, 3), "p90": round(lat[int(len(lat) * 0.9)] * 1e3, 3),
                       "p99": round(lat[int(len(lat) * 0.99)] * 1e3, 3), "max": round(lat[-1] * 1e3, 3)},
        "episodes_finished": episodes, "deaths": deaths,
        "reset_s_mean": round(sum(reset_times) / len(reset_times), 3) if reset_times else None,
        "player_travel_uu": round(chk.player_travel), "boss_montages_seen": len(chk.boss_montages),
        "damage_to_player": round(chk.player_damage, 1), "damage_to_boss": round(chk.boss_damage, 1),
        "events": chk.events, "guards": final_guards, "violations": chk.violations,
        "ok": not chk.violations,
    }
    out = Path(__file__).resolve().parent.parent / "bench_results" / "smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if chk.violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
