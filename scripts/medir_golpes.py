"""
Mide en el juego en que posicion de cada montage del jefe conectan sus golpes y cuales se pueden parar.

Corre episodios de la fase 1 y de la fase 2 con pasos de 1 frame (la resolucion maxima del
puente) y una mezcla de politicas que sobreviven y que se exponen al jefe. Por cada paso con
golpes recibidos registra el montage del jefe, su posicion antes y despues del paso y si el
golpe se podia parar. La posicion de un golpe es el centro del intervalo (previa, actual] del
frame en que ocurrio; el agrupamiento por cercania promedia ese error.

Salidas:
    data/tabla_golpes.json   por montage: lista de golpes {pos, parry, dano, n}
    data/vocabulario_montages.json   montages vistos del jugador y del jefe con su frecuencia

Uso:
    .venv\\Scripts\\python.exe scripts\\medir_golpes.py --fase1 150 --fase2 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.aplicacion.entorno import a_action  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.bot_reglas import BotReglas  # noqa: E402
from dmrl.dominio.tabla_golpes import TablaGolpes  # noqa: E402
from dmrl.infraestructura.datos import cargar_tabla_golpes  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Dos golpes del mismo montage a menos de esta distancia (s de montage) se consideran el mismo.
CLUSTER_GAP = 0.12
# Un grupo necesita al menos estas observaciones para entrar en la tabla.
MIN_OBSERVATIONS = 2
MAX_EPISODE_FRAMES = 30 * 360


def random_walk_policy(rng: random.Random, obs: dict[str, Any], state: dict[str, Any]) -> Action:
    """
    Politica que se acerca al jefe y prueba acciones al azar (expone al jugador a los golpes).

    Contexto:
        Complementa al bot de parry: este esquiva pocos golpes y asi se observan los no parables.

    Args:
        rng (random.Random): Generador con semilla.
        obs (dict[str, Any]): Observacion del puente.
        state (dict[str, Any]): Estado mutable de la politica (accion vigente y contador).

    Returns:
        Action: Accion del paso.

    Raises:
        Ninguna.

    Ejemplo:
        accion = random_walk_policy(rng, obs, {})
    """
    if state.get("left", 0) <= 0:
        dist = float(obs.get("dist", 0.0))
        state["left"] = rng.randint(3, 12)
        state["action"] = Action(
            btn=rng.choice([None, None, "rb", "rb", "b", "y"]),
            lb=rng.random() < 0.15,
            mx=rng.choice([-1.0, 0.0, 0.0, 1.0]) if dist < 300 else 0.0,
            my=1.0 if dist > 250 else rng.choice([-1.0, 0.0, 1.0]),
        )
        action = state["action"]
    else:
        action = Action(lb=state["action"].lb, mx=state["action"].mx, my=state["action"].my)
    state["left"] -= 1
    return action


def cluster(values: list[float]) -> list[list[float]]:
    """
    Agrupa valores ordenados por cercania.

    Contexto:
        Resumen de las posiciones de golpe de un montage.

    Args:
        values (list[float]): Posiciones en segundos de montage.

    Returns:
        list[list[float]]: Grupos de valores a menos de CLUSTER_GAP del anterior.

    Raises:
        Ninguna.

    Ejemplo:
        cluster([0.9, 0.91, 1.8])
    """
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= CLUSTER_GAP:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def summarize(hits: dict[str, list[tuple[float, bool, float]]]) -> dict[str, list[dict[str, Any]]]:
    """
    Convierte los golpes crudos en la tabla por montage.

    Contexto:
        Salida de main hacia data/tabla_golpes.json.

    Args:
        hits (dict): montage -> lista de (posicion, parable, dano).

    Returns:
        dict: montage -> lista de golpes {pos, parry, dano, n} ordenados por posicion.

    Raises:
        Ninguna.

    Ejemplo:
        summarize({"M": [(1.0, True, 25.0)] * 3})
    """
    table: dict[str, list[dict[str, Any]]] = {}
    for name, items in sorted(hits.items()):
        items = sorted(items)
        rows = []
        for group in cluster([i[0] for i in items]):
            members = [i for i in items if group[0] <= i[0] <= group[-1]]
            if len(members) < MIN_OBSERVATIONS:
                continue
            rows.append({
                "pos": round(statistics.median(group), 3),
                "min": round(group[0], 3),
                "max": round(group[-1], 3),
                "parry": sum(1 for m in members if m[1]) * 2 >= len(members),
                "dano": round(statistics.median(m[2] for m in members), 1),
                "n": len(members),
            })
        if rows:
            table[name] = rows
    return table


def run_episode(client: BridgeClient, rng: random.Random, phase: int, hits: dict[str, list[tuple[float, bool, float]]],
                player_names: Counter[str], boss_names: Counter[str], lengths: dict[str, float],
                tabla: TablaGolpes) -> dict[str, Any]:
    """
    Juega un episodio con pasos de 1 frame y acumula golpes y montages vistos.

    Contexto:
        Bucle principal de main.

    Args:
        client (BridgeClient): Cliente conectado.
        rng (random.Random): Generador con semilla.
        phase (int): Fase de inicio (1 o 2).
        hits (dict): Acumulador de golpes por montage.
        tabla (TablaGolpes): Tabla de golpes con la que decide el bot (si es vacia el bot solo ataca).
        player_names (Counter[str]): Frecuencia de montages del jugador.
        boss_names (Counter[str]): Frecuencia de montages del jefe.
        lengths (dict[str, float]): Duracion de cada montage visto.

    Returns:
        dict[str, Any]: Resumen del episodio (frames, resultado).

    Raises:
        BridgeError: Si el puente falla.

    Ejemplo:
        run_episode(client, rng, 1, {}, Counter(), Counter(), {})
    """
    obs = client.reset(start_phase=phase)
    use_bot = rng.random() < 0.85
    bot = BotReglas(tabla)
    state: dict[str, Any] = {}
    previa = acc.ACCION_NADA
    continuar = Action()
    frames = 0
    prev = obs
    outcome = "tiempo"
    while frames < MAX_EPISODE_FRAMES:
        if use_bot:
            # El bot decide cada 2 frames, como el entorno; en el frame intermedio solo se
            # mantiene el movimiento y el bloqueo (los botones de un disparo no se repiten).
            if frames % 2 == 0:
                previa = bot.decidir(obs, acc.mascara_acciones(obs.get("player")))
                action = a_action(acc.comando_de_accion(previa, previa))
                continuar = Action(lb=action.lb, mx=action.mx, my=action.my)
            else:
                action = continuar
        else:
            action = random_walk_policy(rng, obs, state)
        obs = client.step(action, 1)
        frames += 1
        boss, player = obs.get("boss") or {}, obs.get("player") or {}
        for mont, counter, who in ((boss.get("montage"), boss_names, "boss"), (player.get("montage"), player_names, "player")):
            if mont:
                counter[mont["name"]] += 1
                lengths[f"{who}:{mont['name']}"] = round(mont.get("len", -1.0), 3)
        ev = obs.get("ev") or {}
        taken = ev.get("taken")
        pm, cm = (prev.get("boss") or {}).get("montage"), boss.get("montage")
        if taken and pm and cm and pm["name"] == cm["name"] and cm["pos"] >= pm["pos"]:
            centre = (pm["pos"] + cm["pos"]) / 2.0
            for amount, parriable in taken:
                hits[cm["name"]].append((centre, bool(parriable), float(amount)))
        prev = obs
        if ev.get("victory"):
            outcome = "victoria"
            break
        if player.get("dead") or ev.get("player_death"):
            outcome = "muerte"
            break
    return {"frames": frames, "outcome": outcome}


def main() -> int:
    """
    Punto de entrada: mide golpes y vocabulario de montages.

    Contexto:
        Herramienta de linea de comandos; escribe en data/.

    Args:
        Ninguno.

    Returns:
        int: 0 si termino.

    Raises:
        Ninguna.

    Ejemplo:
        python scripts/medir_golpes.py --fase1 50 --fase2 30
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fase1", type=int, default=150)
    ap.add_argument("--fase2", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hits: dict[str, list[tuple[float, bool, float]]] = defaultdict(list)
    player_names: Counter[str] = Counter()
    boss_names: Counter[str] = Counter()
    lengths: dict[str, float] = {}
    tabla = cargar_tabla_golpes()
    client = BridgeClient("med")
    client.launch(LaunchConfig(instance_id="med", affinity=list(range(4, 28))))
    try:
        plan = [1] * args.fase1 + [2] * args.fase2
        rng.shuffle(plan)
        outcomes: Counter[str] = Counter()
        for i, phase in enumerate(plan):
            res = run_episode(client, rng, phase, hits, player_names, boss_names, lengths, tabla)
            outcomes[f"f{phase}:{res['outcome']}"] += 1
            if (i + 1) % 25 == 0:
                print(i + 1, "episodios", dict(outcomes), "golpes", sum(len(v) for v in hits.values()), flush=True)
    finally:
        client.close()
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "tabla_golpes.json").write_text(json.dumps(summarize(hits), indent=1, ensure_ascii=False), encoding="utf-8")
    vocab = {"player": dict(player_names.most_common()), "boss": dict(boss_names.most_common()), "lengths": lengths}
    (DATA_DIR / "vocabulario_montages.json").write_text(json.dumps(vocab, indent=1, ensure_ascii=False), encoding="utf-8")
    print("episodios", dict(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
