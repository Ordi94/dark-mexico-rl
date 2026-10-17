"""
Contrasta la mascara de acciones con lo que el juego realmente acepta.

Juega episodios mezclando decisiones del bot de reglas (estados realistas) y acciones al azar
entre las 22 posibles, ignorando la mascara. Despues de cada accion mide si tuvo efecto:
    roll       el montage de maniobra empezo o IsRolling se activo
    ataque     un montage de ataque empezo o Is Attacking se activo
    especial   la barra de poder bajo a cero o empezo el montage del especial
    parry      empezo el montage de bloqueo
    bloqueo    empezo el montage de bloqueo o IsBlocking se activo
    caguama    empezo la bebida
Reporta, por accion, cuantas veces la mascara la declaro valida/invalida y en cuantas tuvo efecto.
Una accion invalida con efecto es un falso negativo (el agente se pierde opciones); una valida sin
efecto es un falso positivo (el agente aprende que "no sirve"). Con el estado previo desglosado se
decide que condicion agregar a la mascara.

Uso:
    .venv\\Scripts\\python.exe scripts\\validar_mascara.py --episodios 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmbridge import BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.aplicacion.entorno import a_action  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.bot_reglas import BotReglas  # noqa: E402
from dmrl.infraestructura.datos import cargar_tabla_golpes  # noqa: E402
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO  # noqa: E402

ROLL = "ALS_N_LandRoll_F_Montage"
BLOQUEO = "NinjaBlockIntro_Montage1"
BEBIDA = "MM_Idle1_Montage"
ESPECIAL = "Anim_SwordV2_Combo03_05_RM_Montage"
ATAQUES = ("Anim_SwordV2_Combo01_02", "Anim_Sword_Attack_Slash05", "Anim_SwordV2_Combo02_01", "Anim_SwordV2_Combo03_04")


def nombre_montage(jugador: dict[str, Any]) -> str | None:
    """
    Nombre del montage activo del jugador.

    Contexto:
        Medicion de efectos.

    Args:
        jugador (dict[str, Any]): Bloque "player".

    Returns:
        str | None: Nombre o None.

    Raises:
        Ninguna.

    Ejemplo:
        nombre_montage({"montage": {"name": "X"}})
    """
    return (jugador.get("montage") or {}).get("name")


def hubo_efecto(accion: int, antes: dict[str, Any], despues: dict[str, Any]) -> bool:
    """
    Decide si la accion tuvo efecto observable en el jugador.

    Contexto:
        main, tras cada paso.

    Args:
        accion (int): Accion enviada.
        antes (dict[str, Any]): Bloque "player" previo al paso.
        despues (dict[str, Any]): Bloque "player" posterior al paso.

    Returns:
        bool: True si empezo el montage esperado (o cambio la bandera equivalente).

    Raises:
        Ninguna.

    Ejemplo:
        hubo_efecto(21, antes, despues)
    """
    m_antes, m_despues = nombre_montage(antes), nombre_montage(despues)
    nuevo = m_despues is not None and m_despues != m_antes
    if acc.ACCION_ROLL_PRIMERA <= accion < acc.ACCION_ATAQUE:
        return (bool(despues.get("rolling")) and not antes.get("rolling")) or (nuevo and m_despues == ROLL)
    if accion == acc.ACCION_ATAQUE:
        sube_ataque = bool(despues.get("attacking")) and not antes.get("attacking")
        return (nuevo and bool(m_despues) and m_despues.startswith(ATAQUES)) or sube_ataque or (
            despues.get("combo") != antes.get("combo"))
    if accion == acc.ACCION_ESPECIAL:
        return (nuevo and m_despues == ESPECIAL) or (antes.get("power", 0) >= acc.POWER_LLENO and despues.get("power", 1) < 0.5)
    if accion in (acc.ACCION_PARRY, acc.ACCION_BLOQUEO):
        return nuevo and m_despues == BLOQUEO
    if accion == acc.ACCION_CAGUAMA:
        return nuevo and m_despues == BEBIDA
    return False


def main() -> int:
    """
    Punto de entrada: mide validez declarada contra efecto real.

    Contexto:
        Herramienta de linea de comandos; escribe bench_results/mascara.json.

    Args:
        Ninguno.

    Returns:
        int: 0 si termino.

    Raises:
        Ninguna.

    Ejemplo:
        python scripts/validar_mascara.py --episodios 20
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--episodios", type=int, default=100)
    ap.add_argument("--azar", type=float, default=0.4, help="fraccion de pasos con accion al azar")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tabla = cargar_tabla_golpes()
    bot = BotReglas(tabla)
    cliente = BridgeClient("vm")
    cliente.launch(LaunchConfig(instance_id="vm", affinity=list(AFINIDAD_ENTRENAMIENTO)))
    cuenta: dict[tuple[int, bool, bool], int] = Counter()
    detalle: dict[tuple[int, bool, str], list[int]] = defaultdict(lambda: [0, 0])
    previa = acc.ACCION_NADA
    registros: list[dict[str, Any]] = []
    try:
        for _ in range(args.episodios):
            obs = cliente.reset(start_phase=rng.choice([1, 1, 2]))
            bot.reiniciar()
            previa = acc.ACCION_NADA
            for _paso in range(900):
                mascara = acc.mascara_acciones(obs.get("player"))
                azar = rng.random() < args.azar
                accion = rng.randrange(acc.N_ACCIONES) if azar else bot.decidir(obs, mascara)
                antes = obs.get("player") or {}
                obs = cliente.step(a_action(acc.comando_de_accion(accion, previa)), 2)
                previa = accion
                despues = obs.get("player") or {}
                if accion >= acc.ACCION_ROLL_PRIMERA:
                    efecto = hubo_efecto(accion, antes, despues)
                    m = antes.get("montage") or {}
                    registros.append({
                        "a": accion, "valida": mascara[accion], "efecto": efecto, "montage": m.get("name"),
                        "frac": round(m.get("pos", 0.0) / m["len"], 3) if m.get("len", 0) > 0 else None,
                        "pos": m.get("pos"), "stamina": antes.get("stamina"), "power": antes.get("power"),
                        "can_attack": antes.get("can_attack"), "can_roll": antes.get("can_roll"),
                        "can_heal": antes.get("can_heal"), "hp": antes.get("hp"), "caguamas": antes.get("caguamas"),
                        "blocking": antes.get("blocking"), "rolling": antes.get("rolling"),
                        "attacking": antes.get("attacking"), "combo": antes.get("combo"),
                    })
                    grupo = acc.ACCION_ROLL_PRIMERA if accion < acc.ACCION_ATAQUE else accion
                    cuenta[(grupo, mascara[accion], efecto)] += 1
                    clave = (grupo, mascara[accion], nombre_montage(antes) or "-")
                    detalle[clave][0] += 1
                    detalle[clave][1] += int(efecto)
                if despues.get("dead") or (obs.get("ev") or {}).get("player_death") or (obs.get("ev") or {}).get("victory"):
                    break
    finally:
        cliente.close()

    nombres = {acc.ACCION_ROLL_PRIMERA: "roll", acc.ACCION_ATAQUE: "ataque", acc.ACCION_ESPECIAL: "especial",
               acc.ACCION_PARRY: "parry", acc.ACCION_BLOQUEO: "bloqueo", acc.ACCION_CAGUAMA: "caguama"}
    print(f"{'accion':10} {'mascara':8} {'n':>7} {'con efecto':>11} {'tasa':>6}")
    resumen = []
    for grupo, nombre in nombres.items():
        for valida in (True, False):
            n = cuenta[(grupo, valida, True)] + cuenta[(grupo, valida, False)]
            if n:
                ef = cuenta[(grupo, valida, True)]
                print(f"{nombre:10} {'valida' if valida else 'invalida':8} {n:7d} {ef:11d} {ef / n:6.2f}")
                resumen.append({"accion": nombre, "valida": valida, "n": n, "efecto": ef})
    print("\nDesglose por montage previo (accion, mascara, montage, n, con efecto):")
    filas = []
    for (grupo, valida, montage), (n, ef) in sorted(detalle.items()):
        if n >= 5:
            filas.append({"accion": nombres[grupo], "valida": valida, "montage_previo": montage, "n": n, "efecto": ef})
            print(f"  {nombres[grupo]:9} {'V' if valida else 'I'} {montage[:46]:46} n={n:5d} efecto={ef:5d} ({ef / n:.2f})")
    Path("bench_results").mkdir(exist_ok=True)
    Path("bench_results/mascara_filas.jsonl").write_text(chr(10).join(json.dumps(f) for f in registros), encoding="utf-8")
    Path("bench_results/mascara.json").write_text(json.dumps({"resumen": resumen, "detalle": filas}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
