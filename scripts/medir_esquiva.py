"""
Mide si un ataque a distancia se evita desplazando al jugador (geometria del ataque).

Contexto:
    Para cada evento de ataque a distancia (katana lanzada, katanas del piso, rayo, rafaga de fuego,
    golpe de piso del Chamuco) se teletransporta al jugador un desplazamiento elegido al azar
    (0, 150, 300, 600 o 1200 uu, alejandose del jefe o de lado) inmediatamente despues del evento,
    antes del impacto, y se cuenta si el ataque llega. Sirve para saber si el ataque se evita saliendo
    del area (y de cuanto es) o si solo se evita con la invulnerabilidad de la maniobra. Es una
    medicion con teletransporte: no dice que el jugador pueda correr tanto en ese tiempo.

Uso:
    python scripts/medir_esquiva.py --jefe nahual --segundos 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmbridge import Action  # noqa: E402
from scripts.medir_jefes import LUA_CURAR, lanzar  # noqa: E402

DESPLAZAMIENTOS = (0.0, 150.0, 300.0, 600.0, 1200.0)
EVENTOS_POR_JEFE = {
    "amlo": ("amlo_throw_kat", "amlo_katanas_floor"),
    "chamuco": ("chamuco_lava",),
    "nahual": ("nahual_rayo", "nahual_fireball"),
}
# Ventana tras el evento en la que un impacto se atribuye a el (segundos de juego).
VENTANA = {"amlo_throw_kat": 1.4, "amlo_katanas_floor": 2.6, "chamuco_lava": 1.6, "nahual_rayo": 1.9, "nahual_fireball": 5.0}
LUA_MOVER = """
local G = BRIDGE.G
local p = G.pawn()
if not p then return 0 end
local l = p:K2_GetActorLocation()
p:K2_SetActorLocation({X = l.X + %(dx)f, Y = l.Y + %(dy)f, Z = l.Z + 5}, false, {}, true)
return 1
"""
LUA_ESCENA = """
local G = BRIDGE.G
local p, b = G.pawn(), G.boss_actor()
if not (p and b) then return 0 end
local lb = b:K2_GetActorLocation()
local ang = math.rad(%(angulo)f)
p:K2_SetActorLocation({X = lb.X + math.cos(ang) * %(distancia)f, Y = lb.Y + math.sin(ang) * %(distancia)f, Z = lb.Z - 60}, false, {}, true)
b.BPC_DamageSystem.Health = b.BPC_DamageSystem.MaxHealth * %(fraccion)f
return 1
"""


def medir(jefe: str, segundos: float, semilla: int) -> list[dict[str, Any]]:
    """
    Corre la medicion y devuelve una fila por evento.

    Contexto:
        main.

    Args:
        jefe (str): amlo, chamuco o nahual.
        segundos (float): Tiempo de juego.
        semilla (int): Semilla del azar.

    Returns:
        list[dict[str, Any]]: Fila por evento con id, desplazamiento, direccion e impactos.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        medir("nahual", 300.0, 0)
    """
    registro = json.loads((RAIZ / "data" / "jefes" / f"{jefe}.json").read_text(encoding="utf-8"))
    mod = dict(registro["episodio"]["mod"])
    mod["trace_hits"] = True
    azar = random.Random(semilla)
    cliente = lanzar(f"esq_{jefe}")
    filas: list[dict[str, Any]] = []
    pendientes: list[dict[str, Any]] = []
    try:
        obs = cliente.reset(boss=mod)
        t0 = float(obs["t"])
        proximo = t0
        distancias = (400.0, 900.0, 1600.0, 2600.0)
        i = 0
        while float(obs["t"]) - t0 < segundos:
            ahora = float(obs["t"])
            if ahora >= proximo:
                cliente.exec_lua(LUA_ESCENA % {"angulo": azar.uniform(0, 360), "distancia": distancias[i % 4],
                                               "fraccion": 0.5 if jefe == "amlo" else 1.0})
                i += 1
                proximo = ahora + 15.0
            obs = cliente.step(Action(), 2)
            ev = obs.get("ev") or {}
            for golpe in ev.get("hits") or []:
                for fila in pendientes:
                    if 0.0 <= float(golpe["t"]) - fila["t"] <= VENTANA[fila["id"]] and float(golpe["amount"]) < 1e6:
                        fila["impactos"] += 1
            for evento in ev.get("remote") or []:
                if evento["id"] not in EVENTOS_POR_JEFE[jefe]:
                    continue
                despl = azar.choice(DESPLAZAMIENTOS)
                lateral = azar.random() < 0.5
                ux, uy = float(evento["px"]) - float(evento["bx"]), float(evento["py"]) - float(evento["by"])
                n = math.hypot(ux, uy) or 1.0
                ux, uy = ux / n, uy / n
                if lateral:
                    ux, uy = -uy, ux
                if despl > 0:
                    cliente.exec_lua(LUA_MOVER % {"dx": ux * despl, "dy": uy * despl})
                fila = {"id": evento["id"], "t": float(evento["t"]), "desplazamiento": despl,
                        "direccion": "lateral" if lateral else "alejarse", "impactos": 0,
                        "distancia": round(n)}
                pendientes.append(fila)
                filas.append(fila)
            pendientes = [f for f in pendientes if float(obs["t"]) - f["t"] <= VENTANA[f["id"]] + 0.5]
            jugador = obs.get("player") or {}
            if float(jugador.get("hp", 100.0)) < 75.0:
                cliente.exec_lua(LUA_CURAR)
            if jugador.get("dead") or ev.get("player_death"):
                obs = cliente.reset(boss=mod)
    finally:
        cliente.close()
    return filas


def tabular(filas: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """
    Fraccion de eventos con impacto por evento, direccion y desplazamiento.

    Contexto:
        Informe de main.

    Args:
        filas (list[dict[str, Any]]): Salida de medir.

    Returns:
        dict[str, dict[str, dict[str, float]]]: id -> "direccion/desplazamiento" -> n y tasa de impacto.

    Raises:
        Ninguna.

    Ejemplo:
        tabular([{"id": "e", "desplazamiento": 0.0, "direccion": "lateral", "impactos": 1}])
    """
    tabla: dict[str, dict[str, list[int]]] = {}
    for f in filas:
        clave = f"{f['direccion']}/{int(f['desplazamiento'])}"
        tabla.setdefault(f["id"], {}).setdefault(clave, []).append(1 if f["impactos"] > 0 else 0)
    return {e: {c: {"n": float(len(v)), "tasa_impacto": sum(v) / len(v)} for c, v in sorted(d.items())}
            for e, d in tabla.items()}


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/medir_esquiva.py.

    Args:
        Ninguno.

    Returns:
        int: 0 al terminar.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        python scripts/medir_esquiva.py --jefe chamuco
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefe", required=True, choices=tuple(EVENTOS_POR_JEFE))
    parser.add_argument("--segundos", type=float, default=600.0)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    filas = medir(args.jefe, args.segundos, args.semilla)
    tabla = tabular(filas)
    salida = RAIZ / "bench_results" / f"esquiva_{args.jefe}.json"
    salida.write_text(json.dumps({"filas": filas, "tabla": tabla}, indent=1), encoding="utf-8")
    for evento, celdas in tabla.items():
        print(evento)
        for clave, v in celdas.items():
            print(f"  {clave:>16}  n={int(v['n']):3d}  impacto={v['tasa_impacto']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
