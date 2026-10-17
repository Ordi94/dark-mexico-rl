"""
Mide los golpes, los alcances y los retardos de ataques a distancia de un jefe en el juego real.

Contexto:
    Una instancia con exec habilitado reinicia la pelea del jefe con trace_hits y deja al jugador
    quieto (el jefe recorre todo su repertorio). Para que la medicion no termine con la muerte se
    cura al jugador cuando baja de 75 de vida y, cada cierto tiempo de juego, se cambia la
    distancia entre ambos y la fraccion de vida del jefe: varios ataques (teletransporte, katanas,
    rayo) solo aparecen a distancia o con poca vida. El resultado se guarda en
    bench_results/medicion_<jefe>.json y, con --fusionar, se incorpora a data/jefes/<jefe>.json.

Uso:
    python scripts/medir_jefes.py --jefe chamuco --segundos 900
    python scripts/medir_jefes.py --jefe amlo --segundos 1200 --fusionar
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

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.dominio import medicion as med  # noqa: E402

T_INICIO = time.time()
DISTANCIAS = (150.0, 400.0, 900.0, 1600.0, 2600.0)
DISTANCIAS_ESCENARIO = {"juice": (150.0, 400.0, 700.0, 900.0, 1100.0)}
FRACCIONES_DEFECTO = {"amlo": (0.95, 0.75, 0.5, 0.3), "chamuco": (1.0,), "nahual": (1.0, 0.6, 0.3), "juice": (1.0, 0.6, 0.3)}
SEGUNDOS_CAMBIO = 20.0

LUA_CURAR = "local p = BRIDGE.G.pawn() if p then p.BPC_DamageSystem.Health = 100 end return 1"
LUA_ESCENA = """
local G = BRIDGE.G
local p, b = G.pawn(), G.boss_actor()
if not (p and b) then return 0 end
local lb = b:K2_GetActorLocation()
local ang = math.rad(%(angulo)f)
local x, y = lb.X + math.cos(ang) * %(distancia)f, lb.Y + math.sin(ang) * %(distancia)f
p:K2_SetActorLocation({X = x, Y = y, Z = lb.Z - 60}, false, {}, true)
local d = b.BPC_DamageSystem
d.Health = d.MaxHealth * %(fraccion)f
return 1
"""


def lanzar(ident: str) -> BridgeClient:
    """
    Lanza una instancia con exec habilitado.

    Contexto:
        main.

    Args:
        ident (str): Identificador de la instancia.

    Returns:
        BridgeClient: Cliente conectado.

    Raises:
        TimeoutError: Si el juego no se conecta.

    Ejemplo:
        cliente = lanzar("med0")
    """
    cliente = BridgeClient(ident, reply_timeout=120.0)
    cliente.launch(LaunchConfig(instance_id=ident, profile="training", sim_hz=30.0, debug=True,
                                affinity=list(range(4, 28))), connect_timeout=120.0)
    return cliente


def medir(jefe: str, segundos: float, fracciones: tuple[float, ...], semilla: int) -> dict[str, Any]:
    """
    Recoge golpes, eventos y montages del jefe durante `segundos` de juego.

    Contexto:
        main.

    Args:
        jefe (str): Identificador del registro (juice, amlo, chamuco, nahual).
        segundos (float): Tiempo de juego a medir.
        fracciones (tuple[float, ...]): Fracciones de vida del jefe que se alternan.
        semilla (int): Semilla de las posiciones del jugador.

    Returns:
        dict[str, Any]: golpes, eventos, montages y resumen.

    Raises:
        BridgeError: Si el juego rechaza una orden.

    Ejemplo:
        medir("chamuco", 600.0, (1.0,), 0)
    """
    registro = json.loads((RAIZ / "data" / "jefes" / f"{jefe}.json").read_text(encoding="utf-8"))
    mod = dict(registro["episodio"]["mod"])
    mod["trace_hits"] = True
    azar = random.Random(semilla)
    distancias = DISTANCIAS_ESCENARIO.get(jefe, DISTANCIAS)
    cliente = lanzar(f"med_{jefe}")
    golpes: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    montages: dict[str, dict[str, Any]] = {}
    aturdido: set[str] = set()
    try:
        obs = cliente.reset(boss=mod)
        previo = float(obs["t"])
        transcurrido = 0.0
        proximo_cambio = 0.0
        indice = 0
        reinicios = 0
        while transcurrido < segundos:
            # OpenLevel reinicia el reloj del mundo (Juice): el tiempo medido se acumula por pasos.
            actual = float(obs["t"])
            transcurrido += actual - previo if actual >= previo else actual
            previo = actual
            ahora = transcurrido
            if ahora >= proximo_cambio and obs.get("boss"):
                cliente.exec_lua(LUA_ESCENA % {
                    "angulo": azar.uniform(0, 360), "distancia": distancias[indice % len(distancias)],
                    "fraccion": fracciones[(indice // len(distancias)) % len(fracciones)]})
                print(f"escena {indice} t={ahora:.0f} real={time.time() - T_INICIO:.0f}", flush=True)
                indice += 1
                proximo_cambio = ahora + SEGUNDOS_CAMBIO
            obs = cliente.step(Action(), 2)
            ev = obs.get("ev") or {}
            golpes += ev.get("hits") or []
            eventos += ev.get("remote") or []
            jefe_obs = obs.get("boss") or {}
            montage = jefe_obs.get("montage")
            if montage:
                info = montages.setdefault(montage["name"], {"len": montage["len"], "rates": []})
                if montage["rate"] not in info["rates"]:
                    info["rates"].append(montage["rate"])
                if jefe_obs.get("stunned"):
                    aturdido.add(montage["name"])
            jugador = obs.get("player") or {}
            if jugador.get("dead") or ev.get("player_death") or ev.get("victory"):
                reinicios += 1
                obs = cliente.reset(boss=mod)
                previo = float(obs["t"])
                proximo_cambio = transcurrido
            elif float(jugador.get("hp", 100.0)) < 75.0:
                cliente.exec_lua(LUA_CURAR)
    finally:
        cliente.close()
    clases = registro["clases"] + (["BP_Chamuco_C"] if jefe == "amlo" else [])
    retardos = med.retardos_remotos(eventos, golpes, clases)
    return {
        "jefe": jefe, "segundos": segundos, "reinicios": reinicios, "golpes": golpes, "eventos": eventos,
        "montages": montages, "aturdido": sorted(aturdido),
        "golpes_agrupados": med.agrupar_golpes(golpes, clases),
        "retardos": retardos, "retardos_resumen": med.resumir_retardos(retardos),
    }


def fusionar(jefe: str, resultado: dict[str, Any], agregar: bool = True) -> None:
    """
    Incorpora la medicion al registro del jefe.

    Contexto:
        main con --fusionar.

    Args:
        jefe (str): Identificador del registro.
        resultado (dict[str, Any]): Salida de medir.
        agregar (bool): Si es False solo se actualizan los golpes ya registrados.

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        fusionar("chamuco", resultado)
    """
    ruta = RAIZ / "data" / "jefes" / f"{jefe}.json"
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    ignorar = set(registro.get("ignorar", []))
    medidos = {m: g for m, g in resultado["golpes_agrupados"].items() if m not in ignorar}
    registro["golpes"] = med.fusionar_golpes(registro["golpes"], medidos, agregar)
    nuevos = [m for m in resultado["montages"] if m not in registro["vocabulario"]]
    registro["vocabulario"] = sorted({*registro["vocabulario"], *nuevos})
    ruta.write_text(json.dumps(registro, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registro {ruta.name} actualizado; montages nuevos: {nuevos}")


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/medir_jefes.py.

    Args:
        Ninguno.

    Returns:
        int: 0 al terminar.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        python scripts/medir_jefes.py --jefe nahual --segundos 600
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefe", required=True, choices=("juice", "amlo", "chamuco", "nahual"))
    parser.add_argument("--segundos", type=float, default=600.0)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--fusionar", action="store_true")
    parser.add_argument("--solo-existentes", action="store_true",
                        help="al fusionar, no agrega golpes nuevos (solo actualiza los registrados)")
    parser.add_argument("--solo-fusionar", action="store_true",
                        help="no lanza el juego: fusiona bench_results/medicion_<jefe>.json en el registro")
    args = parser.parse_args()
    if args.solo_fusionar:
        guardado = json.loads((RAIZ / "bench_results" / f"medicion_{args.jefe}.json").read_text(encoding="utf-8"))
        fusionar(args.jefe, guardado, not args.solo_existentes)
        return 0
    t0 = time.time()
    resultado = medir(args.jefe, args.segundos, FRACCIONES_DEFECTO[args.jefe], args.semilla)
    salida = RAIZ / "bench_results" / f"medicion_{args.jefe}.json"
    salida.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(f"{args.jefe}: {len(resultado['golpes'])} golpes, {len(resultado['eventos'])} eventos en "
          f"{time.time() - t0:.0f} s reales; escrito {salida.name}")
    for montage, filas in sorted(resultado["golpes_agrupados"].items()):
        print(" ", montage, [(f["pos"], f["parry"], f["dano"], f["n"], f.get("alcance")) for f in filas])
    print("retardos:", json.dumps(resultado["retardos_resumen"]))
    if args.fusionar:
        fusionar(args.jefe, resultado, not args.solo_existentes)
    return 0 if math.isfinite(args.segundos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
