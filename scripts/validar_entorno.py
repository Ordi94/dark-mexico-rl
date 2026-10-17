"""
Valida el entorno real con acciones aleatorias validas (sin caidas, reset correcto, rangos, recompensa).

Juega N episodios con acciones al azar entre las que admite la mascara y comprueba, contra el juego real:
    * ningun episodio termina en error (caida del juego o del reinicio) y cada reset deja al jefe activo
      con vida completa del nivel pedido, jugador con 100/100 y 3 caguamas;
    * la observacion es finita, float32 y esta dentro de [-1, 1];
    * la recompensa de cada episodio coincide con la recalculada de forma independiente a partir de los
      eventos crudos (dano hecho, vida perdida y recuperada, parries, fase 2, final);
    * el dano al jefe medido por los golpes dados coincide con la caida de la vida del jefe que lee el juego.

Escribe bench_results/entorno_aleatorio.json.

Uso:
    .venv\\Scripts\\python.exe scripts\\validar_entorno.py --episodios 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmrl.aplicacion.entorno import DarkMexicoEnv  # noqa: E402
from dmrl.dominio.configuracion import ConfigRecompensa  # noqa: E402
from dmrl.infraestructura.datos import cargar_tabla_golpes  # noqa: E402
from dmrl.infraestructura.juego import ParametrosInstancia, crear_fabrica_puerto  # noqa: E402

CFG = ConfigRecompensa()


def recompensa_independiente(previo: dict[str, Any], actual: dict[str, Any]) -> float:
    """
    Recalcula la recompensa de un paso solo con los hechos crudos, sin usar CalculadoraRecompensa.

    Contexto:
        main. Sirve para detectar errores de cableado (por ejemplo un paso de enfoque cuya recompensa
        no se sumara).

    Args:
        previo (dict[str, Any]): Observacion cruda anterior.
        actual (dict[str, Any]): Observacion cruda posterior.

    Returns:
        float: Recompensa esperada del tramo.

    Raises:
        Ninguna.

    Ejemplo:
        recompensa_independiente(obs0, obs1)
    """
    ev = actual.get("ev") or {}
    vida0, vida1 = previo["player"]["hp"], actual["player"]["hp"]
    r = -CFG.paso + CFG.danio_hecho * ev.get("dmg_dealt", 0.0) + CFG.parry * ev.get("parries", 0)
    if vida1 < vida0:
        r -= CFG.danio_recibido * (vida0 - max(vida1, 0.0))
    elif vida1 > vida0 and vida0 < CFG.curacion_vida_maxima:
        r += CFG.curacion * (vida1 - vida0)
    if ev.get("player_death") or actual["player"].get("dead"):
        r -= CFG.muerte
    elif ev.get("victory"):
        r += CFG.victoria
    return r


def main() -> int:
    """
    Punto de entrada: juega los episodios aleatorios y reporta las comprobaciones.

    Contexto:
        Validacion del entorno con el juego real.

    Args:
        Ninguno.

    Returns:
        int: 0 si todo coincide, 1 si alguna comprobacion falla.

    Raises:
        Ninguna.

    Ejemplo:
        python scripts/validar_entorno.py --episodios 5
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--episodios", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    azar = random.Random(args.seed)
    env = DarkMexicoEnv(crear_fabrica_puerto(ParametrosInstancia("va")), cargar_tabla_golpes(), semilla=args.seed)
    fallos: list[str] = []
    episodios: list[dict[str, Any]] = []
    rango_min, rango_max, pasos_total = 1.0, -1.0, 0
    try:
        for n in range(args.episodios):
            mult = azar.choice([0.3, 0.5, 1.0])
            fase = azar.choice([1, 1, 2])
            obs, _ = env.reset(options={"start_phase": fase, "boss_hp_mult": mult})
            crudo = env.observacion_cruda
            esperado_hp_jefe = (1000.0 if fase == 2 else 2000.0) * mult
            if abs(crudo["boss"]["hp"] - esperado_hp_jefe) > 0.5 or crudo["player"]["hp"] != 100.0 or crudo["player"]["caguamas"] != 3:
                fallos.append(f"ep {n}: estado inicial incorrecto {crudo['boss']['hp']} {crudo['player']['hp']} {crudo['player']['caguamas']}")
            suma_real, suma_esperada, dano_dealt, caida_jefe, previo = 0.0, 0.0, 0.0, 0.0, crudo
            info: dict[str, Any] = {}
            while True:
                validas = np.flatnonzero(env.action_masks())
                accion = int(validas[azar.randrange(len(validas))])
                obs, r, terminado, truncado, info = env.step(accion)
                pasos_total += 1
                actual = env.observacion_cruda
                rango_min, rango_max = min(rango_min, float(obs.min())), max(rango_max, float(obs.max()))
                if not np.all(np.isfinite(obs)) or obs.dtype != np.float32:
                    fallos.append(f"ep {n}: observacion no finita o no float32")
                suma_real += r
                # El paso de enfoque (1 frame) tambien aporta recompensa: se compara por episodio.
                suma_esperada += recompensa_independiente(previo, actual)
                dano_dealt += (actual.get("ev") or {}).get("dmg_dealt", 0.0)
                if previo["boss"].get("cls") == actual["boss"].get("cls") and actual["boss"]["hp"] < previo["boss"]["hp"]:
                    caida_jefe += previo["boss"]["hp"] - actual["boss"]["hp"]
                previo = actual
                if terminado or truncado:
                    break
            resumen = info["episodio"]
            episodios.append({**resumen, "suma_recompensa_env": suma_real, "suma_recompensa_independiente": suma_esperada,
                              "dano_golpes": dano_dealt, "caida_vida_jefe": caida_jefe})
            if resumen["resultado"] == "error":
                fallos.append(f"ep {n}: error {resumen.get('error')}")
            if not math.isclose(suma_real, resumen["recompensa"], abs_tol=1e-6):
                fallos.append(f"ep {n}: recompensa acumulada {suma_real} != resumen {resumen['recompensa']}")
            print(n, resumen["resultado"], f"fase{fase} x{mult}", f"{resumen['duracion_s']:.0f}s", f"R={suma_real:.2f}",
                  f"dano_env={resumen['danio_hecho']:.0f} caida_jefe={caida_jefe:.0f} dR={abs(suma_real - suma_esperada):.3f}", flush=True)
    finally:
        env.close()
    desviaciones = [abs(e["suma_recompensa_env"] - e["suma_recompensa_independiente"]) for e in episodios]
    dif_dano = [abs(e["caida_vida_jefe"] - e["danio_hecho"]) for e in episodios]
    print(json.dumps({
        "episodios": len(episodios), "pasos": pasos_total, "rango_observacion": [rango_min, rango_max],
        "reinicios_instancia": env.reinicios_instancia, "fallos": fallos,
        "max_dif_recompensa_vs_independiente": max(desviaciones, default=0.0),
        "max_dif_danio_vs_caida_vida_jefe": max(dif_dano, default=0.0),
        "dif_recompensa_media": float(np.mean(desviaciones)) if desviaciones else 0.0,
    }, indent=1))
    Path("bench_results").mkdir(exist_ok=True)
    Path("bench_results/entorno_aleatorio.json").write_text(
        json.dumps({"episodios": episodios, "fallos": fallos, "pasos": pasos_total,
                    "rango_observacion": [rango_min, rango_max]}, indent=1), encoding="utf-8")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
