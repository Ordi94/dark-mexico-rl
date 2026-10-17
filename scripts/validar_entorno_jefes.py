"""
Validacion del entorno multi-jefe contra el juego real: episodios con acciones aleatorias validas.

Contexto:
    Por cada jefe juega N episodios eligiendo acciones al azar entre las que admite la mascara y
    comprueba, paso a paso: observacion finita, de la forma y el rango declarados; recompensa finita
    y coherente (la suma de los pasos es la del resumen, la victoria y la muerte cobran su premio al
    terminar y el dano al jefe coincide con la caida de su vida); el episodio termina por muerte,
    victoria o tiempo, sin errores del juego. Escribe bench_results/validacion_jefes_<jefe>.json.

Uso:
    python scripts/validar_entorno_jefes.py --jefe amlo --episodios 10
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

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmrl.aplicacion.evaluacion_jefes import crear_entorno_jefes  # noqa: E402
from dmrl.infraestructura.jefes import cargar_catalogo  # noqa: E402


def validar(jefe: str, episodios: int, semilla: int) -> dict[str, Any]:
    """
    Juega los episodios y devuelve las comprobaciones.

    Contexto:
        main.

    Args:
        jefe (str): Jefe de episodio.
        episodios (int): Episodios a jugar.
        semilla (int): Semilla de las acciones.

    Returns:
        dict[str, Any]: Episodios, pasos, violaciones por tipo y resumen por episodio.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        validar("nahual", 3, 0)
    """
    catalogo = cargar_catalogo()
    env = crear_entorno_jefes(catalogo, (jefe,), f"val_{jefe}", semilla=semilla, trazar=True)
    azar = random.Random(semilla)
    violaciones: dict[str, int] = {}
    filas: list[dict[str, Any]] = []
    pasos_totales = 0
    t0 = time.time()

    def falla(clave: str) -> None:
        violaciones[clave] = violaciones.get(clave, 0) + 1

    try:
        for numero in range(episodios):
            obs, _ = env.reset(options={"jefe": jefe, "boss_hp_mult": 1.0})
            forma = env.observation_space.shape
            if obs.shape != forma or not np.isfinite(obs).all() or obs.min() < -1.0 or obs.max() > 1.0:
                falla("observacion_reset")
            suma = 0.0
            ultima = 0.0
            terminal = False
            dano_vida = 0.0
            previo = env.observacion_cruda.get("boss") or {}
            info: dict[str, Any] = {}
            while True:
                validas = np.flatnonzero(env.action_masks())
                accion = int(validas[azar.randrange(len(validas))])
                obs, recompensa, terminado, truncado, info = env.step(accion)
                pasos_totales += 1
                if obs.shape != forma or obs.dtype != np.float32 or not np.isfinite(obs).all():
                    falla("observacion_forma_o_no_finita")
                if obs.min() < -1.0 or obs.max() > 1.0:
                    falla("observacion_fuera_de_rango")
                if not math.isfinite(recompensa):
                    falla("recompensa_no_finita")
                suma += recompensa
                ultima = recompensa
                cruda = env.observacion_cruda
                actual = cruda.get("boss") or {}
                if previo and actual and previo.get("cls") == actual.get("cls") and previo.get("hp_max") == actual.get("hp_max"):
                    dano_vida += max(0.0, float(previo.get("hp", 0.0)) - float(actual.get("hp", 0.0)))
                previo = actual
                if terminado or truncado:
                    terminal = True
                    break
            resumen = info["episodio"]
            if not terminal or resumen["resultado"] == "error":
                falla("episodio_con_error")
            if abs(suma - resumen["recompensa"]) > 1e-6:
                falla("recompensa_del_resumen_distinta_de_la_suma")
            if resumen["victoria"] and ultima < 8.0:
                falla("victoria_sin_premio")
            if resumen["muerte"] and ultima > -4.0:
                falla("muerte_sin_castigo")
            filas.append({
                "episodio": numero, "resultado": resumen["resultado"], "pasos": resumen["pasos"],
                "duracion_s": resumen["duracion_s"], "danio_hecho": resumen["danio_hecho"],
                "danio_recibido": resumen["danio_recibido"], "recompensa": resumen["recompensa"],
                "golpe_mortal": resumen.get("golpe_mortal"), "reinicio_s": resumen.get("reinicio_s"),
                "caida_vida_jefe": dano_vida,
            })
    finally:
        env.close()
    # El dano de la recompensa (eventos de TakeDamage, incluido el paso de enfoque) contra la caida de la
    # vida que lee el juego mientras el actor jefe es el mismo.
    desfase = [abs(f["danio_hecho"] - f["caida_vida_jefe"]) for f in filas]
    return {
        "jefe": jefe, "episodios": len(filas), "pasos": pasos_totales, "segundos_reales": time.time() - t0,
        "violaciones": violaciones, "desfase_dano_maximo": max(desfase, default=0.0),
        "desfase_dano_medio": sum(desfase) / len(desfase) if desfase else 0.0,
        "filas": filas,
    }


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/validar_entorno_jefes.py.

    Args:
        Ninguno.

    Returns:
        int: 0 si no hay violaciones, 1 si las hay.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        python scripts/validar_entorno_jefes.py --jefe chamuco
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefe", required=True)
    parser.add_argument("--episodios", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    resultado = validar(args.jefe, args.episodios, args.semilla)
    salida = RAIZ / "bench_results" / f"validacion_jefes_{args.jefe}.json"
    salida.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(f"{args.jefe}: {resultado['episodios']} episodios, {resultado['pasos']} pasos en "
          f"{resultado['segundos_reales']:.0f} s; violaciones={resultado['violaciones']}; "
          f"desfase de dano max {resultado['desfase_dano_maximo']:.2f}")
    for f in resultado["filas"]:
        print("  ", f["resultado"], f["pasos"], f"{f['duracion_s']:.0f}s", f"dano {f['danio_hecho']:.0f}",
              f["golpe_mortal"])
    return 1 if resultado["violaciones"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
