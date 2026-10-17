"""
Contrasta la mascara de acciones con lo que el juego acepta en cada jefe del entorno multi-jefe.

Contexto:
    Igual que scripts/validar_mascara.py (acciones del bot y acciones al azar ignorando la mascara,
    efecto medido como inicio de montage o cambio de bandera del jugador) pero reiniciando cada
    episodio contra un jefe con su descripcion de data/jefes. La mascara depende solo del jugador, asi
    que debe comportarse igual que contra Juice; este script lo comprueba contra AMLO, el Chamuco y el
    Chupacabras. Escribe bench_results/mascara_jefes.json.

Uso:
    python scripts/validar_mascara_jefes.py --jefes amlo chamuco nahual --episodios 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmbridge import BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.aplicacion.entorno import a_action  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.amenazas import SeguidorAmenazas  # noqa: E402
from dmrl.dominio.bot_jefes import BotJefes  # noqa: E402
from dmrl.infraestructura.jefes import cargar_catalogo  # noqa: E402
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO  # noqa: E402
from scripts.validar_mascara import hubo_efecto  # noqa: E402

NOMBRES = {acc.ACCION_ROLL_PRIMERA: "roll", acc.ACCION_ATAQUE: "ataque", acc.ACCION_ESPECIAL: "especial",
           acc.ACCION_PARRY: "parry", acc.ACCION_BLOQUEO: "bloqueo", acc.ACCION_CAGUAMA: "caguama"}


def medir(jefe: str, episodios: int, azar_fraccion: float, semilla: int) -> dict[str, dict[str, Any]]:
    """
    Juega episodios contra un jefe y cuenta, por accion, validez declarada y efecto real.

    Contexto:
        main.

    Args:
        jefe (str): Jefe de episodio.
        episodios (int): Episodios a jugar.
        azar_fraccion (float): Fraccion de pasos con accion al azar.
        semilla (int): Semilla.

    Returns:
        dict[str, dict[str, Any]]: accion -> {"valida": {n, efecto}, "invalida": {n, efecto}}.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        medir("nahual", 10, 0.4, 0)
    """
    catalogo = cargar_catalogo()
    mod = dict(catalogo.perfil(jefe).episodio.mod or {})
    azar = random.Random(semilla)
    bot = BotJefes.desde_catalogo(catalogo)
    seguidor = SeguidorAmenazas(catalogo)
    cliente = BridgeClient(f"vmj_{jefe}")
    cliente.launch(LaunchConfig(instance_id=f"vmj_{jefe}", affinity=list(AFINIDAD_ENTRENAMIENTO)))
    cuenta: Counter[tuple[int, bool, bool]] = Counter()
    try:
        for _ in range(episodios):
            obs = cliente.reset(boss=mod)
            bot.reiniciar()
            seguidor.reiniciar()
            previa = acc.ACCION_NADA
            for _paso in range(900):
                amenazas = seguidor.actualizar(obs)
                mascara = acc.mascara_acciones(obs.get("player"))
                al_azar = azar.random() < azar_fraccion
                accion = azar.randrange(acc.N_ACCIONES) if al_azar else bot.decidir(obs, mascara, amenazas)
                antes = obs.get("player") or {}
                obs = cliente.step(a_action(acc.comando_de_accion(accion, previa)), 2)
                bot.confirmar(accion if mascara[accion] else acc.ACCION_NADA)
                previa = accion
                despues = obs.get("player") or {}
                if accion >= acc.ACCION_ROLL_PRIMERA:
                    grupo = acc.ACCION_ROLL_PRIMERA if accion < acc.ACCION_ATAQUE else accion
                    cuenta[(grupo, mascara[accion], hubo_efecto(accion, antes, despues))] += 1
                ev = obs.get("ev") or {}
                if despues.get("dead") or ev.get("player_death") or ev.get("victory"):
                    break
    finally:
        cliente.close()
    salida: dict[str, dict[str, Any]] = {}
    for grupo, nombre in NOMBRES.items():
        salida[nombre] = {}
        for valida in (True, False):
            n = cuenta[(grupo, valida, True)] + cuenta[(grupo, valida, False)]
            salida[nombre]["valida" if valida else "invalida"] = {"n": n, "efecto": cuenta[(grupo, valida, True)]}
    return salida


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/validar_mascara_jefes.py.

    Args:
        Ninguno.

    Returns:
        int: 0 al terminar.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        python scripts/validar_mascara_jefes.py --jefes nahual
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefes", nargs="+", required=True)
    parser.add_argument("--episodios", type=int, default=40)
    parser.add_argument("--azar", type=float, default=0.4)
    parser.add_argument("--semilla", type=int, default=11)
    args = parser.parse_args()
    resultado = {j: medir(j, args.episodios, args.azar, args.semilla) for j in args.jefes}
    (RAIZ / "bench_results" / "mascara_jefes.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    for jefe, acciones in resultado.items():
        print(jefe)
        for accion, v in acciones.items():
            def tasa(x: dict[str, int]) -> str:
                return f"{x['efecto']}/{x['n']} ({x['efecto'] / x['n']:.0%})" if x["n"] else "-"

            print(f"  {accion:9} valida {tasa(v['valida']):>16}   invalida {tasa(v['invalida']):>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
