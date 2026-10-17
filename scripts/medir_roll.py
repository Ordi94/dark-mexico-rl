"""
Mide en que momento antes de un golpe no parable hay que rodar para esquivarlo.

Contra un golpe unico y no parable (por defecto el salto Anim_SwordV2_Combo03_01_RM_Montage2, golpe a
0.683 s de montage) el jugador se queda quieto, y al faltar `d` segundos de juego para el impacto
lanza un roll en una direccion dada. Se registra si el golpe quito vida. Barre `d` y las
direcciones (0 = hacia el jefe, 2 = lateral, 4 = hacia atras) con decisiones de 2 frames, que es
lo que usa el entorno. Escribe bench_results/roll.json.

Uso:
    .venv\\Scripts\\python.exe scripts\\medir_roll.py --repeticiones 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmrl.aplicacion.entorno import DarkMexicoEnv  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.bot_reglas import BotReglas  # noqa: E402
from dmrl.infraestructura.datos import cargar_tabla_golpes  # noqa: E402
from dmrl.infraestructura.juego import ParametrosInstancia, crear_fabrica_puerto  # noqa: E402


def ensayo(env: DarkMexicoEnv, bot: BotReglas, objetivo: str, golpe: float, d: float,
           direccion: int) -> tuple[dict[str, Any] | None, bool]:
    """
    Un ensayo: sobrevive con el bot hasta que aparece el montage objetivo, rueda cuando faltan d s.

    Contexto:
        main. El bot juega todo lo que no es el montage objetivo para que el episodio dure.

    Args:
        env (DarkMexicoEnv): Entorno con un episodio en marcha.
        bot (BotReglas): Bot de reglas que mantiene vivo al jugador.
        objetivo (str): Montage del jefe con el golpe a esquivar.
        golpe (float): Posicion del golpe en el montage (s).
        d (float): Segundos de juego antes del impacto en que se decide rodar.
        direccion (int): Direccion del roll 0..7.

    Returns:
        tuple[dict | None, bool]: Resultado del ensayo (None si no se pudo completar) y si el
            episodio sigue vivo.

    Raises:
        BridgeError: Si el puente falla.

    Ejemplo:
        ensayo(env, bot, "Anim_SwordV2_Combo03_01_RM_Montage2", 0.683, 0.27, 2)
    """
    disparado = False
    datos: dict[str, Any] = {}
    for _ in range(3000):
        crudo = env.observacion_cruda
        jefe = crudo.get("boss") or {}
        montage = jefe.get("montage") or {}
        jugador = crudo["player"]
        pos, rate = float(montage.get("pos", 0.0)), float(montage.get("rate", 1.0)) or 1.0
        mascara = env.action_masks()
        en_objetivo = montage.get("name") == objetivo
        accion = bot.decidir(crudo, mascara.tolist())
        if en_objetivo:
            accion = acc.ACCION_NADA
            falta = (golpe - pos) / rate
            roll = acc.ACCION_ROLL_PRIMERA + direccion
            if not disparado and falta <= d and mascara[roll]:
                accion, disparado = roll, True
                datos = {"d": d, "falta": falta, "direccion": direccion, "stamina": jugador["stamina"],
                         "hp_antes": jugador["hp"]}
        _, _, terminado, truncado, _ = env.step(accion)
        nuevo = env.observacion_cruda
        m_nuevo = (nuevo.get("boss") or {}).get("montage") or {}
        if disparado and m_nuevo.get("name") == objetivo and float(m_nuevo.get("pos", 0)) > golpe + 0.2:
            datos["danio"] = datos["hp_antes"] - nuevo["player"]["hp"]
            return datos, True
        if disparado and m_nuevo.get("name") != objetivo:
            datos["danio"] = datos["hp_antes"] - nuevo["player"]["hp"]
            return datos, not (terminado or truncado)
        if terminado or truncado:
            return None, False
    return None, False


def main() -> int:
    """
    Punto de entrada: barre d y direcciones y resume la tasa de esquiva.

    Contexto:
        Herramienta de linea de comandos.

    Args:
        Ninguno.

    Returns:
        int: 0 si termino.

    Raises:
        Ninguna.

    Ejemplo:
        python scripts/medir_roll.py --repeticiones 4
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--repeticiones", type=int, default=6)
    ap.add_argument("--objetivo", default="Anim_SwordV2_Combo03_01_RM_Montage2")
    ap.add_argument("--golpe", type=float, default=0.683)
    args = ap.parse_args()
    rng = random.Random(3)
    tabla = cargar_tabla_golpes()
    env = DarkMexicoEnv(crear_fabrica_puerto(ParametrosInstancia("mr")), tabla)
    plan = [(d, dire) for d in (0.0, 0.07, 0.13, 0.2, 0.27, 0.33, 0.4, 0.47, 0.53) for dire in (0, 2, 4)] * args.repeticiones
    rng.shuffle(plan)
    resultados: dict[tuple[float, int], list[float]] = defaultdict(list)
    try:
        bot = BotReglas(tabla)
        vivo = False
        for d, dire in plan:
            if not vivo:
                env.reset(options={"start_phase": 1, "boss_hp_mult": 1.0})
                bot.reiniciar()
            r, vivo = ensayo(env, bot, args.objetivo, args.golpe, d, dire)
            if r is not None and "danio" in r:
                resultados[(d, dire)].append(r["danio"])
    finally:
        env.close()
    salida = []
    print("d(s)  dir  ensayos  con_danio")
    for (d, dire), lista in sorted(resultados.items()):
        con = sum(1 for x in lista if x > 0)
        print(f"{d:4.2f}  {dire}    {len(lista):3d}      {con:3d}")
        salida.append({"d": d, "direccion": dire, "ensayos": len(lista), "con_danio": con})
    Path("bench_results").mkdir(exist_ok=True)
    Path("bench_results/roll.json").write_text(json.dumps(salida, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
