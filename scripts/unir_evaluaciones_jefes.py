"""
Une varias evaluaciones en juego de un mismo modelo en una sola, con el resumen por jefe recalculado.

La eleccion final de la arquitectura y la referencia del PPO usan las peleas de la seleccion y de la
confirmacion juntas: una tasa con 40 peleas tiene +-14 puntos de incertidumbre y con 80, +-10.
Uso, desde la raiz del proyecto y con PYTHONPATH=.:

    python scripts/unir_evaluaciones_jefes.py <salida> <entrada1> <entrada2> ...

Los nombres son archivos de runs/multijefe_imitacion/evaluaciones sin la extension.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dmrl.aplicacion.evaluacion_modelo_jefes import resumir_por_jefe, tasas_por_jefe

CARPETA = Path(__file__).resolve().parent.parent / "runs" / "multijefe_imitacion" / "evaluaciones"


def main(argumentos: list[str]) -> int:
    """
    Une las evaluaciones nombradas y escribe la evaluacion combinada.

    Contexto:
        Linea de comandos.

    Args:
        argumentos (list[str]): Nombre de salida seguido de los nombres de entrada.

    Returns:
        int: 0 al terminar, 2 si faltan argumentos.

    Raises:
        FileNotFoundError: Si alguna entrada no existe.

    Ejemplo:
        main(["final_3x256", "3x256", "confirmacion_3x256"])
    """
    if len(argumentos) < 3:
        print(__doc__)
        return 2
    salida, entradas = argumentos[0], argumentos[1:]
    episodios = []
    for nombre in entradas:
        episodios += json.loads((CARPETA / f"{nombre}.json").read_text(encoding="utf-8"))["episodios"]
    por_jefe = resumir_por_jefe(episodios)
    tasas = tasas_por_jefe(por_jefe)
    media = sum(tasas.values()) / len(tasas)
    (CARPETA / f"{salida}.json").write_text(json.dumps({
        "modelo": entradas, "tasas": tasas, "tasa_media": media, "por_jefe": por_jefe, "episodios": episodios,
    }), encoding="utf-8")
    print(json.dumps({"tasas": tasas, "tasa_media": media}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
