"""
Tablas de la imitacion multi-jefe en Markdown: datos por jefe, ajuste por arquitectura y juego por jefe.

Lee runs/multijefe_imitacion (datos, modelos/*.json y evaluaciones/*.json) sin abrir el juego ni las
observaciones. Se ejecuta con PYTHONPATH=. desde la raiz del proyecto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dmrl.aplicacion.imitacion_jefes import RONDAS_MAXIMAS, resumen_de_datos
from dmrl.infraestructura.demostraciones import metadatos_de_ronda

BASE = Path(__file__).resolve().parent.parent / "runs" / "multijefe_imitacion"
JEFES = ("juice", "amlo", "chamuco", "nahual")


def tabla_datos() -> str:
    """
    Datos recolectados por jefe, con las peleas que pasaron por la segunda fase.

    Contexto:
        Reporte de la recoleccion.

    Args:
        Ninguno.

    Returns:
        str: Tabla Markdown.

    Raises:
        Ninguna.

    Ejemplo:
        print(tabla_datos())
    """
    metadatos: list[dict[str, Any]] = []
    for ronda in range(RONDAS_MAXIMAS):
        metadatos += metadatos_de_ronda(BASE / "datos", ronda)
    resumen = resumen_de_datos(metadatos)
    transiciones = {
        jefe: sum(1 for m in metadatos if m["jefe"] == jefe and int(m["fase_inicio"]) == 1 and int(m["fase_max"]) >= 2)
        for jefe in JEFES
    }
    filas = ["| Jefe | Peleas | Pasos | Victorias del bot | Con ruido | Inicio en 2a fase | Transicion natural a la 2a fase |",
             "|---|---|---|---|---|---|---|"]
    total_peleas = total_pasos = 0
    for jefe in JEFES:
        r = resumen[jefe]
        total_peleas += int(r["peleas"])
        total_pasos += int(r["pasos"])
        filas.append(f"| {jefe} | {int(r['peleas'])} | {int(r['pasos']):,} | {int(r['victorias_bot'])} | "
                     f"{int(r['con_ruido'])} | {int(r['inicio_fase2'])} | {transiciones[jefe]} |")
    filas.append(f"| **Total** | **{total_peleas}** | **{total_pasos:,}** | | | | |")
    return "\n".join(filas)


def tabla_ajuste() -> str:
    """
    Resultado del ajuste de cada arquitectura: perdida, precision global y por jefe.

    Contexto:
        Reporte del ajuste.

    Args:
        Ninguno.

    Returns:
        str: Tabla Markdown.

    Raises:
        Ninguna.

    Ejemplo:
        print(tabla_ajuste())
    """
    filas = ["| Arquitectura | Epocas (mejor / total) | Perdida entren. | Perdida valid. | Precision valid. | "
             + " | ".join(f"Prec. {j}" for j in JEFES) + " | Parry recall / prec. | Varianza expl. valor | Segundos |",
             "|---|---|---|---|---|" + "---|" * len(JEFES) + "---|---|---|"]
    for archivo in sorted((BASE / "modelos").glob("*.json")):
        d = json.loads(archivo.read_text(encoding="utf-8"))
        parry = next(a for a in d["por_accion"] if a["accion"] == 19)
        filas.append(
            f"| {d['arquitectura']} | {d['mejor_epoca']} / {len(d['curva'])} | {d['entrenamiento']['perdida']:.4f} | "
            f"{d['validacion']['perdida']:.4f} | {d['validacion']['precision']:.4f} | "
            + " | ".join(f"{d['por_jefe'][j]['precision']:.4f}" for j in JEFES)
            + f" | {parry['recall']:.3f} / {parry['precision']:.3f} | "
              f"{d['valor']['varianza_explicada_validacion']:.2f} | {d['segundos']:.0f} |")
    return "\n".join(filas)


def tabla_juego(nombres: list[str]) -> str:
    """
    Victorias en juego por jefe de las evaluaciones indicadas.

    Contexto:
        Reporte de la comparacion contra el bot.

    Args:
        nombres (list[str]): Nombres de archivo (sin .json) en evaluaciones/.

    Returns:
        str: Tabla Markdown.

    Raises:
        Ninguna.

    Ejemplo:
        print(tabla_juego(["bot", "2x256"]))
    """
    filas = ["| Evaluacion | " + " | ".join(JEFES) + " | Media |", "|---|" + "---|" * (len(JEFES) + 1)]
    for nombre in nombres:
        ruta = BASE / "evaluaciones" / f"{nombre}.json"
        if not ruta.exists():
            continue
        d = json.loads(ruta.read_text(encoding="utf-8"))
        celdas = []
        for j in JEFES:
            g = d["por_jefe"][j]["juego"]
            celdas.append(f"{int(g['victorias'])}/{int(g['peleas'])} ({g['tasa_victoria']:.0%})")
        filas.append(f"| {nombre} | " + " | ".join(celdas) + f" | {d['tasa_media']:.1%} |")
    return "\n".join(filas)


def main() -> int:
    """
    Imprime las tablas de la imitacion multi-jefe.

    Contexto:
        Linea de comandos.

    Args:
        Ninguno.

    Returns:
        int: 0 al terminar.

    Raises:
        FileNotFoundError: Si faltan los datos o los informes.

    Ejemplo:
        python scripts/resumen_imitacion_jefes.py bot 2x256
    """
    nombres = sys.argv[1:] or ["bot"]
    print("### Datos\n\n" + tabla_datos() + "\n\n### Ajuste\n\n" + tabla_ajuste() + "\n\n### Juego\n\n" + tabla_juego(nombres))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
