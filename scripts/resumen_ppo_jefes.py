"""
Estado del PPO multi-jefe en una pantalla: pasos, velocidad, victorias por jefe y evaluaciones.

Lee runs/<nombre> (checkpoints/estado_ppo.json, evaluaciones.jsonl y los eventos de TensorBoard) sin
tocar el entrenamiento. Se ejecuta con PYTHONPATH=. desde la raiz:

    python scripts/resumen_ppo_jefes.py [nombre_de_la_corrida]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RAIZ = Path(__file__).resolve().parent.parent / "runs"
JEFES = ("juice", "amlo", "chamuco", "nahual")
METRICAS = ("tasa_victoria", "danio_al_jefe", "danio_recibido", "parries", "muertes", "duracion_s")


def ultimo_valor(acumulador: EventAccumulator, etiqueta: str) -> float | None:
    """
    Ultimo valor registrado de una serie de TensorBoard.

    Contexto:
        Tabla de metricas por jefe.

    Args:
        acumulador (EventAccumulator): Eventos ya cargados.
        etiqueta (str): Nombre de la serie.

    Returns:
        float | None: Ultimo valor, o None si la serie no existe.

    Raises:
        Ninguna.

    Ejemplo:
        ultimo_valor(acumulador, "dm/amlo/tasa_victoria")
    """
    if etiqueta not in acumulador.Tags()["scalars"]:
        return None
    return float(acumulador.Scalars(etiqueta)[-1].value)


def main(argumentos: list[str]) -> int:
    """
    Imprime el estado de la corrida.

    Contexto:
        Linea de comandos.

    Args:
        argumentos (list[str]): Opcionalmente el nombre de la corrida (por defecto multijefe_ppo).

    Returns:
        int: 0 al terminar, 1 si la corrida no existe.

    Raises:
        Ninguna.

    Ejemplo:
        main(["multijefe_ppo"])
    """
    nombre = argumentos[0] if argumentos else "multijefe_ppo"
    carpeta = RAIZ / nombre
    if not carpeta.is_dir():
        print(f"no existe {carpeta}")
        return 1
    estado: dict[str, Any] = {}
    archivo_estado = carpeta / "checkpoints" / "estado_ppo.json"
    if archivo_estado.exists():
        estado = json.loads(archivo_estado.read_text(encoding="utf-8"))
    carpetas_tb = sorted((carpeta / "tb").glob("ppo_*"))
    if carpetas_tb:
        acumulador = EventAccumulator(str(carpetas_tb[-1]), size_guidance={"scalars": 0})
        acumulador.Reload()
        pasos = acumulador.Scalars("dm/pasos_por_segundo")[-1].step if "dm/pasos_por_segundo" in acumulador.Tags()["scalars"] else 0
        print(f"corrida {nombre}: {pasos} pasos, {ultimo_valor(acumulador, 'dm/pasos_por_segundo') or 0:.0f} pasos/s, "
              f"tasa de victoria media (ventana 50) {ultimo_valor(acumulador, 'dm/tasa_victoria_media') or 0:.3f}")
        print("jefe      " + "  ".join(f"{m:>15}" for m in METRICAS))
        for jefe in JEFES:
            valores = [ultimo_valor(acumulador, f"dm/{jefe}/{m}") for m in METRICAS]
            print(f"{jefe:<9} " + "  ".join(f"{v:>15.3f}" if v is not None else f"{'-':>15}" for v in valores))
    print(f"mejor: media {estado.get('mejor_media', 0):.3f} a los {estado.get('mejor_paso', 0)} pasos; "
          f"promociones {estado.get('promociones', 0)}, reversiones {estado.get('reversiones', 0)}, "
          f"tasa de aprendizaje {estado.get('learning_rate', 0):.2e}")
    print(f"referencia (imitacion): {estado.get('referencia', {})}")
    for evaluacion in estado.get("evaluaciones", []):
        tasas = ", ".join(f"{j} {t:.2f}" for j, t in evaluacion["tasas"].items())
        marcas = ("PROMOVIDO " if evaluacion["promovido"] else "") + ("REVERTIDO" if evaluacion["revertido"] else "")
        print(f"  eval {evaluacion['pasos']:>9} pasos: media {evaluacion['media']:.3f} [{tasas}] {marcas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
