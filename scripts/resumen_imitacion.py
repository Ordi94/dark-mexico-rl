"""
Resume en tablas Markdown lo que dejo una corrida de imitacion en runs/<nombre>.

Uso: .venv\\Scripts\\python.exe scripts\\resumen_imitacion.py [nombre]

Lee los metadatos de las demostraciones, los informes de cada ajuste y las evaluaciones en juego; no
abre el juego ni modifica nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dmrl.dominio.imitacion import intervalo_wilson, resumen_de_juego
from dmrl.infraestructura.demostraciones import leer_json, metadatos_de_ronda

RAIZ = Path(__file__).resolve().parent.parent
EPOCAS_MOSTRADAS = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60)


def porcentaje(valor: float) -> str:
    """
    Formatea una fraccion como porcentaje entero.

    Contexto:
        Todas las tablas.

    Args:
        valor (float): Fraccion en [0, 1].

    Returns:
        str: Texto "NN%".

    Raises:
        Ninguna.

    Ejemplo:
        porcentaje(0.724)  # "72%"
    """
    return f"{valor * 100:.0f}%"


def fila_juego(etiqueta: str, resumen: dict[str, float]) -> str:
    """
    Fila de la tabla de resultados en juego.

    Contexto:
        tabla_juego.

    Args:
        etiqueta (str): Nombre de la fila.
        resumen (dict[str, float]): Salida de resumen_de_juego.

    Returns:
        str: Fila Markdown; vacia si no hay peleas.

    Raises:
        Ninguna.

    Ejemplo:
        fila_juego("ronda 0", resumen)
    """
    if not resumen:
        return ""
    n, v = int(resumen["peleas"]), int(resumen["victorias"])
    return (f"| {etiqueta} | {v}/{n} ({porcentaje(resumen['tasa_victoria'])}, "
            f"{porcentaje(resumen['tasa_victoria_inferior'])}-{porcentaje(resumen['tasa_victoria_superior'])}) "
            f"| {porcentaje(resumen['tasa_fase2'])} | {resumen['danio_hecho']:.0f} | {resumen['danio_recibido']:.0f} "
            f"| {resumen['parries']:.1f} | {resumen['caguamas_usadas']:.1f} / {resumen['caguamas_interrumpidas']:.2f} "
            f"| {int(resumen['muertes_fase1'])} / {int(resumen['muertes_fase2'])} / {int(resumen['tiempos_agotados'])} "
            f"| {resumen['duracion_s']:.0f} |")


def tabla_juego(base: Path, condicion: str, rondas: list[int]) -> list[str]:
    """
    Tabla de resultados en juego de una condicion para el bot y cada ronda.

    Contexto:
        main.

    Args:
        base (Path): Carpeta runs/<nombre>.
        condicion (str): "principal", "fase2" o "ruido".
        rondas (list[int]): Rondas a incluir.

    Returns:
        list[str]: Lineas Markdown de la tabla.

    Raises:
        Ninguna.

    Ejemplo:
        tabla_juego(Path("runs/imitacion"), "principal", [0, 1])
    """
    lineas = [
        "| Politica | Victorias (tasa, IC 95%) | Llegan a fase 2 | Dano al jefe | Dano recibido | Parries | Caguamas usadas / interrumpidas | Muertes f1 / f2 / tiempo | Duracion (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    bot = leer_json(base / "evaluaciones" / "bot.json").get(condicion)
    if bot:
        lineas.append(fila_juego("bot de reglas", bot["resumen"]))
    for ronda in rondas:
        datos = leer_json(base / "evaluaciones" / f"ronda_{ronda:02d}.json").get(condicion)
        if datos:
            lineas.append(fila_juego(f"red, ronda {ronda}", datos["resumen"]))
    return [linea for linea in lineas if linea]


def main() -> int:
    """
    Imprime el resumen de la corrida.

    Contexto:
        Linea de comandos.

    Args:
        Ninguno.

    Returns:
        int: 0 si hay datos; 1 si la carpeta no existe.

    Raises:
        Ninguna.

    Ejemplo:
        main()
    """
    nombre = sys.argv[1] if len(sys.argv) > 1 else "imitacion"
    base = RAIZ / "runs" / nombre
    if not base.is_dir():
        print(f"no existe {base}")
        return 1
    rondas = sorted(int(p.name.split("_")[1]) for p in (base / "datos").glob("ronda_*"))
    print("### Datos recolectados por ronda\n")
    print("| Ronda | Ejecuta | Episodios | Pasos | Pasos por episodio | Victorias del ejecutor | En fase 2 | Con ruido |")
    print("|---|---|---|---|---|---|---|---|")
    acumulado = 0
    for ronda in rondas:
        meta = metadatos_de_ronda(base / "datos", ronda)
        pasos = sum(int(m["pasos"]) for m in meta)
        acumulado += pasos
        vic = sum(1 for m in meta if m.get("victoria"))
        print(f"| {ronda} | {meta[0]['ejecutor']} | {len(meta)} | {pasos} | {pasos / len(meta):.0f} | {vic} "
              f"({porcentaje(vic / len(meta))}) | {porcentaje(sum(1 for m in meta if m['fase_inicio'] == 2) / len(meta))} "
              f"| {porcentaje(sum(1 for m in meta if m['epsilon'] > 0) / len(meta))} |")
    print(f"\nTotal: {acumulado} pasos.\n")
    print("### Ajuste supervisado por ronda\n")
    print("| Ronda | Pasos (entren. / valid.) | Episodios (entren. / valid.) | Mejor epoca / epocas | Perdida train | Perdida val | Brecha | Precision train | Precision val | Valor: varianza explicada val |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    informes = {r: leer_json(base / "modelos" / f"ronda_{r:02d}.json") for r in rondas}
    for ronda, inf in informes.items():
        if not inf:
            continue
        print(f"| {ronda} | {inf['pasos_entrenamiento']} / {inf['pasos_validacion']} | {inf['episodios_entrenamiento']} / "
              f"{inf['episodios_validacion']} | {inf['mejor_epoca']} / {len(inf['curva'])} | {inf['entrenamiento']['perdida']:.4f} "
              f"| {inf['validacion']['perdida']:.4f} | {inf['brecha_perdida']:+.1%} | {inf['entrenamiento']['precision']:.3f} "
              f"| {inf['validacion']['precision']:.3f} | {inf['valor']['varianza_explicada_validacion']:.2f} |")
    for ronda, inf in informes.items():
        if not inf:
            continue
        print(f"\n#### Curva de aprendizaje, ronda {ronda}\n")
        print("| Epoca | Perdida train | Perdida val | Precision train | Precision val |")
        print("|---|---|---|---|---|")
        curva = inf["curva"]
        for punto in curva:
            if punto["epoca"] in EPOCAS_MOSTRADAS or punto["epoca"] in (inf["mejor_epoca"], len(curva)):
                print(f"| {punto['epoca']} | {punto['perdida_entrenamiento']:.4f} | {punto['perdida_validacion']:.4f} "
                      f"| {punto['precision_entrenamiento']:.3f} | {punto['precision_validacion']:.3f} |")
    for ronda, inf in informes.items():
        if not inf:
            continue
        print(f"\n#### Precision por accion en validacion, ronda {ronda}\n")
        print("| Accion | Veces (bot) | Recall (la red la elige cuando el bot la elige) | Precision (cuando la red la elige, el bot tambien) |")
        print("|---|---|---|---|")
        for a in inf["por_accion"]:
            if a["soporte"]:
                print(f"| {a['accion']} {a['nombre']} | {a['soporte']} | {a['recall']:.3f} | {a['precision']:.3f} |")
    for condicion, titulo in (("principal", "Jefe al 100%, fase 1"), ("fase2", "Inicio en fase 2"),
                              ("ruido", "Jefe al 100% con ruido de ejecucion")):
        print(f"\n### En juego: {titulo}\n")
        print("\n".join(tabla_juego(base, condicion, rondas)))
    elegido = leer_json(base / "mejor.json")
    evaluaciones = base / "evaluaciones"
    print("\n### Seleccion y confirmacion (jefe al 100%, fase 1)\n")
    print("| Politica | Seleccion (50) | Confirmacion (100, peleas nuevas) | Juntas | IC 95% de las juntas |")
    print("|---|---|---|---|---|")

    def fila_agrupada(etiqueta: str, principal: dict[str, object] | None, confirmacion: dict[str, object] | None) -> None:
        """
        Imprime la fila de una politica con sus dos muestras y la suma.

        Contexto:
            main.

        Args:
            etiqueta (str): Nombre de la fila.
            principal (dict, opcional): Evaluacion principal (con episodios).
            confirmacion (dict, opcional): Confirmacion (con episodios).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if not principal or not confirmacion:
            return
        a, b = principal["resumen"], confirmacion["resumen"]  # type: ignore[index]
        juntos = resumen_de_juego([*principal["episodios"], *confirmacion["episodios"]])  # type: ignore[misc]
        bajo, alto = intervalo_wilson(int(juntos["victorias"]), int(juntos["peleas"]))
        print(f"| {etiqueta} | {int(a['victorias'])}/{int(a['peleas'])} ({porcentaje(a['tasa_victoria'])}) "
              f"| {int(b['victorias'])}/{int(b['peleas'])} ({porcentaje(b['tasa_victoria'])}) "
              f"| {int(juntos['victorias'])}/{int(juntos['peleas'])} ({porcentaje(juntos['tasa_victoria'])}) "
              f"| {porcentaje(bajo)}-{porcentaje(alto)} |")

    fila_agrupada("bot de reglas", leer_json(evaluaciones / "bot.json").get("principal"),
                  leer_json(evaluaciones / "bot_confirmacion.json"))
    for ronda in rondas:
        fila_agrupada(f"red, ronda {ronda}", leer_json(evaluaciones / f"ronda_{ronda:02d}.json").get("principal"),
                      leer_json(evaluaciones / f"confirmacion_ronda_{ronda:02d}.json"))
    if elegido:
        print(f"\nModelo elegido: ronda {elegido['ronda']} (criterio: {elegido['criterio']}).")
        confirmacion = leer_json(evaluaciones / f"confirmacion_ronda_{elegido['ronda']:02d}.json")
        if confirmacion:
            print("\n| Politica | Victorias (tasa, IC 95%) | Llegan a fase 2 | Dano al jefe | Dano recibido | Parries | "
                  "Caguamas usadas / interrumpidas | Muertes f1 / f2 / tiempo | Duracion (s) |")
            print("|---|---|---|---|---|---|---|---|---|")
            print(fila_juego("elegido, solo confirmacion", confirmacion["resumen"]))
            bot = leer_json(evaluaciones / "bot_confirmacion.json")
            if bot:
                print(fila_juego("bot, solo confirmacion", bot["resumen"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
