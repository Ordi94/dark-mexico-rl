"""
Linea de comandos: python -m dmrl <comando>.

Comandos:
    entrenar        inicia o reanuda un entrenamiento MaskablePPO.
    evaluar         juega N peleas con el bot de reglas, un modelo o el azar y reporta metricas.
    ver             una instancia con ventana a velocidad normal para ver pelear a una politica.
    imitar          clonacion de comportamiento del bot de reglas con DAgger (reanudable).
    verificar       comprueba el entorno real con check_env de Stable-Baselines3.
    especificacion  escribe docs/ESPECIFICACION_OBSERVACION.md.
    evaluar-jefes   peleas completas contra un jefe del entorno multi-jefe con el bot de reglas.
    verificar-jefes check_env del entorno multi-jefe contra un jefe real.
    especificacion-v2  escribe docs/ESPECIFICACION_OBSERVACION_V2.md.
    recolectar-jefes   demostraciones del bot de reglas contra uno o varios jefes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from dmrl.aplicacion import entrenamiento as ent
from dmrl.aplicacion import entrenamiento_jefes as entj
from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion import evaluacion_modelo_jefes as evm
from dmrl.aplicacion import imitacion_ciclo as ciclo
from dmrl.aplicacion import imitacion_jefes as imj
from dmrl.aplicacion import panel_vivo as pv
from dmrl.aplicacion import recoleccion_jefes as recj
from dmrl.aplicacion import visualizacion as vis
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import ConfigAprendizaje, ConfigImitacion
from dmrl.dominio.imitacion_jefes import ARQUITECTURAS_CANDIDATAS
from dmrl.dominio.observacion import especificacion_markdown
from dmrl.dominio.observacion_v2 import especificacion_markdown_v2
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from dmrl.infraestructura.datos import CARPETA_DATOS, RAIZ_PROYECTO, cargar_tabla_golpes
from dmrl.infraestructura.demostraciones import escribir_json, leer_json
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import (
    AFINIDAD_ENTRENAMIENTO,
    ParametrosInstancia,
    crear_fabrica_puerto,
)
from dmrl.infraestructura.procesos import ajustar_proceso_actual, escribir_pid
from dmrl.infraestructura.servidor_panel import PUERTO_DEFECTO as PUERTO_DEFECTO_PANEL

ARCHIVO_ESPECIFICACION = RAIZ_PROYECTO / "docs" / "ESPECIFICACION_OBSERVACION.md"
ARCHIVO_ESPECIFICACION_V2 = RAIZ_PROYECTO / "docs" / "ESPECIFICACION_OBSERVACION_V2.md"


def construir_parser() -> argparse.ArgumentParser:
    """
    Define los comandos y sus opciones.

    Contexto:
        main y las pruebas de la linea de comandos.

    Args:
        Ninguno.

    Returns:
        argparse.ArgumentParser: Parser con un subcomando por caso de uso.

    Raises:
        Ninguna.

    Ejemplo:
        construir_parser().parse_args(["evaluar", "--politica", "bot"])
    """
    parser = argparse.ArgumentParser(
        prog="dmrl", description="Entrenamiento por refuerzo contra el jefe de Dark Mexico"
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    e = sub.add_parser("entrenar", help="inicia o reanuda un entrenamiento")
    e.add_argument("--nombre", default="principal")
    e.add_argument("--instancias", type=int, default=4)
    e.add_argument("--pasos", type=int, default=50_000_000, help="techo de pasos de los entornos")
    e.add_argument("--dispositivo", choices=("cpu", "cuda", "auto"), default="auto")
    e.add_argument("--n-steps", type=int, default=1024)
    e.add_argument("--batch-size", type=int, default=512)
    e.add_argument("--ent-coef", type=float, default=0.01)
    e.add_argument("--checkpoint-cada", type=int, default=100_000)
    e.add_argument("--desde-cero", action="store_true", help="ignora el checkpoint existente")
    e.add_argument("--semilla", type=int, default=0)

    v = sub.add_parser("evaluar", help="juega N peleas completas y reporta metricas")
    v.add_argument("--politica", choices=("bot", "modelo", "azar"), default="bot")
    v.add_argument("--modelo", type=Path, default=None,
                   help="archivo .zip (por defecto runs/<nombre>/mejor.zip)")
    v.add_argument("--nombre", default="principal")
    v.add_argument("--episodios", type=int, default=50)
    v.add_argument("--instancias", type=int, default=4)
    v.add_argument("--semilla", type=int, default=0)
    v.add_argument("--salida", type=Path, default=None, help="JSON con los episodios y el resumen")
    v.add_argument("--epsilon-ruido", type=float, default=0.0,
                   help="probabilidad por paso de ejecutar una accion valida al azar (prueba de robustez)")

    r = sub.add_parser("ver", help="una ventana del juego a velocidad normal")
    r.add_argument("--politica", choices=("bot", "modelo"), default="modelo")
    r.add_argument("--modelo", type=Path, default=None)
    r.add_argument("--nombre", default="principal")
    r.add_argument("--episodios", type=int, default=1)
    r.add_argument("--max-segundos", type=float, default=None,
                   help="tiempo real maximo (para comprobar que abre y cierra)")

    i = sub.add_parser("imitar", help="clonacion de comportamiento del bot con DAgger (reanudable)")
    i.add_argument("--nombre", default="imitacion")
    i.add_argument("--instancias", type=int, default=4)
    i.add_argument("--episodios-inicial", type=int, default=600, help="peleas del bot en la ronda 0")
    i.add_argument("--episodios-dagger", type=int, default=300, help="peleas por ronda de DAgger")
    i.add_argument("--rondas", type=int, default=3, help="rondas de DAgger")
    i.add_argument("--episodios-evaluacion", type=int, default=50, help="peleas de la evaluacion principal por ronda")
    i.add_argument("--episodios-secundarios", type=int, default=20,
                   help="peleas de cada evaluacion de robustez (inicio en fase 2 y con ruido)")
    i.add_argument("--episodios-confirmacion", type=int, default=100, help="peleas nuevas por candidata y para el bot")
    i.add_argument("--candidatas", type=int, default=2, help="rondas que se confirman (las mejores en juego)")
    i.add_argument("--semilla", type=int, default=0)
    i.add_argument("--accion", choices=("todo", "entrenar"), default="todo",
                   help="todo = ciclo completo; entrenar = solo reajustar la ronda --ronda con los datos existentes")
    i.add_argument("--ronda", type=int, default=0)

    sub.add_parser("verificar", help="check_env de Stable-Baselines3 con el juego real")
    sub.add_parser("especificacion", help="escribe docs/ESPECIFICACION_OBSERVACION.md")

    j = sub.add_parser("evaluar-jefes", help="peleas completas contra un jefe con el bot de reglas")
    j.add_argument("--jefe", required=True, help="juice, amlo, chamuco o nahual")
    j.add_argument("--episodios", type=int, default=50)
    j.add_argument("--instancias", type=int, default=4)
    j.add_argument("--semilla", type=int, default=0)
    j.add_argument("--fase-inicio", type=int, choices=(1, 2), default=None,
                   help="fuerza la fase de inicio (jefes que la admiten)")
    j.add_argument("--epsilon-ruido", type=float, default=0.0)
    j.add_argument("--salida", type=Path, default=None, help="JSON con los episodios y el resumen")

    vj = sub.add_parser("verificar-jefes", help="check_env del entorno multi-jefe con el juego real")
    vj.add_argument("--jefe", required=True)

    sub.add_parser("especificacion-v2", help="escribe docs/ESPECIFICACION_OBSERVACION_V2.md")

    rj = sub.add_parser("recolectar-jefes", help="demostraciones del bot contra uno o varios jefes")
    rj.add_argument("--jefes", nargs="+", required=True)
    rj.add_argument("--episodios", type=int, default=5, help="peleas por jefe")
    rj.add_argument("--instancias", type=int, default=4)
    rj.add_argument("--nombre", default="imitacion_v2")
    rj.add_argument("--ronda", type=int, default=0)
    rj.add_argument("--semilla", type=int, default=0)
    rj.add_argument("--max-minutos", type=float, default=None,
                    help="tope de tiempo real; pasado el plazo no se inician peleas nuevas")

    ij = sub.add_parser("imitar-jefes", help="ajuste supervisado multi-jefe con las demostraciones recolectadas")
    ij.add_argument("--nombre", default="multijefe_imitacion")
    ij.add_argument("--accion", choices=("resumen", "dispositivos", "entrenar", "publicar"), default="resumen")
    ij.add_argument("--arquitectura", choices=sorted(ARQUITECTURAS_CANDIDATAS), default="2x256")
    ij.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto",
                    help="auto = el mas rapido segun la medicion de la accion dispositivos")
    ij.add_argument("--semilla", type=int, default=0)
    ij.add_argument("--evaluacion", default=None,
                    help="publicar: evaluaciones/<nombre>.json con las tasas de victoria que fijan la referencia")

    em = sub.add_parser("evaluar-modelo-jefes", help="peleas completas de un modelo (o del bot) contra varios jefes")
    em.add_argument("--modelo", type=Path, default=None)
    em.add_argument("--bot", action="store_true", help="juega el bot de reglas en lugar de un modelo")
    em.add_argument("--jefes", nargs="+", default=list(evm.JEFES_EVALUADOS))
    em.add_argument("--episodios", type=int, default=20, help="peleas por jefe")
    em.add_argument("--instancias", type=int, default=4)
    em.add_argument("--semilla", type=int, default=0)
    em.add_argument("--prefijo", default="ev", help="prefijo de los ids de instancia del juego")
    em.add_argument("--epsilon-ruido", type=float, default=0.0)
    em.add_argument("--salida", type=Path, default=None, help="JSON con las tasas por jefe y los episodios")

    ej = sub.add_parser("entrenar-jefes", help="PPO multi-jefe partiendo de la politica clonada (reanudable)")
    ej.add_argument("--nombre", default="multijefe_ppo")
    ej.add_argument("--instancias", type=int, default=6)
    ej.add_argument("--pasos", type=int, default=200_000_000)
    ej.add_argument("--learning-rate", type=float, default=1e-4)
    ej.add_argument("--minutos-evaluacion", type=float, default=45.0)
    ej.add_argument("--minutos-primera-evaluacion", type=float, default=20.0)
    ej.add_argument("--episodios-evaluacion", type=int, default=20)
    ej.add_argument("--beta-kl", type=float, default=0.3)
    ej.add_argument("--pasos-calentamiento", type=int, default=200_000)
    ej.add_argument("--imitacion", type=Path, default=None, help="modelo clonado de partida (por defecto mejor.zip)")
    ej.add_argument("--semilla", type=int, default=0)

    vw = sub.add_parser("ver-jefes", help="una ventana del juego para ver pelear a un modelo contra un jefe")
    vw.add_argument("--jefe", required=True, choices=list(evm.JEFES_EVALUADOS))
    vw.add_argument("--modelo", type=Path, default=None)
    vw.add_argument("--nombre", default="multijefe_ppo")
    vw.add_argument("--episodios", type=int, default=1)
    vw.add_argument("--max-segundos", type=float, default=None)
    vw.add_argument("--panel", action="store_true", help="panel web en vivo con las neuronas de la red")
    vw.add_argument("--panel-puerto", type=int, default=PUERTO_DEFECTO_PANEL)
    vw.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador con el panel")
    return parser


def ruta_modelo_por_defecto(nombre: str) -> Path:
    """
    Elige el modelo a usar: mejor.zip si existe y si no el ultimo checkpoint.

    Contexto:
        evaluar y ver sin --modelo.

    Args:
        nombre (str): Nombre de la corrida.

    Returns:
        Path: Archivo del modelo.

    Raises:
        FileNotFoundError: Si la corrida no tiene ningun modelo guardado.

    Ejemplo:
        ruta_modelo_por_defecto("principal")
    """
    base = ent.CARPETA_RUNS / nombre
    for candidato in (base / "mejor.zip", base / "checkpoints" / "ultimo.zip"):
        if candidato.exists():
            return candidato
    raise FileNotFoundError(f"la corrida {nombre!r} no tiene modelo guardado en {base}")


def comando_entrenar(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando entrenar.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar de forma ordenada.

    Raises:
        Exception: Errores del entrenamiento, tras guardar y apagar.

    Ejemplo:
        comando_entrenar(args)
    """
    config = ent.ConfigEntrenamiento(
        nombre=args.nombre, instancias=args.instancias, pasos_totales=args.pasos, dispositivo=args.dispositivo,
        n_steps=args.n_steps, batch_size=args.batch_size, ent_coef=args.ent_coef,
        checkpoint_cada=args.checkpoint_cada, reanudar=not args.desde_cero, semilla=args.semilla,
    )
    ultimo = ent.entrenar(config)
    print(f"entrenamiento terminado; ultimo modelo en {ultimo}")
    return 0


def comando_evaluar(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando evaluar.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        FileNotFoundError: Si se pide un modelo y no hay ninguno.

    Ejemplo:
        comando_evaluar(args)
    """
    modelo = None
    if args.politica == "modelo":
        modelo = args.modelo or ruta_modelo_por_defecto(args.nombre)
    extra = {"epsilon_ruido": args.epsilon_ruido} if args.epsilon_ruido > 0 else {}
    episodios = evl.evaluar_en_paralelo(args.politica, args.episodios, args.instancias, modelo, args.semilla, **extra)
    resumen = evl.resumir(episodios)
    print(json.dumps(resumen, indent=1))
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps({"resumen": resumen, "episodios": episodios}, indent=1), encoding="utf-8")
    return 0


def comando_ver(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando ver.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        FileNotFoundError: Si se pide un modelo y no hay ninguno.

    Ejemplo:
        comando_ver(args)
    """
    tabla = cargar_tabla_golpes()
    modelo = None
    if args.politica == "modelo":
        modelo = str(args.modelo or ruta_modelo_por_defecto(args.nombre))
    politica = evl.construir_politica(args.politica, tabla, modelo, 0)
    env = vis.crear_entorno_ventana(tabla)
    try:
        resumenes = vis.ver_pelea(env, politica, args.episodios, args.max_segundos)
    finally:
        env.close()
    print(json.dumps(evl.resumir(resumenes), indent=1))
    return 0


def comando_imitar(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando imitar.

    Contexto:
        main. Registra el PID en runs/entrenamiento.pid para que detener.ps1 pueda pedir la
        parada y entrenar.ps1 no compita por las mismas CPUs.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 si el ciclo termino, 2 si se interrumpio por una parada.

    Raises:
        Exception: Errores del ciclo, tras limpiar el archivo de PID.

    Ejemplo:
        comando_imitar(args)
    """
    config = ConfigImitacion(
        nombre=args.nombre, instancias=args.instancias, episodios_ronda0=args.episodios_inicial,
        episodios_dagger=args.episodios_dagger, rondas_dagger=args.rondas, semilla=args.semilla,
        episodios_evaluacion=args.episodios_evaluacion, episodios_fase2=args.episodios_secundarios,
        episodios_ruido=args.episodios_secundarios, episodios_confirmacion=args.episodios_confirmacion,
        candidatas_confirmacion=args.candidatas,
    )
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    escribir_pid(ent.ARCHIVO_PID)
    rutas = ciclo.rutas_de(config.nombre)
    rutas.parada.unlink(missing_ok=True)
    try:
        if args.accion == "entrenar":
            ciclo.entrenar_ronda(config, rutas, args.ronda, ciclo.elegir_dispositivo_ajuste())
            return 0
        estado = ciclo.ejecutar_ciclo(config)
    finally:
        rutas.parada.unlink(missing_ok=True)
        ent.ARCHIVO_PID.unlink(missing_ok=True)
    print(json.dumps(estado))
    return 0 if estado["estado"] == "completo" else 2


def comando_verificar(args: argparse.Namespace) -> int:
    """
    Ejecuta check_env de SB3 contra el entorno real.

    Contexto:
        main. Es la validacion (a): confirma espacios, tipos, rangos y reset/step reales.

    Args:
        args (argparse.Namespace): Opciones del comando (sin uso).

    Returns:
        int: 0 si check_env pasa.

    Raises:
        AssertionError: Si el entorno no cumple la API.

    Ejemplo:
        comando_verificar(args)
    """
    from stable_baselines3.common.env_checker import check_env

    from dmrl.aplicacion.entorno import DarkMexicoEnv

    fabrica = crear_fabrica_puerto(ParametrosInstancia(id_instancia="chk"))
    env = DarkMexicoEnv(fabrica, cargar_tabla_golpes(), ConfigEntorno())
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()
    print("check_env: OK")
    return 0


def comando_especificacion(args: argparse.Namespace) -> int:
    """
    Escribe la especificacion de la observacion.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando (sin uso).

    Returns:
        int: 0 al terminar.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        comando_especificacion(args)
    """
    ARCHIVO_ESPECIFICACION.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_ESPECIFICACION.write_text(especificacion_markdown(ConfigEntorno().apilado), encoding="utf-8")
    print(f"escrito {ARCHIVO_ESPECIFICACION} (datos en {CARPETA_DATOS})")
    return 0


def comando_evaluar_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando evaluar-jefes.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        KeyError: Si el jefe no existe en el catalogo.

    Ejemplo:
        comando_evaluar_jefes(args)
    """
    cargar_catalogo().perfil(args.jefe)
    opciones = {"start_phase": args.fase_inicio} if args.fase_inicio else {}
    episodios = evj.evaluar_jefe_en_paralelo(
        args.jefe, args.episodios, args.instancias, args.semilla, opciones, epsilon_ruido=args.epsilon_ruido)
    resumen = evj.resumir_jefe(episodios)
    print(json.dumps(resumen, indent=1))
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps({"resumen": resumen, "episodios": episodios}, indent=1), encoding="utf-8")
    return 0


def comando_verificar_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta check_env de SB3 contra el entorno multi-jefe real.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 si check_env pasa.

    Raises:
        AssertionError: Si el entorno no cumple la API.

    Ejemplo:
        comando_verificar_jefes(args)
    """
    from stable_baselines3.common.env_checker import check_env

    catalogo = cargar_catalogo()
    env = evj.crear_entorno_jefes(catalogo, (args.jefe,), "chk", ConfigEntorno(), trazar=False)
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()
    print(f"check_env ({args.jefe}): OK")
    return 0


def comando_especificacion_v2(args: argparse.Namespace) -> int:
    """
    Escribe la especificacion de la observacion v2.

    Contexto:
        main.

    Args:
        args (argparse.Namespace): Opciones del comando (sin uso).

    Returns:
        int: 0 al terminar.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        comando_especificacion_v2(args)
    """
    ARCHIVO_ESPECIFICACION_V2.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_ESPECIFICACION_V2.write_text(
        especificacion_markdown_v2(cargar_catalogo(), ConfigEntorno().apilado), encoding="utf-8")
    print(f"escrito {ARCHIVO_ESPECIFICACION_V2}")
    return 0


def comando_recolectar_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando recolectar-jefes.

    Contexto:
        main. Registra el PID para que detener.ps1 pueda pedir la parada.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 si no queda ningun episodio sin guardar, 2 si faltan.

    Raises:
        KeyError: Si algun jefe no existe en el catalogo.

    Ejemplo:
        comando_recolectar_jefes(args)
    """
    catalogo = cargar_catalogo()
    for jefe in args.jefes:
        catalogo.perfil(jefe)
    carpeta = ent.CARPETA_RUNS / args.nombre / "datos"
    parada = ent.CARPETA_RUNS / args.nombre / "PARAR"
    parada.unlink(missing_ok=True)
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    escribir_pid(ent.ARCHIVO_PID)
    limite = args.max_minutos * 60.0 if args.max_minutos is not None else None
    try:
        faltan = recj.recolectar_jefes(carpeta, args.ronda, args.jefes, args.episodios, args.instancias,
                                       args.semilla, archivo_parada=parada, limite_segundos=limite)
    finally:
        parada.unlink(missing_ok=True)
        ent.ARCHIVO_PID.unlink(missing_ok=True)
    resumen = recj.resumen_de_ronda(carpeta, args.ronda)
    print(json.dumps({"carpeta": str(carpeta), "faltan": faltan, "resumen": resumen}))
    return 0 if faltan == 0 else 2


def comando_imitar_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando imitar-jefes.

    Contexto:
        main. Cuatro acciones sobre las demostraciones multi-jefe ya recolectadas: resumen de datos
        por jefe, medicion CPU contra GPU, ajuste de una arquitectura y publicacion del modelo elegido.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        ValueError: Si falta el dato que pide la accion (por ejemplo --evaluacion en publicar).

    Ejemplo:
        comando_imitar_jefes(args)
    """
    rutas = imj.RutasImitacionJefes(ent.CARPETA_RUNS / args.nombre)
    if args.accion == "resumen":
        print(json.dumps(imj.datos_por_jefe_en_disco(rutas.datos), indent=1))
        return 0
    if args.accion == "publicar":
        if not args.evaluacion:
            raise ValueError("publicar necesita --evaluacion con el nombre de la evaluacion en juego")
        tasas = imj.publicar_modelo(rutas, args.arquitectura, leer_json(rutas.evaluacion(args.evaluacion)))
        print(json.dumps({"mejor": str(rutas.mejor), "referencia": tasas}))
        return 0
    catalogo = cargar_catalogo()
    conjunto = imj.cargar_conjunto_jefes(rutas.datos)
    aprendizaje = ConfigAprendizaje()
    if args.accion == "dispositivos":
        tiempos = imj.medir_dispositivos(catalogo, args.arquitectura, conjunto.demostraciones, aprendizaje)
        escribir_json(rutas.base / "dispositivos.json",
                      {"arquitectura": args.arquitectura, "segundos_por_epoca": tiempos})
        print(json.dumps({"segundos_por_epoca": tiempos}))
        return 0
    if args.dispositivo == "auto":
        tiempos = imj.medir_dispositivos(catalogo, args.arquitectura, conjunto.demostraciones, aprendizaje)
        dispositivo = imj.elegir_dispositivo_mas_rapido(tiempos)
    else:
        dispositivo = torch.device(args.dispositivo)
    informe = imj.entrenar_arquitectura(catalogo, conjunto, args.arquitectura, aprendizaje, args.semilla,
                                        dispositivo, rutas, registrar=lambda m: print(m, flush=True))
    print(json.dumps({"arquitectura": args.arquitectura, "validacion": informe["validacion"],
                      "segundos": informe["segundos"]}))
    return 0


def comando_evaluar_modelo_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando evaluar-modelo-jefes.

    Contexto:
        main, la evaluacion automatica del PPO y la comparacion de arquitecturas.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        ValueError: Si no se da ni --modelo ni --bot, o se dan los dos.
        KeyError: Si algun jefe no existe en el catalogo.

    Ejemplo:
        comando_evaluar_modelo_jefes(args)
    """
    if args.bot == (args.modelo is not None):
        raise ValueError("indica --modelo o --bot, no ambos ni ninguno")
    catalogo = cargar_catalogo()
    for jefe in args.jefes:
        catalogo.perfil(jefe)
    episodios = evm.evaluar_en_jefes(
        None if args.bot else str(args.modelo), args.jefes, args.episodios, args.instancias, args.semilla,
        prefijo_id=args.prefijo, epsilon_ruido=args.epsilon_ruido, catalogo=catalogo)
    por_jefe = evm.resumir_por_jefe(episodios)
    tasas = evm.tasas_por_jefe(por_jefe)
    media = sum(tasas.values()) / len(tasas) if tasas else 0.0
    salida = {"modelo": None if args.bot else str(args.modelo), "tasas": tasas, "tasa_media": media,
              "por_jefe": por_jefe, "episodios": episodios}
    print(json.dumps({"tasas": tasas, "tasa_media": media}))
    if args.salida:
        escribir_json(args.salida, salida)
    return 0


def comando_entrenar_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando entrenar-jefes.

    Contexto:
        main y entrenar.ps1 -Multijefe.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar de forma ordenada.

    Raises:
        FileNotFoundError: Si falta el modelo clonado o la referencia de la imitacion.
        Exception: Errores del entrenamiento, tras guardar y apagar.

    Ejemplo:
        comando_entrenar_jefes(args)
    """
    config = ConfigPPOJefes(
        nombre=args.nombre, instancias=args.instancias, pasos_totales=args.pasos, learning_rate=args.learning_rate,
        minutos_evaluacion=args.minutos_evaluacion, minutos_primera_evaluacion=args.minutos_primera_evaluacion,
        episodios_evaluacion=args.episodios_evaluacion,
        beta_kl=args.beta_kl, pasos_calentamiento=args.pasos_calentamiento, semilla=args.semilla)
    ultimo = entj.entrenar_jefes(config, ruta_imitacion=args.imitacion)
    print(f"entrenamiento terminado; ultimo modelo en {ultimo}")
    return 0


def comando_ver_jefes(args: argparse.Namespace) -> int:
    """
    Ejecuta el comando ver-jefes.

    Contexto:
        main y ver_pelea.ps1 -Multijefe.

    Args:
        args (argparse.Namespace): Opciones del comando.

    Returns:
        int: 0 al terminar.

    Raises:
        FileNotFoundError: Si no hay modelo guardado para la corrida.

    Ejemplo:
        comando_ver_jefes(args)
    """
    from sb3_contrib import MaskablePPO

    modelo = args.modelo or entj.rutas_ppo(args.nombre).mejor
    if not Path(modelo).exists():
        raise FileNotFoundError(f"no existe el modelo {modelo}")
    catalogo = cargar_catalogo()
    cargado = MaskablePPO.load(str(modelo), device="cpu")
    politica: evl.Politica = evl.PoliticaModelo(cargado)
    sesion = None
    extras: dict[str, Any] = {}
    if args.panel:
        # El panel se abre antes que el juego: un puerto ocupado o un modelo que no es v2 fallan
        # sin lanzar la ventana, y el navegador ya esta esperando cuando empieza la pelea.
        sesion = pv.iniciar_sesion_panel(cargado, politica, catalogo, args.jefe, Path(modelo).parent.name,
                                         args.panel_puerto, not args.sin_navegador)
        politica, extras = sesion.politica, {"al_terminar": sesion.al_terminar}
        print(f"panel en {sesion.url}", flush=True)
    try:
        env = evj.crear_entorno_jefes(catalogo, (args.jefe,), "ver", ConfigEntorno(), trazar=False,
                                      params=vis.PARAMETROS_VENTANA)
        try:
            opciones = {"jefe": args.jefe, "start_phase": catalogo.perfil(args.jefe).fases_inicio[0],
                        "boss_hp_mult": 1.0}
            resumenes = vis.ver_pelea(env, politica, args.episodios, args.max_segundos, opciones, **extras)
        finally:
            env.close()
    finally:
        if sesion is not None:
            sesion.cerrar()
    print(json.dumps(evl.resumir(resumenes), indent=1))
    return 0

COMANDOS = {
    "entrenar": comando_entrenar, "evaluar": comando_evaluar, "ver": comando_ver, "imitar": comando_imitar,
    "verificar": comando_verificar, "especificacion": comando_especificacion,
    "evaluar-jefes": comando_evaluar_jefes, "verificar-jefes": comando_verificar_jefes,
    "especificacion-v2": comando_especificacion_v2, "recolectar-jefes": comando_recolectar_jefes,
    "imitar-jefes": comando_imitar_jefes, "evaluar-modelo-jefes": comando_evaluar_modelo_jefes,
    "entrenar-jefes": comando_entrenar_jefes, "ver-jefes": comando_ver_jefes,
}


def main(argv: Sequence[str] | None = None) -> int:
    """
    Punto de entrada de la linea de comandos.

    Contexto:
        python -m dmrl y los scripts .ps1.

    Args:
        argv (Sequence[str], opcional): Argumentos; por defecto sys.argv[1:].

    Returns:
        int: Codigo de salida.

    Raises:
        SystemExit: Si los argumentos no son validos (argparse).

    Ejemplo:
        main(["especificacion"])
    """
    args = construir_parser().parse_args(list(argv) if argv is not None else None)
    return COMANDOS[args.comando](args)

