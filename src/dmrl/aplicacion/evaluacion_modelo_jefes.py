"""
Evaluacion en juego de una politica (modelo MaskablePPO o bot de reglas) contra varios jefes.

Cada instancia del juego queda fija a un mapa, asi que las peleas se reparten por mapa (al menos una
instancia por mapa) y dentro de un mapa se intercalan los jefes. Todas las peleas son completas: jefe
al 100% de vida y la fase de inicio normal de cada jefe (la primera que admite). La politica del
modelo es determinista (la accion de mayor probabilidad entre las validas), igual que al jugar de
verdad.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

from dmrl.aplicacion.evaluacion_jefes import crear_entorno_jefes, jugar_episodio_bot, resumir_jefe
from dmrl.aplicacion.recoleccion_jefes import PlanJefe, intercalar_por_jefe, repartir_por_mapa
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import PlanEpisodio, resumen_de_juego, ruido_dart
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO
from dmrl.infraestructura.procesos import ajustar_proceso_actual

JEFES_EVALUADOS = ("juice", "amlo", "chamuco", "nahual")


@dataclass(frozen=True)
class TrabajoEvaluacionModelo:
    """
    Encargo para un proceso de evaluacion (debe poder viajar a otro proceso).

    Contexto:
        evaluar_en_jefes. Todas las peleas de un trabajo son del mismo mapa.

    Atributos:
        indice: numero del trabajador; parte del id de su instancia del juego.
        planes: peleas que le toca jugar (jefe, fase de inicio y semilla).
        ruta_modelo: archivo .zip del modelo, o None para jugar con el bot de reglas.
        prefijo_id: prefijo del id de instancia (distinto del de entrenamiento para no chocar).
        config: configuracion del entorno (la misma con que se recolectaron las demostraciones).
        epsilon_ruido: probabilidad por paso de ejecutar una accion valida al azar.
        recarga_cada: episodios entre recargas completas del mapa.
    """

    indice: int
    planes: tuple[PlanJefe, ...]
    ruta_modelo: str | None = None
    prefijo_id: str = "ev"
    config: ConfigEntorno = field(default_factory=ConfigEntorno)
    epsilon_ruido: float = 0.0
    recarga_cada: int = 150


def planificar_evaluacion(catalogo: CatalogoJefes, jefes: Sequence[str], por_jefe: int,
                          semilla: int = 0) -> list[PlanJefe]:
    """
    Peleas completas de la evaluacion: por_jefe contra cada jefe, en su fase de inicio normal.

    Contexto:
        evaluar_en_jefes. Sin sorteo de fase: la comparacion entre modelos, y contra el bot, debe
        hacerse en las mismas condiciones.

    Args:
        catalogo (CatalogoJefes): Jefes disponibles.
        jefes (Sequence[str]): Jefes de episodio a evaluar.
        por_jefe (int): Peleas por jefe.
        semilla (int): Semilla base del ruido de cada pelea.

    Returns:
        list[PlanJefe]: Peleas con indices unicos, jefe por jefe.

    Raises:
        KeyError: Si algun jefe no existe.
        ValueError: Si por_jefe es negativo.

    Ejemplo:
        planificar_evaluacion(catalogo, ("amlo", "nahual"), 20)
    """
    if por_jefe < 0:
        raise ValueError("por_jefe no puede ser negativo")
    planes: list[PlanJefe] = []
    for orden, jefe in enumerate(jefes):
        fase = catalogo.perfil(jefe).fases_inicio[0]
        planes += [PlanJefe(PlanEpisodio(orden * por_jefe + i, fase, 0.0, semilla + orden * por_jefe + i), jefe)
                   for i in range(por_jefe)]
    return planes


def jugar_episodio_modelo(env: Any, modelo: Any, opciones: dict[str, Any], epsilon: float = 0.0,
                          azar: random.Random | None = None) -> dict[str, Any]:
    """
    Juega un episodio completo con un modelo MaskablePPO determinista.

    Contexto:
        Trabajadores de evaluacion y evaluacion periodica del entrenamiento.

    Args:
        env (EntornoMultiJefe): Entorno ya creado.
        modelo (Any): MaskablePPO con predict(obs, action_masks=..., deterministic=...).
        opciones (dict[str, Any]): Opciones de reset (jefe, start_phase, boss_hp_mult).
        epsilon (float): Probabilidad por paso de ejecutar una accion valida al azar.
        azar (random.Random, opcional): Generador del ruido.

    Returns:
        dict[str, Any]: Resumen del episodio (el "episodio" de la info del entorno).

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.
        ValueError: Si epsilon esta fuera de [0, 1].

    Ejemplo:
        resumen = jugar_episodio_modelo(env, modelo, {"jefe": "amlo"})
    """
    generador = azar or random.Random()
    obs, _ = env.reset(options=opciones)
    while True:
        mascara = env.mascara_lista()
        accion, _ = modelo.predict(obs, action_masks=env.action_masks(), deterministic=True)
        ejecutada = ruido_dart(int(accion), mascara, epsilon, generador)
        obs, _recompensa, terminado, truncado, info = env.step(ejecutada)
        if terminado or truncado:
            resumen: dict[str, Any] = info["episodio"]
            return resumen


def ejecutar_trabajo_modelo(trabajo: TrabajoEvaluacionModelo) -> list[dict[str, Any]]:
    """
    Juega las peleas de un trabajador con su propia instancia del juego.

    Contexto:
        Funcion de nivel superior ejecutada en cada proceso de evaluar_en_jefes.

    Args:
        trabajo (TrabajoEvaluacionModelo): Encargo del trabajador.

    Returns:
        list[dict[str, Any]]: Resumenes de sus episodios.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_trabajo_modelo(TrabajoEvaluacionModelo(0, (plan,), "runs/multijefe_imitacion/mejor.zip"))
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    catalogo = cargar_catalogo()
    jefes = tuple(dict.fromkeys(p.jefe for p in trabajo.planes))
    env = crear_entorno_jefes(catalogo, jefes, f"{trabajo.prefijo_id}{trabajo.indice}", trabajo.config,
                              semilla=trabajo.indice, trazar=True, recarga_cada=trabajo.recarga_cada)
    try:
        modelo = None
        if trabajo.ruta_modelo is not None:
            import torch
            from sb3_contrib import MaskablePPO

            torch.set_num_threads(1)
            modelo = MaskablePPO.load(trabajo.ruta_modelo, device="cpu")
        bot = BotJefes.desde_catalogo(catalogo)
        resumenes: list[dict[str, Any]] = []
        for plan in trabajo.planes:
            opciones = {"jefe": plan.jefe, "start_phase": plan.plan.fase_inicio, "boss_hp_mult": 1.0}
            azar = random.Random(plan.plan.semilla)
            if modelo is None:
                resumenes.append(jugar_episodio_bot(env, bot, opciones, trabajo.epsilon_ruido, azar))
            else:
                resumenes.append(jugar_episodio_modelo(env, modelo, opciones, trabajo.epsilon_ruido, azar))
        return resumenes
    finally:
        env.close()


def evaluar_en_jefes(ruta_modelo: str | None, jefes: Sequence[str], por_jefe: int, instancias: int,
                     semilla: int = 0, config: ConfigEntorno | None = None, prefijo_id: str = "ev",
                     ejecutor: Executor | None = None, epsilon_ruido: float = 0.0,
                     recarga_cada: int = 150, catalogo: CatalogoJefes | None = None) -> list[dict[str, Any]]:
    """
    Juega las peleas de la evaluacion repartidas entre instancias del juego.

    Contexto:
        Comando evaluar-modelo-jefes, seleccion de la arquitectura y evaluacion periodica del PPO.

    Args:
        ruta_modelo (str, opcional): Modelo a evaluar; None usa el bot de reglas.
        jefes (Sequence[str]): Jefes de episodio.
        por_jefe (int): Peleas por jefe.
        instancias (int): Instancias del juego en paralelo (al menos una por mapa).
        semilla (int): Semilla base.
        config (ConfigEntorno, opcional): Configuracion del entorno.
        prefijo_id (str): Prefijo de los ids de instancia.
        ejecutor (Executor, opcional): Ejecutor a usar (por defecto procesos con spawn).
        epsilon_ruido (float): Ruido de ejecucion por paso (evaluacion de robustez).
        recarga_cada (int): Episodios entre recargas completas del mapa.
        catalogo (CatalogoJefes, opcional): Catalogo; por defecto el de data/jefes.

    Returns:
        list[dict[str, Any]]: Resumenes de todos los episodios.

    Raises:
        KeyError: Si algun jefe no existe.
        ValueError: Si hay menos instancias que mapas.

    Ejemplo:
        episodios = evaluar_en_jefes("runs/multijefe_imitacion/mejor.zip", ("amlo", "nahual"), 20, 4)
    """
    catalogo = catalogo or cargar_catalogo()
    planes = intercalar_por_jefe(planificar_evaluacion(catalogo, jefes, por_jefe, semilla))
    grupos = repartir_por_mapa(planes, instancias, catalogo)
    trabajos = [
        TrabajoEvaluacionModelo(indice=i, planes=grupo, ruta_modelo=ruta_modelo, prefijo_id=prefijo_id,
                                config=config or ConfigEntorno(), epsilon_ruido=epsilon_ruido,
                                recarga_cada=recarga_cada)
        for i, grupo in enumerate(grupos)
    ]
    propio = ejecutor is None
    pool = ejecutor or ProcessPoolExecutor(max_workers=max(1, len(trabajos)), mp_context=get_context("spawn"))
    try:
        resultados = list(pool.map(ejecutar_trabajo_modelo, trabajos))
    finally:
        if propio:
            pool.shutdown(wait=True, cancel_futures=True)
    return [resumen for lista in resultados for resumen in lista]


def resumir_por_jefe(episodios: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Resumen de juego de cada jefe y su tasa de victoria.

    Contexto:
        Salida de evaluar-modelo-jefes y entrada de la compuerta de promocion del PPO.

    Args:
        episodios (Sequence[dict[str, Any]]): Resumenes de todos los episodios.

    Returns:
        dict[str, dict[str, Any]]: jefe -> "juego" (resumen_de_juego con intervalo), "detalle"
            (resumir_jefe con muertes por ataque) y "tasa" (victorias entre peleas validas).

    Raises:
        Ninguna.

    Ejemplo:
        resumir_por_jefe(episodios)["amlo"]["tasa"]
    """
    por_jefe: dict[str, list[dict[str, Any]]] = {}
    for episodio in episodios:
        por_jefe.setdefault(str(episodio.get("jefe", "?")), []).append(episodio)
    salida: dict[str, dict[str, Any]] = {}
    for jefe, lista in sorted(por_jefe.items()):
        juego = resumen_de_juego(lista)
        salida[jefe] = {"juego": juego, "detalle": resumir_jefe(lista), "tasa": float(juego.get("tasa_victoria", 0.0))}
    return salida


def tasas_por_jefe(resumen: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    Tasa de victoria de cada jefe.

    Contexto:
        Compuerta de promocion del PPO.

    Args:
        resumen (dict[str, dict[str, Any]]): Salida de resumir_por_jefe.

    Returns:
        dict[str, float]: jefe -> tasa de victoria.

    Raises:
        Ninguna.

    Ejemplo:
        tasas_por_jefe(resumir_por_jefe(episodios))  # {"amlo": 0.9, ...}
    """
    return {jefe: float(datos["tasa"]) for jefe, datos in resumen.items()}
