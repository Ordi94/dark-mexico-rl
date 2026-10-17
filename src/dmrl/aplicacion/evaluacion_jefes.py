"""
Evaluacion por jefe: juega N peleas completas contra un jefe con el bot de reglas (o el azar) y
resume tasa de victoria, dano, parries y de que ataque muere el jugador.

Mismo esquema que la evaluacion v1 (evaluacion.py): las peleas se reparten entre instancias del
juego, cada una en su proceso, pero el entorno es el multi-jefe, el bot es BotJefes (con las marcas de
impacto atendido confirmadas por lo que de verdad se ejecuto) y el resumen agrega las muertes por
tipo de ataque.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any

from dmrl.aplicacion.entorno_jefes import EntornoMultiJefe
from dmrl.aplicacion.evaluacion import repartir, resumir
from dmrl.dominio import acciones as acc
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import ruido_dart
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.seleccion_jefes import ConfigSeleccion
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import (
    AFINIDAD_ENTRENAMIENTO,
    ParametrosInstancia,
    crear_fabrica_puerto,
)
from dmrl.infraestructura.procesos import ajustar_proceso_actual

Registrador = Callable[[Any, list[bool], int, int, float], None]


def jugar_episodio_bot(env: EntornoMultiJefe, bot: BotJefes, opciones: dict[str, Any], epsilon: float = 0.0,
                       azar: random.Random | None = None, registrar: Registrador | None = None) -> dict[str, Any]:
    """
    Juega un episodio con el bot y, opcionalmente, registra cada decision.

    Contexto:
        Evaluacion por jefe y recoleccion de demostraciones: ambas necesitan el mismo bucle.

    Args:
        env (EntornoMultiJefe): Entorno ya creado.
        bot (BotJefes): Experto.
        opciones (dict[str, Any]): Opciones de reset (jefe, start_phase, boss_hp_mult, reload).
        epsilon (float): Probabilidad por paso de ejecutar una accion valida al azar (ruido DART);
            la etiqueta del bot sigue siendo su propia decision.
        azar (random.Random, opcional): Generador del ruido.
        registrar (Registrador, opcional): Se llama con (observacion apilada, mascara, etiqueta,
            ejecutada, recompensa) en cada paso.

    Returns:
        dict[str, Any]: Resumen del episodio (el "episodio" de la info del entorno).

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.
        ValueError: Si epsilon esta fuera de [0, 1].

    Ejemplo:
        resumen = jugar_episodio_bot(env, bot, {"jefe": "amlo"})
    """
    generador = azar or random.Random()
    obs, _ = env.reset(options=opciones)
    bot.reiniciar()
    while True:
        mascara = env.mascara_lista()
        etiqueta = bot.decidir(env.observacion_cruda, mascara, env.amenazas)
        if not mascara[etiqueta]:
            etiqueta = acc.ACCION_NADA
        ejecutada = ruido_dart(etiqueta, mascara, epsilon, generador)
        siguiente, recompensa, terminado, truncado, info = env.step(ejecutada)
        bot.confirmar(env.accion_efectiva(ejecutada, info))
        if registrar is not None:
            registrar(obs, mascara, etiqueta, ejecutada, recompensa)
        obs = siguiente
        if terminado or truncado:
            resumen: dict[str, Any] = info["episodio"]
            return resumen


def resumir_jefe(episodios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Metricas de una evaluacion contra un jefe.

    Contexto:
        Salida de evaluar_jefe_en_paralelo y de las validaciones.

    Args:
        episodios (Sequence[dict[str, Any]]): Resumenes de jugar_episodio_bot.

    Returns:
        dict[str, Any]: Las metricas de resumir mas muertes, muertes por golpe mortal, tiempo medio de
            reinicio y recargas; vacio si no hay episodios.

    Raises:
        Ninguna.

    Ejemplo:
        resumir_jefe([{"victoria": False, "muerte": True, "golpe_mortal": "B:M"}])
    """
    if not episodios:
        return {}
    resumen: dict[str, Any] = dict(resumir(episodios))
    muertes = [e for e in episodios if e.get("muerte")]
    resumen["muertes"] = len(muertes)
    resumen["muertes_por_golpe"] = dict(Counter(str(e.get("golpe_mortal")) for e in muertes).most_common())
    resumen["reinicio_s"] = sum(float(e.get("reinicio_s", 0.0)) for e in episodios) / len(episodios)
    resumen["recargas"] = max((int(e.get("recargas", 0)) for e in episodios), default=0)
    resumen["inicios_inconsistentes"] = max((int(e.get("inicios_inconsistentes", 0)) for e in episodios), default=0)
    return resumen


@dataclass(frozen=True)
class TrabajoEvaluacionJefe:
    """
    Encargo para un proceso de evaluacion contra un jefe (debe poder viajar a otro proceso).

    Contexto:
        evaluar_jefe_en_paralelo.

    Atributos:
        indice: numero del trabajador, parte del id de la instancia del juego.
        jefe: jefe de episodio.
        episodios: peleas que le toca jugar.
        semilla: semilla base del trabajador.
        opciones: opciones de reset de cada episodio.
        prefijo_id: prefijo del id de instancia.
        config: configuracion del entorno.
        epsilon_ruido: ruido de ejecucion por paso.
        recarga_cada: episodios entre recargas completas del mapa.
    """

    indice: int
    jefe: str
    episodios: int
    semilla: int = 0
    opciones: dict[str, Any] = field(default_factory=dict)
    prefijo_id: str = "ej"
    config: ConfigEntorno = field(default_factory=ConfigEntorno)
    epsilon_ruido: float = 0.0
    recarga_cada: int = 150


def crear_entorno_jefes(catalogo: CatalogoJefes, jefes: Sequence[str] | None, id_instancia: str,
                        config: ConfigEntorno | None = None, semilla: int | None = None,
                        seleccion: ConfigSeleccion | None = None, trazar: bool = True,
                        recarga_cada: int = 150, params: ParametrosInstancia | None = None) -> EntornoMultiJefe:
    """
    Crea un entorno multi-jefe con su propia instancia del juego.

    Contexto:
        Trabajadores de evaluacion y recoleccion, y verificaciones.

    Args:
        catalogo (CatalogoJefes): Jefes y vocabulario union.
        jefes (Sequence[str], opcional): Jefes que esta instancia enfrenta (todos del mismo mapa).
        id_instancia (str): Identificador de la instancia (pipes y registro).
        config (ConfigEntorno, opcional): Parametros del entorno.
        semilla (int, opcional): Semilla del azar propio.
        seleccion (ConfigSeleccion, opcional): Pesos y modo adaptativo.
        trazar (bool): Registrar cada golpe recibido (para la causa de muerte).
        recarga_cada (int): Episodios entre recargas completas del mapa.
        params (ParametrosInstancia, opcional): Lanzamiento del juego; por defecto el perfil de
            entrenamiento.

    Returns:
        EntornoMultiJefe: Entorno con el juego ya lanzado.

    Raises:
        TimeoutError: Si el juego no se conecta.
        ValueError: Si los jefes no son validos.

    Ejemplo:
        env = crear_entorno_jefes(catalogo, ("amlo",), "ej0")
    """
    parametros = params or ParametrosInstancia(id_instancia=id_instancia)
    return EntornoMultiJefe(crear_fabrica_puerto(parametros), catalogo, config, semilla, jefes, seleccion,
                            recarga_cada=recarga_cada, trazar=trazar)


def ejecutar_trabajo_jefe(trabajo: TrabajoEvaluacionJefe) -> list[dict[str, Any]]:
    """
    Juega las peleas de un trabajador con su propia instancia del juego.

    Contexto:
        Funcion de nivel superior ejecutada en cada proceso de evaluar_jefe_en_paralelo.

    Args:
        trabajo (TrabajoEvaluacionJefe): Encargo del trabajador.

    Returns:
        list[dict[str, Any]]: Resumenes de sus episodios.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_trabajo_jefe(TrabajoEvaluacionJefe(0, "amlo", 5))
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    catalogo = cargar_catalogo()
    semilla = trabajo.semilla + trabajo.indice
    env = crear_entorno_jefes(catalogo, (trabajo.jefe,), f"{trabajo.prefijo_id}{trabajo.indice}", trabajo.config,
                              semilla, recarga_cada=trabajo.recarga_cada)
    try:
        bot = BotJefes.desde_catalogo(catalogo)
        azar = random.Random(semilla)
        opciones = {"jefe": trabajo.jefe, "boss_hp_mult": 1.0, **trabajo.opciones}
        return [jugar_episodio_bot(env, bot, opciones, trabajo.epsilon_ruido, azar) for _ in range(trabajo.episodios)]
    finally:
        env.close()


def evaluar_jefe_en_paralelo(jefe: str, episodios: int, instancias: int, semilla: int = 0,
                             opciones: dict[str, Any] | None = None, config: ConfigEntorno | None = None,
                             prefijo_id: str = "ej", ejecutor: Executor | None = None,
                             epsilon_ruido: float = 0.0, recarga_cada: int = 150) -> list[dict[str, Any]]:
    """
    Juega las peleas contra un jefe repartidas entre varias instancias del juego.

    Contexto:
        Comando evaluar-jefes y validaciones de cada jefe.

    Args:
        jefe (str): Jefe de episodio.
        episodios (int): Total de peleas.
        instancias (int): Instancias del juego en paralelo (todas en el mapa del jefe).
        semilla (int): Semilla base.
        opciones (dict, opcional): Opciones de reset adicionales (por ejemplo start_phase).
        config (ConfigEntorno, opcional): Configuracion del entorno.
        prefijo_id (str): Prefijo de los ids de instancia.
        ejecutor (Executor, opcional): Ejecutor a usar (por defecto procesos con spawn).
        epsilon_ruido (float): Ruido de ejecucion por paso (evaluacion de robustez).
        recarga_cada (int): Episodios entre recargas completas del mapa.

    Returns:
        list[dict[str, Any]]: Resumenes de todos los episodios.

    Raises:
        ValueError: Si instancias es menor que 1.

    Ejemplo:
        resumenes = evaluar_jefe_en_paralelo("nahual", 50, 4)
    """
    reparto = repartir(episodios, instancias)
    trabajos = [
        TrabajoEvaluacionJefe(
            indice=i, jefe=jefe, episodios=n, semilla=semilla, opciones=dict(opciones or {}),
            prefijo_id=prefijo_id, config=config or ConfigEntorno(), epsilon_ruido=epsilon_ruido,
            recarga_cada=recarga_cada,
        )
        for i, n in enumerate(reparto)
    ]
    propio = ejecutor is None
    pool = ejecutor or ProcessPoolExecutor(max_workers=max(1, len(trabajos)), mp_context=get_context("spawn"))
    try:
        resultados = list(pool.map(ejecutar_trabajo_jefe, trabajos))
    finally:
        if propio:
            pool.shutdown(wait=True, cancel_futures=True)
    return [resumen for lista in resultados for resumen in lista]
