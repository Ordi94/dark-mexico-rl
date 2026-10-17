"""
Recoleccion de demostraciones multi-jefe: el bot de reglas juega contra cada jefe y etiqueta cada estado.

Reutiliza el formato y el almacenamiento de la recoleccion de imitacion (lotes .npz atomicos,
reanudable) con tres diferencias: el entorno es el multi-jefe (observacion v2, un mapa por
instancia), el bot confirma lo que de verdad se ejecuto antes de dar un impacto por atendido (con
ruido DART la etiqueta sigue siendo la decision del bot y el impacto sigue pendiente, asi que la
etiqueta ya no pide "nada" mientras el golpe sigue sin parar) y cada episodio lleva su jefe.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from dmrl.aplicacion.entorno import RESULTADO_ERROR
from dmrl.aplicacion.evaluacion_jefes import crear_entorno_jefes, jugar_episodio_bot
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import (
    Demostraciones,
    PlanEpisodio,
    concatenar,
    identificador_episodio,
    planificar_episodios,
)
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.seleccion_jefes import repartir_instancias
from dmrl.infraestructura.demostraciones import guardar_lote, metadatos_de_ronda
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO
from dmrl.infraestructura.procesos import ajustar_proceso_actual

MAX_PASADAS = 3
CAMPOS_RESUMEN = ("resultado", "victoria", "muerte", "danio_hecho", "danio_recibido", "parries",
                  "caguamas_usadas", "caguamas_interrumpidas", "fase_max", "progreso_jefe", "duracion_s")


@dataclass(frozen=True)
class PlanJefe:
    """
    Un episodio de recoleccion contra un jefe concreto.

    Contexto:
        planificar_jefes; el trabajador lo recibe ya decidido para que la recoleccion sea reproducible
        y reanudable.

    Atributos:
        plan: fase de inicio, ruido y semilla; su indice es unico en la ronda.
        jefe: jefe de episodio.
    """

    plan: PlanEpisodio
    jefe: str


@dataclass(frozen=True)
class TrabajoRecoleccionJefes:
    """
    Encargo de recoleccion para un proceso (debe poder viajar a otro proceso).

    Contexto:
        recolectar_jefes reparte los episodios pendientes entre estos trabajos; todos los de un
        trabajo son del mismo mapa.

    Atributos:
        indice: numero del trabajador; forma el id de su instancia del juego.
        ronda: ronda a la que pertenecen los episodios.
        planes: episodios que le toca jugar.
        carpeta_datos: carpeta raiz donde escribe sus lotes.
        prefijo_id: prefijo del id de instancia.
        config: configuracion del entorno.
        lote_episodios: episodios por archivo de lote.
        archivo_parada: si este archivo existe, termina tras el episodio en curso.
        recarga_cada: episodios entre recargas completas del mapa.
        hora_limite: instante (segundos desde la epoca) a partir del cual no empieza episodios nuevos.
    """

    indice: int
    ronda: int
    planes: tuple[PlanJefe, ...]
    carpeta_datos: str
    prefijo_id: str = "rj"
    config: ConfigEntorno = field(default_factory=ConfigEntorno)
    lote_episodios: int = 10
    archivo_parada: str | None = None
    recarga_cada: int = 150
    hora_limite: float | None = None


def intercalar_por_jefe(planes: Sequence[PlanJefe]) -> list[PlanJefe]:
    """
    Ordena los episodios en rondas: uno de cada jefe antes del segundo de cualquiera.

    Contexto:
        recolectar_jefes. Con un tope de tiempo la recoleccion puede cortarse en cualquier punto; si
        los episodios estuvieran agrupados por jefe, los ultimos jefes quedarian sin datos. Intercalados,
        un corte deja a todos los jefes con la misma cantidad de peleas (salvo una).

    Args:
        planes (Sequence[PlanJefe]): Episodios pendientes, agrupados por jefe o en cualquier orden.

    Returns:
        list[PlanJefe]: Los mismos episodios, ordenados por posicion dentro de su jefe y despues por
            el orden de aparicion del jefe.

    Raises:
        Ninguna.

    Ejemplo:
        intercalar_por_jefe([a1, a2, n1, n2])  # [a1, n1, a2, n2]
    """
    contador: dict[str, int] = {}
    orden_jefe: dict[str, int] = {}
    claves: list[tuple[int, int, int]] = []
    for posicion, p in enumerate(planes):
        rango = contador.get(p.jefe, 0)
        contador[p.jefe] = rango + 1
        claves.append((rango, orden_jefe.setdefault(p.jefe, len(orden_jefe)), posicion))
    return [planes[k[2]] for k in sorted(claves)]


def planificar_jefes(catalogo: CatalogoJefes, jefes: Sequence[str], por_jefe: int, prob_fase2: float = 0.2,
                     fraccion_ruido: float = 0.5, epsilon_min: float = 0.05, epsilon_max: float = 0.20,
                     semilla: int = 0) -> list[PlanJefe]:
    """
    Decide de antemano la fase de inicio y el ruido de cada episodio contra cada jefe.

    Contexto:
        Inicio de la recoleccion. Los conteos son exactos por jefe (ver planificar_episodios); los
        jefes que no admiten la segunda fase empiezan siempre en la suya.

    Args:
        catalogo (CatalogoJefes): Jefes disponibles.
        jefes (Sequence[str]): Jefes de episodio a recolectar.
        por_jefe (int): Episodios por jefe.
        prob_fase2 (float): Fraccion que empieza en la fase 2 en los jefes que la admiten.
        fraccion_ruido (float): Fraccion de episodios con ruido de ejecucion.
        epsilon_min (float): Ruido minimo de un episodio ruidoso.
        epsilon_max (float): Ruido maximo.
        semilla (int): Semilla del plan.

    Returns:
        list[PlanJefe]: Planes con indices 0..len(jefes)*por_jefe-1, jefe por jefe.

    Raises:
        KeyError: Si algun jefe no existe.
        ValueError: Si algun parametro esta fuera de rango.

    Ejemplo:
        planificar_jefes(catalogo, ("amlo", "nahual"), 5)
    """
    planes: list[PlanJefe] = []
    for orden, jefe in enumerate(jefes):
        perfil = catalogo.perfil(jefe)
        admite = len(perfil.fases_inicio) > 1
        base = planificar_episodios(por_jefe, prob_fase2 if admite else 0.0, fraccion_ruido, epsilon_min,
                                    epsilon_max, semilla + orden)
        for p in base:
            fase = p.fase_inicio if admite else perfil.fases_inicio[0]
            planes.append(PlanJefe(PlanEpisodio(orden * por_jefe + p.indice, fase, p.epsilon, p.semilla), jefe))
    return planes


def recolectar_episodio_jefe(env: Any, bot: BotJefes, plan: PlanJefe, ronda: int) -> tuple[Demostraciones | None,
                                                                                         dict[str, Any]]:
    """
    Juega un episodio y registra (estado, accion del bot) en cada decision.

    Contexto:
        Bucle de cada trabajador de recoleccion.

    Args:
        env (EntornoMultiJefe): Entorno ya creado.
        bot (BotJefes): Experto que etiqueta.
        plan (PlanJefe): Jefe, fase de inicio, ruido y semilla del episodio.
        ronda (int): Ronda de la recoleccion (parte del identificador del episodio).

    Returns:
        tuple[Demostraciones | None, dict[str, Any]]: Los pasos (None si el juego se cayo y el
            episodio es inutilizable) y los metadatos del episodio.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        demostraciones, meta = recolectar_episodio_jefe(env, bot, plan, 0)
    """
    identificador = identificador_episodio(ronda, plan.plan.indice)
    estados: list[Any] = []
    mascaras: list[Any] = []
    etiquetas: list[int] = []
    ejecutadas: list[int] = []
    recompensas: list[float] = []

    def registrar(obs: Any, mascara: list[bool], etiqueta: int, ejecutada: int, recompensa: float) -> None:
        """
        Guarda una decision del episodio.

        Contexto:
            Llamada por jugar_episodio_bot en cada paso.

        Args:
            obs (Any): Observacion apilada previa a la accion.
            mascara (list[bool]): Acciones validas.
            etiqueta (int): Decision del bot.
            ejecutada (int): Accion que se envio al juego.
            recompensa (float): Recompensa del paso.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        estados.append(obs)
        mascaras.append(mascara)
        etiquetas.append(etiqueta)
        ejecutadas.append(ejecutada)
        recompensas.append(recompensa)

    opciones = {"jefe": plan.jefe, "start_phase": plan.plan.fase_inicio, "boss_hp_mult": 1.0}
    resumen = jugar_episodio_bot(env, bot, opciones, plan.plan.epsilon, random.Random(plan.plan.semilla), registrar)
    meta: dict[str, Any] = {
        "id": identificador, "indice": plan.plan.indice, "ronda": ronda, "jefe": plan.jefe,
        "fase_inicio": plan.plan.fase_inicio, "epsilon": plan.plan.epsilon, "ejecutor": "bot",
        "pasos": len(etiquetas),
    }
    meta.update({k: resumen[k] for k in CAMPOS_RESUMEN if k in resumen})
    if resumen.get("resultado") == RESULTADO_ERROR:
        return None, meta
    demostraciones = Demostraciones(
        np.asarray(estados, dtype=np.float32), np.asarray(mascaras, dtype=np.bool_),
        np.asarray(etiquetas, dtype=np.int16), np.asarray(ejecutadas, dtype=np.int16),
        np.full(len(etiquetas), identificador, dtype=np.int64), np.asarray(recompensas, dtype=np.float32),
    )
    return demostraciones, meta


def ejecutar_recoleccion_jefes(trabajo: TrabajoRecoleccionJefes) -> list[dict[str, Any]]:
    """
    Juega los episodios de un trabajador con su propia instancia del juego y guarda los lotes.

    Contexto:
        Funcion de nivel superior ejecutada en cada proceso de recolectar_jefes.

    Args:
        trabajo (TrabajoRecoleccionJefes): Encargo del trabajador.

    Returns:
        list[dict[str, Any]]: Metadatos de los episodios guardados.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_recoleccion_jefes(TrabajoRecoleccionJefes(0, 0, (plan,), "runs/imitacion_v2/datos"))
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    catalogo = cargar_catalogo()
    jefes = tuple(dict.fromkeys(p.jefe for p in trabajo.planes))
    env = crear_entorno_jefes(catalogo, jefes, f"{trabajo.prefijo_id}{trabajo.indice}", trabajo.config,
                              semilla=trabajo.indice, trazar=False, recarga_cada=trabajo.recarga_cada)
    bot = BotJefes.desde_catalogo(catalogo)
    carpeta = Path(trabajo.carpeta_datos) / f"ronda_{trabajo.ronda:02d}"
    guardados: list[dict[str, Any]] = []
    pendientes_lote: list[Demostraciones] = []
    meta_lote: list[dict[str, Any]] = []

    def volcar() -> None:
        """
        Escribe en disco el lote acumulado.

        Contexto:
            ejecutar_recoleccion_jefes, cada lote_episodios episodios y al terminar.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            (uso interno)
        """
        if not pendientes_lote:
            return
        primero = meta_lote[0]["indice"]
        guardar_lote(carpeta / f"w{trabajo.indice}_{primero:05d}.npz", concatenar(pendientes_lote), list(meta_lote))
        guardados.extend(meta_lote)
        pendientes_lote.clear()
        meta_lote.clear()

    try:
        for plan in trabajo.planes:
            if trabajo.archivo_parada and Path(trabajo.archivo_parada).exists():
                break
            if trabajo.hora_limite is not None and time.time() >= trabajo.hora_limite:
                break
            demostraciones, meta = recolectar_episodio_jefe(env, bot, plan, trabajo.ronda)
            if demostraciones is None:
                continue
            pendientes_lote.append(demostraciones)
            meta_lote.append(meta)
            if len(meta_lote) >= trabajo.lote_episodios:
                volcar()
    finally:
        try:
            volcar()
        finally:
            env.close()
    return guardados


def repartir_por_mapa(planes: Sequence[PlanJefe], instancias: int,
                      catalogo: CatalogoJefes) -> list[tuple[PlanJefe, ...]]:
    """
    Reparte los episodios entre trabajadores sin mezclar mapas (una instancia queda fija a un mapa).

    Contexto:
        recolectar_jefes. Las instancias se reparten entre mapas segun los episodios de cada uno
        (al menos una por mapa) y dentro de un mapa los episodios se alternan para mezclar jefes.

    Args:
        planes (Sequence[PlanJefe]): Episodios pendientes.
        instancias (int): Instancias del juego disponibles (>= mapas con episodios).
        catalogo (CatalogoJefes): Para saber el mapa de cada jefe.

    Returns:
        list[tuple[PlanJefe, ...]]: Un grupo por trabajador; sin grupos vacios.

    Raises:
        ValueError: Si hay menos instancias que mapas.

    Ejemplo:
        repartir_por_mapa(planes, 4, catalogo)
    """
    por_mapa: dict[str, list[PlanJefe]] = {}
    for p in planes:
        por_mapa.setdefault(catalogo.perfil(p.jefe).mapa, []).append(p)
    if not por_mapa:
        return []
    cuentas = repartir_instancias({m: float(len(v)) for m, v in por_mapa.items()},
                                  {m: 1.0 for m in por_mapa}, max(instancias, len(por_mapa)))
    grupos: list[tuple[PlanJefe, ...]] = []
    for mapa, lista in por_mapa.items():
        n = cuentas[mapa]
        grupos += [tuple(lista[i::n]) for i in range(n)]
    return [g for g in grupos if g]


def recolectar_jefes(carpeta_datos: Path, ronda: int, jefes: Sequence[str], por_jefe: int, instancias: int,
                     semilla: int = 0, config: ConfigEntorno | None = None, archivo_parada: Path | None = None,
                     ejecutor: Executor | None = None, prefijo_id: str = "rj",
                     limite_segundos: float | None = None) -> int:
    """
    Recolecta los episodios de una ronda que aun no estan guardados (reanudable).

    Contexto:
        Comando recolectar-jefes. Un episodio que termina en error del juego no se guarda; se
        reintenta en una pasada siguiente (hasta MAX_PASADAS).

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.
        ronda (int): Numero de ronda.
        jefes (Sequence[str]): Jefes de episodio.
        por_jefe (int): Episodios por jefe.
        instancias (int): Instancias del juego en paralelo.
        semilla (int): Semilla del plan.
        config (ConfigEntorno, opcional): Configuracion del entorno.
        archivo_parada (Path, opcional): Si existe, la recoleccion termina ordenadamente.
        ejecutor (Executor, opcional): Ejecutor a usar (por defecto procesos con spawn).
        prefijo_id (str): Prefijo de los ids de instancia.
        limite_segundos (float, opcional): Tope de tiempo real; pasado ese plazo no se inician
            episodios nuevos (los que estan en curso terminan y se guardan).

    Returns:
        int: Episodios del plan que siguen sin guardarse (0 = ronda completa).

    Raises:
        KeyError: Si algun jefe no existe.
        ValueError: Si hay menos instancias que mapas.

    Ejemplo:
        faltan = recolectar_jefes(Path("runs/imitacion_v2/datos"), 0, ("amlo", "nahual"), 5, 4)
    """
    catalogo = cargar_catalogo()
    config = config or ConfigEntorno()
    plan = planificar_jefes(catalogo, jefes, por_jefe, semilla=semilla)
    hora_limite = time.time() + limite_segundos if limite_segundos is not None else None
    for _ in range(MAX_PASADAS):
        hechos = {int(m["indice"]) for m in metadatos_de_ronda(carpeta_datos, ronda)}
        pendientes = intercalar_por_jefe([p for p in plan if p.plan.indice not in hechos])
        if not pendientes or (archivo_parada and archivo_parada.exists()):
            break
        if hora_limite is not None and time.time() >= hora_limite:
            break
        trabajos = [
            TrabajoRecoleccionJefes(
                indice=i, ronda=ronda, planes=grupo, carpeta_datos=str(carpeta_datos), prefijo_id=prefijo_id,
                config=config, archivo_parada=str(archivo_parada) if archivo_parada else None,
                hora_limite=hora_limite,
            )
            for i, grupo in enumerate(repartir_por_mapa(pendientes, instancias, catalogo))
        ]
        propio = ejecutor is None
        pool = ejecutor or ProcessPoolExecutor(max_workers=max(1, len(trabajos)), mp_context=get_context("spawn"))
        try:
            list(pool.map(ejecutar_recoleccion_jefes, trabajos))
        finally:
            if propio:
                pool.shutdown(wait=True, cancel_futures=True)
    hechos = {int(m["indice"]) for m in metadatos_de_ronda(carpeta_datos, ronda)}
    return sum(1 for p in plan if p.plan.indice not in hechos)


def resumen_de_ronda(carpeta_datos: Path, ronda: int) -> dict[str, dict[str, float]]:
    """
    Resumen por jefe de lo recolectado en una ronda.

    Contexto:
        Informe de recolectar-jefes y verificacion del formato.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.
        ronda (int): Numero de ronda.

    Returns:
        dict[str, dict[str, float]]: jefe -> episodios, pasos, victorias, episodios con ruido y que empiezan en fase 2.

    Raises:
        Ninguna.

    Ejemplo:
        resumen_de_ronda(Path("runs/imitacion_v2/datos"), 0)
    """
    por_jefe: dict[str, list[dict[str, Any]]] = {}
    for meta in metadatos_de_ronda(carpeta_datos, ronda):
        por_jefe.setdefault(str(meta.get("jefe", "?")), []).append(meta)
    return {
        jefe: {
            "episodios": float(len(m)), "pasos": float(sum(int(e.get("pasos", 0)) for e in m)),
            "victorias": float(sum(1 for e in m if e.get("victoria"))),
            "con_ruido": float(sum(1 for e in m if float(e.get("epsilon", 0.0)) > 0.0)),
            "fase2": float(sum(1 for e in m if int(e.get("fase_inicio", 1)) == 2)),
        }
        for jefe, m in sorted(por_jefe.items())
    }
