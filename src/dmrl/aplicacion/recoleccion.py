"""
Recoleccion de demostraciones: el bot de reglas etiqueta cada estado que se visita.

Quien ejecuta las acciones puede ser el propio bot (ronda inicial) o la red aprendida (rondas de
DAgger), con o sin ruido de ejecucion. Sea quien sea, la etiqueta guardada es siempre la accion que
el bot elegiria en ese estado. El bot se consulta en cada paso aunque su accion no se ejecute: su
memoria interna (golpes ya atendidos) debe seguir el combate real.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from dmrl.aplicacion.entorno import RESULTADO_ERROR, DarkMexicoEnv
from dmrl.aplicacion.evaluacion import Politica, PoliticaModelo
from dmrl.dominio import acciones as acc
from dmrl.dominio.bot_reglas import BotReglas
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import (
    Demostraciones,
    PlanEpisodio,
    concatenar,
    identificador_episodio,
    ruido_dart,
)
from dmrl.infraestructura.datos import cargar_tabla_golpes
from dmrl.infraestructura.demostraciones import guardar_lote, metadatos_de_ronda
from dmrl.infraestructura.juego import (
    AFINIDAD_ENTRENAMIENTO,
    ParametrosInstancia,
    crear_fabrica_puerto,
)
from dmrl.infraestructura.procesos import ajustar_proceso_actual

MAX_PASADAS = 3
CAMPOS_RESUMEN = ("resultado", "victoria", "muerte", "danio_hecho", "danio_recibido", "parries",
                  "caguamas_usadas", "caguamas_interrumpidas", "fase_max", "progreso_jefe", "duracion_s")


@dataclass(frozen=True)
class TrabajoRecoleccion:
    """
    Encargo de recoleccion para un proceso (debe poder viajar a otro proceso).

    Contexto:
        recolectar_ronda reparte los episodios pendientes entre estos trabajos.

    Atributos:
        indice: numero del trabajador; forma el id de su instancia del juego.
        ronda: ronda a la que pertenecen los episodios.
        planes: episodios que le toca jugar.
        carpeta_datos: carpeta raiz donde escribe sus lotes.
        ruta_modelo: modelo que ejecuta las acciones, o None si las ejecuta el bot.
        prefijo_id: prefijo del id de instancia, para no chocar con otros procesos.
        config: configuracion del entorno.
        lote_episodios: episodios por archivo de lote.
        archivo_parada: si este archivo existe, termina tras el episodio en curso.
    """

    indice: int
    ronda: int
    planes: tuple[PlanEpisodio, ...]
    carpeta_datos: str
    ruta_modelo: str | None = None
    prefijo_id: str = "im"
    config: ConfigEntorno = field(default_factory=ConfigEntorno)
    lote_episodios: int = 10
    archivo_parada: str | None = None


def recolectar_episodio(env: DarkMexicoEnv, bot: BotReglas, ejecutor: Politica | None, plan: PlanEpisodio,
                        ronda: int) -> tuple[Demostraciones | None, dict[str, Any]]:
    """
    Juega un episodio y registra (estado, accion del bot) en cada decision.

    Contexto:
        Bucle de cada trabajador de recoleccion.

    Args:
        env (DarkMexicoEnv): Entorno ya creado.
        bot (BotReglas): Experto que etiqueta.
        ejecutor (Politica, opcional): Quien ejecuta las acciones; None = el propio bot.
        plan (PlanEpisodio): Fase de inicio, ruido y semilla del episodio.
        ronda (int): Ronda de la recoleccion (parte del identificador del episodio).

    Returns:
        tuple[Demostraciones | None, dict[str, Any]]: Los pasos (None si el juego se cayo y el
            episodio es inutilizable) y los metadatos del episodio.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        demostraciones, meta = recolectar_episodio(env, bot, None, plan, 0)
    """
    azar = random.Random(plan.semilla)
    identificador = identificador_episodio(ronda, plan.indice)
    obs, _ = env.reset(options={"start_phase": plan.fase_inicio, "boss_hp_mult": 1.0})
    bot.reiniciar()
    if ejecutor is not None:
        ejecutor.reiniciar()
    estados: list[Any] = []
    mascaras: list[Any] = []
    etiquetas: list[int] = []
    ejecutadas: list[int] = []
    recompensas: list[float] = []
    resumen: dict[str, Any] = {}
    while True:
        mascara = env.action_masks()
        validas = mascara.tolist()
        etiqueta = bot.decidir(env.observacion_cruda, validas)
        if not validas[etiqueta]:
            etiqueta = acc.ACCION_NADA
        base = etiqueta if ejecutor is None else ejecutor.decidir(env, obs)
        ejecutada = ruido_dart(base, validas, plan.epsilon, azar)
        siguiente, recompensa, terminado, truncado, info = env.step(ejecutada)
        estados.append(obs)
        mascaras.append(mascara)
        etiquetas.append(etiqueta)
        ejecutadas.append(ejecutada)
        recompensas.append(recompensa)
        obs = siguiente
        if terminado or truncado:
            resumen = info["episodio"]
            break
    meta: dict[str, Any] = {
        "id": identificador, "indice": plan.indice, "ronda": ronda, "fase_inicio": plan.fase_inicio,
        "epsilon": plan.epsilon, "ejecutor": "bot" if ejecutor is None else "modelo", "pasos": len(etiquetas),
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


def ejecutar_recoleccion(trabajo: TrabajoRecoleccion) -> list[dict[str, Any]]:
    """
    Juega los episodios de un trabajador con su propia instancia del juego y guarda los lotes.

    Contexto:
        Funcion de nivel superior ejecutada en cada proceso de recolectar_ronda.

    Args:
        trabajo (TrabajoRecoleccion): Encargo del trabajador.

    Returns:
        list[dict[str, Any]]: Metadatos de los episodios guardados.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_recoleccion(TrabajoRecoleccion(0, 0, (plan,), "runs/imitacion/datos"))
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    tabla = cargar_tabla_golpes()
    fabrica = crear_fabrica_puerto(ParametrosInstancia(id_instancia=f"{trabajo.prefijo_id}{trabajo.indice}"))
    env = DarkMexicoEnv(fabrica, tabla, trabajo.config, semilla=trabajo.indice)
    ejecutor: Politica | None = None
    if trabajo.ruta_modelo:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        ejecutor = PoliticaModelo(MaskablePPO.load(trabajo.ruta_modelo, device="cpu"))
    bot = BotReglas(tabla)
    carpeta = Path(trabajo.carpeta_datos) / f"ronda_{trabajo.ronda:02d}"
    guardados: list[dict[str, Any]] = []
    pendientes_lote: list[Demostraciones] = []
    meta_lote: list[dict[str, Any]] = []

    def volcar() -> None:
        """
        Escribe en disco el lote acumulado.

        Contexto:
            ejecutar_recoleccion, cada lote_episodios episodios y al terminar.

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
            demostraciones, meta = recolectar_episodio(env, bot, ejecutor, plan, trabajo.ronda)
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


def repartir_planes(planes: Sequence[PlanEpisodio], trabajadores: int) -> list[tuple[PlanEpisodio, ...]]:
    """
    Reparte los episodios entre trabajadores alternando, para mezclar fases y ruido en cada uno.

    Contexto:
        recolectar_ronda.

    Args:
        planes (Sequence[PlanEpisodio]): Episodios pendientes.
        trabajadores (int): Procesos disponibles (>= 1).

    Returns:
        list[tuple[PlanEpisodio, ...]]: Un grupo por trabajador; sin grupos vacios.

    Raises:
        ValueError: Si trabajadores es menor que 1.

    Ejemplo:
        repartir_planes(planes, 4)
    """
    if trabajadores < 1:
        raise ValueError("trabajadores debe ser >= 1")
    grupos = [tuple(planes[i::trabajadores]) for i in range(trabajadores)]
    return [g for g in grupos if g]


def recolectar_ronda(carpeta_datos: Path, ronda: int, planes: Sequence[PlanEpisodio], instancias: int,
                     ruta_modelo: Path | None = None, config: ConfigEntorno | None = None,
                     archivo_parada: Path | None = None, prefijo_id: str = "im",
                     ejecutor: Executor | None = None) -> int:
    """
    Recolecta los episodios de una ronda que aun no estan guardados (reanudable).

    Contexto:
        Ciclo de imitacion, una vez por ronda. Un episodio que termina en error del juego no se
        guarda; se reintenta en una pasada siguiente (hasta MAX_PASADAS).

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.
        ronda (int): Numero de ronda.
        planes (Sequence[PlanEpisodio]): Plan completo de la ronda.
        instancias (int): Instancias del juego en paralelo.
        ruta_modelo (Path, opcional): Modelo que ejecuta las acciones (DAgger); None = el bot.
        config (ConfigEntorno, opcional): Configuracion del entorno.
        archivo_parada (Path, opcional): Si existe, la recoleccion termina ordenadamente.
        prefijo_id (str): Prefijo de los ids de instancia.
        ejecutor (Executor, opcional): Ejecutor a usar (por defecto procesos con spawn).

    Returns:
        int: Episodios del plan que siguen sin guardarse (0 = ronda completa).

    Raises:
        ValueError: Si instancias es menor que 1.

    Ejemplo:
        faltan = recolectar_ronda(Path("runs/imitacion/datos"), 0, planes, 4)
    """
    config = config or ConfigEntorno()
    for _ in range(MAX_PASADAS):
        hechos = {int(m["indice"]) for m in metadatos_de_ronda(carpeta_datos, ronda)}
        pendientes = [p for p in planes if p.indice not in hechos]
        if not pendientes or (archivo_parada and archivo_parada.exists()):
            break
        trabajos = [
            TrabajoRecoleccion(
                indice=i, ronda=ronda, planes=grupo, carpeta_datos=str(carpeta_datos),
                ruta_modelo=str(ruta_modelo) if ruta_modelo else None, prefijo_id=prefijo_id, config=config,
                archivo_parada=str(archivo_parada) if archivo_parada else None,
            )
            for i, grupo in enumerate(repartir_planes(pendientes, instancias))
        ]
        propio = ejecutor is None
        pool = ejecutor or ProcessPoolExecutor(max_workers=max(1, len(trabajos)), mp_context=get_context("spawn"))
        try:
            list(pool.map(ejecutar_recoleccion, trabajos))
        finally:
            if propio:
                pool.shutdown(wait=True, cancel_futures=True)
    hechos = {int(m["indice"]) for m in metadatos_de_ronda(carpeta_datos, ronda)}
    return sum(1 for p in planes if p.indice not in hechos)
