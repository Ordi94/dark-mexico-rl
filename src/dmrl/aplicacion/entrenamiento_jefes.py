"""
Entrenamiento PPO multi-jefe partiendo de la politica clonada del bot.

Cada entorno es una instancia del juego fija a un mapa (LVL_Boss con Juice, LVL_AMLO con AMLO, el
Chamuco y el Chupacabras); las instancias se reparten entre mapas con repartir_instancias y dentro
del mapa de AMLO el selector da mas peso a los jefes que menos gana (seleccion adaptativa).

Para no destruir lo aprendido por imitacion se usa MaskablePPOKL (calentamiento del critico con la
politica congelada y penalizacion KL que decae), una tasa de aprendizaje baja, recorte 0.1 y una
evaluacion automatica en juego con compuerta de promocion y correccion por regresion
(callbacks_jefes). Todo se reanuda desde checkpoints/ultimo.zip y estado_ppo.json.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from dmrl.aplicacion.callbacks import CallbackParada
from dmrl.aplicacion.callbacks_jefes import (
    CallbackCheckpointJefes,
    CallbackEvaluacionPeriodica,
    CallbackMetricasJefes,
    Lanzador,
    lanzar_evaluacion,
)
from dmrl.aplicacion.entrenamiento import ARCHIVO_PID, CARPETA_RUNS
from dmrl.aplicacion.evaluacion_jefes import crear_entorno_jefes
from dmrl.aplicacion.ppo_kl import MaskablePPOKL
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from dmrl.dominio.seleccion_jefes import ConfigSeleccion, repartir_instancias
from dmrl.infraestructura.demostraciones import leer_json
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO
from dmrl.infraestructura.procesos import ajustar_proceso_actual, apagar_descendientes, escribir_pid

CARPETA_IMITACION = CARPETA_RUNS / "multijefe_imitacion"
SELECCION_ADAPTATIVA = ConfigSeleccion(adaptativo=True)
RECARGA_CADA = 150


@dataclass(frozen=True)
class RutasPPO:
    """
    Ubicacion de los archivos de una corrida de PPO multi-jefe.

    Contexto:
        Entrenamiento, evaluacion, reanudacion y scripts de usuario.

    Atributos:
        base: carpeta runs/<nombre>.
    """

    base: Path

    @property
    def ultimo(self) -> Path:
        """
        Checkpoint mas reciente (con el que se reanuda).

        Contexto:
            Entrenamiento y reanudacion.

        Args:
            Ninguno.

        Returns:
            Path: base/checkpoints/ultimo.zip.

        Raises:
            Ninguna.

        Ejemplo:
            RutasPPO(Path("runs/a")).ultimo
        """
        return self.base / "checkpoints" / "ultimo.zip"

    @property
    def mejor(self) -> Path:
        """
        Mejor modelo segun la compuerta de promocion.

        Contexto:
            Evaluacion, ver_pelea y correccion por regresion.

        Args:
            Ninguno.

        Returns:
            Path: base/mejor.zip.

        Raises:
            Ninguna.

        Ejemplo:
            RutasPPO(Path("runs/a")).mejor
        """
        return self.base / "mejor.zip"

    @property
    def parada(self) -> Path:
        """
        Archivo cuya existencia pide terminar ordenadamente.

        Contexto:
            detener.ps1 y CallbackParada.

        Args:
            Ninguno.

        Returns:
            Path: base/PARAR.

        Raises:
            Ninguna.

        Ejemplo:
            RutasPPO(Path("runs/a")).parada
        """
        return self.base / "PARAR"


def rutas_ppo(nombre: str) -> RutasPPO:
    """
    Rutas de la corrida con ese nombre.

    Contexto:
        CLI y scripts.

    Args:
        nombre (str): Nombre de la corrida.

    Returns:
        RutasPPO: Rutas bajo runs/<nombre>.

    Raises:
        Ninguna.

    Ejemplo:
        rutas_ppo("multijefe_ppo").mejor
    """
    return RutasPPO(CARPETA_RUNS / nombre)


def repartir_entornos(catalogo: CatalogoJefes, instancias: int) -> list[tuple[str, ...]]:
    """
    Jefes que enfrenta cada instancia del entrenamiento.

    Contexto:
        Arranque del entrenamiento. SubprocVecEnv avanza todos los entornos a la vez y recoge una
        muestra de cada uno por paso, asi que la parte de las muestras de un mapa es la parte de sus
        instancias, sin importar lo rapido que corra cada mapa. Para que cada jefe aporte lo mismo,
        las instancias se reparten en proporcion al numero de jefes de cada mapa (con 4 instancias:
        1 en LVL_Boss con Juice y 3 en LVL_AMLO; con 6: 2 y 4). Todas las instancias de un mapa
        enfrentan a todos los jefes de ese mapa.

    Args:
        catalogo (CatalogoJefes): Jefes de episodio y su mapa.
        instancias (int): Instancias del juego disponibles.

    Returns:
        list[tuple[str, ...]]: Un grupo de jefes por instancia, mapa por mapa.

    Raises:
        ValueError: Si hay menos instancias que mapas.

    Ejemplo:
        repartir_entornos(catalogo, 4)  # [("juice",), ("amlo", "chamuco", "nahual"), ...]
    """
    por_mapa: dict[str, tuple[str, ...]] = {}
    for jefe in catalogo.jefes_de_episodio():
        mapa = catalogo.perfil(jefe).mapa
        por_mapa[mapa] = (*por_mapa.get(mapa, ()), jefe)
    cuentas = repartir_instancias({m: float(len(j)) for m, j in por_mapa.items()}, {m: 1.0 for m in por_mapa},
                                  instancias)
    return [por_mapa[mapa] for mapa in por_mapa for _ in range(cuentas[mapa])]


def crear_entorno_jefe(indice: int, jefes: tuple[str, ...], config: ConfigEntorno, semilla: int,
                       seleccion: ConfigSeleccion = SELECCION_ADAPTATIVA, prefijo: str = "mj",
                       trazar: bool = True) -> Monitor:
    """
    Crea un entorno multi-jefe completo (instancia del juego incluida); se ejecuta en cada subproceso.

    Contexto:
        Funcion de nivel superior para que SubprocVecEnv (spawn) pueda enviarla a los subprocesos.
        Registra los golpes recibidos (trazar) para publicar de que ataque muere el jugador.

    Args:
        indice (int): Numero del entorno; forma el id de la instancia.
        jefes (tuple[str, ...]): Jefes que enfrenta (todos del mismo mapa).
        config (ConfigEntorno): Configuracion del entorno (la misma de las demostraciones).
        semilla (int): Semilla base.
        seleccion (ConfigSeleccion): Pesos y modo adaptativo del selector.
        prefijo (str): Prefijo del id de instancia.
        trazar (bool): Si es True el mod registra cada golpe recibido, lo que permite publicar de que
            ataque muere el jugador a costa de algo de velocidad por paso.

    Returns:
        Monitor: Entorno envuelto para las estadisticas de SB3.

    Raises:
        TimeoutError: Si el juego no se conecta.

    Ejemplo:
        env = crear_entorno_jefe(0, ("juice",), ConfigEntorno(), 0)
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    env = crear_entorno_jefes(cargar_catalogo(), jefes, f"{prefijo}{indice}", config, semilla + indice, seleccion,
                              trazar=trazar, recarga_cada=RECARGA_CADA)
    return Monitor(env)


def crear_vec_env_jefes(grupos: list[tuple[str, ...]], config: ConfigEntorno, semilla: int,
                        fabricas: list[Callable[[], gym.Env[Any, Any]]] | None = None,
                        trazar: bool = True) -> VecEnv:
    """
    Crea el entorno vectorizado con una instancia del juego por entorno.

    Contexto:
        entrenar_jefes. Con `fabricas` (pruebas) usa un DummyVecEnv en proceso.

    Args:
        grupos (list[tuple[str, ...]]): Jefes de cada instancia (salida de repartir_entornos).
        config (ConfigEntorno): Configuracion del entorno.
        semilla (int): Semilla base.
        fabricas (list, opcional): Funciones que crean cada entorno (sustituyen a las reales).
        trazar (bool): Si los entornos registran los golpes recibidos (ver crear_entorno_jefe).

    Returns:
        VecEnv: Entorno vectorizado.

    Raises:
        ValueError: Si no hay instancias.

    Ejemplo:
        venv = crear_vec_env_jefes([("juice",), ("amlo", "nahual")], ConfigEntorno(), 0)
    """
    if not grupos:
        raise ValueError("se necesita al menos una instancia")
    if fabricas is not None:
        return DummyVecEnv(fabricas)
    return SubprocVecEnv(
        [functools.partial(crear_entorno_jefe, i, jefes, config, semilla, trazar=trazar)
         for i, jefes in enumerate(grupos)],
        start_method="spawn")


def parametros_de_carga(config: ConfigPPOJefes, carpeta: Path) -> dict[str, Any]:
    """
    Hiperparametros de PPO que sustituyen a los guardados en el zip al cargarlo.

    Contexto:
        Crear la corrida desde el modelo clonado y reanudarla: la imitacion se guardo con los
        hiperparametros del PPO desde cero (tasa 3e-4, recorte 0.2), que aqui no sirven.

    Args:
        config (ConfigPPOJefes): Parametros de la corrida.
        carpeta (Path): Carpeta de la corrida (para TensorBoard).

    Returns:
        dict[str, Any]: custom_objects de MaskablePPO.load.

    Raises:
        Ninguna.

    Ejemplo:
        parametros_de_carga(ConfigPPOJefes(), Path("runs/multijefe_ppo"))["clip_range"]  # 0.1
    """
    return {
        "learning_rate": config.learning_rate, "clip_range": config.clip_range, "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef, "max_grad_norm": config.max_grad_norm, "target_kl": config.target_kl,
        "n_steps": config.n_steps, "batch_size": config.batch_size, "n_epochs": config.n_epochs,
        "gamma": config.gamma, "gae_lambda": config.gae_lambda, "tensorboard_log": str(carpeta / "tb"),
    }


def cargar_modelo_ppo(ruta: Path, entorno: VecEnv, config: ConfigPPOJefes, carpeta: Path) -> MaskablePPOKL:
    """
    Carga un zip (la imitacion o un checkpoint) como MaskablePPOKL con los hiperparametros de la corrida.

    Contexto:
        Inicio y reanudacion del entrenamiento.

    Args:
        ruta (Path): Zip de un MaskablePPO.
        entorno (VecEnv): Entorno vectorizado con los mismos espacios.
        config (ConfigPPOJefes): Parametros de la corrida.
        carpeta (Path): Carpeta de la corrida.

    Returns:
        MaskablePPOKL: Modelo listo para learn (falta configurar_kl).

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si los espacios del entorno no coinciden con los del zip.

    Ejemplo:
        modelo = cargar_modelo_ppo(Path("mejor.zip"), venv, ConfigPPOJefes(), Path("runs/a"))
    """
    if not ruta.exists():
        raise FileNotFoundError(f"no existe el modelo {ruta}")
    modelo = MaskablePPOKL.load(str(ruta), env=entorno, device=config.dispositivo,
                                custom_objects=parametros_de_carga(config, carpeta))
    return modelo


def referencia_de_imitacion(ruta: Path) -> dict[str, float]:
    """
    Tasas de victoria por jefe de la imitacion, piso de la compuerta de promocion.

    Contexto:
        Arranque del entrenamiento. Las escribe la evaluacion final de la imitacion.

    Args:
        ruta (Path): Archivo referencia.json de la corrida de imitacion.

    Returns:
        dict[str, float]: Jefe -> tasa de victoria.

    Raises:
        FileNotFoundError: Si el archivo falta o no trae tasas.

    Ejemplo:
        referencia_de_imitacion(Path("runs/multijefe_imitacion/referencia.json"))
    """
    datos = leer_json(ruta)
    tasas = {str(j): float(t) for j, t in dict(datos.get("tasas", {})).items()}
    if not tasas:
        raise FileNotFoundError(f"{ruta} no existe o no trae tasas de victoria por jefe")
    return tasas


def entrenar_jefes(config: ConfigPPOJefes, entorno: VecEnv | None = None, ruta_imitacion: Path | None = None,
                   catalogo: CatalogoJefes | None = None, lanzador: Lanzador = lanzar_evaluacion) -> Path:
    """
    Ejecuta (o reanuda) el PPO multi-jefe hasta agotar los pasos o recibir la orden de parar.

    Contexto:
        Comando entrenar-jefes. Con `entorno` (pruebas) usa ese entorno vectorizado en lugar de
        lanzar el juego. Una corrida nueva parte de mejor.zip de la imitacion y copia ese modelo como
        mejor.zip propio (el primer "mejor" es la imitacion); una existente se reanuda desde
        checkpoints/ultimo.zip.

    Args:
        config (ConfigPPOJefes): Hiperparametros y reglas.
        entorno (VecEnv, opcional): Entorno ya creado; entrenar_jefes no lo cierra si se lo dan.
        ruta_imitacion (Path, opcional): Modelo clonado de partida y referencia del KL; por defecto
            runs/multijefe_imitacion/mejor.zip.
        catalogo (CatalogoJefes, opcional): Catalogo de jefes; por defecto el de data/jefes.
        lanzador (Lanzador): Funcion que lanza la evaluacion automatica (inyectable en pruebas).

    Returns:
        Path: Ruta de ultimo.zip guardado al terminar.

    Raises:
        FileNotFoundError: Si falta el modelo clonado o la referencia de la imitacion.
        Exception: Cualquier error del entrenamiento tras guardar el estado y apagar todo.

    Ejemplo:
        entrenar_jefes(ConfigPPOJefes(nombre="prueba", pasos_totales=50_000))
    """
    import torch

    torch.set_num_threads(config.hilos_torch)
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    catalogo = catalogo or cargar_catalogo()
    imitacion = ruta_imitacion or CARPETA_IMITACION / "mejor.zip"
    rutas = rutas_ppo(config.nombre)
    carpeta = rutas.base
    referencia = referencia_de_imitacion(imitacion.parent / "referencia.json")
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(json.dumps(asdict(config), indent=1), encoding="utf-8")
    escribir_pid(ARCHIVO_PID)
    rutas.parada.unlink(missing_ok=True)
    propio = entorno is None
    vec = entorno or crear_vec_env_jefes(repartir_entornos(catalogo, config.instancias), ConfigEntorno(),
                                         config.semilla, trazar=config.trazar_golpes)
    reanudar = rutas.ultimo.exists()
    guardador = CallbackCheckpointJefes(carpeta / "checkpoints", config.checkpoint_cada)
    evaluacion = CallbackEvaluacionPeriodica(config, carpeta, referencia, lanzador)
    try:
        modelo = cargar_modelo_ppo(rutas.ultimo if reanudar else imitacion, vec, config, carpeta)
        modelo.configurar_kl(config, imitacion)
        if not rutas.mejor.exists():
            shutil.copyfile(imitacion, rutas.mejor)
        fase_normal = {j: catalogo.perfil(j).fases_inicio[0] for j in catalogo.jefes_de_episodio()}
        callbacks = [CallbackMetricasJefes(fase_normal), guardador, evaluacion, CallbackParada(rutas.parada)]
        try:
            modelo.learn(total_timesteps=config.pasos_totales, callback=callbacks,
                         reset_num_timesteps=not reanudar, tb_log_name="ppo")
        finally:
            guardador.model = modelo
            guardador.guardar(numerado=True)
    finally:
        rutas.parada.unlink(missing_ok=True)
        ARCHIVO_PID.unlink(missing_ok=True)
        if propio:
            try:
                vec.close()
            finally:
                apagar_descendientes(os.getpid())
    return rutas.ultimo
