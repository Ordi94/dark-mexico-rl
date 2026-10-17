"""
Entrenamiento MaskablePPO con varias instancias del juego en paralelo.

Cada entorno corre en su propio subproceso (SubprocVecEnv) y dueno de una instancia del juego.
El proceso principal lleva la red, el curriculo y las metricas. Todo se apaga con Ctrl+C, con el
archivo de parada de detener.ps1 o al terminar los pasos pedidos: se guarda ultimo.zip, se cierran
los juegos con su orden de cierre y, como ultima red de seguridad, se matan los procesos hijos.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from dmrl.aplicacion.callbacks import (
    CallbackCheckpoint,
    CallbackCurriculo,
    CallbackMejorModelo,
    CallbackMetricas,
    CallbackParada,
    cargar_estado_curriculo,
    guardar_estado_curriculo,
)
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.curriculo import Curriculo
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.datos import RAIZ_PROYECTO, cargar_tabla_golpes
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO, FabricaPuerto, ParametrosInstancia, crear_fabrica_puerto
from dmrl.infraestructura.procesos import ajustar_proceso_actual, apagar_descendientes, escribir_pid

CARPETA_RUNS = RAIZ_PROYECTO / "runs"
ARCHIVO_PID = CARPETA_RUNS / "entrenamiento.pid"


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """
    Hiperparametros y opciones de una corrida.

    Contexto:
        Se guarda como config.json en la carpeta de la corrida; los valores por defecto son
        los de la primera corrida (documentados en docs/).

    Atributos:
        nombre: carpeta de la corrida dentro de runs/.
        instancias: instancias del juego (un entorno cada una).
        pasos_totales: techo de pasos de los entornos; se puede parar antes.
        n_steps: pasos por entorno en cada rollout (rollout total = n_steps * instancias).
        batch_size: tamano de minibatch.
        n_epochs: pasadas por rollout.
        learning_rate: tasa de aprendizaje constante.
        gamma: descuento (0.995: horizonte de ~13 s de juego a 15 Hz).
        gae_lambda: lambda de la ventaja generalizada.
        clip_range: recorte de PPO.
        ent_coef: coeficiente de entropia (exploracion).
        vf_coef: peso de la perdida del valor.
        max_grad_norm: recorte del gradiente.
        capas: neuronas de cada capa oculta (politica y valor).
        dispositivo: "cpu", "cuda" o "auto".
        hilos_torch: hilos de CPU de PyTorch en el proceso principal.
        checkpoint_cada: pasos entre checkpoints.
        semilla: semilla base.
        reanudar: continuar desde ultimo.zip si existe.
        entorno: configuracion del entorno.
    """

    nombre: str = "principal"
    instancias: int = 4
    pasos_totales: int = 50_000_000
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    capas: tuple[int, ...] = (256, 256)
    dispositivo: str = "auto"
    hilos_torch: int = 2
    checkpoint_cada: int = 100_000
    semilla: int = 0
    reanudar: bool = True
    entorno: ConfigEntorno = field(default_factory=ConfigEntorno)

    @property
    def carpeta(self) -> Path:
        """
        Carpeta de la corrida.

        Contexto:
            Checkpoints, TensorBoard y archivo de parada.

        Args:
            Ninguno.

        Returns:
            Path: runs/<nombre>.

        Raises:
            Ninguna.

        Ejemplo:
            ConfigEntrenamiento().carpeta
        """
        return CARPETA_RUNS / self.nombre

    def a_dict(self) -> dict[str, Any]:
        """
        Representacion serializable.

        Contexto:
            Se guarda en config.json de la corrida.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Valores de tipos JSON.

        Raises:
            Ninguna.

        Ejemplo:
            ConfigEntrenamiento().a_dict()["gamma"]
        """
        return asdict(self)


def crear_entorno(indice: int, config: ConfigEntorno, semilla: int, prefijo: str = "tr",
                  fabrica: FabricaPuerto | None = None, tabla: TablaGolpes | None = None) -> Monitor:
    """
    Crea un entorno completo (instancia del juego incluida); se ejecuta en cada subproceso.

    Contexto:
        Funcion de nivel superior para que SubprocVecEnv (spawn) pueda enviarla a los
        subprocesos. Deja al proceso con prioridad Below Normal y sin las CPUs 0-3.

    Args:
        indice (int): Numero del entorno; forma el id de la instancia.
        config (ConfigEntorno): Configuracion del entorno.
        semilla (int): Semilla base.
        prefijo (str): Prefijo del id de instancia.
        fabrica (FabricaPuerto, opcional): Fabrica de puertos (pruebas); por defecto el juego real.
        tabla (TablaGolpes, opcional): Tabla de golpes (pruebas); por defecto la medida.

    Returns:
        Monitor: Entorno envuelto para las estadisticas de SB3.

    Raises:
        TimeoutError: Si el juego no se conecta.

    Ejemplo:
        env = crear_entorno(0, ConfigEntorno(), 0)
    """
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    puerto = fabrica or crear_fabrica_puerto(ParametrosInstancia(id_instancia=f"{prefijo}{indice}"))
    env = DarkMexicoEnv(puerto, tabla or cargar_tabla_golpes(), config, semilla=semilla + indice)
    return Monitor(env)


def crear_vec_env(instancias: int, config: ConfigEntorno, semilla: int,
                  fabricas: list[Callable[[], gym.Env[Any, Any]]] | None = None) -> VecEnv:
    """
    Crea el entorno vectorizado con una instancia del juego por entorno.

    Contexto:
        entrenar. Con `fabricas` (pruebas) usa un DummyVecEnv en proceso en lugar de subprocesos.

    Args:
        instancias (int): Numero de entornos.
        config (ConfigEntorno): Configuracion del entorno.
        semilla (int): Semilla base.
        fabricas (list, opcional): Funciones que crean cada entorno (sustituyen a las reales).

    Returns:
        VecEnv: Entorno vectorizado.

    Raises:
        ValueError: Si instancias es menor que 1.

    Ejemplo:
        venv = crear_vec_env(4, ConfigEntorno(), 0)
    """
    if instancias < 1:
        raise ValueError("instancias debe ser >= 1")
    if fabricas is not None:
        return DummyVecEnv(fabricas)
    return SubprocVecEnv(
        [functools.partial(crear_entorno, i, config, semilla) for i in range(instancias)], start_method="spawn"
    )


def elegir_dispositivo(preferido: str) -> str:
    """
    Resuelve "auto" al dispositivo mas rapido medido para esta red: la CPU.

    Contexto:
        entrenar. Medido con 4 instancias y la red de 2x256: CPU 437 pasos/s contra 358 en la
        RTX 3060 (cada paso transfiere un lote diminuto a la GPU y de vuelta, y el muestreo del
        juego es el cuello de botella, no la red). "cuda" sigue disponible de forma explicita.

    Args:
        preferido (str): "cpu", "cuda" o "auto".

    Returns:
        str: "cpu" o "cuda".

    Raises:
        ValueError: Si el valor no existe.

    Ejemplo:
        elegir_dispositivo("auto")  # "cpu"
    """
    if preferido not in ("cpu", "cuda", "auto"):
        raise ValueError(f"dispositivo desconocido: {preferido!r}")
    return "cpu" if preferido == "auto" else preferido


def crear_modelo(config: ConfigEntrenamiento, venv: VecEnv, dispositivo: str) -> Any:
    """
    Crea un MaskablePPO nuevo con los hiperparametros de la corrida.

    Contexto:
        entrenar, cuando no hay checkpoint que reanudar.

    Args:
        config (ConfigEntrenamiento): Hiperparametros.
        venv (VecEnv): Entorno vectorizado.
        dispositivo (str): "cpu" o "cuda".

    Returns:
        MaskablePPO: Modelo listo para learn.

    Raises:
        ValueError: Si los hiperparametros no son validos para SB3.

    Ejemplo:
        modelo = crear_modelo(config, venv, "cpu")
    """
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy", venv,
        learning_rate=config.learning_rate, n_steps=config.n_steps, batch_size=config.batch_size,
        n_epochs=config.n_epochs, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": {"pi": list(config.capas), "vf": list(config.capas)}},
        tensorboard_log=str(config.carpeta / "tb"), seed=config.semilla, device=dispositivo, verbose=0,
    )


def cargar_modelo(ruta: Path, venv: VecEnv, dispositivo: str, tensorboard_log: str | None = None) -> Any:
    """
    Carga un MaskablePPO guardado y lo conecta al entorno.

    Contexto:
        entrenar al reanudar, y evaluacion/ver_pelea con entorno propio. El zip recuerda la carpeta
        de TensorBoard con que se creo; al partir de un modelo de otra corrida (por ejemplo el de
        imitacion) hay que redirigirla a la corrida actual o las curvas se escribirian en la ajena.

    Args:
        ruta (Path): Archivo .zip.
        venv (VecEnv): Entorno vectorizado.
        dispositivo (str): "cpu" o "cuda".
        tensorboard_log (str, opcional): Carpeta de TensorBoard; None conserva la guardada en el zip.

    Returns:
        MaskablePPO: Modelo cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe.

    Ejemplo:
        modelo = cargar_modelo(Path("runs/a/checkpoints/ultimo.zip"), venv, "cpu")
    """
    from sb3_contrib import MaskablePPO

    if tensorboard_log is None:
        return MaskablePPO.load(ruta, env=venv, device=dispositivo)
    return MaskablePPO.load(ruta, env=venv, device=dispositivo, tensorboard_log=tensorboard_log)


def entrenar(config: ConfigEntrenamiento, venv: VecEnv | None = None) -> Path:
    """
    Ejecuta (o reanuda) un entrenamiento hasta agotar los pasos o recibir la orden de parar.

    Contexto:
        Comando entrenar. Con `venv` (pruebas) usa ese entorno vectorizado en lugar de lanzar
        el juego.

    Args:
        config (ConfigEntrenamiento): Hiperparametros y opciones.
        venv (VecEnv, opcional): Entorno ya creado; entrenar no lo cierra si se lo dan.

    Returns:
        Path: Ruta de ultimo.zip guardado al terminar.

    Raises:
        Exception: Cualquier error del entrenamiento tras guardar el estado y apagar todo.

    Ejemplo:
        entrenar(ConfigEntrenamiento(nombre="prueba", pasos_totales=50_000))
    """
    import torch

    torch.set_num_threads(config.hilos_torch)
    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    carpeta = config.carpeta
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(json.dumps(config.a_dict(), indent=1), encoding="utf-8")
    escribir_pid(ARCHIVO_PID)
    parada = carpeta / "PARAR"
    parada.unlink(missing_ok=True)
    propio = venv is None
    entorno = venv or crear_vec_env(config.instancias, config.entorno, config.semilla)
    dispositivo = elegir_dispositivo(config.dispositivo)
    curriculo = Curriculo(config.entorno.curriculo)
    checkpoints = carpeta / "checkpoints"
    ultimo = checkpoints / "ultimo.zip"
    try:
        if config.reanudar and ultimo.exists():
            modelo = cargar_modelo(ultimo, entorno, dispositivo, str(carpeta / "tb"))
            cargar_estado_curriculo(curriculo, checkpoints / "curriculo.json")
        else:
            modelo = crear_modelo(config, entorno, dispositivo)
        guardador = CallbackCheckpoint(checkpoints, curriculo, config.checkpoint_cada)
        callbacks = [
            CallbackMetricas(curriculo=curriculo), CallbackCurriculo(curriculo), guardador,
            CallbackMejorModelo(carpeta / "mejor.zip", curriculo), CallbackParada(parada),
        ]
        try:
            modelo.learn(total_timesteps=config.pasos_totales, callback=callbacks,
                         reset_num_timesteps=not (config.reanudar and ultimo.exists()),
                         tb_log_name="ppo")
        finally:
            guardador.model = modelo
            guardador.guardar()
            guardar_estado_curriculo(curriculo, checkpoints / "curriculo.json")
    finally:
        parada.unlink(missing_ok=True)
        if propio:
            try:
                entorno.close()
            finally:
                apagar_descendientes(os.getpid())
    return ultimo
