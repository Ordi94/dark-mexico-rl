"""
Utilidades de prueba del PPO multi-jefe: un entorno con recompensas, modelos pequenos guardados en
disco y dobles del proceso de evaluacion, su lanzador y el reloj.

Todo corre sin el juego: los modelos se crean sobre espacios identicos a los del entorno real pero con
una observacion minima y capas chicas, de modo que un rollout dura milisegundos.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento
from dmrl.aplicacion.entrenamiento_jefes import cargar_modelo_ppo
from dmrl.aplicacion.imitacion_red import EntornoEspacios, crear_modelo_base, guardar_modelo
from dmrl.aplicacion.ppo_kl import MaskablePPOKL
from dmrl.dominio.ppo_jefes import ConfigPPOJefes

TAMANO_SIMPLE_PRUEBA = 6
CAPAS_PRUEBA = (8, 8)


class EntornoAleatorio(EntornoEspacios):
    """
    Entorno con observaciones y recompensas al azar que publica el resumen de episodio de un jefe.

    Contexto:
        Entrenamiento de MaskablePPOKL y de los callbacks sin el juego. Con la recompensa nula de
        EntornoEspacios las ventajas serian cero y la politica casi no recibiria gradiente.
    """

    def __init__(self, jefe: str = "amlo", largo: int = 4, semilla: int = 0) -> None:
        """
        Crea el entorno.

        Contexto:
            vec_prueba.

        Args:
            jefe (str): Jefe que declara el resumen de cada episodio.
            largo (int): Pasos por episodio.
            semilla (int): Semilla del generador de observaciones y recompensas.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoAleatorio("nahual", largo=3)
        """
        super().__init__(4, TAMANO_SIMPLE_PRUEBA)
        self._jefe = jefe
        self._largo = largo
        self._azar = np.random.default_rng(semilla)
        self._n = 0

    def _obs(self) -> NDArray[np.float32]:
        """
        Observacion al azar dentro del espacio.

        Contexto:
            reset y step.

        Args:
            Ninguno.

        Returns:
            NDArray[np.float32]: Valores en [-1, 1].

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self._azar.uniform(-1.0, 1.0, self._forma).astype(np.float32)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Empieza un episodio.

        Contexto:
            API de Gymnasium.

        Args:
            seed (int, opcional): Ignorada.
            options (dict, opcional): Ignoradas.

        Returns:
            tuple[NDArray[np.float32], dict[str, Any]]: Observacion al azar e info vacia.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoAleatorio().reset()
        """
        self._n = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Avanza un paso; el ultimo del episodio publica el resumen.

        Contexto:
            API de Gymnasium.

        Args:
            action (int): Ignorada.

        Returns:
            tuple: Observacion, recompensa al azar, terminado, truncado e info con el episodio.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoAleatorio().step(0)
        """
        self._n += 1
        terminado = self._n >= self._largo
        info: dict[str, Any] = {}
        if terminado:
            info["episodio"] = {"jefe": self._jefe, "resultado": "victoria", "victoria": True, "fase_inicio": 1}
        return self._obs(), float(self._azar.normal()), terminado, False, info


def vec_prueba(jefes: tuple[str, ...] = ("amlo",), semilla: int = 0) -> DummyVecEnv:
    """
    Entorno vectorizado con un EntornoAleatorio por jefe.

    Contexto:
        Pruebas de entrenamiento sin juego.

    Args:
        jefes (tuple[str, ...]): Un entorno por cada jefe.
        semilla (int): Semilla base.

    Returns:
        DummyVecEnv: Entorno en proceso.

    Raises:
        Ninguna.

    Ejemplo:
        venv = vec_prueba(("amlo", "nahual"))
    """
    return DummyVecEnv([
        (lambda j=j, i=i: EntornoAleatorio(j, semilla=semilla + i)) for i, j in enumerate(jefes)  # type: ignore[misc]
    ])


def config_ppo(**cambios: Any) -> ConfigPPOJefes:
    """
    Configuracion de PPO con rollouts diminutos.

    Contexto:
        Pruebas de entrenamiento.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        ConfigPPOJefes: 32 pasos por rollout, minibatch de 32 y dos epocas.

    Raises:
        ValueError: Si los cambios no son validos.

    Ejemplo:
        config_ppo(pasos_calentamiento=0)
    """
    cambios.setdefault("minutos_primera_evaluacion", cambios.get("minutos_evaluacion", 1.0))
    base: dict[str, Any] = {
        "nombre": "prueba_ppo", "n_steps": 32, "batch_size": 32, "n_epochs": 2, "learning_rate": 1e-3,
        "lr_minimo": 1e-4, "pasos_calentamiento": 0, "beta_kl": 0.0,
        "ent_coef": 0.01, "target_kl": 100.0,
    }
    base.update(cambios)
    return ConfigPPOJefes(**base)


def guardar_modelo_pequeno(ruta: Path, semilla: int = 0) -> Path:
    """
    Guarda un MaskablePPO nuevo con la forma de la observacion de prueba.

    Contexto:
        Hace de modelo clonado de partida y de referencia del KL.

    Args:
        ruta (Path): Zip de destino.
        semilla (int): Semilla de los pesos iniciales.

    Returns:
        Path: La misma ruta.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        guardar_modelo_pequeno(tmp_path / "mejor.zip")
    """
    modelo = crear_modelo_base(ConfigEntrenamiento(nombre="x", capas=CAPAS_PRUEBA, semilla=semilla),
                               TAMANO_SIMPLE_PRUEBA)
    guardar_modelo(modelo, ruta)
    return ruta


def cargar_modelo_prueba(carpeta: Path, config: ConfigPPOJefes, semilla: int = 0,
                         jefes: tuple[str, ...] = ("amlo",)) -> tuple[MaskablePPOKL, Path]:
    """
    Carga un modelo pequeno como MaskablePPOKL, sin referencia y con un logger sin salidas.

    Contexto:
        Pruebas de callbacks que necesitan una politica y un optimizador reales.

    Args:
        carpeta (Path): Carpeta donde se guarda el zip de partida y se apunta TensorBoard.
        config (ConfigPPOJefes): Regla de la corrida.
        semilla (int): Semilla de los pesos iniciales.
        jefes (tuple[str, ...]): Jefes de los entornos de entrenamiento.

    Returns:
        tuple[MaskablePPOKL, Path]: Modelo listo para learn y zip de partida.

    Raises:
        OSError: Si no se puede escribir el zip.

    Ejemplo:
        modelo, inicial = cargar_modelo_prueba(tmp_path, config_ppo())
    """
    inicial = guardar_modelo_pequeno(carpeta / f"inicial_{semilla}.zip", semilla)
    modelo = cargar_modelo_ppo(inicial, vec_prueba(jefes), config, carpeta / "corrida")
    modelo.configurar_kl(config, None)
    modelo.set_logger(configure(None, []))
    return modelo, inicial


def pesos_de(parametros: Any) -> list[torch.Tensor]:
    """
    Copia de los pesos de un conjunto de parametros.

    Contexto:
        Comparar antes y despues de entrenar.

    Args:
        parametros (Any): Iterable de parametros de torch.

    Returns:
        list[torch.Tensor]: Copias independientes.

    Raises:
        Ninguna.

    Ejemplo:
        antes = pesos_de(modelo.policy.action_net.parameters())
    """
    return [p.detach().clone() for p in parametros]


def son_iguales(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    """
    Indica si dos listas de pesos coinciden exactamente.

    Contexto:
        Comprobar congelamiento o restauracion de pesos.

    Args:
        a (list[torch.Tensor]): Primera lista.
        b (list[torch.Tensor]): Segunda lista.

    Returns:
        bool: True si todos los tensores son identicos.

    Raises:
        Ninguna.

    Ejemplo:
        son_iguales(antes, despues)
    """
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def observaciones_fijas(n: int = 64, semilla: int = 123) -> torch.Tensor:
    """
    Lote fijo de observaciones para medir divergencias entre politicas.

    Contexto:
        Comparar politicas sobre los mismos estados.

    Args:
        n (int): Filas.
        semilla (int): Semilla.

    Returns:
        torch.Tensor: Observaciones de forma (n, 4 * TAMANO_SIMPLE_PRUEBA).

    Raises:
        Ninguna.

    Ejemplo:
        obs = observaciones_fijas()
    """
    azar = np.random.default_rng(semilla)
    return torch.from_numpy(azar.uniform(-1.0, 1.0, (n, 4 * TAMANO_SIMPLE_PRUEBA)).astype(np.float32))


class LoggerFalso:
    """Guarda los valores registrados."""

    def __init__(self) -> None:
        """
        Crea el diccionario de valores.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            LoggerFalso()
        """
        self.valores: dict[str, float] = {}

    def record(self, clave: str, valor: float, exclude: Any = None) -> None:
        """
        Registra un valor.

        Contexto:
            Sustituye a Logger.record.

        Args:
            clave (str): Nombre de la metrica.
            valor (float): Valor.
            exclude (Any, opcional): Ignorado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            logger.record("dm/x", 1.0)
        """
        self.valores[clave] = valor


class ModeloFalso:
    """Modelo minimo para los callbacks que solo cuentan pasos, registran y guardan."""

    def __init__(self) -> None:
        """
        Crea el modelo con contadores en cero.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ModeloFalso()
        """
        self.num_timesteps = 0
        self.logger = LoggerFalso()
        self.contenido = b"modelo"
        self.policy = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]))

    def get_env(self) -> None:
        """
        Sin entorno.

        Contexto:
            Sustituye a BaseAlgorithm.get_env.

        Args:
            Ninguno.

        Returns:
            None: Siempre None.

        Raises:
            Ninguna.

        Ejemplo:
            modelo.get_env()
        """
        return None

    def save(self, ruta: Path | str) -> None:
        """
        Escribe un archivo para poder comprobar que se guardo.

        Contexto:
            Sustituye a BaseAlgorithm.save.

        Args:
            ruta (Path | str): Destino.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            modelo.save("m.zip")
        """
        Path(ruta).write_bytes(self.contenido)


def resumen_jefe(jefe: str, victoria: bool, **extra: Any) -> dict[str, Any]:
    """
    Resumen de episodio de un jefe como el que publica el entorno.

    Contexto:
        Entrada de CallbackMetricasJefes.

    Args:
        jefe (str): Jefe del episodio.
        victoria (bool): Si gano el jugador.
        **extra (Any): Campos a sobrescribir.

    Returns:
        dict[str, Any]: Resumen con dano, parries, duracion, recompensa y fase de inicio.

    Raises:
        Ninguna.

    Ejemplo:
        resumen_jefe("amlo", True)
    """
    base: dict[str, Any] = {
        "jefe": jefe, "resultado": "victoria" if victoria else "muerte", "victoria": victoria,
        "muerte": not victoria, "fase_inicio": 1, "danio_hecho": 100.0, "danio_recibido": 40.0,
        "parries": 2, "duracion_s": 30.0, "progreso_jefe": 0.5, "recompensa": 1.0,
    }
    base.update(extra)
    return base


class ProcesoFalso:
    """Proceso de evaluacion que termina cuando la prueba lo indica."""

    def __init__(self, codigo: int = 0) -> None:
        """
        Crea el proceso en marcha.

        Contexto:
            LanzadorFalso.

        Args:
            codigo (int): Codigo de salida que tendra al terminar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ProcesoFalso(1)
        """
        self.codigo = codigo
        self.terminado = False
        self.returncode: int | None = None
        self.terminaciones = 0

    def terminar_ya(self) -> None:
        """
        Marca el proceso como terminado.

        Contexto:
            La prueba simula que la evaluacion acabo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            proceso.terminar_ya()
        """
        self.terminado = True
        self.returncode = self.codigo

    def poll(self) -> int | None:
        """
        Codigo de salida si termino.

        Contexto:
            Sustituye a Popen.poll.

        Args:
            Ninguno.

        Returns:
            int | None: Codigo o None mientras corre.

        Raises:
            Ninguna.

        Ejemplo:
            proceso.poll()
        """
        return self.returncode

    def terminate(self) -> None:
        """
        Anota la peticion de terminar.

        Contexto:
            Sustituye a Popen.terminate.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            proceso.terminate()
        """
        self.terminaciones += 1


class LanzadorFalso:
    """Lanzador que escribe el resultado de la evaluacion y devuelve un proceso falso."""

    def __init__(self, tasas: Mapping[str, float] | None = None, codigo: int = 0,
                 terminar_al_lanzar: bool = False) -> None:
        """
        Crea el lanzador.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            tasas (Mapping[str, float], opcional): Tasas que deja el resultado; None no escribe archivo.
            codigo (int): Codigo de salida del proceso.
            terminar_al_lanzar (bool): Si es True el proceso nace ya terminado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            LanzadorFalso({"amlo": 0.9})
        """
        self.tasas = dict(tasas) if tasas is not None else None
        self.codigo = codigo
        self.terminar_al_lanzar = terminar_al_lanzar
        self.llamadas: list[tuple[Path, Path]] = []
        self.procesos: list[ProcesoFalso] = []

    def __call__(self, modelo: Path, salida: Path, config: ConfigPPOJefes) -> ProcesoFalso:
        """
        Registra la llamada, deja el resultado y devuelve el proceso.

        Contexto:
            Lo invoca CallbackEvaluacionPeriodica.

        Args:
            modelo (Path): Copia del modelo a evaluar.
            salida (Path): JSON de resultado.
            config (ConfigPPOJefes): Configuracion de la corrida.

        Returns:
            ProcesoFalso: Proceso simulado.

        Raises:
            OSError: Si no se puede escribir el resultado.

        Ejemplo:
            proceso = lanzador(Path("m.zip"), Path("e.json"), ConfigPPOJefes())
        """
        self.llamadas.append((modelo, salida))
        if self.tasas is not None:
            salida.write_text(json.dumps({"por_jefe": {j: {"tasa": t} for j, t in self.tasas.items()}}),
                              encoding="utf-8")
        proceso = ProcesoFalso(self.codigo)
        if self.terminar_al_lanzar:
            proceso.terminar_ya()
        self.procesos.append(proceso)
        return proceso


class RelojFalso:
    """Reloj monotono que solo avanza cuando la prueba lo indica."""

    def __init__(self) -> None:
        """
        Crea el reloj en cero.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            RelojFalso()
        """
        self.ahora = 0.0

    def __call__(self) -> float:
        """
        Hora actual.

        Contexto:
            Sustituye a time.monotonic.

        Args:
            Ninguno.

        Returns:
            float: Segundos acumulados.

        Raises:
            Ninguna.

        Ejemplo:
            reloj()
        """
        return self.ahora

    def avanzar(self, segundos: float) -> None:
        """
        Adelanta el reloj.

        Contexto:
            La prueba hace vencer el plazo de evaluacion.

        Args:
            segundos (float): Segundos a sumar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            reloj.avanzar(3600)
        """
        self.ahora += segundos
