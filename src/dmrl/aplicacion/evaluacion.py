"""
Evaluacion de politicas: juega N peleas completas y resume tasa de victoria, dano y parries.

Las politicas se envuelven con la misma interfaz (reiniciar/decidir) para comparar el bot de
reglas, un modelo entrenado y el azar con las mismas condiciones: jefe al 100% de vida y
episodios que empiezan en la fase 1, salvo que se pida otra cosa.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio.bot_reglas import BotReglas
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import ruido_dart
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.datos import cargar_tabla_golpes
from dmrl.infraestructura.juego import ParametrosInstancia, crear_fabrica_puerto
from dmrl.infraestructura.procesos import ajustar_proceso_actual

OPCIONES_PELEA_COMPLETA: dict[str, Any] = {"start_phase": 1, "boss_hp_mult": 1.0}


class Politica(Protocol):
    """Interfaz comun de las politicas evaluables."""

    def reiniciar(self) -> None:
        """Prepara la politica para un episodio nuevo."""
        ...

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """Elige la accion del siguiente paso."""
        ...


class PoliticaBot:
    """
    El bot de reglas como politica evaluable.

    Contexto:
        evaluar --politica bot; usa la observacion cruda del entorno, no el vector.
    """

    def __init__(self, tabla: TablaGolpes) -> None:
        """
        Crea la politica con la tabla de golpes.

        Contexto:
            Arranque de la evaluacion.

        Args:
            tabla (TablaGolpes): Golpes medidos del jefe.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica = PoliticaBot(tabla)
        """
        self._bot = BotReglas(tabla)

    def reiniciar(self) -> None:
        """
        Olvida el estado del episodio anterior.

        Contexto:
            Antes de cada episodio.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica.reiniciar()
        """
        self._bot.reiniciar()

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """
        Accion del bot para el estado actual.

        Contexto:
            Bucle de jugar_episodios.

        Args:
            env (DarkMexicoEnv): Entorno (de el se leen la observacion cruda y la mascara).
            obs (NDArray[np.float32]): Observacion apilada (no usada por el bot).

        Returns:
            int: Accion valida.

        Raises:
            Ninguna.

        Ejemplo:
            accion = politica.decidir(env, obs)
        """
        return self._bot.decidir(env.observacion_cruda, env.action_masks().tolist())


class PoliticaAleatoria:
    """
    Accion al azar entre las validas segun la mascara.

    Contexto:
        Validacion del entorno (episodios aleatorios) y linea base minima.
    """

    def __init__(self, semilla: int | None = None) -> None:
        """
        Crea la politica con su generador.

        Contexto:
            Arranque de la validacion.

        Args:
            semilla (int, opcional): Semilla del generador.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica = PoliticaAleatoria(1)
        """
        self._azar = random.Random(semilla)

    def reiniciar(self) -> None:
        """
        Sin estado entre episodios.

        Contexto:
            Cumple la interfaz Politica.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica.reiniciar()
        """

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """
        Accion valida al azar.

        Contexto:
            Bucle de jugar_episodios.

        Args:
            env (DarkMexicoEnv): Entorno (de el se lee la mascara).
            obs (NDArray[np.float32]): Observacion apilada (no usada).

        Returns:
            int: Accion valida.

        Raises:
            Ninguna.

        Ejemplo:
            accion = politica.decidir(env, obs)
        """
        validas = np.flatnonzero(env.action_masks())
        return int(validas[self._azar.randrange(len(validas))])


class PoliticaModelo:
    """
    Un modelo MaskablePPO como politica evaluable (determinista).

    Contexto:
        evaluar --politica modelo y ver_pelea.
    """

    def __init__(self, modelo: Any, deterministico: bool = True) -> None:
        """
        Envuelve un modelo ya cargado.

        Contexto:
            Arranque de la evaluacion o de ver_pelea.

        Args:
            modelo (Any): MaskablePPO con predict(obs, action_masks=..., deterministic=...).
            deterministico (bool): Si es True usa la accion de mayor probabilidad.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica = PoliticaModelo(MaskablePPO.load("mejor.zip"))
        """
        self._modelo = modelo
        self._deterministico = deterministico

    def reiniciar(self) -> None:
        """
        Sin estado entre episodios (la memoria corta va en la observacion apilada).

        Contexto:
            Cumple la interfaz Politica.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica.reiniciar()
        """

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """
        Accion del modelo respetando la mascara.

        Contexto:
            Bucle de jugar_episodios.

        Args:
            env (DarkMexicoEnv): Entorno (de el se lee la mascara).
            obs (NDArray[np.float32]): Observacion apilada.

        Returns:
            int: Accion elegida.

        Raises:
            Ninguna.

        Ejemplo:
            accion = politica.decidir(env, obs)
        """
        accion, _ = self._modelo.predict(obs, action_masks=env.action_masks(), deterministic=self._deterministico)
        return int(accion)


class PoliticaConRuido:
    """
    Envuelve una politica y, con probabilidad epsilon por paso, ejecuta una accion valida al azar.

    Contexto:
        Evaluacion de robustez: una politica que solo funciona sobre la trayectoria exacta que
        aprendio (sobreajuste a estados identicos) se degrada al perturbar su ejecucion.
    """

    def __init__(self, base: Politica, epsilon: float, semilla: int | None = None) -> None:
        """
        Crea la politica con ruido.

        Contexto:
            construir_politica cuando se pide epsilon_ruido > 0.

        Args:
            base (Politica): Politica envuelta.
            epsilon (float): Probabilidad de ruido por paso, en [0, 1].
            semilla (int, opcional): Semilla del generador.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si epsilon esta fuera de [0, 1].

        Ejemplo:
            politica = PoliticaConRuido(PoliticaBot(tabla), 0.1, 0)
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon fuera de [0, 1]: {epsilon}")
        self._base = base
        self._epsilon = epsilon
        self._azar = random.Random(semilla)

    def reiniciar(self) -> None:
        """
        Reinicia la politica envuelta.

        Contexto:
            Antes de cada episodio.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica.reiniciar()
        """
        self._base.reiniciar()

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """
        Accion de la politica base o, con probabilidad epsilon, una valida al azar.

        Contexto:
            Bucle de jugar_episodios. La politica base se consulta siempre para que su memoria
            interna siga el combate.

        Args:
            env (DarkMexicoEnv): Entorno (de el se lee la mascara).
            obs (NDArray[np.float32]): Observacion apilada.

        Returns:
            int: Accion valida.

        Raises:
            Ninguna.

        Ejemplo:
            accion = politica.decidir(env, obs)
        """
        accion = self._base.decidir(env, obs)
        return ruido_dart(accion, env.action_masks().tolist(), self._epsilon, self._azar)


def jugar_episodios(env: DarkMexicoEnv, politica: Politica, episodios: int,
                    opciones: dict[str, Any] | None = None,
                    al_terminar: Callable[[int, dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """
    Juega episodios completos con una politica.

    Contexto:
        Evaluacion (bot, modelo, azar) y ver_pelea.

    Args:
        env (DarkMexicoEnv): Entorno ya creado.
        politica (Politica): Politica a evaluar.
        episodios (int): Cantidad de episodios.
        opciones (dict, opcional): Opciones de reset (por defecto, pelea completa al 100%).
        al_terminar (Callable, opcional): Se llama con (numero, resumen) tras cada episodio.

    Returns:
        list[dict[str, Any]]: Un resumen por episodio (el "episodio" de la info del entorno).

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        resumenes = jugar_episodios(env, PoliticaBot(tabla), 50)
    """
    opciones = OPCIONES_PELEA_COMPLETA if opciones is None else opciones
    resumenes: list[dict[str, Any]] = []
    for numero in range(episodios):
        obs, _ = env.reset(options=opciones)
        politica.reiniciar()
        while True:
            obs, _, terminado, truncado, info = env.step(politica.decidir(env, obs))
            if terminado or truncado:
                resumen = info["episodio"]
                resumenes.append(resumen)
                if al_terminar:
                    al_terminar(numero + 1, resumen)
                break
    return resumenes


def resumir(episodios: Sequence[dict[str, Any]]) -> dict[str, float]:
    """
    Agrega los resumenes de episodios en las metricas de la evaluacion.

    Contexto:
        Salida de evaluar y del informe del bot.

    Args:
        episodios (Sequence[dict[str, Any]]): Resumenes de jugar_episodios.

    Returns:
        dict[str, float]: Episodios, tasa de victoria, dano medio, parries medios, etc.; vacio
            si no hay episodios.

    Raises:
        Ninguna.

    Ejemplo:
        resumir([{"victoria": True, "danio_hecho": 2700.0}])
    """
    if not episodios:
        return {}

    def media(clave: str) -> float:
        """
        Media de una clave numerica entre los episodios.

        Contexto:
            resumir.

        Args:
            clave (str): Nombre del campo.

        Returns:
            float: Media; 0.0 si ningun episodio lo trae.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        valores = [float(e[clave]) for e in episodios if clave in e]
        return statistics.fmean(valores) if valores else 0.0

    n = len(episodios)
    return {
        "episodios": float(n),
        "tasa_victoria": sum(1 for e in episodios if e.get("victoria")) / n,
        "tasa_fase2": sum(1 for e in episodios if e.get("fase_max", 1) >= 2) / n,
        "danio_hecho": media("danio_hecho"),
        "danio_recibido": media("danio_recibido"),
        "progreso_jefe": media("progreso_jefe"),
        "parries": media("parries"),
        "caguamas_usadas": media("caguamas_usadas"),
        "caguamas_interrumpidas": media("caguamas_interrumpidas"),
        "duracion_s": media("duracion_s"),
        "recompensa": media("recompensa"),
        "errores": float(sum(1 for e in episodios if e.get("resultado") == "error")),
    }


@dataclass(frozen=True)
class TrabajoEvaluacion:
    """
    Encargo para un proceso de evaluacion (debe poder viajar a otro proceso).

    Contexto:
        evaluar_en_paralelo.

    Atributos:
        indice: numero del trabajador, parte del id de la instancia del juego.
        episodios: peleas que le toca jugar.
        politica: "bot", "azar" o "modelo".
        ruta_modelo: archivo .zip del modelo cuando politica es "modelo".
        semilla: semilla base del trabajador.
        opciones: opciones de reset de cada episodio.
        prefijo_id: prefijo del id de instancia, para no chocar con otros procesos.
        config: configuracion del entorno.
    """

    indice: int
    episodios: int
    politica: str
    ruta_modelo: str | None = None
    semilla: int = 0
    opciones: dict[str, Any] = field(default_factory=lambda: dict(OPCIONES_PELEA_COMPLETA))
    prefijo_id: str = "ev"
    config: ConfigEntorno = field(default_factory=ConfigEntorno)
    epsilon_ruido: float = 0.0


def construir_politica(nombre: str, tabla: TablaGolpes, ruta_modelo: str | None, semilla: int) -> Politica:
    """
    Crea la politica pedida por nombre.

    Contexto:
        Trabajadores de evaluacion y ver_pelea.

    Args:
        nombre (str): "bot", "azar" o "modelo".
        tabla (TablaGolpes): Golpes medidos del jefe.
        ruta_modelo (str, opcional): Archivo del modelo para "modelo".
        semilla (int): Semilla para "azar".

    Returns:
        Politica: Politica lista para jugar_episodios.

    Raises:
        ValueError: Si el nombre no existe o falta la ruta del modelo.

    Ejemplo:
        politica = construir_politica("bot", tabla, None, 0)
    """
    if nombre == "bot":
        return PoliticaBot(tabla)
    if nombre == "azar":
        return PoliticaAleatoria(semilla)
    if nombre == "modelo":
        if not ruta_modelo:
            raise ValueError("la politica 'modelo' necesita ruta_modelo")
        from sb3_contrib import MaskablePPO

        return PoliticaModelo(MaskablePPO.load(ruta_modelo, device="cpu"))
    raise ValueError(f"politica desconocida: {nombre!r}")


def ejecutar_trabajo(trabajo: TrabajoEvaluacion) -> list[dict[str, Any]]:
    """
    Juega las peleas de un trabajador con su propia instancia del juego.

    Contexto:
        Funcion de nivel superior ejecutada en cada proceso de evaluar_en_paralelo.

    Args:
        trabajo (TrabajoEvaluacion): Encargo del trabajador.

    Returns:
        list[dict[str, Any]]: Resumenes de sus episodios.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_trabajo(TrabajoEvaluacion(0, 5, "bot"))
    """
    from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO

    ajustar_proceso_actual(AFINIDAD_ENTRENAMIENTO)
    tabla = cargar_tabla_golpes()
    fabrica = crear_fabrica_puerto(ParametrosInstancia(id_instancia=f"{trabajo.prefijo_id}{trabajo.indice}"))
    env = DarkMexicoEnv(fabrica, tabla, trabajo.config, semilla=trabajo.semilla + trabajo.indice)
    try:
        politica = construir_politica(trabajo.politica, tabla, trabajo.ruta_modelo, trabajo.semilla + trabajo.indice)
        if trabajo.epsilon_ruido > 0.0:
            politica = PoliticaConRuido(politica, trabajo.epsilon_ruido, trabajo.semilla + trabajo.indice)
        return jugar_episodios(env, politica, trabajo.episodios, trabajo.opciones)
    finally:
        env.close()


def repartir(episodios: int, trabajadores: int) -> list[int]:
    """
    Reparte episodios entre trabajadores lo mas parejo posible.

    Contexto:
        evaluar_en_paralelo.

    Args:
        episodios (int): Total de peleas.
        trabajadores (int): Procesos disponibles (>= 1).

    Returns:
        list[int]: Peleas por trabajador; sin ceros.

    Raises:
        ValueError: Si trabajadores es menor que 1.

    Ejemplo:
        repartir(50, 4)  # [13, 13, 12, 12]
    """
    if trabajadores < 1:
        raise ValueError("trabajadores debe ser >= 1")
    base, resto = divmod(episodios, trabajadores)
    reparto = [base + (1 if i < resto else 0) for i in range(trabajadores)]
    return [r for r in reparto if r > 0]


def evaluar_en_paralelo(politica: str, episodios: int, instancias: int, ruta_modelo: Path | None = None,
                        semilla: int = 0, opciones: dict[str, Any] | None = None,
                        config: ConfigEntorno | None = None, prefijo_id: str = "ev",
                        ejecutor: Executor | None = None, epsilon_ruido: float = 0.0) -> list[dict[str, Any]]:
    """
    Juega las peleas repartidas entre varias instancias del juego, cada una en su proceso.

    Contexto:
        Comando evaluar. Con una sola instancia el trabajo corre igual, en un proceso aparte.

    Args:
        politica (str): "bot", "azar" o "modelo".
        episodios (int): Total de peleas.
        instancias (int): Instancias del juego en paralelo.
        ruta_modelo (Path, opcional): Modelo para la politica "modelo".
        semilla (int): Semilla base.
        opciones (dict, opcional): Opciones de reset de cada episodio.
        config (ConfigEntorno, opcional): Configuracion del entorno.
        prefijo_id (str): Prefijo de los ids de instancia.
        ejecutor (Executor, opcional): Ejecutor a usar (por defecto procesos con spawn).
        epsilon_ruido (float): Probabilidad por paso de ejecutar una accion valida al azar
            (evaluacion de robustez); 0 = la politica pura.

    Returns:
        list[dict[str, Any]]: Resumenes de todos los episodios.

    Raises:
        ValueError: Si instancias es menor que 1.

    Ejemplo:
        resumenes = evaluar_en_paralelo("bot", 50, 4)
    """
    reparto = repartir(episodios, instancias)
    trabajos = [
        TrabajoEvaluacion(
            indice=i, episodios=n, politica=politica, ruta_modelo=str(ruta_modelo) if ruta_modelo else None,
            semilla=semilla, opciones=dict(OPCIONES_PELEA_COMPLETA if opciones is None else opciones),
            prefijo_id=prefijo_id, config=config or ConfigEntorno(), epsilon_ruido=epsilon_ruido,
        )
        for i, n in enumerate(reparto)
    ]
    propio = ejecutor is None
    pool = ejecutor or ProcessPoolExecutor(max_workers=max(1, len(trabajos)), mp_context=get_context("spawn"))
    try:
        resultados = list(pool.map(ejecutar_trabajo, trabajos))
    finally:
        if propio:
            pool.shutdown(wait=True, cancel_futures=True)
    return [resumen for lista in resultados for resumen in lista]
