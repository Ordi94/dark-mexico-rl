"""
Entorno Gymnasium "DarkMexicoEnv": una instancia del juego por entorno.

Un episodio es una pelea completa contra el jefe, desde que el jefe esta activo hasta la muerte
del jugador, la victoria o el truncado por tiempo de juego. El reinicio, el teleport frente a la
niebla y el boton de interactuar los hace el mod (ver docs/); aqui se decide la
dificultad del episodio (curriculo), se mantiene el enfoque al jefe, se traduce la accion discreta,
se calcula la recompensa y se arma la observacion apilada.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from dmbridge import Action, BridgeError
from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.curriculo import vida_efectiva_jefe
from dmrl.dominio.observacion import TAMANO_SIMPLE, ApiladorObservaciones, ConstructorObservacion
from dmrl.dominio.recompensa import (
    RESULTADO_MUERTE,
    RESULTADO_TIEMPO,
    RESULTADO_VICTORIA,
    CalculadoraRecompensa,
    resultado_final,
)
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.juego import FabricaPuerto, PuertoJuego, cerrar_puerto

RESULTADO_ERROR = "error"
INTENTOS_RELANZAR = 3


def a_action(comando: acc.ComandoJuego) -> Action:
    """
    Convierte la orden elemental del dominio a la accion del protocolo del puente.

    Contexto:
        DarkMexicoEnv.step. Es el unico punto donde el dominio toca el formato de dmbridge.

    Args:
        comando (acc.ComandoJuego): Orden del paso.

    Returns:
        Action: Accion validada del protocolo (movimiento en el marco del jefe).

    Raises:
        ValueError: Si el movimiento esta fuera de [-1, 1].

    Ejemplo:
        a_action(acc.comando_de_accion(acc.ACCION_ATAQUE))
    """
    return Action(btn=comando.btn, lb=comando.lb, mx=comando.mx, my=comando.my, mf="boss")


class DarkMexicoEnv(gym.Env[NDArray[np.float32], int]):
    """
    Entorno de una instancia del juego.

    Contexto:
        Lo crean las fabricas de entrenamiento (SubprocVecEnv), la evaluacion y ver_pelea.
        Expone action_masks() para MaskablePPO.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(self, fabrica: FabricaPuerto, tabla: TablaGolpes, config: ConfigEntorno | None = None,
                 semilla: int | None = None, multiplicador: float = 1.0, prob_fase2: float = 0.0) -> None:
        """
        Crea el entorno y lanza su instancia del juego.

        Contexto:
            Construccion de cada entorno del entrenamiento; la instancia queda lista para reset.

        Args:
            fabrica (FabricaPuerto): Funcion que lanza el juego y devuelve su cliente.
            tabla (TablaGolpes): Golpes medidos del jefe (para la observacion).
            config (ConfigEntorno, opcional): Parametros del entorno.
            semilla (int, opcional): Semilla del azar propio (fase de inicio).
            multiplicador (float): Multiplicador inicial de la vida del jefe, en [0.05, 1].
            prob_fase2 (float): Probabilidad inicial de empezar directo en la fase 2.

        Returns:
            None: No retorna valor.

        Raises:
            TimeoutError: Si el juego no se conecta.
            BridgeError: Si el saludo falla.

        Ejemplo:
            env = DarkMexicoEnv(fabrica, tabla)
        """
        super().__init__()
        self.config = config or ConfigEntorno()
        self._fabrica = fabrica
        self._constructor = ConstructorObservacion(tabla, self.config)
        self._apilador = ApiladorObservaciones(TAMANO_SIMPLE, self.config.apilado)
        self._calculadora = CalculadoraRecompensa(self.config.recompensa)
        self._azar = random.Random(semilla)
        self.observation_space = spaces.Box(-1.0, 1.0, (TAMANO_SIMPLE * self.config.apilado,), np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(acc.N_ACCIONES)  # type: ignore[assignment]
        self._multiplicador = multiplicador
        self._prob_fase2 = prob_fase2
        self.reinicios_instancia = 0
        self._puerto: PuertoJuego | None = None
        self._cruda: dict[str, Any] = {}
        self._accion_previa = acc.ACCION_NADA
        self._frames = 0
        self._pasos_desde_lock = 0
        self._fase_inicio = 1
        self._multiplicador_episodio = multiplicador
        self._recompensa_total = 0.0
        self._puerto = self._fabrica()

    @property
    def observacion_cruda(self) -> dict[str, Any]:
        """
        Ultima observacion cruda del puente (la que ve el bot de reglas).

        Contexto:
            Politicas de evaluacion basadas en reglas.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Observacion cruda; vacia antes del primer reset.

        Raises:
            Ninguna.

        Ejemplo:
            env.observacion_cruda["player"]["hp"]
        """
        return self._cruda

    def fijar_multiplicador(self, multiplicador: float) -> None:
        """
        Cambia el multiplicador de vida del jefe de los proximos episodios.

        Contexto:
            CallbackCurriculo, via env_method, cuando sube el nivel.

        Args:
            multiplicador (float): Valor en [0.05, 1].

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si esta fuera de rango.

        Ejemplo:
            env.fijar_multiplicador(0.5)
        """
        if not 0.05 <= multiplicador <= 1.0:
            raise ValueError(f"multiplicador fuera de [0.05, 1]: {multiplicador}")
        self._multiplicador = multiplicador

    def fijar_prob_fase2(self, probabilidad: float) -> None:
        """
        Cambia la probabilidad de empezar directo en la fase 2.

        Contexto:
            Configuracion del curriculo (CallbackCurriculo y evaluacion).

        Args:
            probabilidad (float): Valor en [0, 1].

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si esta fuera de rango.

        Ejemplo:
            env.fijar_prob_fase2(0.2)
        """
        if not 0.0 <= probabilidad <= 1.0:
            raise ValueError(f"probabilidad fuera de [0, 1]: {probabilidad}")
        self._prob_fase2 = probabilidad

    def action_masks(self) -> NDArray[np.bool_]:
        """
        Mascara de acciones validas en el estado actual.

        Contexto:
            MaskablePPO la consulta antes de muestrear cada accion.

        Args:
            Ninguno.

        Returns:
            NDArray[np.bool_]: 22 valores; solo "nada" antes del primer reset.

        Raises:
            Ninguna.

        Ejemplo:
            env.action_masks()
        """
        return np.asarray(acc.mascara_acciones(self._cruda.get("player")), dtype=np.bool_)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Empieza una pelea nueva con el jefe ya activo.

        Contexto:
            Gymnasium/SB3 al empezar y tras cada episodio terminado.

        Args:
            seed (int, opcional): Reinicia el azar propio.
            options (dict, opcional): "start_phase" (1|2) y "boss_hp_mult" fuerzan la dificultad
                del episodio; sin ellas se usan el curriculo y la probabilidad de fase 2.

        Returns:
            tuple[NDArray[np.float32], dict[str, Any]]: Observacion apilada e info vacia.

        Raises:
            RuntimeError: Si tras varios relanzamientos el juego no consigue reiniciar.

        Ejemplo:
            obs, info = env.reset()
        """
        super().reset(seed=seed)
        if seed is not None:
            self._azar = random.Random(seed)
        opciones = options or {}
        fase = opciones.get("start_phase")
        if fase is None:
            fase = 2 if self._azar.random() < self._prob_fase2 else 1
        multiplicador = float(opciones.get("boss_hp_mult", self._multiplicador))
        self._cruda = self._reiniciar_juego(multiplicador, int(fase))
        self._al_observar()
        self._fase_inicio = int(fase)
        self._multiplicador_episodio = multiplicador
        self._calculadora.reiniciar(self._cruda)
        self._accion_previa = acc.ACCION_NADA
        self._frames = 0
        self._pasos_desde_lock = self.config.pasos_entre_intentos_lock
        self._recompensa_total = 0.0
        vector = self._constructor.construir(self._cruda, self._accion_previa, 0.0)
        return self._apilador.iniciar(vector), {}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Ejecuta una decision del agente (frames_por_paso frames de juego).

        Contexto:
            Gymnasium/SB3, una vez por decision.

        Args:
            action (int): Indice de accion 0..21. Una accion que la mascara no admite se
                sustituye por "nada" (MaskablePPO nunca las elige).

        Returns:
            tuple: (observacion apilada, recompensa, terminado, truncado, info). La info trae
                "episodio" con las metricas al terminar o truncar.

        Raises:
            ValueError: Si la accion esta fuera de 0..21.

        Ejemplo:
            obs, recompensa, terminado, truncado, info = env.step(17)
        """
        if not 0 <= int(action) < acc.N_ACCIONES:
            raise ValueError(f"accion fuera de 0..{acc.N_ACCIONES - 1}: {action}")
        info: dict[str, Any] = {}
        efectiva = int(action)
        if not self.action_masks()[efectiva]:
            efectiva = acc.ACCION_NADA
            info["accion_invalida"] = True
        try:
            recompensa, final = self._avanzar(efectiva)
        except (BridgeError, TimeoutError, OSError) as exc:
            return self._tras_caida(exc)
        self._recompensa_total += recompensa
        self._accion_previa = efectiva
        segundos = self._frames / self.config.sim_hz
        terminado = final is not None
        truncado = (not terminado) and self._frames >= self._limite_frames()
        if terminado or truncado:
            info["episodio"] = self._resumen(final or RESULTADO_TIEMPO)
        vector = self._constructor.construir(self._cruda, self._accion_previa, segundos)
        return self._apilador.agregar(vector), float(recompensa), terminado, truncado, info

    def close(self) -> None:
        """
        Cierra la instancia del juego.

        Contexto:
            Fin del entrenamiento o de la evaluacion.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de cierre se ignoran.

        Ejemplo:
            env.close()
        """
        cerrar_puerto(self._puerto)
        self._puerto = None

    def _puerto_activo(self) -> PuertoJuego:
        """
        Instancia viva del juego.

        Contexto:
            Todas las llamadas al puente.

        Args:
            Ninguno.

        Returns:
            PuertoJuego: Cliente conectado.

        Raises:
            RuntimeError: Si el entorno ya se cerro.

        Ejemplo:
            (uso interno)
        """
        if self._puerto is None:
            raise RuntimeError("el entorno esta cerrado")
        return self._puerto

    def _reiniciar_juego(self, multiplicador: float, fase: int) -> dict[str, Any]:
        """
        Pide el reinicio al juego y relanza la instancia si falla.

        Contexto:
            reset. Una caida del juego durante el entrenamiento no debe tumbar la corrida: se
            vuelve a lanzar la instancia (hasta INTENTOS_RELANZAR veces seguidas).

        Args:
            multiplicador (float): Multiplicador de vida del jefe.
            fase (int): Fase de inicio.

        Returns:
            dict[str, Any]: Observacion cruda inicial.

        Raises:
            RuntimeError: Si todos los intentos fallan.

        Ejemplo:
            (uso interno)
        """
        ultimo: Exception | None = None
        for _ in range(INTENTOS_RELANZAR):
            try:
                return self._pedir_reset(multiplicador, fase)
            except (BridgeError, TimeoutError, OSError) as exc:
                ultimo = exc
                self._relanzar()
        raise RuntimeError(f"el juego no consigue reiniciar tras {INTENTOS_RELANZAR} relanzamientos: {ultimo!r}")

    def _pedir_reset(self, multiplicador: float, fase: int) -> dict[str, Any]:
        """
        Envia la orden de reinicio al juego.

        Contexto:
            _reiniciar_juego. Punto de extension: el entorno multi-jefe agrega el jefe y la recarga.

        Args:
            multiplicador (float): Multiplicador de vida del jefe.
            fase (int): Fase de inicio.

        Returns:
            dict[str, Any]: Observacion cruda inicial.

        Raises:
            BridgeError: Si el juego rechaza el reinicio.

        Ejemplo:
            (uso interno)
        """
        return self._puerto_activo().reset(boss_hp_mult=multiplicador, start_phase=fase)

    def _al_observar(self) -> None:
        """
        Gancho que se llama cada vez que llega una observacion cruda nueva.

        Contexto:
            reset y _avanzar. El entorno multi-jefe lo usa para actualizar el seguimiento de amenazas;
            aqui no hace nada.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """

    def _resultado_final(self, cruda: dict[str, Any]) -> str | None:
        """
        Resultado del episodio segun la observacion cruda.

        Contexto:
            _avanzar. Punto de extension: cada jefe tiene su propia regla de victoria.

        Args:
            cruda (dict[str, Any]): Observacion cruda.

        Returns:
            str | None: "victoria", "muerte" o None.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return resultado_final(cruda)

    def _limite_frames(self) -> float:
        """
        Frames de juego tras los cuales el episodio se trunca por tiempo.

        Contexto:
            step. Punto de extension: cada jefe puede tener su propio limite.

        Args:
            Ninguno.

        Returns:
            float: segundos_maximos * sim_hz.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self.config.segundos_maximos * self.config.sim_hz

    def _vida_efectiva(self) -> float:
        """
        Vida que hay que quitar para ganar el episodio en curso.

        Contexto:
            _resumen (progreso contra el jefe). Punto de extension.

        Args:
            Ninguno.

        Returns:
            float: Puntos de vida.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return vida_efectiva_jefe(self._multiplicador_episodio, self._fase_inicio)

    def _relanzar(self) -> None:
        """
        Cierra la instancia caida y lanza una nueva.

        Contexto:
            reset y step ante un error del puente.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            TimeoutError: Si la instancia nueva no se conecta.

        Ejemplo:
            (uso interno)
        """
        cerrar_puerto(self._puerto)
        self._puerto = None
        self.reinicios_instancia += 1
        self._puerto = self._fabrica()

    def _tras_caida(self, exc: Exception) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Cierra el episodio como truncado tras una caida del juego y relanza la instancia.

        Contexto:
            step. El episodio no cuenta como victoria ni derrota: el resultado es "error".

        Args:
            exc (Exception): Error del puente que provoco la caida.

        Returns:
            tuple: Misma forma que step, con terminado=False y truncado=True.

        Raises:
            TimeoutError: Si la instancia nueva no se conecta.

        Ejemplo:
            (uso interno)
        """
        resumen = self._resumen(RESULTADO_ERROR)
        resumen["error"] = repr(exc)
        self._relanzar()
        ultimo = self._apilador.apilado()
        return ultimo, 0.0, False, True, {"episodio": resumen}

    def _avanzar(self, accion: int) -> tuple[float, str | None]:
        """
        Ejecuta la accion, con el enfoque al jefe asegurado antes, y acumula la recompensa.

        Contexto:
            step.

        Args:
            accion (int): Accion efectiva (ya validada con la mascara).

        Returns:
            tuple[float, str | None]: Recompensa del paso y resultado final si lo hubo.

        Raises:
            BridgeError: Si el juego rechaza una orden.

        Ejemplo:
            (uso interno)
        """
        puerto = self._puerto_activo()
        recompensa = 0.0
        if self._necesita_enfoque():
            self._pasos_desde_lock = 0
            self._cruda = puerto.step(Action(btn="rs"), 1)
            self._al_observar()
            self._frames += 1
            recompensa += self._calculadora.calcular(self._cruda).recompensa
            final = self._resultado_final(self._cruda)
            if final is not None:
                return recompensa, final
        else:
            self._pasos_desde_lock += 1
        comando = acc.comando_de_accion(accion, self._accion_previa)
        self._cruda = puerto.step(a_action(comando), self.config.frames_por_paso)
        self._al_observar()
        self._frames += self.config.frames_por_paso
        recompensa += self._calculadora.calcular(self._cruda).recompensa
        return recompensa, self._resultado_final(self._cruda)

    def _necesita_enfoque(self) -> bool:
        """
        Decide si hay que pulsar el enfoque (lock-on) antes de la accion.

        Contexto:
            _avanzar. El enfoque hace que el personaje mire al jefe; solo funciona dentro de su
            alcance (~500 uu) y es un interruptor, por eso solo se pulsa si esta apagado y se
            espera unos pasos entre intentos.

        Args:
            Ninguno.

        Returns:
            bool: True si conviene pulsar el enfoque ahora.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        jugador = self._cruda.get("player") or {}
        if jugador.get("target_lock") or jugador.get("dead"):
            return False
        if float(self._cruda.get("dist", 1e9)) > self.config.distancia_lock:
            return False
        return self._pasos_desde_lock >= self.config.pasos_entre_intentos_lock

    def _resumen(self, resultado: str) -> dict[str, Any]:
        """
        Metricas del episodio que acaba de terminar.

        Contexto:
            step y _tras_caida; las consumen los callbacks de TensorBoard y el curriculo.

        Args:
            resultado (str): "victoria", "muerte", "tiempo" o "error".

        Returns:
            dict[str, Any]: Resultado, dano, parries, caguamas, fase, duracion y dificultad.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        est = self._calculadora.estadisticas
        efectiva = self._vida_efectiva()
        victoria = resultado == RESULTADO_VICTORIA
        return {
            "resultado": resultado,
            "victoria": victoria,
            "muerte": resultado == RESULTADO_MUERTE,
            "danio_hecho": est.danio_hecho,
            "danio_recibido": est.danio_recibido,
            "curado": est.curado,
            "parries": est.parries,
            "caguamas_usadas": est.caguamas_usadas,
            "caguamas_interrumpidas": est.caguamas_interrumpidas,
            "fase_max": est.fase_max,
            "fase_inicio": self._fase_inicio,
            "multiplicador": self._multiplicador_episodio,
            "progreso_jefe": 1.0 if victoria else min(1.0, est.danio_hecho / efectiva),
            "pasos": est.pasos,
            "duracion_s": self._frames / self.config.sim_hz,
            "recompensa": self._recompensa_total,
        }
