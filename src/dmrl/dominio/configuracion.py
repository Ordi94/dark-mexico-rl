"""
Parametros configurables del entorno, la recompensa y el curriculo.

Todos los coeficientes viven aqui para poder cambiarlos sin tocar la logica; los valores por
defecto son los usados en la primera corrida de entrenamiento (documentados en docs/).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConfigRecompensa:
    """
    Coeficientes de la funcion de recompensa.

    Contexto:
        CalculadoraRecompensa. Cada termino se suma por paso de decision.

    Atributos:
        danio_hecho: recompensa por punto de dano al jefe (1/100).
        danio_recibido: castigo por punto de vida perdida del jugador (1/50): sobrevivir pesa
            el doble que golpear.
        parry: premio por parry exitoso.
        curacion: recompensa por punto de vida efectivamente recuperada (1/100: una caguama
            completa vale +0.5, menos que lo que cuesta el dano que la hizo necesaria).
        curacion_vida_maxima: solo se premia curar si la vida previa era menor que esto; evita
            que el agente se deje golpear para beber.
        bebida_interrumpida: castigo pequeno por una bebida cortada antes de curar.
        fase2: premio unico al pasar a la fase 2.
        victoria: premio por matar al jefe.
        muerte: castigo por morir.
        paso: costo por paso de decision (presion de tiempo).
    """

    danio_hecho: float = 1.0 / 100.0
    danio_recibido: float = 1.0 / 50.0
    parry: float = 0.5
    curacion: float = 1.0 / 100.0
    curacion_vida_maxima: float = 60.0
    bebida_interrumpida: float = 0.05
    fase2: float = 2.0
    victoria: float = 10.0
    muerte: float = 5.0
    paso: float = 0.001


@dataclass(frozen=True)
class ConfigCurriculo:
    """
    Parametros del curriculo de dificultad.

    Contexto:
        Curriculo; los niveles son multiplicadores de la vida del jefe.

    Atributos:
        niveles: multiplicadores crecientes; el ultimo debe ser 1.0.
        tasa_victoria_subida: tasa de victoria en la ventana que habilita subir de nivel.
        ventana: episodios (de fase 1 completa) sobre los que se mide la tasa.
        minimo_episodios: episodios minimos en un nivel antes de poder subir.
        prob_fase2: probabilidad de que un episodio arranque directo en la fase 2.
    """

    niveles: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 1.0)
    tasa_victoria_subida: float = 0.6
    ventana: int = 100
    minimo_episodios: int = 100
    prob_fase2: float = 0.2


@dataclass(frozen=True)
class ConfigEntorno:
    """
    Parametros del entorno Gymnasium.

    Contexto:
        DarkMexicoEnv y las fabricas de entornos del entrenamiento.

    Atributos:
        frames_por_paso: frames de simulacion por decision (2 a 30 Hz = 15 Hz de decision).
        sim_hz: frecuencia de simulacion del juego (paso fijo de 1/30 s).
        segundos_maximos: duracion maxima de un episodio en segundos de juego (truncado).
        apilado: observaciones consecutivas que ve la politica.
        distancia_lock: por debajo de esta distancia (uu) se intenta activar el enfoque.
        pasos_entre_intentos_lock: decisiones minimas entre dos intentos de enfoque.
        recompensa: coeficientes de recompensa.
        curriculo: parametros del curriculo.
    """

    frames_por_paso: int = 2
    sim_hz: float = 30.0
    segundos_maximos: float = 360.0
    apilado: int = 4
    distancia_lock: float = 480.0
    pasos_entre_intentos_lock: int = 5
    recompensa: ConfigRecompensa = field(default_factory=ConfigRecompensa)
    curriculo: ConfigCurriculo = field(default_factory=ConfigCurriculo)

    @property
    def segundos_por_paso(self) -> float:
        """
        Segundos de juego que dura una decision.

        Contexto:
            Tiempo de episodio y normalizacion de la observacion.

        Args:
            Ninguno.

        Returns:
            float: frames_por_paso / sim_hz.

        Raises:
            ZeroDivisionError: Si sim_hz es cero.

        Ejemplo:
            ConfigEntorno().segundos_por_paso  # 0.0667
        """
        return self.frames_por_paso / self.sim_hz

    @property
    def pasos_maximos(self) -> int:
        """
        Numero maximo de decisiones por episodio.

        Contexto:
            Truncado por tiempo en DarkMexicoEnv.

        Args:
            Ninguno.

        Returns:
            int: Techo de segundos_maximos / segundos_por_paso.

        Raises:
            ZeroDivisionError: Si sim_hz es cero.

        Ejemplo:
            ConfigEntorno().pasos_maximos  # 5400
        """
        return int(round(self.segundos_maximos / self.segundos_por_paso))

    def a_dict(self) -> dict[str, Any]:
        """
        Representacion serializable de la configuracion.

        Contexto:
            Se guarda junto a cada corrida de entrenamiento.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Estructura anidada de tipos JSON.

        Raises:
            Ninguna.

        Ejemplo:
            ConfigEntorno().a_dict()["recompensa"]["victoria"]
        """
        return asdict(self)
