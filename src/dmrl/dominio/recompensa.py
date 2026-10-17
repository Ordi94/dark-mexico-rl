"""
Funcion de recompensa y deteccion del final de episodio a partir de la observacion cruda.

La recompensa se calcula con hechos medidos en el juego, no con estimaciones:
    * dano al jefe: suma de los golpes dados en el paso (hook de TakeDamage del mod);
    * dano recibido: caida real de la vida del jugador (un golpe parado o esquivado con la
      invulnerabilidad del roll no cuenta aunque el juego haya intentado aplicarlo);
    * curacion: subida real de la vida, que llega de golpe a 1.054 s de empezar a beber;
    * bebida interrumpida: el montage de la bebida termino sin que la caguama se gastara
      (un golpe antes del notify la cancela sin curar).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dmrl.dominio.configuracion import ConfigRecompensa

RESULTADO_VICTORIA = "victoria"
RESULTADO_MUERTE = "muerte"
RESULTADO_TIEMPO = "tiempo"
CLASE_JEFE_FASE2 = "BP_Enemy_JuicePhase2_C"


def resultado_final(obs: dict[str, Any]) -> str | None:
    """
    Determina si la observacion cierra el episodio por victoria o muerte.

    Contexto:
        DarkMexicoEnv.step y la calculadora de recompensa. Si el jugador muere en el mismo
        paso en que cae el jefe se cuenta la muerte: el juego ya lo reinicia.

    Args:
        obs (dict[str, Any]): Observacion cruda del puente, con "ev" y bloques de entidades.

    Returns:
        str | None: "victoria", "muerte" o None si la pelea sigue.

    Raises:
        Ninguna.

    Ejemplo:
        resultado_final({"ev": {"victory": True}, "player": {"dead": False}})  # "victoria"
    """
    ev = obs.get("ev") or {}
    jugador = obs.get("player") or {}
    jefe = obs.get("boss") or {}
    if jugador.get("dead") or ev.get("player_death"):
        return RESULTADO_MUERTE
    if ev.get("victory") or (jefe.get("cls") == CLASE_JEFE_FASE2 and jefe.get("dead")):
        return RESULTADO_VICTORIA
    return None


def fase_de(obs: dict[str, Any], por_defecto: int = 1) -> int:
    """
    Fase actual de la pelea (1 o 2) segun el jefe observado.

    Contexto:
        Calculadora de recompensa y metricas.

    Args:
        obs (dict[str, Any]): Observacion cruda.
        por_defecto (int): Valor si no hay jefe en la observacion.

    Returns:
        int: 1 o 2.

    Raises:
        Ninguna.

    Ejemplo:
        fase_de({"boss": {"phase": 1}})  # 2
    """
    jefe = obs.get("boss")
    if not jefe:
        return por_defecto
    return 2 if int(jefe.get("phase", 0)) >= 1 or jefe.get("cls") == CLASE_JEFE_FASE2 else 1


@dataclass
class EstadisticasEpisodio:
    """
    Acumulados de un episodio, publicados al terminar para las metricas de TensorBoard.

    Contexto:
        CalculadoraRecompensa.estadisticas.

    Atributos:
        danio_hecho: dano total al jefe.
        danio_recibido: vida perdida por el jugador.
        curado: vida recuperada con caguamas.
        parries: parries exitosos.
        caguamas_usadas: bebidas completadas (la caguama se gasto).
        caguamas_interrumpidas: bebidas cortadas antes de curar.
        fase_max: fase mas alta alcanzada (1 o 2).
        pasos: decisiones tomadas.
    """

    danio_hecho: float = 0.0
    danio_recibido: float = 0.0
    curado: float = 0.0
    parries: int = 0
    caguamas_usadas: int = 0
    caguamas_interrumpidas: int = 0
    fase_max: int = 1
    pasos: int = 0


@dataclass
class ResultadoPaso:
    """
    Recompensa de un paso con su desglose por termino.

    Contexto:
        Salida de CalculadoraRecompensa.calcular; el desglose sirve para depurar y validar.

    Atributos:
        recompensa: suma de los terminos.
        terminos: valor aportado por cada termino.
    """

    recompensa: float = 0.0
    terminos: dict[str, float] = field(default_factory=dict)


class CalculadoraRecompensa:
    """
    Calcula la recompensa paso a paso comparando cada observacion con la anterior.

    Contexto:
        DarkMexicoEnv la reinicia en cada reset y la consulta en cada step.
    """

    def __init__(self, config: ConfigRecompensa | None = None,
                 resolver_final: Callable[[dict[str, Any]], str | None] | None = None,
                 resolver_fase: Callable[[dict[str, Any], int], int] | None = None) -> None:
        """
        Crea la calculadora con sus coeficientes.

        Contexto:
            Construccion del entorno. El entorno multi-jefe inyecta sus propias reglas de victoria
            y de fase; sin ellas se usan las de Juice.

        Args:
            config (ConfigRecompensa, opcional): Coeficientes; por defecto los de fabrica.
            resolver_final (Callable, opcional): Decide "victoria", "muerte" o None a partir de la
                observacion; por defecto resultado_final.
            resolver_fase (Callable, opcional): Fase (1 o 2) a partir de la observacion y la fase
                por defecto; por defecto fase_de.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            calculadora = CalculadoraRecompensa()
        """
        self.config = config or ConfigRecompensa()
        self._final = resolver_final or resultado_final
        self._fase_de = resolver_fase or fase_de
        self.estadisticas = EstadisticasEpisodio()
        self._vida = 0.0
        self._fase = 1
        self._bebiendo = False
        self._caguamas = 0
        self._caguamas_al_beber = 0

    def reiniciar(self, obs: dict[str, Any]) -> None:
        """
        Toma la observacion inicial como punto de partida del episodio.

        Contexto:
            DarkMexicoEnv.reset, con la primera observacion del jefe activo.

        Args:
            obs (dict[str, Any]): Observacion cruda inicial.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            calculadora.reiniciar(obs)
        """
        jugador = obs.get("player") or {}
        self.estadisticas = EstadisticasEpisodio(fase_max=self._fase_de(obs, 1))
        self._vida = float(jugador.get("hp", 0.0))
        self._fase = self._fase_de(obs, 1)
        self._bebiendo = bool(jugador.get("drinking"))
        self._caguamas = int(jugador.get("caguamas", 0))
        self._caguamas_al_beber = self._caguamas

    def calcular(self, obs: dict[str, Any]) -> ResultadoPaso:
        """
        Recompensa del paso que produjo la observacion dada.

        Contexto:
            DarkMexicoEnv.step, una vez por decision.

        Args:
            obs (dict[str, Any]): Observacion cruda al final del paso.

        Returns:
            ResultadoPaso: Recompensa total y desglose.

        Raises:
            Ninguna.

        Ejemplo:
            paso = calculadora.calcular(obs)
        """
        cfg = self.config
        ev = obs.get("ev") or {}
        jugador = obs.get("player") or {}
        terminos: dict[str, float] = {"paso": -cfg.paso}
        self.estadisticas.pasos += 1

        danio_hecho = float(ev.get("dmg_dealt", 0.0))
        if danio_hecho > 0:
            self.estadisticas.danio_hecho += danio_hecho
            terminos["danio_hecho"] = cfg.danio_hecho * danio_hecho
        parries = int(ev.get("parries", 0))
        if parries > 0:
            self.estadisticas.parries += parries
            terminos["parry"] = cfg.parry * parries

        vida = float(jugador.get("hp", self._vida))
        perdida = self._vida - max(vida, 0.0)
        if perdida > 0:
            self.estadisticas.danio_recibido += perdida
            terminos["danio_recibido"] = -cfg.danio_recibido * perdida
        elif vida > self._vida:
            curado = vida - self._vida
            self.estadisticas.curado += curado
            if self._vida < cfg.curacion_vida_maxima:
                terminos["curacion"] = cfg.curacion * curado
        self._vida = vida

        self._actualizar_bebida(jugador, terminos)

        fase = self._fase_de(obs, self._fase)
        if fase > self._fase:
            terminos["fase2"] = cfg.fase2
        self._fase = fase
        self.estadisticas.fase_max = max(self.estadisticas.fase_max, fase)

        final = self._final(obs)
        if final == RESULTADO_VICTORIA:
            terminos["victoria"] = cfg.victoria
        elif final == RESULTADO_MUERTE:
            terminos["muerte"] = -cfg.muerte
        return ResultadoPaso(recompensa=sum(terminos.values()), terminos=terminos)

    def _actualizar_bebida(self, jugador: dict[str, Any], terminos: dict[str, float]) -> None:
        """
        Sigue el montage de la bebida para contar bebidas completas e interrumpidas.

        Contexto:
            calcular. Una bebida termina completa si la caguama se gasto (el juego la gasta
            en el notify de curacion) e interrumpida si el montage acabo sin gastarse.

        Args:
            jugador (dict[str, Any]): Bloque "player" de la observacion.
            terminos (dict[str, float]): Desglose del paso, que se modifica con el castigo.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        bebiendo = bool(jugador.get("drinking"))
        caguamas = int(jugador.get("caguamas", self._caguamas))
        if not bebiendo and self._bebiendo:
            if caguamas < self._caguamas_al_beber:
                self.estadisticas.caguamas_usadas += 1
            else:
                self.estadisticas.caguamas_interrumpidas += 1
                terminos["bebida_interrumpida"] = -self.config.bebida_interrumpida
        if bebiendo and not self._bebiendo:
            self._caguamas_al_beber = self._caguamas
        self._bebiendo = bebiendo
        self._caguamas = caguamas

