"""
Bot de reglas de referencia: linea base contra la que se mide al agente entrenado.

Usa solo lo que ve el agente (la observacion cruda del puente) mas la tabla de golpes medida:
    1. Golpe parable inminente: parry, con la ventana centrada (LB se pulsa 0.235 s de juego
       antes del impacto; la ventana abre a los 0.17 s y dura 0.12 s).
    2. Golpe no parable inminente: roll lateral cuando faltan <= 0.20 s (la invulnerabilidad del
       roll cubre de ~0.03 a ~0.27 s despues de decidir).
    3. Jefe aturdido por parry: contraataque (especial si hay barra, si no ataque).
    4. Curacion: con vida < 50 solo si no hay golpes pendientes y el jefe esta a >= 450 uu (o
       aturdido, que dura mas que la bebida); si esta mas cerca se aleja para crear distancia.
    5. Ofensiva: ataque en ventanas sin golpes pendientes cercanos, acercandose si esta lejos.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dmrl.dominio import acciones as acc
from dmrl.dominio.tabla_golpes import GolpePendiente, TablaGolpes

# Tiempo de juego antes del impacto en que se pulsa LB: centra la ventana (0.17-0.29 s).
PARRY_DISPARO = 0.262
# Margen del roll respecto al golpe. Medido (scripts/medir_roll.py, decisiones de 2 frames): el
# golpe se esquiva si faltan entre 0 y ~0.27 s al decidir; con 0.33 s o mas el roll ya termino su
# invulnerabilidad. Se dispara con 0.20 s para quedar centrado entre pasos de 0.067 s.
ROLL_MIN = 0.0
ROLL_DISPARO = 0.20
# A partir de este tiempo a un golpe el bot deja de hacer otras cosas y espera la ocasion.
VIGILANCIA = 0.45
# Sin golpes pendientes en este plazo se considera ventana segura para atacar.
VENTANA_SEGURA = 1.2
VIDA_PARA_BEBER = 50.0
DISTANCIA_PARA_BEBER = 450.0
# El roll es la defensa contra los golpes no parables y cuesta 0.1 de Resistencia: el bot solo
# ataca si le queda reserva para esquivar (medido: sin Resistencia el roll se ignora).
ESTAMINA_RESERVA = 0.4
DISTANCIA_ATAQUE = 230.0
DISTANCIA_ACERCARSE = 250.0
DIRECCION_HACIA_JEFE = 0
DIRECCION_LATERAL = 2
DIRECCION_ATRAS = 4


class BotReglas:
    """
    Politica de reglas sobre la observacion cruda.

    Contexto:
        aplicacion.evaluacion la usa como politica de referencia ("bot de reglas").
    """

    def __init__(self, tabla: TablaGolpes) -> None:
        """
        Crea el bot con la tabla de golpes del jefe.

        Contexto:
            Arranque de la evaluacion o de una prueba.

        Args:
            tabla (TablaGolpes): Golpes medidos por montage.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot = BotReglas(TablaGolpes())
        """
        self.tabla = tabla
        self.reiniciar()

    def reiniciar(self) -> None:
        """
        Olvida el estado del episodio anterior.

        Contexto:
            Tras cada reset del entorno.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot.reiniciar()
        """
        self._montage: str | None = None
        self._pos = 0.0
        self._atendidos: set[int] = set()

    def decidir(self, obs: dict[str, Any], mascara: Sequence[bool]) -> int:
        """
        Elige la accion del siguiente paso.

        Contexto:
            Bucle de evaluacion, una vez por decision.

        Args:
            obs (dict[str, Any]): Observacion cruda actual.
            mascara (Sequence[bool]): Acciones validas (22 valores).

        Returns:
            int: Indice de accion valido segun la mascara.

        Raises:
            Ninguna.

        Ejemplo:
            accion = bot.decidir(obs, mascara)
        """
        jugador = obs.get("player") or {}
        jefe = obs.get("boss") or {}
        if not jugador or jugador.get("dead") or not jefe:
            return acc.ACCION_NADA
        if jugador.get("drinking"):
            return self._mover(DIRECCION_ATRAS)
        montage = jefe.get("montage") or {}
        pendiente = self._seguir_montage(montage)
        distancia = float(obs.get("dist", 0.0))
        for decision in (
            self._reaccion(pendiente, mascara),
            self._contraataque(jefe, jugador, distancia, mascara),
            self._curacion(jugador, jefe, pendiente, distancia, mascara),
            self._ofensiva(pendiente, distancia, float(jugador.get("stamina", 1.0)), mascara),
        ):
            if decision is not None:
                return decision
        return acc.ACCION_NADA

    def _seguir_montage(self, montage: dict[str, Any]) -> GolpePendiente | None:
        """
        Actualiza el seguimiento del montage del jefe y devuelve el golpe pendiente.

        Contexto:
            decidir. Un montage que reinicia (la posicion retrocede o cambia el nombre) vuelve
            a tener todos sus golpes por atender.

        Args:
            montage (dict[str, Any]): Bloque "montage" del jefe, vacio si no hay.

        Returns:
            GolpePendiente | None: Proximo golpe por delante.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        nombre = montage.get("name")
        pos = float(montage.get("pos", 0.0))
        if nombre != self._montage or pos < self._pos - 0.05:
            self._atendidos = set()
        self._montage, self._pos = nombre, pos
        return self.tabla.proximo_golpe(nombre, pos, float(montage.get("rate", 1.0)))

    def _reaccion(self, pendiente: GolpePendiente | None, mascara: Sequence[bool]) -> int | None:
        """
        Parry o roll ante un golpe inminente; espera quieto mientras se acerca.

        Contexto:
            decidir, regla 1 y 2.

        Args:
            pendiente (GolpePendiente | None): Proximo golpe del jefe.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si no hay golpe a vigilar.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if pendiente is None or pendiente.segundos > VIGILANCIA:
            return None
        ya_atendido = pendiente.indice in self._atendidos
        if pendiente.golpe.parry and mascara[acc.ACCION_PARRY]:
            if not ya_atendido and pendiente.segundos <= PARRY_DISPARO:
                self._atendidos.add(pendiente.indice)
                return acc.ACCION_PARRY
            return acc.ACCION_NADA
        roll = acc.ACCION_ROLL_PRIMERA + DIRECCION_LATERAL
        if mascara[roll] and not ya_atendido and ROLL_MIN <= pendiente.segundos <= ROLL_DISPARO:
            self._atendidos.add(pendiente.indice)
            return roll
        return acc.ACCION_NADA

    def _contraataque(self, jefe: dict[str, Any], jugador: dict[str, Any], distancia: float,
                      mascara: Sequence[bool]) -> int | None:
        """
        Aprovecha el aturdimiento del jefe tras un parry.

        Contexto:
            decidir, regla 3. Aturdido el jefe no golpea durante ~2.3 s: da para atacar y,
            si la vida esta baja, tambien para beber entera.

        Args:
            jefe (dict[str, Any]): Bloque "boss".
            jugador (dict[str, Any]): Bloque "player".
            distancia (float): Distancia 2D al jefe.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si el jefe no esta aturdido.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if not jefe.get("stunned"):
            return None
        if float(jugador.get("hp", 100.0)) < VIDA_PARA_BEBER and mascara[acc.ACCION_CAGUAMA]:
            return acc.ACCION_CAGUAMA
        if distancia > DISTANCIA_ATAQUE:
            return self._mover(DIRECCION_HACIA_JEFE)
        for accion in (acc.ACCION_ESPECIAL, acc.ACCION_ATAQUE):
            if mascara[accion]:
                return accion
        return acc.ACCION_NADA

    def _curacion(self, jugador: dict[str, Any], jefe: dict[str, Any], pendiente: GolpePendiente | None,
                  distancia: float, mascara: Sequence[bool]) -> int | None:
        """
        Beber cuando es seguro; alejarse para poder hacerlo si no lo es.

        Contexto:
            decidir, regla 4 (medida: beber con un golpe pendiente falla siempre).

        Args:
            jugador (dict[str, Any]): Bloque "player".
            jefe (dict[str, Any]): Bloque "boss".
            pendiente (GolpePendiente | None): Proximo golpe del jefe.
            distancia (float): Distancia 2D al jefe.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si no toca curarse.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if float(jugador.get("hp", 100.0)) >= VIDA_PARA_BEBER or int(jugador.get("caguamas", 0)) <= 0:
            return None
        if pendiente is not None or jefe.get("attacking"):
            return None
        if distancia < DISTANCIA_PARA_BEBER:
            return self._mover(DIRECCION_ATRAS)
        return acc.ACCION_CAGUAMA if mascara[acc.ACCION_CAGUAMA] else None

    def _ofensiva(self, pendiente: GolpePendiente | None, distancia: float, estamina: float,
                  mascara: Sequence[bool]) -> int | None:
        """
        Ataca en ventanas seguras y se acerca si el jefe esta lejos.

        Contexto:
            decidir, regla 5.

        Args:
            pendiente (GolpePendiente | None): Proximo golpe del jefe.
            distancia (float): Distancia 2D al jefe.
            estamina (float): Resistencia del jugador.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si conviene esperar.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if pendiente is not None and pendiente.segundos < VENTANA_SEGURA:
            return None
        if distancia > DISTANCIA_ACERCARSE:
            return self._mover(DIRECCION_HACIA_JEFE)
        if estamina < ESTAMINA_RESERVA:
            return None
        for accion in (acc.ACCION_ESPECIAL, acc.ACCION_ATAQUE):
            if mascara[accion]:
                return accion
        return None

    @staticmethod
    def _mover(direccion: int) -> int:
        """
        Indice de accion de movimiento para una direccion 0..7.

        Contexto:
            Reglas de acercamiento y retirada.

        Args:
            direccion (int): 0 hacia el jefe, 4 en sentido contrario.

        Returns:
            int: Accion de movimiento.

        Raises:
            Ninguna.

        Ejemplo:
            BotReglas._mover(4)  # 5
        """
        return acc.ACCION_MOVER_PRIMERA + direccion
