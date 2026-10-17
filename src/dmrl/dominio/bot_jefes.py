"""
Bot de reglas multi-jefe: una sola implementacion parametrizada por el perfil del jefe.

Usa lo que ve el agente (la observacion cruda) mas las amenazas que calcula el seguidor con la tabla
del jefe, de modo que un jefe nuevo solo necesita su registro:
    1. Impacto parable inminente: parry, con la ventana centrada (LB se pulsa parry_disparo segundos
       de juego antes del impacto; la ventana abre a los 0.17 s y dura 0.12 s).
    2. Impacto no parable inminente (golpe fuerte, proyectil, area, rayo): maniobra (roll) cuando
       faltan <= roll_disparo segundos, que cae dentro de la invulnerabilidad del roll. La direccion
       depende del tipo de amenaza.
    3. Jefe aturdido por parry: contraataque (especial si hay barra, si no ataque).
    4. Curacion: con poca vida solo si no hay impactos pendientes en lo que dura la bebida y el jefe
       esta lejos; si esta cerca se aleja para crear distancia.
    5. Ofensiva: ataque en ventanas sin impactos cercanos, acercandose si el jefe esta lejos.

Marca de impactos atendidos: el bot recuerda a que impactos ya respondio para no repetir un parry
(dos parries seguidos serian pulsar y soltar). La marca solo se confirma cuando la accion se
EJECUTO (confirmar): si el entorno la sustituyo (mascara) o un ruido de ejecucion la cambio, el
impacto sigue sin atender y el bot vuelve a pedirla; sin esto, las etiquetas de una pelea con ruido
decian "nada" mientras el golpe seguia sin parar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenaza, Amenazas
from dmrl.dominio.jefes import CatalogoJefes, PerfilJefe

DIRECCION_HACIA_JEFE = 0
DIRECCION_LATERAL = 2
DIRECCION_ATRAS = 4


@dataclass(frozen=True)
class ParametrosBot:
    """
    Constantes de las reglas, ajustables por jefe desde su registro (seccion "bot").

    Contexto:
        BotJefes. Los valores de fabrica son los medidos con Juice.

    Atributos:
        parry_disparo: segundos de juego antes del impacto en que se pulsa LB.
        roll_disparo: segundos antes del impacto en que se pulsa la maniobra.
        vigilancia: desde este tiempo al impacto el bot deja de hacer otras cosas y espera la ocasion.
        ventana_segura: sin impactos en este plazo se considera ventana segura para atacar.
        vida_para_beber: por debajo de esta vida se intenta beber.
        horizonte_beber: la bebida (2.15 s) solo empieza si ningun impacto llega en este plazo.
        distancia_beber: distancia minima al jefe para beber.
        estamina_reserva: Resistencia que el bot reserva para esquivar.
        distancia_ataque: distancia maxima a la que ataca.
        distancia_acercarse: a mas de esto se acerca al jefe.
        direccion_roll_melee: direccion (0..7, 0 hacia el jefe) de la maniobra ante golpes melee.
        direccion_roll_remoto: direccion de la maniobra ante proyectiles, areas y rayos.
        ignora_fuera_de_alcance: 1.0 = no reacciona a golpes melee que no pueden alcanzarlo.
        especial_seguro: el ataque especial (2.55 s sin poder parar ni esquivar) solo se usa si el
            siguiente impacto esta a esta distancia en el tiempo; 0 = sin restriccion.
        pausa_bajo_vida: con la vida del jefe por debajo de esta fraccion el bot deja de atacar para que
            el jefe pueda cambiar de fase (la IA de AMLO solo la cambia cuando no esta aturdido y
            golpearlo sin pausa lo deja aturdido para siempre, con vida negativa); 0 = no se pausa.
        pausa_maxima: segundos como mucho que dura esa pausa antes de volver a atacar.
    """

    parry_disparo: float = 0.262
    roll_disparo: float = 0.20
    vigilancia: float = 0.45
    ventana_segura: float = 1.2
    vida_para_beber: float = 50.0
    horizonte_beber: float = 2.2
    distancia_beber: float = 450.0
    estamina_reserva: float = 0.4
    distancia_ataque: float = 230.0
    distancia_acercarse: float = 250.0
    direccion_roll_melee: int = DIRECCION_LATERAL
    direccion_roll_remoto: int = DIRECCION_LATERAL
    ignora_fuera_de_alcance: float = 0.0
    especial_seguro: float = 0.0
    pausa_bajo_vida: float = 0.0
    pausa_maxima: float = 25.0

    @classmethod
    def desde_perfil(cls, perfil: PerfilJefe) -> ParametrosBot:
        """
        Parametros de fabrica con los ajustes propios del jefe.

        Contexto:
            Construccion del bot.

        Args:
            perfil (PerfilJefe): Jefe cuyo registro puede traer la seccion "bot".

        Returns:
            ParametrosBot: Parametros resultantes.

        Raises:
            ValueError: Si el registro trae un parametro que el bot no tiene.

        Ejemplo:
            ParametrosBot.desde_perfil(catalogo.perfil("amlo"))
        """
        validos = {f.name for f in fields(cls)}
        desconocidos = set(perfil.bot) - validos
        if desconocidos:
            raise ValueError(f"{perfil.id}: parametros de bot desconocidos {sorted(desconocidos)}")
        base = cls()
        ajustes: dict[str, Any] = {}
        for nombre, valor in perfil.bot.items():
            ajustes[nombre] = type(getattr(base, nombre))(valor)
        return replace(base, **ajustes)


class BotJefes:
    """
    Politica de reglas sobre la observacion cruda y las amenazas.

    Contexto:
        aplicacion.evaluacion_jefes la usa como politica de referencia y como maestra de la
        recoleccion de demostraciones.
    """

    def __init__(self, parametros: ParametrosBot | None = None,
                 por_clase: Mapping[str, ParametrosBot] | None = None) -> None:
        """
        Crea el bot.

        Contexto:
            Arranque de la evaluacion, de la recoleccion o de una prueba.

        Args:
            parametros (ParametrosBot, opcional): Constantes por defecto; las de fabrica si no se dan.
            por_clase (Mapping[str, ParametrosBot], opcional): Constantes propias de cada clase de
                jefe; en una pelea que cambia de actor (AMLO a Chamuco) el bot usa las del actor
                que tiene delante.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot = BotJefes.desde_catalogo(catalogo)
        """
        self._defecto = parametros or ParametrosBot()
        self._por_clase = dict(por_clase or {})
        self.parametros = self._defecto
        self.reiniciar()

    @classmethod
    def desde_catalogo(cls, catalogo: CatalogoJefes) -> BotJefes:
        """
        Bot con los ajustes de cada jefe del catalogo.

        Contexto:
            Evaluacion y recoleccion por jefe.

        Args:
            catalogo (CatalogoJefes): Jefes con su seccion "bot" opcional.

        Returns:
            BotJefes: Bot parametrizado por clase de actor.

        Raises:
            ValueError: Si un registro trae un parametro desconocido.

        Ejemplo:
            bot = BotJefes.desde_catalogo(catalogo)
        """
        por_clase = {clase: ParametrosBot.desde_perfil(p) for p in catalogo.perfiles for clase in p.clases}
        return cls(por_clase=por_clase)

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
        self._atendidos: set[tuple[Any, ...]] = set()
        self._propuesta: tuple[int, tuple[Any, ...]] | None = None
        self._t_pausa: float | None = None

    def confirmar(self, ejecutada: int) -> None:
        """
        Informa de la accion que se ejecuto de verdad tras la ultima decision.

        Contexto:
            Bucle de evaluacion y de recoleccion, justo despues de decidir y antes del siguiente
            decidir. El impacto solo se da por atendido si se ejecuto la accion que el bot pidio.

        Args:
            ejecutada (int): Accion que llego al juego (la pedida, una sustituida por la mascara o
                una alterada por ruido).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot.confirmar(accion_ejecutada)
        """
        if self._propuesta is not None and self._propuesta[0] == ejecutada:
            self._atendidos.add(self._propuesta[1])
        self._propuesta = None

    def decidir(self, obs: dict[str, Any], mascara: Sequence[bool], amenazas: Amenazas) -> int:
        """
        Elige la accion del siguiente paso.

        Contexto:
            Bucle de evaluacion o de recoleccion, una vez por decision.

        Args:
            obs (dict[str, Any]): Observacion cruda actual.
            mascara (Sequence[bool]): Acciones validas (22 valores).
            amenazas (Amenazas): Impactos pendientes del jefe.

        Returns:
            int: Indice de accion valido segun la mascara.

        Raises:
            Ninguna.

        Ejemplo:
            accion = bot.decidir(obs, mascara, amenazas)
        """
        if self._propuesta is not None:
            # Un cliente que no llama a confirmar da por ejecutada la accion pedida.
            self.confirmar(self._propuesta[0])
        self._atendidos &= {a.clave for a in amenazas.pendientes}
        jugador = obs.get("player") or {}
        jefe = obs.get("boss") or {}
        self.parametros = self._por_clase.get(str(jefe.get("cls")), self._defecto)
        if not jugador or jugador.get("dead") or not jefe:
            return acc.ACCION_NADA
        if jugador.get("drinking"):
            return self._mover(DIRECCION_ATRAS)
        distancia = float(obs.get("dist", 0.0))
        pausado = self._en_pausa(jefe, float(obs.get("t", 0.0)))
        for decision in (
            self._reaccion(amenazas, mascara),
            None if pausado else self._contraataque(jefe, jugador, distancia, amenazas, mascara),
            self._curacion(jugador, jefe, amenazas, distancia, mascara),
            None if pausado else self._ofensiva(amenazas, distancia, float(jugador.get("stamina", 1.0)), mascara),
        ):
            if decision is not None:
                return decision
        return acc.ACCION_NADA

    def _en_pausa(self, jefe: dict[str, Any], ahora: float) -> bool:
        """
        Indica si el bot debe dejar de atacar para que el jefe pueda cambiar de fase.

        Contexto:
            decidir. La pausa empieza cuando la vida del jefe baja de pausa_bajo_vida y dura como
            mucho pausa_maxima segundos; si la vida vuelve a subir se reinicia.

        Args:
            jefe (dict[str, Any]): Bloque "boss".
            ahora (float): Reloj de juego.

        Returns:
            bool: True mientras dure la pausa.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        p = self.parametros
        hp_max = float(jefe.get("hp_max", 0.0))
        if p.pausa_bajo_vida <= 0 or hp_max <= 0 or float(jefe.get("hp", hp_max)) / hp_max >= p.pausa_bajo_vida:
            self._t_pausa = None
            return False
        if self._t_pausa is None:
            self._t_pausa = ahora
        return ahora - self._t_pausa < p.pausa_maxima

    def _reaccion(self, amenazas: Amenazas, mascara: Sequence[bool]) -> int | None:
        """
        Parry o maniobra ante un impacto inminente; espera quieto mientras se acerca.

        Contexto:
            decidir, reglas 1 y 2.

        Args:
            amenazas (Amenazas): Impactos pendientes.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si no hay impacto a vigilar.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        p = self.parametros
        amenaza = amenazas.proximo
        if amenaza is None or amenaza.segundos > p.vigilancia:
            return None
        if p.ignora_fuera_de_alcance and not amenaza.en_alcance(amenazas.distancia):
            return None
        ya_atendido = amenaza.clave in self._atendidos
        if amenaza.parry and mascara[acc.ACCION_PARRY]:
            if not ya_atendido and amenaza.segundos <= p.parry_disparo:
                return self._proponer(acc.ACCION_PARRY, amenaza)
            return acc.ACCION_NADA
        roll = acc.ACCION_ROLL_PRIMERA + self._direccion_roll(amenaza)
        if mascara[roll] and not ya_atendido and 0.0 <= amenaza.segundos <= p.roll_disparo:
            return self._proponer(roll, amenaza)
        return acc.ACCION_NADA

    def _direccion_roll(self, amenaza: Amenaza) -> int:
        """
        Direccion de la maniobra segun el tipo de amenaza.

        Contexto:
            _reaccion.

        Args:
            amenaza (Amenaza): Impacto que se esquiva.

        Returns:
            int: Direccion 0..7 (0 hacia el jefe).

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if amenaza.tipo == "melee":
            return self.parametros.direccion_roll_melee
        return self.parametros.direccion_roll_remoto

    def _proponer(self, accion: int, amenaza: Amenaza) -> int:
        """
        Registra la accion pedida y el impacto al que responde, pendiente de confirmacion.

        Contexto:
            _reaccion.

        Args:
            accion (int): Accion que se pide.
            amenaza (Amenaza): Impacto al que responde.

        Returns:
            int: La misma accion.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self._propuesta = (accion, amenaza.clave)
        return accion

    def _contraataque(self, jefe: dict[str, Any], jugador: dict[str, Any], distancia: float,
                      amenazas: Amenazas, mascara: Sequence[bool]) -> int | None:
        """
        Aprovecha el aturdimiento del jefe tras un parry.

        Contexto:
            decidir, regla 3. Aturdido el jefe no golpea durante ~2.3 s: da para atacar y, si la
            vida esta baja, tambien para beber entera.

        Args:
            jefe (dict[str, Any]): Bloque "boss".
            jugador (dict[str, Any]): Bloque "player".
            distancia (float): Distancia 2D al jefe.
            amenazas (Amenazas): Impactos pendientes (para decidir si cabe el especial).
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
        if float(jugador.get("hp", 100.0)) < self.parametros.vida_para_beber and mascara[acc.ACCION_CAGUAMA]:
            return acc.ACCION_CAGUAMA
        if distancia > self.parametros.distancia_ataque:
            return self._mover(DIRECCION_HACIA_JEFE)
        for accion in self._ataques(amenazas):
            if mascara[accion]:
                return accion
        return acc.ACCION_NADA

    def _curacion(self, jugador: dict[str, Any], jefe: dict[str, Any], amenazas: Amenazas, distancia: float,
                  mascara: Sequence[bool]) -> int | None:
        """
        Beber cuando es seguro; alejarse para poder hacerlo si no lo es.

        Contexto:
            decidir, regla 4. Una bebida con un impacto pendiente dentro de su duracion falla siempre.

        Args:
            jugador (dict[str, Any]): Bloque "player".
            jefe (dict[str, Any]): Bloque "boss".
            amenazas (Amenazas): Impactos pendientes.
            distancia (float): Distancia 2D al jefe.
            mascara (Sequence[bool]): Acciones validas.

        Returns:
            int | None: Accion, o None si no toca curarse.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        p = self.parametros
        if float(jugador.get("hp", 100.0)) >= p.vida_para_beber or int(jugador.get("caguamas", 0)) <= 0:
            return None
        if jefe.get("attacking") or any(a.segundos < p.horizonte_beber for a in amenazas.pendientes):
            return None
        if distancia < p.distancia_beber:
            return self._mover(DIRECCION_ATRAS)
        return acc.ACCION_CAGUAMA if mascara[acc.ACCION_CAGUAMA] else None

    def _ofensiva(self, amenazas: Amenazas, distancia: float, estamina: float,
                  mascara: Sequence[bool]) -> int | None:
        """
        Ataca en ventanas seguras y se acerca si el jefe esta lejos.

        Contexto:
            decidir, regla 5.

        Args:
            amenazas (Amenazas): Impactos pendientes.
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
        p = self.parametros
        proximo = amenazas.proximo
        if proximo is not None and proximo.segundos < p.ventana_segura:
            return None
        if distancia > p.distancia_acercarse:
            return self._mover(DIRECCION_HACIA_JEFE)
        if estamina < p.estamina_reserva:
            return None
        for accion in self._ataques(amenazas):
            if mascara[accion]:
                return accion
        return None

    def _ataques(self, amenazas: Amenazas) -> tuple[int, ...]:
        """
        Ataques por orden de preferencia segun el margen que deja el siguiente impacto.

        Contexto:
            _contraataque y _ofensiva. El especial deja al jugador 2.55 s sin parry ni maniobra: solo
            se prefiere si el siguiente impacto esta suficientemente lejos (especial_seguro).

        Args:
            amenazas (Amenazas): Impactos pendientes.

        Returns:
            tuple[int, ...]: (especial, ataque) o solo (ataque,).

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        proximo = amenazas.proximo
        if proximo is not None and proximo.segundos < self.parametros.especial_seguro:
            return (acc.ACCION_ATAQUE,)
        return (acc.ACCION_ESPECIAL, acc.ACCION_ATAQUE)

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
            BotJefes._mover(4)  # 5
        """
        return acc.ACCION_MOVER_PRIMERA + direccion
