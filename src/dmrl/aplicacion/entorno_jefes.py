"""
Entorno Gymnasium multi-jefe (version 2): una instancia del juego fija a un mapa y varios jefes.

Reutiliza DarkMexicoEnv (acciones, mascara, enfoque, recompensa, relanzamiento ante caidas) y cambia
lo que depende del jefe:
    * cada episodio elige un jefe del mapa de la instancia (SelectorJefes, con pesos configurables y
      la opcion de dar mas peso a los jefes con menor tasa de victoria) y le pide al mod su reinicio
      con la descripcion del registro (data/jefes/<jefe>.json);
    * el mod reinicia en sitio, sin recargar el mapa; el entorno pide la recarga completa cada
      `recarga_cada` episodios como respaldo y cuando el estado inicial no es el esperado;
    * la observacion es la v2 (identidad del jefe, vocabulario union de montages y amenazas
      calculadas con la tabla del jefe, ver observacion_v2);
    * la victoria, la fase y la vida efectiva salen del registro del jefe.
La observacion v1 y DarkMexicoEnv no cambian: los modelos entrenados con ellos siguen cargando.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from dmbridge import BridgeError
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenazas, SeguidorAmenazas
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.jefes import CatalogoJefes, PerfilJefe
from dmrl.dominio.observacion import ApiladorObservaciones
from dmrl.dominio.observacion_v2 import ConstructorObservacionV2, tamano_simple_v2
from dmrl.dominio.recompensa import CalculadoraRecompensa
from dmrl.dominio.reglas_jefes import resolver_fase_de, resolver_final_de
from dmrl.dominio.seleccion_jefes import ConfigSeleccion, SelectorJefes
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.juego import FabricaPuerto

RECARGA_CADA_DEFECTO = 150
# Duracion maxima del episodio que normaliza el tiempo de la observacion; cada jefe trunca con la suya.
SEGUNDOS_MAXIMOS_V2 = 600.0
VENTANA_EVENTO_MORTAL = 6.0
HP_INICIAL_MINIMO = 0.95
# El enfoque (lock-on) puede quedar fijado en un objetivo que ya no existe (el AMLO que se transforma en
# Chamuco): el personaje deja de mirar al jefe y sus ataques golpean el aire. Si esta activo pero mira a
# mas de este angulo del jefe durante estas decisiones seguidas, se pulsa para soltarlo y volver a fijarlo.
ANGULO_ENFOQUE_PERDIDO = 70.0
DECISIONES_ENFOQUE_PERDIDO = 3
TOLERANCIA_VIDA_MAX = 0.02
CAGUAMAS_INICIALES = 3


class _ConstructorConAmenazas:
    """
    Adapta ConstructorObservacionV2 a la interfaz (obs, accion_previa, segundos) de DarkMexicoEnv.

    Contexto:
        EntornoMultiJefe: el vector necesita ademas las amenazas, que el entorno mantiene al dia.
    """

    def __init__(self, entorno: EntornoMultiJefe) -> None:
        """
        Crea el adaptador.

        Contexto:
            Construccion del entorno.

        Args:
            entorno (EntornoMultiJefe): Entorno que posee el constructor y las amenazas.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            _ConstructorConAmenazas(entorno)
        """
        self._entorno = entorno

    def construir(self, obs: dict[str, Any], accion_previa: int, segundos: float) -> NDArray[np.float32]:
        """
        Vector de observacion del instante actual con las amenazas vigentes.

        Contexto:
            DarkMexicoEnv.reset y step.

        Args:
            obs (dict[str, Any]): Observacion cruda.
            accion_previa (int): Accion del paso anterior.
            segundos (float): Segundos de juego del episodio.

        Returns:
            NDArray[np.float32]: Observacion simple v2.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self._entorno.constructor_v2.construir(obs, accion_previa, segundos, self._entorno.amenazas)


class EntornoMultiJefe(DarkMexicoEnv):
    """
    Entorno de una instancia del juego con varios jefes del mismo mapa.

    Contexto:
        Lo crean las fabricas de evaluacion, recoleccion y entrenamiento multi-jefe. Expone
        action_masks() para MaskablePPO y amenazas para el bot de reglas.
    """

    def __init__(self, fabrica: FabricaPuerto, catalogo: CatalogoJefes, config: ConfigEntorno | None = None,
                 semilla: int | None = None, jefes: Sequence[str] | None = None,
                 seleccion: ConfigSeleccion | None = None, multiplicador: float = 1.0,
                 recarga_cada: int = RECARGA_CADA_DEFECTO, trazar: bool = False) -> None:
        """
        Crea el entorno y lanza su instancia del juego.

        Contexto:
            Construccion de cada entorno multi-jefe.

        Args:
            fabrica (FabricaPuerto): Funcion que lanza el juego y devuelve su cliente.
            catalogo (CatalogoJefes): Jefes y vocabulario union.
            config (ConfigEntorno, opcional): Parametros del entorno; por defecto con segundos_maximos
                = 600 (el tiempo de la observacion se normaliza con el; el limite de cada episodio
                lo da el registro del jefe).
            semilla (int, opcional): Semilla del azar propio (jefe y fase de inicio).
            jefes (Sequence[str], opcional): Jefes de episodio que esta instancia puede enfrentar, todos
                del mismo mapa; por defecto todos los de LVL_AMLO si hay alguno, si no los de su mapa.
            seleccion (ConfigSeleccion, opcional): Pesos, modo adaptativo y probabilidad de fase 2.
            multiplicador (float): Multiplicador inicial de la vida del jefe, en [0.05, 1].
            recarga_cada (int): Episodios entre dos recargas completas del mapa (respaldo del reinicio
                en sitio); 0 recarga en cada episodio.
            trazar (bool): Si es True el mod registra cada golpe recibido (para medir de que ataque
                murio el jugador); cuesta un poco de tiempo por paso.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si un jefe no inicia episodios, los jefes son de mas de un mapa o la lista
                queda vacia.
            TimeoutError: Si el juego no se conecta.
            BridgeError: Si el saludo falla.

        Ejemplo:
            env = EntornoMultiJefe(fabrica, catalogo, jefes=("amlo", "nahual"))
        """
        config = config or ConfigEntorno(segundos_maximos=SEGUNDOS_MAXIMOS_V2)
        todos = catalogo.jefes_de_episodio()
        if jefes is None:
            jefes = catalogo.jefes_de_episodio("LVL_AMLO") or todos
        permitidos = tuple(jefes)
        if not permitidos or any(j not in todos for j in permitidos):
            raise ValueError(f"jefes de episodio no validos: {permitidos}; admitidos: {todos}")
        mapas = {catalogo.perfil(j).mapa for j in permitidos}
        if len(mapas) != 1:
            raise ValueError(f"una instancia queda fija a un mapa; los jefes {permitidos} son de {sorted(mapas)}")
        self.catalogo = catalogo
        self.mapa = mapas.pop()
        self.constructor_v2 = ConstructorObservacionV2(catalogo, config)
        self.amenazas = Amenazas()
        self._seguidor = SeguidorAmenazas(catalogo)
        self._azar_jefes = random.Random(semilla)
        self.selector = SelectorJefes(permitidos, seleccion, self._azar_jefes)
        self._recarga_cada = recarga_cada
        self._desde_recarga = recarga_cada
        self._recargar = False
        self._trazar = trazar
        self._jefe = permitidos[0]
        self._fase_pedida = 1
        self._golpes: deque[dict[str, Any]] = deque(maxlen=8)
        self._enfoque_perdido = 0
        self._ultimo_evento: tuple[str, float] | None = None
        self.recargas = 0
        self.inicios_inconsistentes = 0
        self.reinicio_s = 0.0
        super().__init__(fabrica, TablaGolpes(), config, semilla, multiplicador, 0.0)
        tamano = tamano_simple_v2(catalogo)
        self._constructor = _ConstructorConAmenazas(self)  # type: ignore[assignment]
        self._apilador = ApiladorObservaciones(tamano, self.config.apilado)
        self.observation_space = spaces.Box(-1.0, 1.0, (tamano * self.config.apilado,), np.float32)
        self._preparar_episodio(self._jefe)

    @property
    def jefe(self) -> str:
        """
        Jefe de episodio en curso (o el ultimo jugado).

        Contexto:
            Evaluacion y recoleccion por jefe.

        Args:
            Ninguno.

        Returns:
            str: Identificador del jefe.

        Raises:
            Ninguna.

        Ejemplo:
            env.jefe
        """
        return self._jefe

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Empieza una pelea nueva contra el jefe elegido, con el jefe ya activo.

        Contexto:
            Gymnasium/SB3 al empezar y tras cada episodio terminado.

        Args:
            seed (int, opcional): Reinicia el azar propio.
            options (dict, opcional): "jefe" (jefe de episodio), "start_phase" (1|2, solo jefes que
                admiten la segunda fase), "boss_hp_mult" y "reload" (fuerza la recarga del mapa). Sin
                ellas el jefe lo elige el selector y la fase la probabilidad configurada.

        Returns:
            tuple[NDArray[np.float32], dict[str, Any]]: Observacion apilada e info vacia.

        Raises:
            ValueError: Si el jefe pedido no esta entre los permitidos o la fase no es valida para el.
            RuntimeError: Si tras varios relanzamientos el juego no consigue reiniciar.

        Ejemplo:
            obs, info = env.reset(options={"jefe": "chamuco"})
        """
        if seed is not None:
            self._azar_jefes.seed(seed)
        opciones = dict(options or {})
        jefe = str(opciones.get("jefe") or self.selector.elegir())
        if jefe not in self.selector.permitidos:
            raise ValueError(f"jefe {jefe!r} fuera de los permitidos {self.selector.permitidos}")
        perfil = self.catalogo.perfil(jefe)
        fase = opciones.get("start_phase")
        if fase is None:
            admite = len(perfil.fases_inicio) > 1
            en_fase2 = admite and self._azar_jefes.random() < self.selector.config.prob_fase2
            fase = 2 if en_fase2 else perfil.fases_inicio[0]
        if int(fase) not in perfil.fases_inicio:
            raise ValueError(f"{jefe}: fase de inicio {fase} no admitida; admitidas {perfil.fases_inicio}")
        self._preparar_episodio(jefe)
        self._fase_pedida = int(fase)
        self._recargar = bool(opciones.get("reload")) or self._desde_recarga >= self._recarga_cada
        multiplicador = float(opciones.get("boss_hp_mult", self._multiplicador))
        return super().reset(seed=seed, options={"start_phase": int(fase), "boss_hp_mult": multiplicador})

    def _preparar_episodio(self, jefe: str) -> None:
        """
        Deja listas las reglas del jefe elegido (victoria, fase y recompensa).

        Contexto:
            reset y construccion.

        Args:
            jefe (str): Jefe de episodio.

        Returns:
            None: No retorna valor.

        Raises:
            KeyError: Si el jefe no existe en el catalogo.

        Ejemplo:
            (uso interno)
        """
        self._jefe = jefe
        perfil = self.catalogo.perfil(jefe)
        self._resolver_final = resolver_final_de(perfil.episodio.clases_victoria)
        self._calculadora = CalculadoraRecompensa(
            self.config.recompensa, self._resolver_final, resolver_fase_de(self.catalogo))

    def _pedir_reset(self, multiplicador: float, fase: int) -> dict[str, Any]:
        """
        Pide al mod el reinicio del jefe elegido y comprueba que el estado inicial sea el esperado.

        Contexto:
            _reiniciar_juego. Si el estado inicial no es coherente (jugador sin vida llena, jefe que
            no es de este episodio o con vida incompleta) se repite el reinicio recargando el mapa.

        Args:
            multiplicador (float): Multiplicador de vida del jefe.
            fase (int): Fase de inicio con la que se cuenta el episodio.

        Returns:
            dict[str, Any]: Observacion cruda inicial.

        Raises:
            BridgeError: Si el mod rechaza el reinicio o el estado sigue siendo incoherente tras la
                recarga completa.

        Ejemplo:
            (uso interno)
        """
        perfil = self.catalogo.perfil(self._jefe)
        descripcion = dict(perfil.episodio.mod or {})
        descripcion["trace_hits"] = self._trazar
        puerto = self._puerto_activo()
        inicio = time.monotonic()
        recarga = self._recargar
        for _ in range(2):
            crudo = puerto.reset(boss_hp_mult=multiplicador, start_phase=perfil.fase_para_mod(fase),
                                 boss=descripcion, reload=recarga)
            if recarga:
                self._desde_recarga = 0
                self.recargas += 1
            else:
                self._desde_recarga += 1
            if self._inicio_valido(crudo, multiplicador):
                self.reinicio_s = time.monotonic() - inicio
                self._seguidor.reiniciar()
                self._golpes.clear()
                self._ultimo_evento = None
                return crudo
            self.inicios_inconsistentes += 1
            recarga = True
        raise BridgeError(f"estado inicial incoherente contra {self._jefe} tras recargar el mapa")

    def _inicio_valido(self, crudo: dict[str, Any], multiplicador: float) -> bool:
        """
        Comprueba que un reinicio dejo un estado inicial limpio.

        Contexto:
            _pedir_reset. Es la unica defensa contra un reinicio en sitio que deje restos del episodio
            anterior (jugador sin curar, jefe viejo o a medias).

        Args:
            crudo (dict[str, Any]): Observacion cruda del reinicio.
            multiplicador (float): Multiplicador de vida pedido.

        Returns:
            bool: True si jugador y jefe estan como al empezar.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        jugador, jefe = crudo.get("player") or {}, crudo.get("boss") or {}
        if not jugador or not jefe or jugador.get("dead") or jefe.get("dead"):
            return False
        if float(jugador.get("hp", 0.0)) < float(jugador.get("hp_max", 1.0)) or \
                int(jugador.get("caguamas", 0)) != CAGUAMAS_INICIALES:
            return False
        perfil_actor = self.catalogo.por_clase(jefe.get("cls"))
        if perfil_actor is None or perfil_actor.mapa != self.mapa:
            return False
        esperada = perfil_actor.vida_max * multiplicador
        hp_max, hp = float(jefe.get("hp_max", 0.0)), float(jefe.get("hp", 0.0))
        return abs(hp_max - esperada) <= TOLERANCIA_VIDA_MAX * perfil_actor.vida_max and \
            hp >= HP_INICIAL_MINIMO * hp_max

    def _necesita_enfoque(self) -> bool:
        """
        Decide si hay que pulsar el enfoque, tambien cuando esta activo pero perdio al jefe.

        Contexto:
            _avanzar. Fuera del caso de enfoque perdido es la regla del entorno base.

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
        jefe = self._cruda.get("boss") or {}
        if not (jugador.get("target_lock") and not jugador.get("dead") and jefe.get("pos") and jugador.get("pos")):
            self._enfoque_perdido = 0
            return super()._necesita_enfoque()
        if float(self._cruda.get("dist", 1e9)) > self.config.distancia_lock:
            self._enfoque_perdido = 0
            return False
        dx = float(jefe["pos"][0]) - float(jugador["pos"][0])
        dy = float(jefe["pos"][1]) - float(jugador["pos"][1])
        desvio = abs((math.degrees(math.atan2(dy, dx)) - float(jugador.get("yaw", 0.0)) + 180.0) % 360.0 - 180.0)
        self._enfoque_perdido = self._enfoque_perdido + 1 if desvio > ANGULO_ENFOQUE_PERDIDO else 0
        return (self._enfoque_perdido >= DECISIONES_ENFOQUE_PERDIDO
                and self._pasos_desde_lock >= self.config.pasos_entre_intentos_lock)

    def _al_observar(self) -> None:
        """
        Actualiza las amenazas y el registro de golpes con la observacion cruda nueva.

        Contexto:
            reset y cada paso al juego, incluido el de enfoque.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.amenazas = self._seguidor.actualizar(self._cruda)
        ev = self._cruda.get("ev") or {}
        self._golpes.extend(ev.get("hits") or [])
        for registro in ev.get("remote") or []:
            self._ultimo_evento = (str(registro.get("id")), float(registro.get("t", 0.0)))

    def _resultado_final(self, cruda: dict[str, Any]) -> str | None:
        """
        Resultado del episodio con la regla de victoria del jefe en curso.

        Contexto:
            _avanzar.

        Args:
            cruda (dict[str, Any]): Observacion cruda.

        Returns:
            str | None: "victoria", "muerte" o None.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self._resolver_final(cruda)

    def _limite_frames(self) -> float:
        """
        Frames tras los cuales el episodio se trunca, segun el jefe en curso.

        Contexto:
            step.

        Args:
            Ninguno.

        Returns:
            float: segundos_maximos del registro del jefe * sim_hz.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self.catalogo.perfil(self._jefe).episodio.segundos_maximos * self.config.sim_hz

    def _vida_efectiva(self) -> float:
        """
        Vida que hay que quitar para ganar el episodio en curso.

        Contexto:
            _resumen.

        Args:
            Ninguno.

        Returns:
            float: Puntos de vida segun el registro del jefe.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return self.catalogo.vida_efectiva(self._jefe, self._multiplicador_episodio, self._fase_inicio)

    def _resumen(self, resultado: str) -> dict[str, Any]:
        """
        Metricas del episodio que acaba de terminar, con el jefe y la causa de la muerte.

        Contexto:
            step y _tras_caida.

        Args:
            resultado (str): "victoria", "muerte", "tiempo" o "error".

        Returns:
            dict[str, Any]: Metricas del entorno base mas jefe, mapa, golpe mortal y recargas.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        resumen = super()._resumen(resultado)
        resumen["jefe"] = self._jefe
        resumen["mapa"] = self.mapa
        resumen["golpe_mortal"] = self._golpe_mortal() if resultado == "muerte" else None
        resumen["recargas"] = self.recargas
        resumen["inicios_inconsistentes"] = self.inicios_inconsistentes
        resumen["reinicio_s"] = self.reinicio_s
        if resultado != "error":
            self.selector.registrar(self._jefe, resultado == "victoria")
        return resumen

    def _golpe_mortal(self) -> str:
        """
        Describe el ultimo golpe recibido: montage del atacante, ataque a distancia o desconocido.

        Contexto:
            _resumen, en las muertes. Necesita trazar=True.

        Args:
            Ninguno.

        Returns:
            str: "clase:montage", "evento:<id>", "remoto" o "sin_traza".

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if not self._golpes:
            return "sin_traza"
        golpe = self._golpes[-1]
        if golpe.get("am"):
            return f"{golpe.get('att', '?')}:{golpe['am']}"
        if self._ultimo_evento and float(golpe.get("t", 0.0)) - self._ultimo_evento[1] <= VENTANA_EVENTO_MORTAL:
            return f"evento:{self._ultimo_evento[0]}"
        return "remoto"

    def perfil_actual(self) -> PerfilJefe:
        """
        Perfil del jefe de episodio en curso.

        Contexto:
            Evaluacion y recoleccion.

        Args:
            Ninguno.

        Returns:
            PerfilJefe: Perfil del jefe de episodio.

        Raises:
            Ninguna.

        Ejemplo:
            env.perfil_actual().mapa
        """
        return self.catalogo.perfil(self._jefe)

    def mascara_lista(self) -> list[bool]:
        """
        Mascara de acciones como lista de bool.

        Contexto:
            Bot de reglas y recoleccion.

        Args:
            Ninguno.

        Returns:
            list[bool]: 22 valores.

        Raises:
            Ninguna.

        Ejemplo:
            env.mascara_lista()[acc.ACCION_NADA]  # True
        """
        return [bool(v) for v in self.action_masks()]

    def accion_efectiva(self, accion: int, info: dict[str, Any]) -> int:
        """
        Accion que llego realmente al juego tras step.

        Contexto:
            Bucles de evaluacion y recoleccion: el bot necesita saber si su accion se ejecuto.

        Args:
            accion (int): Accion que se envio a step.
            info (dict[str, Any]): Info devuelta por step.

        Returns:
            int: La misma accion, o "nada" si el entorno la sustituyo por no ser valida.

        Raises:
            Ninguna.

        Ejemplo:
            env.accion_efectiva(19, info)
        """
        return acc.ACCION_NADA if info.get("accion_invalida") else int(accion)
