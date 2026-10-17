"""
Amenazas del jefe: los golpes que van a conectar, cuando y de que tipo.

El seguidor convierte la observacion cruda en una lista de impactos pendientes con tiempo absoluto
de juego. Hay dos origenes:
    * montage: la tabla del jefe dice en que posicion del montage conecta cada golpe; al ver
      empezar un montage se programan todos sus golpes. Un golpe marcado post_montage (el rayo)
      sigue pendiente aunque el montage termine, que es justo cuando una lectura "golpes por
      delante en el montage actual" lo perderia de vista. Los golpes con velocidad (rafaga de fuego)
      se retrasan distancia / velocidad.
    * evento: el mod avisa cuando el jefe lanza un ataque cuyo impacto no pertenece a ningun montage
      (katanas lanzadas, katanas del piso) y el catalogo dice cuanto tarda en llegar.
Un montage que se interrumpe (parry, cambio de ataque) cancela los golpes programados que no sean
post_montage. Mientras el montage sigue, el tiempo de cada golpe se recalcula en cada observacion con
su posicion actual (el juego detiene la animacion unas decimas al conectar o recibir golpes y una hora
de inicio fija se desfasaria); al terminar el montage el golpe post_montage conserva su ultimo tiempo.
Un proyectil vuela distancia / velocidad desde que se lanza: antes del lanzamiento se usa la distancia
actual y despues la que habia al lanzarse. Las mismas amenazas alimentan la observacion v2 y el bot,
de modo que ven lo mismo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dmrl.dominio.jefes import ALCANCE_GLOBAL, CatalogoJefes, EventoDistancia, GolpeJefe, PerfilJefe

# Retroceso de la posicion del montage que indica que se reinicio (mismo nombre, otra vez).
TOLERANCIA_REINICIO = 0.05
# Un golpe programado se descarta cuando paso hace mas que esto (segundos).
RETENCION = 0.3
# Un golpe melee solo conecta si el jugador esta a menos de alcance * FACTOR + MARGEN (el jefe se
# adelanta con el movimiento del ataque; el alcance medido es el percentil 95 al impactar).
FACTOR_ALCANCE = 1.5
MARGEN_ALCANCE = 100.0


@dataclass(frozen=True)
class Amenaza:
    """
    Un impacto pendiente del jefe.

    Contexto:
        Amenazas.pendientes; lo consumen la observacion v2 y el bot de reglas.

    Atributos:
        clave: identidad estable del impacto mientras dure (el bot marca con ella los atendidos).
        segundos: segundos de juego hasta el impacto (el reloj del juego, play rate ya aplicado).
        parry: True si se puede parar.
        danio: dano tipico.
        alcance: distancia 2D por debajo de la cual conecta (melee); global en los demas tipos.
        tipo: melee, area, proyectil o rayo.
        esquivable: True si la invulnerabilidad de la maniobra lo evita.
        origen: "montage" o "evento".
        indice: posicion del impacto dentro de su ataque.
        origen_xy: donde estaba el origen (jefe o jugador) al programarse el impacto; None si no se supo.
        centro: "jefe" (viene del jefe) o "jugador" (cae sobre el jugador).
    """

    clave: tuple[Any, ...]
    segundos: float
    parry: bool
    danio: float
    alcance: float
    tipo: str
    esquivable: bool
    origen: str
    indice: int
    origen_xy: tuple[float, float] | None = None
    centro: str = "jefe"

    def en_alcance(self, distancia: float) -> bool:
        """
        Indica si el impacto puede alcanzar al jugador a esa distancia del jefe.

        Contexto:
            Observacion v2 y bot.

        Args:
            distancia (float): Distancia 2D jugador-jefe.

        Returns:
            bool: True siempre para area, proyectil y rayo; para melee, si el jugador esta dentro del
                alcance ampliado.

        Raises:
            Ninguna.

        Ejemplo:
            amenaza.en_alcance(120.0)
        """
        if self.tipo != "melee":
            return True
        return distancia <= self.alcance * FACTOR_ALCANCE + MARGEN_ALCANCE


@dataclass(frozen=True)
class Amenazas:
    """
    Impactos pendientes en un instante, ordenados por tiempo.

    Contexto:
        Salida de SeguidorAmenazas.actualizar.

    Atributos:
        pendientes: impactos que aun no ocurrieron, del mas cercano al mas lejano.
        distancia: distancia 2D jugador-jefe en ese instante.
    """

    pendientes: tuple[Amenaza, ...] = ()
    distancia: float = 0.0

    @property
    def proximo(self) -> Amenaza | None:
        """
        El impacto mas cercano en el tiempo.

        Contexto:
            Decision del bot y observacion.

        Args:
            Ninguno.

        Returns:
            Amenaza | None: None si no hay ninguno pendiente.

        Raises:
            Ninguna.

        Ejemplo:
            amenazas.proximo
        """
        return self.pendientes[0] if self.pendientes else None

    @property
    def proximo_parry(self) -> Amenaza | None:
        """
        El impacto parable mas cercano.

        Contexto:
            Decision del bot y observacion.

        Args:
            Ninguno.

        Returns:
            Amenaza | None: None si no hay ninguno.

        Raises:
            Ninguna.

        Ejemplo:
            amenazas.proximo_parry
        """
        return next((a for a in self.pendientes if a.parry), None)

    @property
    def remota(self) -> Amenaza | None:
        """
        El impacto a distancia mas cercano (evento, area, proyectil o rayo).

        Contexto:
            Bloque de amenaza remota de la observacion v2.

        Args:
            Ninguno.

        Returns:
            Amenaza | None: None si no hay ninguno.

        Raises:
            Ninguna.

        Ejemplo:
            amenazas.remota
        """
        return next((a for a in self.pendientes if a.origen == "evento" or a.tipo != "melee"), None)

    @property
    def del_ataque(self) -> int:
        """
        Cuantos impactos quedan en el ataque en curso (los programados desde un montage).

        Contexto:
            Observacion v2 (golpes pendientes).

        Args:
            Ninguno.

        Returns:
            int: Cantidad de impactos de origen montage por delante.

        Raises:
            Ninguna.

        Ejemplo:
            amenazas.del_ataque
        """
        return sum(1 for a in self.pendientes if a.origen == "montage")


@dataclass
class _Programada:
    """
    Impacto programado con su tiempo de juego.

    Contexto:
        SeguidorAmenazas.

    Atributos:
        t_abs: instante de juego del impacto (se recalcula mientras el montage siga).
        base: Amenaza con segundos en cero (se recalcula en cada consulta).
        persistente: sobrevive a la interrupcion o fin del montage que lo programo.
        pos_golpe: posicion del impacto en su montage (None en los eventos).
        t_lanzamiento: instante de juego en que sale el proyectil o cae el ataque (sin vuelo).
        velocidad: uu/s del proyectil (0 = instantaneo).
        d_fija: distancia jugador-jefe al lanzarse, una vez lanzado; None antes.
        montage_id: contador del montage de origen (0 en los eventos).
        vigencia: segundos que se retiene tras el impacto.
    """

    t_abs: float
    base: Amenaza
    persistente: bool
    pos_golpe: float | None = None
    t_lanzamiento: float = 0.0
    velocidad: float = 0.0
    d_fija: float | None = None
    montage_id: int = 0
    vigencia: float = RETENCION

    def recalcular(self, ahora: float, distancia: float, pos_montage: float | None, rate: float) -> None:
        """
        Actualiza el instante del impacto con lo ultimo observado.

        Contexto:
            SeguidorAmenazas.actualizar, en cada observacion.

        Args:
            ahora (float): Reloj de juego.
            distancia (float): Distancia 2D jugador-jefe actual.
            pos_montage (float | None): Posicion actual del montage de origen si sigue activo; None
                si ya termino (el impacto conserva su tiempo) o es un evento.
            rate (float): Play rate del montage.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        if pos_montage is not None and self.pos_golpe is not None:
            self.t_lanzamiento = ahora + (self.pos_golpe - pos_montage) / rate
        if self.velocidad <= 0:
            self.t_abs = self.t_lanzamiento
            return
        if self.d_fija is None and self.t_lanzamiento <= ahora:
            self.d_fija = distancia
        d = self.d_fija if self.d_fija is not None else distancia
        self.t_abs = self.t_lanzamiento + d / self.velocidad


@dataclass
class SeguidorAmenazas:
    """
    Sigue los ataques del jefe y mantiene la lista de impactos pendientes.

    Contexto:
        Lo posee el entorno multi-jefe (una instancia por entorno) y se actualiza con cada
        observacion cruda, incluida la del paso de enfoque. Tambien puede usarse suelto en el bot.
    """

    catalogo: CatalogoJefes
    _programadas: list[_Programada] = field(default_factory=list)
    _nombre: str | None = None
    _pos: float = 0.0
    _montage_id: int = 0
    _clase: str | None = None
    _secuencia_eventos: int = 0

    def reiniciar(self) -> None:
        """
        Olvida todo (episodio nuevo).

        Contexto:
            Tras cada reset del entorno.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            seguidor.reiniciar()
        """
        self._programadas = []
        self._nombre = None
        self._pos = 0.0
        self._clase = None

    def actualizar(self, obs: dict[str, Any]) -> Amenazas:
        """
        Incorpora una observacion cruda y devuelve los impactos pendientes.

        Contexto:
            DarkMexicoEnv (version multi-jefe), tras cada paso al juego.

        Args:
            obs (dict[str, Any]): Observacion cruda del puente (t, dist, boss, ev.remote).

        Returns:
            Amenazas: Impactos por delante, del mas cercano al mas lejano.

        Raises:
            Ninguna: Una observacion sin jefe devuelve la lista de eventos aun pendientes.

        Ejemplo:
            amenazas = seguidor.actualizar(obs)
        """
        ahora = float(obs.get("t", 0.0))
        distancia = float(obs.get("dist", 0.0))
        jefe = obs.get("boss") or {}
        perfil = self.catalogo.por_clase(jefe.get("cls")) if jefe else None
        clase = jefe.get("cls") if jefe else None
        if clase != self._clase:
            self._cancelar_montages(forzar=True)
            self._clase = clase
            self._nombre = None
        montage = jefe.get("montage") or {}
        nombre = montage.get("name")
        pos = float(montage.get("pos", 0.0))
        if nombre != self._nombre or (nombre is not None and pos < self._pos - TOLERANCIA_REINICIO):
            self._cancelar_montages(forzar=False)
            self._montage_id += 1
            if perfil is not None and nombre is not None and not jefe.get("dead"):
                self._programar_montage(perfil, montage, ahora, jefe)
        self._nombre, self._pos = nombre, pos
        for registro in (obs.get("ev") or {}).get("remote") or []:
            self._programar_evento(registro, distancia)
        rate = float(montage.get("rate", 1.0)) or 1.0
        rate = rate if rate > 0 else 1.0
        for programada in self._programadas:
            activo = nombre is not None and programada.montage_id == self._montage_id
            programada.recalcular(ahora, distancia, pos if activo else None, rate)
        self._programadas = [p for p in self._programadas if p.t_abs - ahora >= -p.vigencia]
        pendientes = sorted(
            (self._instante(p, ahora) for p in self._programadas if p.t_abs - ahora >= 0.0),
            key=lambda a: a.segundos,
        )
        return Amenazas(tuple(pendientes), distancia)

    @staticmethod
    def _instante(programada: _Programada, ahora: float) -> Amenaza:
        """
        Amenaza con los segundos que faltan en este instante.

        Contexto:
            actualizar.

        Args:
            programada (_Programada): Impacto programado.
            ahora (float): Reloj de juego.

        Returns:
            Amenaza: Copia con segundos actualizados.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        base = programada.base
        return Amenaza(base.clave, programada.t_abs - ahora, base.parry, base.danio, base.alcance, base.tipo,
                       base.esquivable, base.origen, base.indice, base.origen_xy, base.centro)

    def _cancelar_montages(self, forzar: bool) -> None:
        """
        Cancela los impactos de montage que ya no van a ocurrir.

        Contexto:
            actualizar, al cambiar o reiniciarse el montage del jefe, o al cambiar de jefe.

        Args:
            forzar (bool): True cancela tambien los persistentes (cambio de jefe).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self._programadas = [
            p for p in self._programadas
            if p.base.origen == "evento" or (p.persistente and not forzar)
        ]

    def _programar_montage(self, perfil: PerfilJefe, montage: dict[str, Any], ahora: float,
                           jefe: dict[str, Any]) -> None:
        """
        Programa los golpes que quedan en un montage recien visto.

        Contexto:
            actualizar. El tiempo exacto de cada golpe lo fija recalcular con la posicion del montage.

        Args:
            perfil (PerfilJefe): Jefe que ejecuta el montage.
            montage (dict[str, Any]): Bloque montage (name, pos, rate).
            ahora (float): Reloj de juego.
            jefe (dict[str, Any]): Bloque boss (posicion del origen).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        pos = float(montage.get("pos", 0.0))
        origen = _xy(jefe)
        for indice, golpe in enumerate(perfil.golpes_de(montage.get("name"))):
            if golpe.pos < pos:
                continue
            self._programadas.append(_Programada(
                t_abs=ahora, base=_amenaza_de_golpe(golpe, ("m", self._montage_id, indice), indice, origen),
                persistente=golpe.post_montage, pos_golpe=golpe.pos, velocidad=golpe.velocidad,
                montage_id=self._montage_id,
            ))

    def _programar_evento(self, registro: dict[str, Any], distancia: float) -> None:
        """
        Programa los impactos de un ataque a distancia anunciado por el mod.

        Contexto:
            actualizar. Los eventos que ningun jefe declara (hooks solo de medicion) se ignoran.

        Args:
            registro (dict[str, Any]): Evento del mod (id, t y, si se supo, px, py, bx, by).
            distancia (float): Distancia 2D actual, si el evento no trae posiciones.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        evento = self.catalogo.evento(str(registro.get("id")))
        if evento is None:
            return
        jugador = _par(registro, "px", "py")
        jefe = _par(registro, "bx", "by")
        lanzado = float(registro.get("t", 0.0))
        d = math.hypot(jugador[0] - jefe[0], jugador[1] - jefe[1]) if jugador and jefe else distancia
        origen = jefe if evento.centro == "jefe" else jugador
        self._secuencia_eventos += 1
        for indice, impacto in enumerate(evento.impactos):
            lanzamiento = lanzado + impacto.retardo
            self._programadas.append(_Programada(
                t_abs=lanzamiento, base=_amenaza_de_evento(
                    evento, impacto.danio, ("e", self._secuencia_eventos, indice), indice, origen),
                persistente=True, t_lanzamiento=lanzamiento, velocidad=evento.velocidad, d_fija=d,
                vigencia=evento.vigencia,
            ))


def _xy(entidad: dict[str, Any]) -> tuple[float, float] | None:
    """
    Posicion 2D de una entidad de la observacion.

    Contexto:
        SeguidorAmenazas.

    Args:
        entidad (dict[str, Any]): Bloque con "pos" [x, y, z].

    Returns:
        tuple[float, float] | None: Posicion, o None si no la trae.

    Raises:
        Ninguna.

    Ejemplo:
        _xy({"pos": [1.0, 2.0, 3.0]})  # (1.0, 2.0)
    """
    pos = entidad.get("pos")
    return (float(pos[0]), float(pos[1])) if pos else None


def _par(registro: dict[str, Any], x: str, y: str) -> tuple[float, float] | None:
    """
    Par de coordenadas de un evento, si el mod las envio.

    Contexto:
        SeguidorAmenazas._programar_evento.

    Args:
        registro (dict[str, Any]): Evento del mod.
        x (str): Clave de la coordenada X.
        y (str): Clave de la coordenada Y.

    Returns:
        tuple[float, float] | None: Par, o None si falta alguna.

    Raises:
        Ninguna.

    Ejemplo:
        _par({"px": 1.0, "py": 2.0}, "px", "py")  # (1.0, 2.0)
    """
    if x in registro and y in registro:
        return float(registro[x]), float(registro[y])
    return None


def _amenaza_de_golpe(golpe: GolpeJefe, clave: tuple[Any, ...], indice: int,
                      origen: tuple[float, float] | None) -> Amenaza:
    """
    Amenaza base de un golpe de montage.

    Contexto:
        SeguidorAmenazas._programar_montage.

    Args:
        golpe (GolpeJefe): Golpe de la tabla.
        clave (tuple[Any, ...]): Identidad del impacto.
        indice (int): Posicion en el montage.
        origen (tuple[float, float] | None): Posicion del jefe.

    Returns:
        Amenaza: Amenaza con segundos en cero.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    return Amenaza(clave, 0.0, golpe.parry, golpe.danio, golpe.alcance, golpe.tipo, not golpe.parry, "montage",
                   indice, origen, "jefe" if golpe.tipo in ("melee", "proyectil") else "jugador")


def _amenaza_de_evento(evento: EventoDistancia, danio: float, clave: tuple[Any, ...], indice: int,
                       origen: tuple[float, float] | None) -> Amenaza:
    """
    Amenaza base de un impacto de ataque a distancia.

    Contexto:
        SeguidorAmenazas._programar_evento.

    Args:
        evento (EventoDistancia): Ataque declarado en el catalogo.
        danio (float): Dano del impacto.
        clave (tuple[Any, ...]): Identidad del impacto.
        indice (int): Posicion entre los impactos del evento.
        origen (tuple[float, float] | None): Posicion del origen del ataque.

    Returns:
        Amenaza: Amenaza con segundos en cero.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    return Amenaza(clave, 0.0, evento.parry, danio, ALCANCE_GLOBAL, evento.tipo, evento.esquivable, "evento",
                   indice, origen, evento.centro)
