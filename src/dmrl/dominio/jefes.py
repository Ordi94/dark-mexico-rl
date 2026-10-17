"""
Jefes como datos: perfil de cada jefe (mapa, clases, vida, vocabulario de montages, tabla de golpes,
ataques a distancia y como empieza su episodio) y catalogo que los reune.

Los perfiles salen de los registros data/jefes/<jefe>.json (ver infraestructura.jefes). Nada del
resto del paquete conoce jefes por nombre: la observacion, el bot y el entorno trabajan con un
PerfilJefe, de modo que agregar un jefe es agregar un registro.

Identidades: cada actor distinto que el juego pone como rival es una identidad (Juice y su actor de
fase 2 son dos; el Chamuco, fase 2 de AMLO, es otra). Un episodio empieza siempre en un jefe
"de episodio" (juice, amlo, chamuco, nahual), que puede pasar a otra identidad durante la pelea.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TIPOS_GOLPE = ("melee", "area", "proyectil", "rayo")
CENTROS = ("jefe", "jugador")
MAPAS = ("LVL_Boss", "LVL_AMLO")
# Alcance por encima del cual un golpe llega al jugador esté donde esté.
ALCANCE_GLOBAL = 6000.0


@dataclass(frozen=True)
class GolpeJefe:
    """
    Un golpe de un montage del jefe.

    Contexto:
        PerfilJefe.golpes. La posicion es la mediana medida en segundos de montage (el play rate
        la acelera); minimo y maximo dan la dispersion observada.

    Atributos:
        pos: posicion tipica del impacto en segundos de montage.
        parry: True si se puede parar.
        minimo: menor posicion observada.
        maximo: mayor posicion observada.
        danio: dano tipico.
        alcance: distancia 2D jefe-jugador (percentil 95 al impactar) por debajo de la cual conecta;
            ALCANCE_GLOBAL si llega a cualquier distancia.
        tipo: melee, area, proyectil o rayo.
        post_montage: el impacto sigue pendiente aunque el montage termine (el rayo cae despues).
        velocidad: uu/s del proyectil; el impacto se retrasa distancia / velocidad (0 = instantaneo).
    """

    pos: float
    parry: bool
    minimo: float
    maximo: float
    danio: float
    alcance: float = ALCANCE_GLOBAL
    tipo: str = "melee"
    post_montage: bool = False
    velocidad: float = 0.0


@dataclass(frozen=True)
class ImpactoEvento:
    """
    Un impacto de un ataque a distancia, medido desde el evento que lo lanza.

    Contexto:
        EventoDistancia.impactos.

    Atributos:
        retardo: segundos de juego entre el evento y el impacto (sin tiempo de vuelo).
        danio: dano del impacto.
    """

    retardo: float
    danio: float


@dataclass(frozen=True)
class EventoDistancia:
    """
    Ataque cuyo impacto no pertenece al montage que lo lanza (katanas, area en el piso).

    Contexto:
        PerfilJefe.eventos. El mod avisa cuando ocurre (hook sobre la funcion Blueprint) y el
        seguidor de amenazas programa sus impactos.

    Atributos:
        id: identificador del evento (el mismo que usa el hook del mod).
        impactos: impactos que provoca.
        velocidad: uu/s del proyectil (0 = el tiempo de vuelo no depende de la distancia).
        parry: True si se puede parar.
        tipo: area, proyectil o rayo.
        esquivable: True si la invulnerabilidad de la maniobra lo evita.
        centro: de donde viene ("jefe" = proyectil desde el jefe, "jugador" = area sobre el jugador).
        vigencia: segundos tras el impacto en que todavia cuenta como amenaza (tolerancia de reloj).
    """

    id: str
    impactos: tuple[ImpactoEvento, ...]
    velocidad: float = 0.0
    parry: bool = False
    tipo: str = "proyectil"
    esquivable: bool = True
    centro: str = "jefe"
    vigencia: float = 0.25


@dataclass(frozen=True)
class EpisodioJefe:
    """
    Como empieza el episodio de un jefe y como se gana.

    Contexto:
        PerfilJefe.episodio. Los jefes que no inician un episodio por si mismos (el actor de fase 2
        de Juice) solo apuntan a quien lo hace.

    Atributos:
        via: identificador del jefe cuyo reinicio arranca este episodio.
        start_phase: fase de inicio que se pide en reset (1 o 2).
        vida_fase1: vida util que hay que quitar en la fase 1 (tras descartar lo que se pierde al
            transformarse).
        vida_fase2: vida de la fase siguiente.
        clases_victoria: la muerte de una de estas clases gana el episodio.
        segundos_maximos: segundos de juego tras los cuales el episodio se trunca (una pelea de AMLO
            con sus dos fases dura mas que una de Juice).
        mod: descripcion que valida el mod (seccion "mod" del registro); None si no inicia episodio.
    """

    via: str
    start_phase: int
    vida_fase1: float
    vida_fase2: float
    clases_victoria: tuple[str, ...] = ()
    segundos_maximos: float = 360.0
    mod: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PerfilJefe:
    """
    Todo lo que se sabe de una identidad de jefe.

    Contexto:
        CatalogoJefes. Se carga de data/jefes/<id>.json.

    Atributos:
        id: identificador (juice, juice_fase2, amlo, chamuco, nahual).
        nombre: nombre para mostrar.
        mapa: mapa del juego donde pelea.
        indice: posicion en el one-hot de identidad de la observacion.
        clases: clases Blueprint de esta identidad.
        vida_max: vida maxima del actor.
        fase_avanzada: True si es la segunda fase de su pelea.
        reacciones: montages de aturdimiento tras un parry.
        alcance_defecto: alcance de un golpe sin alcance propio.
        vocabulario: montages de este jefe (para el one-hot de la observacion).
        golpes: montage -> golpes, ordenados por posicion.
        eventos: id -> ataque a distancia.
        episodio: como empieza y como se gana su episodio.
        bot: parametros propios del bot de reglas (los que no esten usan los de fabrica).
    """

    id: str
    nombre: str
    mapa: str
    indice: int
    clases: tuple[str, ...]
    vida_max: float
    fase_avanzada: bool
    reacciones: frozenset[str]
    alcance_defecto: float
    vocabulario: tuple[str, ...]
    golpes: Mapping[str, tuple[GolpeJefe, ...]]
    eventos: Mapping[str, EventoDistancia]
    episodio: EpisodioJefe
    bot: Mapping[str, float] = field(default_factory=dict)

    @property
    def fases_inicio(self) -> tuple[int, ...]:
        """
        Fases en las que puede empezar un episodio de este jefe de episodio.

        Contexto:
            Selector de fase del entorno multi-jefe.

        Args:
            Ninguno.

        Returns:
            tuple[int, ...]: (1, 2) si el mod sabe arrancar la pelea en la segunda fase (Juice por
                el flujo original, AMLO por la transicion natural al Chamuco); si no, la unica fase
                con la que empieza (el Chamuco directo arranca ya en su fase).

        Raises:
            Ninguna.

        Ejemplo:
            catalogo.perfil("amlo").fases_inicio  # (1, 2)
        """
        return (1, 2) if self._admite_fase2 else (self.episodio.start_phase,)

    @property
    def _admite_fase2(self) -> bool:
        """
        Indica si el mod puede arrancar este jefe directamente en su segunda fase.

        Contexto:
            fases_inicio y fase_para_mod.

        Args:
            Ninguno.

        Returns:
            bool: True con arranque por la niebla (Juice) o con la seccion phase2 en la descripcion.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        mod = self.episodio.mod or {}
        return mod.get("start", {}).get("kind") == "legacy_fog" or "phase2" in mod

    def fase_para_mod(self, fase_inicio: int) -> int:
        """
        start_phase que se envia al mod para pedir una fase de inicio.

        Contexto:
            EntornoMultiJefe, al pedir el reinicio.

        Args:
            fase_inicio (int): Fase con la que el entorno cuenta el episodio (1 o 2).

        Returns:
            int: 2 solo si el mod sabe arrancar en la segunda fase; 1 en los demas casos.

        Raises:
            Ninguna.

        Ejemplo:
            catalogo.perfil("chamuco").fase_para_mod(2)  # 1
        """
        return 2 if fase_inicio == 2 and self._admite_fase2 else 1

    def golpes_de(self, montage: str | None) -> tuple[GolpeJefe, ...]:
        """
        Golpes de un montage.

        Contexto:
            SeguidorAmenazas, al empezar cada montage del jefe.

        Args:
            montage (str | None): Nombre del montage o None.

        Returns:
            tuple[GolpeJefe, ...]: Vacia si no es un montage de ataque conocido.

        Raises:
            Ninguna.

        Ejemplo:
            perfil.golpes_de("Boss_Attack_Uppercut_InP_Montage")
        """
        return self.golpes.get(montage or "", ())

    @classmethod
    def desde_dict(cls, datos: Mapping[str, Any]) -> PerfilJefe:
        """
        Construye y valida un perfil desde el formato de data/jefes/<id>.json.

        Contexto:
            infraestructura.jefes.cargar_catalogo.

        Args:
            datos (Mapping[str, Any]): Contenido del registro.

        Returns:
            PerfilJefe: Perfil validado.

        Raises:
            ValueError: Si falta un campo, un valor esta fuera de lo admitido, un golpe esta mal
                formado o un montage con golpes no esta en el vocabulario del jefe.

        Ejemplo:
            PerfilJefe.desde_dict(json.load(open("data/jefes/amlo.json")))
        """
        try:
            identificador = str(datos["id"])
            mapa = str(datos["mapa"])
            if mapa not in MAPAS:
                raise ValueError(f"{identificador}: mapa no admitido {mapa!r}")
            vocabulario = tuple(str(m) for m in datos["vocabulario"])
            if len(set(vocabulario)) != len(vocabulario):
                raise ValueError(f"{identificador}: vocabulario con montages repetidos")
            sin_vocabulario = sorted(set(datos["golpes"]) - set(vocabulario))
            if sin_vocabulario:
                # Un montage con golpes fuera del vocabulario caeria en "otro" y la red no distinguiria el ataque.
                raise ValueError(f"{identificador}: montages con golpes fuera del vocabulario {sin_vocabulario}")
            return cls(
                id=identificador, nombre=str(datos.get("nombre", identificador)), mapa=mapa,
                indice=int(datos["indice"]), clases=tuple(str(c) for c in datos["clases"]),
                vida_max=float(datos["vida_max"]), fase_avanzada=bool(datos["fase_avanzada"]),
                reacciones=frozenset(str(m) for m in datos.get("reacciones", ())),
                alcance_defecto=float(datos.get("alcance_defecto", 350.0)), vocabulario=vocabulario,
                golpes={str(m): _golpes(identificador, m, filas) for m, filas in datos["golpes"].items()},
                eventos={str(k): _evento(identificador, str(k), v)
                         for k, v in datos.get("eventos_distancia", {}).items()},
                episodio=_episodio(identificador, datos["episodio"]),
                bot={str(k): float(v) for k, v in datos.get("bot", {}).items()},
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"registro de jefe mal formado: {exc!r}") from exc


def _golpes(jefe: str, montage: str, filas: Sequence[Mapping[str, Any]]) -> tuple[GolpeJefe, ...]:
    """
    Convierte las filas de un montage en golpes validados y ordenados.

    Contexto:
        PerfilJefe.desde_dict.

    Args:
        jefe (str): Identificador del jefe (para los mensajes).
        montage (str): Nombre del montage.
        filas (Sequence[Mapping[str, Any]]): Filas del registro.

    Returns:
        tuple[GolpeJefe, ...]: Golpes por posicion creciente.

    Raises:
        ValueError: Si el tipo no se admite o un valor numerico es negativo.
        KeyError: Si falta pos o parry.

    Ejemplo:
        (uso interno)
    """
    golpes = []
    for f in filas:
        tipo = str(f.get("tipo", "melee"))
        if tipo not in TIPOS_GOLPE:
            raise ValueError(f"{jefe}/{montage}: tipo de golpe no admitido {tipo!r}")
        golpe = GolpeJefe(
            pos=float(f["pos"]), parry=bool(f["parry"]), minimo=float(f.get("min", f["pos"])),
            maximo=float(f.get("max", f["pos"])), danio=float(f.get("dano", 0.0)),
            alcance=float(f.get("alcance", ALCANCE_GLOBAL)), tipo=tipo,
            post_montage=bool(f.get("post_montage", False)), velocidad=float(f.get("velocidad", 0.0)),
        )
        if min(golpe.pos, golpe.danio, golpe.alcance, golpe.velocidad) < 0:
            raise ValueError(f"{jefe}/{montage}: valor negativo en un golpe")
        golpes.append(golpe)
    return tuple(sorted(golpes, key=lambda g: g.pos))


def _evento(jefe: str, identificador: str, datos: Mapping[str, Any]) -> EventoDistancia:
    """
    Convierte un ataque a distancia del registro.

    Contexto:
        PerfilJefe.desde_dict.

    Args:
        jefe (str): Identificador del jefe (para los mensajes).
        identificador (str): Id del evento.
        datos (Mapping[str, Any]): Campos del registro.

    Returns:
        EventoDistancia: Evento validado.

    Raises:
        ValueError: Si el centro o el tipo no se admiten o no hay impactos.
        KeyError: Si falta la lista de impactos.

    Ejemplo:
        (uso interno)
    """
    centro = str(datos.get("centro", "jefe"))
    tipo = str(datos.get("tipo", "proyectil"))
    if centro not in CENTROS:
        raise ValueError(f"{jefe}/{identificador}: centro no admitido {centro!r}")
    if tipo not in TIPOS_GOLPE or tipo == "melee":
        raise ValueError(f"{jefe}/{identificador}: tipo de ataque a distancia no admitido {tipo!r}")
    impactos = tuple(ImpactoEvento(float(i["retardo"]), float(i.get("danio", 0.0))) for i in datos["impactos"])
    if not impactos:
        raise ValueError(f"{jefe}/{identificador}: sin impactos")
    return EventoDistancia(
        id=identificador, impactos=impactos, velocidad=float(datos.get("velocidad", 0.0)),
        parry=bool(datos.get("parry", False)), tipo=tipo, esquivable=bool(datos.get("esquivable", True)),
        centro=centro, vigencia=float(datos.get("vigencia", 0.25)),
    )


def _episodio(jefe: str, datos: Mapping[str, Any]) -> EpisodioJefe:
    """
    Convierte la seccion de episodio del registro.

    Contexto:
        PerfilJefe.desde_dict.

    Args:
        jefe (str): Identificador del jefe (para los mensajes).
        datos (Mapping[str, Any]): Seccion "episodio".

    Returns:
        EpisodioJefe: Episodio validado.

    Raises:
        ValueError: Si la fase de inicio no es 1 o 2.
        KeyError: Si falta algun campo.

    Ejemplo:
        (uso interno)
    """
    fase = int(datos["start_phase"])
    if fase not in (1, 2):
        raise ValueError(f"{jefe}: start_phase debe ser 1 o 2")
    if float(datos.get("segundos_maximos", 360.0)) <= 0:
        raise ValueError(f"{jefe}: segundos_maximos debe ser positivo")
    mod = datos.get("mod")
    return EpisodioJefe(
        via=str(datos["via"]), start_phase=fase, vida_fase1=float(datos["vida_fase1"]),
        vida_fase2=float(datos["vida_fase2"]), clases_victoria=tuple(str(c) for c in datos.get("clases_victoria", ())),
        segundos_maximos=float(datos.get("segundos_maximos", 360.0)), mod=dict(mod) if mod is not None else None,
    )


class CatalogoJefes:
    """
    Conjunto de perfiles con las consultas que necesitan la observacion, el entorno y el bot.

    Contexto:
        Lo construye infraestructura.jefes.cargar_catalogo y lo comparten el entorno, el seguidor
        de amenazas, el constructor de la observacion y el bot.
    """

    def __init__(self, perfiles: Sequence[PerfilJefe]) -> None:
        """
        Crea el catalogo y valida la coherencia de los perfiles.

        Contexto:
            Arranque del entorno, de la evaluacion o de una prueba.

        Args:
            perfiles (Sequence[PerfilJefe]): Perfiles de todas las identidades.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si no hay perfiles, un identificador, un indice o una clase se repiten, los
                indices no son 0..n-1, o un evento de distancia se repite entre jefes.

        Ejemplo:
            catalogo = CatalogoJefes(perfiles)
        """
        if not perfiles:
            raise ValueError("el catalogo necesita al menos un perfil")
        self._perfiles = {p.id: p for p in perfiles}
        if len(self._perfiles) != len(perfiles):
            raise ValueError("identificadores de jefe repetidos")
        if sorted(p.indice for p in perfiles) != list(range(len(perfiles))):
            raise ValueError("los indices de identidad deben ser 0..n-1 sin repetir")
        self._por_clase: dict[str, PerfilJefe] = {}
        for p in perfiles:
            for clase in p.clases:
                if clase in self._por_clase:
                    raise ValueError(f"la clase {clase} pertenece a dos jefes")
                self._por_clase[clase] = p
        self._eventos: dict[str, EventoDistancia] = {}
        for p in perfiles:
            for identificador, evento in p.eventos.items():
                if identificador in self._eventos:
                    raise ValueError(f"evento de distancia repetido: {identificador}")
                self._eventos[identificador] = evento
        self._orden = tuple(sorted(perfiles, key=lambda p: p.indice))
        union: list[str] = []
        for p in self._orden:
            union += [m for m in p.vocabulario if m not in union]
        self._vocabulario = tuple(union)

    @property
    def perfiles(self) -> tuple[PerfilJefe, ...]:
        """
        Perfiles ordenados por indice de identidad.

        Contexto:
            Observacion y documentacion.

        Args:
            Ninguno.

        Returns:
            tuple[PerfilJefe, ...]: Un perfil por identidad.

        Raises:
            Ninguna.

        Ejemplo:
            [p.id for p in catalogo.perfiles]
        """
        return self._orden

    @property
    def vocabulario(self) -> tuple[str, ...]:
        """
        Vocabulario UNION de montages de todos los jefes (orden de identidad, sin repetir).

        Contexto:
            One-hot del montage del jefe en la observacion v2.

        Args:
            Ninguno.

        Returns:
            tuple[str, ...]: Nombres de montage.

        Raises:
            Ninguna.

        Ejemplo:
            len(catalogo.vocabulario)
        """
        return self._vocabulario

    def perfil(self, identificador: str) -> PerfilJefe:
        """
        Perfil de una identidad.

        Contexto:
            Entorno y bot.

        Args:
            identificador (str): Id del jefe.

        Returns:
            PerfilJefe: Perfil.

        Raises:
            KeyError: Si no existe.

        Ejemplo:
            catalogo.perfil("amlo")
        """
        return self._perfiles[identificador]

    def por_clase(self, clase: str | None) -> PerfilJefe | None:
        """
        Perfil al que pertenece una clase Blueprint.

        Contexto:
            Seguidor de amenazas y observacion: la identidad se deduce de la clase del actor, nunca
            de la bandera IS BOSS?.

        Args:
            clase (str | None): Clase del actor jefe.

        Returns:
            PerfilJefe | None: Perfil, o None si la clase no es de ningun jefe conocido.

        Raises:
            Ninguna.

        Ejemplo:
            catalogo.por_clase("BP_Chamuco_C")
        """
        return self._por_clase.get(clase or "")

    def evento(self, identificador: str) -> EventoDistancia | None:
        """
        Ataque a distancia por id.

        Contexto:
            Seguidor de amenazas, al recibir un evento del mod.

        Args:
            identificador (str): Id del evento.

        Returns:
            EventoDistancia | None: Evento, o None si ningun jefe lo declara.

        Raises:
            Ninguna.

        Ejemplo:
            catalogo.evento("amlo_throw_kat")
        """
        return self._eventos.get(identificador)

    def jefes_de_episodio(self, mapa: str | None = None) -> tuple[str, ...]:
        """
        Jefes que pueden iniciar un episodio (opcionalmente solo los de un mapa).

        Contexto:
            Selector de jefes del entorno.

        Args:
            mapa (str, opcional): Si se da, solo los jefes de ese mapa.

        Returns:
            tuple[str, ...]: Ids ordenados por indice.

        Raises:
            Ninguna.

        Ejemplo:
            catalogo.jefes_de_episodio("LVL_AMLO")  # ("amlo", "chamuco", "nahual")
        """
        return tuple(
            p.id for p in self._orden
            if p.episodio.mod is not None and p.episodio.via == p.id and (mapa is None or p.mapa == mapa)
        )

    def vida_efectiva(self, jefe: str, multiplicador: float, fase_inicio: int) -> float:
        """
        Vida total que el agente debe quitar para ganar un episodio.

        Contexto:
            Metrica de progreso contra el jefe, comparable entre jefes y niveles de dificultad.

        Args:
            jefe (str): Jefe de episodio.
            multiplicador (float): Multiplicador de vida del episodio.
            fase_inicio (int): Fase en la que arranca (1 o 2).

        Returns:
            float: Puntos de vida a quitar.

        Raises:
            KeyError: Si el jefe no existe.

        Ejemplo:
            catalogo.vida_efectiva("juice", 1.0, 1)  # 2700.0
        """
        episodio = self.perfil(jefe).episodio
        base = episodio.vida_fase2 + (episodio.vida_fase1 if fase_inicio == 1 else 0.0)
        return base * multiplicador
