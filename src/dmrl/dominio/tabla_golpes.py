"""
Tabla de golpes del jefe por montage, medida en el juego.

Cada montage de ataque del jefe tiene golpes que conectan en posiciones fijas del montage (en
segundos de montage, no de juego: el play rate los acelera, p. ej. SwingAndSlam va a 2.0).
Saber cuanto falta para el proximo golpe y si se puede parar es lo que decide entre parry,
roll o beber, tanto para el bot de reglas como para la observacion del agente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Golpe:
    """
    Un golpe del jefe dentro de un montage.

    Contexto:
        TablaGolpes. La posicion es la mediana de las observaciones; minimo y maximo dan la
        dispersion medida (algunos golpes finales de combo varian con la distancia).

    Atributos:
        pos: posicion tipica del golpe en segundos de montage.
        parry: True si el golpe se puede parar.
        minimo: menor posicion observada.
        maximo: mayor posicion observada.
        danio: dano tipico.
    """

    pos: float
    parry: bool
    minimo: float
    maximo: float
    danio: float


@dataclass(frozen=True)
class GolpePendiente:
    """
    El proximo golpe que va a conectar y el tiempo de juego que falta.

    Contexto:
        TablaGolpes.proximo_golpe.

    Atributos:
        golpe: golpe de la tabla.
        segundos: segundos de juego hasta el golpe tipico (play rate ya aplicado).
        indice: posicion del golpe dentro de su montage.
    """

    golpe: Golpe
    segundos: float
    indice: int


class TablaGolpes:
    """
    Consulta de golpes pendientes sobre una tabla ya cargada.

    Contexto:
        ConstructorObservacion y BotReglas. La carga desde archivo la hace la capa de
        infraestructura; aqui solo hay datos.
    """

    def __init__(self, golpes_por_montage: dict[str, tuple[Golpe, ...]] | None = None) -> None:
        """
        Crea la tabla.

        Contexto:
            Arranque del entorno o del bot.

        Args:
            golpes_por_montage (dict, opcional): Nombre de montage -> golpes ordenados por
                posicion. Vacia si no se da.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            tabla = TablaGolpes({"M": (Golpe(0.9, True, 0.88, 0.92, 10.0),)})
        """
        self._golpes = dict(golpes_por_montage or {})

    @classmethod
    def desde_dict(cls, datos: dict[str, Any]) -> TablaGolpes:
        """
        Construye la tabla desde el formato de data/tabla_golpes.json.

        Contexto:
            infraestructura.datos.cargar_tabla_golpes.

        Args:
            datos (dict[str, Any]): montage -> lista de {pos, min, max, parry, dano}.

        Returns:
            TablaGolpes: Tabla con los golpes ordenados por posicion.

        Raises:
            KeyError: Si a un golpe le falta pos o parry.

        Ejemplo:
            TablaGolpes.desde_dict({"M": [{"pos": 1.0, "parry": True}]})
        """
        tabla: dict[str, tuple[Golpe, ...]] = {}
        for nombre, filas in datos.items():
            golpes = [
                Golpe(
                    pos=float(f["pos"]), parry=bool(f["parry"]),
                    minimo=float(f.get("min", f["pos"])), maximo=float(f.get("max", f["pos"])),
                    danio=float(f.get("dano", 0.0)),
                )
                for f in filas
            ]
            tabla[nombre] = tuple(sorted(golpes, key=lambda g: g.pos))
        return cls(tabla)

    def montages(self) -> tuple[str, ...]:
        """
        Montages con golpes conocidos.

        Contexto:
            Pruebas y documentacion.

        Args:
            Ninguno.

        Returns:
            tuple[str, ...]: Nombres ordenados.

        Raises:
            Ninguna.

        Ejemplo:
            TablaGolpes().montages()  # ()
        """
        return tuple(sorted(self._golpes))

    def golpes(self, montage: str | None) -> tuple[Golpe, ...]:
        """
        Golpes de un montage.

        Contexto:
            Consulta directa del bot.

        Args:
            montage (str | None): Nombre del montage o None.

        Returns:
            tuple[Golpe, ...]: Vacia si el montage no es de ataque o no se midio.

        Raises:
            Ninguna.

        Ejemplo:
            tabla.golpes("Boss_Attack_Uppercut_InP_Montage")
        """
        return self._golpes.get(montage or "", ())

    def proximo_golpe(self, montage: str | None, pos: float, rate: float,
                      solo_parry: bool = False) -> GolpePendiente | None:
        """
        Proximo golpe que aun no ocurrio en el montage.

        Contexto:
            Observacion del agente y reglas del bot, en cada decision.

        Args:
            montage (str | None): Montage activo del jefe.
            pos (float): Posicion actual en segundos de montage.
            rate (float): Play rate del montage; los segundos de juego son (golpe - pos) / rate.
            solo_parry (bool): Si es True solo considera golpes que se pueden parar.

        Returns:
            GolpePendiente | None: El golpe mas cercano por delante; None si no queda ninguno.

        Raises:
            Ninguna.

        Ejemplo:
            tabla.proximo_golpe("Boss_Attack_Uppercut_InP_Montage", 0.5, 1.0)
        """
        velocidad = rate if rate > 0 else 1.0
        for indice, golpe in enumerate(self.golpes(montage)):
            if solo_parry and not golpe.parry:
                continue
            if golpe.pos >= pos:
                return GolpePendiente(golpe, (golpe.pos - pos) / velocidad, indice)
        return None
