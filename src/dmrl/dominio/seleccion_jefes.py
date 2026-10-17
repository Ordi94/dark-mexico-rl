"""
Reparto de episodios entre jefes y de instancias del juego entre mapas.

Cada instancia del juego queda fija a un mapa (LVL_Boss o LVL_AMLO), asi que el reparto tiene dos
niveles: cuantas instancias atiende cada mapa (repartir_instancias, segun el peso de sus jefes y la
velocidad medida de cada mapa) y, dentro de una instancia, que jefe toca en cada episodio
(SelectorJefes, con pesos configurables y la opcion de dar mas peso a los jefes con menor tasa de
victoria).
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Muestras minimas de un jefe antes de fiarse de su tasa de victoria.
MUESTRAS_MINIMAS = 10
TASA_SIN_DATOS = 0.5


@dataclass(frozen=True)
class ConfigSeleccion:
    """
    Como se reparten los episodios entre jefes.

    Contexto:
        SelectorJefes y repartir_instancias.

    Atributos:
        pesos: peso relativo de cada jefe de episodio; los que no aparecen pesan 1.0.
        adaptativo: si es True el peso efectivo crece para los jefes con menor tasa de victoria.
        fuerza_adaptativa: con adaptativo, peso efectivo = peso * (1 + fuerza * (1 - tasa de victoria)).
        ventana: episodios recientes por jefe con los que se mide la tasa.
        prob_fase2: probabilidad de que un episodio de un jefe con segunda fase arranque en ella.
    """

    pesos: Mapping[str, float] = field(default_factory=dict)
    adaptativo: bool = False
    fuerza_adaptativa: float = 2.0
    ventana: int = 50
    prob_fase2: float = 0.2

    def __post_init__(self) -> None:
        """
        Valida los pesos y los parametros.

        Contexto:
            Construccion del dataclass.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si un peso es negativo, la fuerza es negativa, la ventana es menor que 1 o la
                probabilidad esta fuera de [0, 1].

        Ejemplo:
            ConfigSeleccion(pesos={"amlo": 2.0})
        """
        if any(p < 0 for p in self.pesos.values()):
            raise ValueError("los pesos no pueden ser negativos")
        if self.fuerza_adaptativa < 0 or self.ventana < 1 or not 0.0 <= self.prob_fase2 <= 1.0:
            raise ValueError("parametros de seleccion fuera de rango")

    def peso_de(self, jefe: str) -> float:
        """
        Peso base de un jefe.

        Contexto:
            SelectorJefes y repartir_instancias.

        Args:
            jefe (str): Jefe de episodio.

        Returns:
            float: Peso configurado, 1.0 si no se configuro.

        Raises:
            Ninguna.

        Ejemplo:
            ConfigSeleccion(pesos={"amlo": 2.0}).peso_de("amlo")  # 2.0
        """
        return float(self.pesos.get(jefe, 1.0))


class SelectorJefes:
    """
    Elige el jefe de cada episodio entre los permitidos de una instancia.

    Contexto:
        EntornoMultiJefe, en cada reset. Los resultados de los episodios (registrar) alimentan la
        tasa de victoria de la que depende el peso adaptativo.
    """

    def __init__(self, permitidos: Sequence[str], config: ConfigSeleccion | None = None,
                 azar: random.Random | None = None) -> None:
        """
        Crea el selector.

        Contexto:
            Construccion del entorno.

        Args:
            permitidos (Sequence[str]): Jefes que esta instancia puede enfrentar (los de su mapa).
            config (ConfigSeleccion, opcional): Pesos y modo adaptativo.
            azar (random.Random, opcional): Generador.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si no hay jefes permitidos o todos tienen peso cero.

        Ejemplo:
            selector = SelectorJefes(("amlo", "nahual"), ConfigSeleccion(adaptativo=True))
        """
        if not permitidos:
            raise ValueError("el selector necesita al menos un jefe")
        self.permitidos = tuple(permitidos)
        self.config = config or ConfigSeleccion()
        if sum(self.config.peso_de(j) for j in self.permitidos) <= 0:
            raise ValueError("todos los jefes permitidos tienen peso cero")
        self._azar = azar or random.Random()
        self._resultados: dict[str, deque[bool]] = {j: deque(maxlen=self.config.ventana) for j in self.permitidos}

    def registrar(self, jefe: str, victoria: bool) -> None:
        """
        Anota el resultado de un episodio.

        Contexto:
            EntornoMultiJefe, al terminar cada episodio (los que acaban en error no se anotan).

        Args:
            jefe (str): Jefe del episodio.
            victoria (bool): Si el agente gano.

        Returns:
            None: No retorna valor.

        Raises:
            KeyError: Si el jefe no esta entre los permitidos.

        Ejemplo:
            selector.registrar("amlo", False)
        """
        self._resultados[jefe].append(bool(victoria))

    def tasa_victoria(self, jefe: str) -> float:
        """
        Tasa de victoria reciente de un jefe.

        Contexto:
            Peso adaptativo y metricas.

        Args:
            jefe (str): Jefe de episodio.

        Returns:
            float: Fraccion de victorias de la ventana; TASA_SIN_DATOS si hay menos de
                MUESTRAS_MINIMAS episodios.

        Raises:
            KeyError: Si el jefe no esta entre los permitidos.

        Ejemplo:
            selector.tasa_victoria("amlo")
        """
        r = self._resultados[jefe]
        return sum(r) / len(r) if len(r) >= MUESTRAS_MINIMAS else TASA_SIN_DATOS

    def probabilidades(self) -> dict[str, float]:
        """
        Probabilidad de elegir cada jefe en el siguiente episodio.

        Contexto:
            elegir y diagnostico.

        Args:
            Ninguno.

        Returns:
            dict[str, float]: Jefe -> probabilidad (suman 1).

        Raises:
            Ninguna.

        Ejemplo:
            selector.probabilidades()
        """
        efectivos = {}
        for j in self.permitidos:
            peso = self.config.peso_de(j)
            if self.config.adaptativo:
                peso *= 1.0 + self.config.fuerza_adaptativa * (1.0 - self.tasa_victoria(j))
            efectivos[j] = peso
        total = sum(efectivos.values())
        return {j: p / total for j, p in efectivos.items()}

    def elegir(self) -> str:
        """
        Elige el jefe del siguiente episodio.

        Contexto:
            EntornoMultiJefe.reset.

        Args:
            Ninguno.

        Returns:
            str: Jefe de episodio.

        Raises:
            Ninguna.

        Ejemplo:
            jefe = selector.elegir()
        """
        probabilidades = self.probabilidades()
        return self._azar.choices(list(probabilidades), weights=list(probabilidades.values()), k=1)[0]


def repartir_instancias(pesos_por_mapa: Mapping[str, float], pasos_por_segundo: Mapping[str, float],
                        total: int) -> dict[str, int]:
    """
    Reparte las instancias del juego entre mapas.

    Contexto:
        Planificacion del entrenamiento y la recoleccion. Una instancia en un mapa aporta
        pasos_por_segundo[mapa] pasos; para que la fraccion de pasos de cada mapa siga el peso de sus
        jefes, las instancias del mapa son proporcionales a peso / pasos_por_segundo. Todo mapa con
        peso positivo recibe al menos una instancia.

    Args:
        pesos_por_mapa (Mapping[str, float]): Suma de los pesos de los jefes de cada mapa.
        pasos_por_segundo (Mapping[str, float]): Velocidad medida por instancia en cada mapa.
        total (int): Instancias disponibles.

    Returns:
        dict[str, int]: Instancias por mapa (suman total).

    Raises:
        ValueError: Si total es menor que la cantidad de mapas con peso, o si falta la velocidad de
            un mapa o no es positiva.

    Ejemplo:
        repartir_instancias({"LVL_Boss": 1.0, "LVL_AMLO": 3.0}, {"LVL_Boss": 300, "LVL_AMLO": 200}, 4)
    """
    activos = {m: w for m, w in pesos_por_mapa.items() if w > 0}
    if total < len(activos):
        raise ValueError("hay menos instancias que mapas con peso")
    for mapa in activos:
        if pasos_por_segundo.get(mapa, 0.0) <= 0:
            raise ValueError(f"falta la velocidad (pasos/s) del mapa {mapa}")
    ideal = {m: w / pasos_por_segundo[m] for m, w in activos.items()}
    escala = total / sum(ideal.values())
    cuota = {m: ideal[m] * escala for m in ideal}
    reparto = {m: max(1, int(c)) for m, c in cuota.items()}
    while sum(reparto.values()) > total:
        sobrante = max((m for m in reparto if reparto[m] > 1), key=lambda m: reparto[m] - cuota[m])
        reparto[sobrante] -= 1
    while sum(reparto.values()) < total:
        falta = max(reparto, key=lambda m: cuota[m] - reparto[m])
        reparto[falta] += 1
    return reparto
