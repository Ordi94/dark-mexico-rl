"""
Curriculo de dificultad por multiplicador de la vida del jefe.

Empieza con un jefe de poca vida (el agente aprende a golpear y a sobrevivir) y sube por
niveles cuando la tasa de victoria medida alcanza el umbral. Nunca modifica la vida del
jugador. Las victorias que cuentan son las de episodios que arrancan en la fase 1: los que
arrancan en la fase 2 son un complemento de entrenamiento y distorsionarian la tasa.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from dmrl.dominio.configuracion import ConfigCurriculo


class Curriculo:
    """
    Estado del curriculo compartido por todos los entornos del entrenamiento.

    Contexto:
        CallbackCurriculo lo alimenta con el resultado de cada episodio y publica el nivel
        vigente a los entornos; se guarda junto a los checkpoints para reanudar.
    """

    def __init__(self, config: ConfigCurriculo | None = None, nivel: int = 0) -> None:
        """
        Crea el curriculo en un nivel dado.

        Contexto:
            Inicio de un entrenamiento o reanudacion desde un checkpoint.

        Args:
            config (ConfigCurriculo, opcional): Parametros; por defecto los de fabrica.
            nivel (int): Indice de nivel inicial.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si no hay niveles, el ultimo no es 1.0, los niveles no crecen o el
                indice esta fuera de rango.

        Ejemplo:
            curriculo = Curriculo()
        """
        self.config = config or ConfigCurriculo()
        niveles = self.config.niveles
        if not niveles or niveles[-1] != 1.0:
            raise ValueError("el ultimo nivel del curriculo debe ser 1.0")
        if niveles[0] <= 0 or any(b <= a for a, b in zip(niveles, niveles[1:], strict=False)):
            raise ValueError("los niveles deben ser positivos y crecientes")
        if not 0 <= nivel < len(niveles):
            raise ValueError(f"nivel fuera de rango: {nivel}")
        self.nivel = nivel
        self._resultados: deque[bool] = deque(maxlen=self.config.ventana)
        self.episodios_en_nivel = 0

    @property
    def multiplicador(self) -> float:
        """
        Multiplicador de vida del jefe del nivel vigente.

        Contexto:
            Lo aplican los entornos en cada reset.

        Args:
            Ninguno.

        Returns:
            float: Valor en (0, 1].

        Raises:
            Ninguna.

        Ejemplo:
            Curriculo().multiplicador  # 0.3
        """
        return self.config.niveles[self.nivel]

    @property
    def completo(self) -> bool:
        """
        Indica si ya se esta en el ultimo nivel (jefe al 100%).

        Contexto:
            Metricas y decision del mejor modelo.

        Args:
            Ninguno.

        Returns:
            bool: True en el ultimo nivel.

        Raises:
            Ninguna.

        Ejemplo:
            Curriculo().completo  # False
        """
        return self.nivel == len(self.config.niveles) - 1

    @property
    def tasa_victoria(self) -> float:
        """
        Tasa de victoria de la ventana actual (episodios de fase 1 del nivel vigente).

        Contexto:
            Decision de subida y metricas de TensorBoard.

        Args:
            Ninguno.

        Returns:
            float: Fraccion en [0, 1]; 0.0 si la ventana esta vacia.

        Raises:
            Ninguna.

        Ejemplo:
            Curriculo().tasa_victoria  # 0.0
        """
        if not self._resultados:
            return 0.0
        return sum(self._resultados) / len(self._resultados)

    def registrar_episodio(self, victoria: bool, fase_inicio: int) -> bool:
        """
        Registra un episodio terminado y sube de nivel si corresponde.

        Contexto:
            CallbackCurriculo, una vez por episodio terminado de cualquier entorno.

        Args:
            victoria (bool): Si el agente mato al jefe.
            fase_inicio (int): Fase en la que arranco el episodio (1 o 2).

        Returns:
            bool: True si este episodio provoco una subida de nivel.

        Raises:
            Ninguna.

        Ejemplo:
            subio = curriculo.registrar_episodio(True, 1)
        """
        if fase_inicio != 1:
            return False
        self._resultados.append(bool(victoria))
        self.episodios_en_nivel += 1
        if self.completo or self.episodios_en_nivel < self.config.minimo_episodios:
            return False
        if len(self._resultados) < self.config.ventana:
            return False
        if self.tasa_victoria < self.config.tasa_victoria_subida:
            return False
        self.nivel += 1
        self._resultados.clear()
        self.episodios_en_nivel = 0
        return True

    def a_dict(self) -> dict[str, Any]:
        """
        Estado serializable para guardarlo junto al checkpoint.

        Contexto:
            Reanudacion del entrenamiento.

        Args:
            Ninguno.

        Returns:
            dict[str, Any]: Nivel, episodios del nivel y resultados de la ventana.

        Raises:
            Ninguna.

        Ejemplo:
            Curriculo().a_dict()
        """
        return {"nivel": self.nivel, "episodios_en_nivel": self.episodios_en_nivel,
                "resultados": [int(r) for r in self._resultados]}

    def cargar(self, estado: dict[str, Any]) -> None:
        """
        Restaura el estado guardado con a_dict.

        Contexto:
            Reanudacion del entrenamiento.

        Args:
            estado (dict[str, Any]): Resultado previo de a_dict.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si el nivel guardado no existe en la configuracion actual.

        Ejemplo:
            curriculo.cargar({"nivel": 2, "episodios_en_nivel": 10, "resultados": [1, 0]})
        """
        nivel = int(estado["nivel"])
        if not 0 <= nivel < len(self.config.niveles):
            raise ValueError(f"nivel guardado fuera de rango: {nivel}")
        self.nivel = nivel
        self.episodios_en_nivel = int(estado.get("episodios_en_nivel", 0))
        self._resultados = deque((bool(r) for r in estado.get("resultados", [])), maxlen=self.config.ventana)


# Vida que hay que quitar para ganar una pelea completa (medido): la fase 1 se transforma al
# bajar de 15% de 2000 (se descartan 300) y la fase 2 tiene 1000.
VIDA_FASE1_UTIL = 1700.0
VIDA_FASE2 = 1000.0


def vida_efectiva_jefe(multiplicador: float, fase_inicio: int) -> float:
    """
    Vida total que el agente debe quitar para ganar un episodio.

    Contexto:
        Metrica de progreso contra el jefe, comparable entre niveles del curriculo.

    Args:
        multiplicador (float): Multiplicador de vida del episodio.
        fase_inicio (int): Fase en la que arranco el episodio (1 o 2).

    Returns:
        float: Puntos de vida a quitar.

    Raises:
        Ninguna.

    Ejemplo:
        vida_efectiva_jefe(0.3, 1)  # 810.0
    """
    base = VIDA_FASE2 + (VIDA_FASE1_UTIL if fase_inicio == 1 else 0.0)
    return base * multiplicador
