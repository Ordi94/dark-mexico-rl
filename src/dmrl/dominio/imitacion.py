"""
Reglas puras de la clonacion de comportamiento: demostraciones, planes de recoleccion, ruido de
ejecucion, particion de entrenamiento y validacion, pesos de clase y metricas.

Todo aqui es calculo sobre arreglos y numeros: no lee archivos, no abre el juego y no usa torch.
La imitacion de un experto con estado oculto (el bot de reglas recuerda que golpes ya atendio)
tiene dos riesgos que estas reglas atacan de frente:

* Desfase de distribucion: la red visita estados que el experto nunca visita. Se mitiga con
  episodios que ejecutan acciones con ruido pero se etiquetan siempre con la accion del experto
  (ruido_dart) y con rondas que ejecutan a la propia red (DAgger).
* Fuga entre entrenamiento y validacion: estados consecutivos de un mismo episodio son casi
  identicos, asi que repartir pasos sueltos inflaria la validacion. La particion es por episodio.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmrl.dominio import acciones as acc

# Los identificadores de episodio codifican la ronda para que sean unicos al juntar rondas.
EPISODIOS_POR_RONDA_MAX = 100_000
Z_NORMAL_95 = 1.959964


@dataclass(frozen=True)
class ConfigAprendizaje:
    """
    Hiperparametros del ajuste supervisado.

    Contexto:
        entrenar_politica y entrenar_valor (aplicacion.imitacion_red).

    Atributos:
        epocas_maximas: tope de pasadas por el conjunto de entrenamiento.
        paciencia: epocas sin mejora de la perdida de validacion antes de parar.
        tamano_lote: ejemplos por paso de optimizacion.
        tasa_aprendizaje: tasa inicial de AdamW (baja a la mitad si la validacion se estanca).
        decaimiento_pesos: weight decay desacoplado de AdamW; es la regularizacion principal porque
            la politica de SB3 no admite dropout sin romper la compatibilidad.
        potencia_pesos: exponente de la frecuencia inversa para balancear clases (0 = sin balance).
        tope_peso: la razon maxima entre el peso de una clase y el neutro.
        gamma: descuento de los retornos con que se ajusta el valor (el mismo de PPO).
        epocas_valor: tope de pasadas del ajuste de la cabeza de valor.
    """

    epocas_maximas: int = 60
    paciencia: int = 6
    tamano_lote: int = 1024
    tasa_aprendizaje: float = 5e-4
    decaimiento_pesos: float = 1e-2
    potencia_pesos: float = 0.5
    tope_peso: float = 3.0
    gamma: float = 0.995
    epocas_valor: int = 30


@dataclass(frozen=True)
class ConfigImitacion:
    """
    Parametros de todo el ciclo de imitacion.

    Contexto:
        CLI imitar y el ciclo de rondas (aplicacion.imitacion_ciclo).

    Atributos:
        nombre: carpeta de la corrida dentro de runs/.
        instancias: instancias del juego en paralelo.
        episodios_ronda0: peleas del experto en la ronda inicial.
        episodios_dagger: peleas por ronda de DAgger.
        rondas_dagger: rondas en que la red juega y el experto etiqueta.
        prob_fase2: fraccion de episodios que empieza en la fase 2 (la del curriculo).
        fraccion_ruido: fraccion de episodios que se ejecutan con ruido.
        epsilon_min: ruido minimo de un episodio ruidoso.
        epsilon_max: ruido maximo de un episodio ruidoso.
        fraccion_validacion: fraccion de episodios reservada para validar.
        episodios_evaluacion: peleas de la evaluacion principal (jefe al 100%, fase 1).
        episodios_fase2: peleas de la evaluacion con inicio en fase 2.
        episodios_ruido: peleas de la evaluacion con ruido de ejecucion.
        epsilon_evaluacion: ruido de esa evaluacion de robustez.
        episodios_confirmacion: peleas nuevas con que se confirma a cada candidata y al bot.
        candidatas_confirmacion: cuantas rondas (las mejores de la evaluacion principal) se confirman.
        semilla: semilla base de planes, particiones e inicializacion.
        aprendizaje: hiperparametros del ajuste supervisado.
    """

    nombre: str = "imitacion"
    instancias: int = 4
    episodios_ronda0: int = 600
    episodios_dagger: int = 300
    rondas_dagger: int = 3
    prob_fase2: float = 0.2
    fraccion_ruido: float = 0.5
    epsilon_min: float = 0.05
    epsilon_max: float = 0.20
    fraccion_validacion: float = 0.15
    episodios_evaluacion: int = 50
    episodios_fase2: int = 20
    episodios_ruido: int = 20
    epsilon_evaluacion: float = 0.10
    episodios_confirmacion: int = 100
    candidatas_confirmacion: int = 2
    semilla: int = 0
    aprendizaje: ConfigAprendizaje = field(default_factory=ConfigAprendizaje)


@dataclass(frozen=True)
class PlanEpisodio:
    """
    Condiciones de un episodio de recoleccion.

    Contexto:
        planificar_episodios; el trabajador de recoleccion lo recibe ya decidido para que la
        recoleccion sea reproducible y reanudable.

    Atributos:
        indice: posicion del episodio en el plan de su ronda (identificador estable).
        fase_inicio: 1 o 2.
        epsilon: probabilidad por paso de ejecutar una accion valida al azar (0 = sin ruido).
        semilla: semilla del azar del episodio.
    """

    indice: int
    fase_inicio: int
    epsilon: float
    semilla: int


@dataclass
class Demostraciones:
    """
    Conjunto de pares (estado, accion del experto) con su episodio.

    Contexto:
        Salida de la recoleccion y entrada del ajuste. Las filas de un episodio son contiguas y
        estan en orden de tiempo (el calculo de retornos lo necesita).

    Atributos:
        obs: observaciones apiladas, forma (n, dimension).
        mascaras: acciones validas en cada estado, forma (n, N_ACCIONES).
        etiquetas: accion que habria elegido el experto en ese estado.
        ejecutadas: accion que se ejecuto de verdad (distinta de la etiqueta en pasos ruidosos
            o cuando juega la red).
        episodios: identificador global del episodio de cada fila.
        recompensas: recompensa recibida tras ejecutar la accion.
    """

    obs: NDArray[np.float32]
    mascaras: NDArray[np.bool_]
    etiquetas: NDArray[np.int16]
    ejecutadas: NDArray[np.int16]
    episodios: NDArray[np.int64]
    recompensas: NDArray[np.float32]

    def __len__(self) -> int:
        """
        Cantidad de pasos.

        Contexto:
            Reportes y particion.

        Args:
            Ninguno.

        Returns:
            int: Filas del conjunto.

        Raises:
            Ninguna.

        Ejemplo:
            len(demostraciones)
        """
        return int(self.etiquetas.shape[0])

    def validar(self) -> None:
        """
        Comprueba la coherencia interna del conjunto.

        Contexto:
            Al guardar y al cargar: una etiqueta que la mascara no admite entrenaria a la red a
            elegir una accion que el entorno sustituye por "nada".

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si las formas no coinciden, hay valores no finitos o una etiqueta es invalida.

        Ejemplo:
            demostraciones.validar()
        """
        n = len(self)
        formas = (self.obs.shape[0], self.mascaras.shape[0], self.ejecutadas.shape[0],
                  self.episodios.shape[0], self.recompensas.shape[0])
        if any(f != n for f in formas):
            raise ValueError(f"longitudes distintas en las demostraciones: {formas} contra {n}")
        if self.mascaras.shape[1:] != (acc.N_ACCIONES,):
            raise ValueError(f"las mascaras deben tener {acc.N_ACCIONES} columnas: {self.mascaras.shape}")
        if n == 0:
            return
        if not bool(np.isfinite(self.obs).all()) or not bool(np.isfinite(self.recompensas).all()):
            raise ValueError("las demostraciones contienen valores no finitos")
        if int(self.etiquetas.min()) < 0 or int(self.etiquetas.max()) >= acc.N_ACCIONES:
            raise ValueError("hay etiquetas fuera de 0..N_ACCIONES-1")
        if not bool(self.mascaras[np.arange(n), self.etiquetas.astype(np.int64)].all()):
            raise ValueError("hay etiquetas que la mascara del mismo estado no admite")

    def seleccionar(self, filas: NDArray[np.bool_]) -> Demostraciones:
        """
        Subconjunto de filas.

        Contexto:
            Particion de entrenamiento y validacion.

        Args:
            filas (NDArray[np.bool_]): Mascara booleana de longitud n.

        Returns:
            Demostraciones: Copia con las filas elegidas, en el mismo orden.

        Raises:
            ValueError: Si la mascara no tiene longitud n.

        Ejemplo:
            entrenamiento = demostraciones.seleccionar(es_entrenamiento)
        """
        if filas.shape != (len(self),):
            raise ValueError("la mascara de filas debe tener una entrada por paso")
        return Demostraciones(self.obs[filas], self.mascaras[filas], self.etiquetas[filas],
                              self.ejecutadas[filas], self.episodios[filas], self.recompensas[filas])


def concatenar(partes: Sequence[Demostraciones]) -> Demostraciones:
    """
    Une varios conjuntos en uno.

    Contexto:
        Agregacion de DAgger: cada ronda suma sus episodios a todos los anteriores.

    Args:
        partes (Sequence[Demostraciones]): Conjuntos a unir (al menos uno).

    Returns:
        Demostraciones: Conjunto con las filas de todos, en el orden dado.

    Raises:
        ValueError: Si no hay partes.

    Ejemplo:
        todo = concatenar([ronda0, ronda1])
    """
    if not partes:
        raise ValueError("no hay demostraciones que unir")
    return Demostraciones(
        np.concatenate([p.obs for p in partes]), np.concatenate([p.mascaras for p in partes]),
        np.concatenate([p.etiquetas for p in partes]), np.concatenate([p.ejecutadas for p in partes]),
        np.concatenate([p.episodios for p in partes]), np.concatenate([p.recompensas for p in partes]),
    )


def identificador_episodio(ronda: int, indice: int) -> int:
    """
    Identificador global de un episodio.

    Contexto:
        Recoleccion y particion: ronda e indice del plan en un solo entero.

    Args:
        ronda (int): Numero de ronda (0 = experto, 1.. = DAgger).
        indice (int): Posicion del episodio en el plan de la ronda.

    Returns:
        int: ronda * EPISODIOS_POR_RONDA_MAX + indice.

    Raises:
        ValueError: Si el indice no cabe en la ronda o algun valor es negativo.

    Ejemplo:
        identificador_episodio(1, 7)  # 100007
    """
    if ronda < 0 or not 0 <= indice < EPISODIOS_POR_RONDA_MAX:
        raise ValueError(f"ronda/indice fuera de rango: {ronda}/{indice}")
    return ronda * EPISODIOS_POR_RONDA_MAX + indice


def ronda_de_episodio(episodio: int) -> int:
    """
    Ronda a la que pertenece un identificador de episodio.

    Contexto:
        Reportes por ronda.

    Args:
        episodio (int): Identificador global.

    Returns:
        int: Numero de ronda.

    Raises:
        Ninguna.

    Ejemplo:
        ronda_de_episodio(100007)  # 1
    """
    return int(episodio) // EPISODIOS_POR_RONDA_MAX


def planificar_episodios(cantidad: int, prob_fase2: float, fraccion_ruido: float, epsilon_min: float,
                         epsilon_max: float, semilla: int) -> list[PlanEpisodio]:
    """
    Decide de antemano la fase de inicio y el ruido de cada episodio de una ronda.

    Contexto:
        Inicio de cada ronda. Los conteos son exactos (no sorteados por episodio) para garantizar
        que ambas fases y ambos regimenes de ejecucion quedan cubiertos aunque la ronda sea corta.

    Args:
        cantidad (int): Episodios de la ronda.
        prob_fase2 (float): Fraccion que empieza en la fase 2, en [0, 1].
        fraccion_ruido (float): Fraccion que se ejecuta con ruido, en [0, 1].
        epsilon_min (float): Ruido minimo de un episodio ruidoso, en [0, 1].
        epsilon_max (float): Ruido maximo, mayor o igual que epsilon_min.
        semilla (int): Semilla del plan.

    Returns:
        list[PlanEpisodio]: Un plan por episodio, con indices 0..cantidad-1.

    Raises:
        ValueError: Si algun parametro esta fuera de rango.

    Ejemplo:
        planificar_episodios(10, 0.2, 0.5, 0.05, 0.2, 0)
    """
    if cantidad < 0 or cantidad > EPISODIOS_POR_RONDA_MAX:
        raise ValueError(f"cantidad fuera de rango: {cantidad}")
    for nombre, valor in (("prob_fase2", prob_fase2), ("fraccion_ruido", fraccion_ruido),
                          ("epsilon_min", epsilon_min), ("epsilon_max", epsilon_max)):
        if not 0.0 <= valor <= 1.0:
            raise ValueError(f"{nombre} fuera de [0, 1]: {valor}")
    if epsilon_max < epsilon_min:
        raise ValueError("epsilon_max debe ser >= epsilon_min")
    azar = random.Random(semilla)
    fases = [2] * round(cantidad * prob_fase2) + [1] * (cantidad - round(cantidad * prob_fase2))
    ruidosos = round(cantidad * fraccion_ruido)
    epsilons = [azar.uniform(epsilon_min, epsilon_max) for _ in range(ruidosos)] + [0.0] * (cantidad - ruidosos)
    azar.shuffle(fases)
    azar.shuffle(epsilons)
    return [PlanEpisodio(i, fases[i], epsilons[i], azar.randrange(2**31)) for i in range(cantidad)]


def ruido_dart(accion_base: int, mascara: Sequence[bool], epsilon: float, azar: random.Random) -> int:
    """
    Accion que se ejecuta de verdad en un paso con ruido de ejecucion (enfoque DART).

    Contexto:
        Recoleccion. La etiqueta guardada es siempre la del experto; lo que cambia es el estado
        siguiente, que se desvia de la trayectoria del experto y obliga a registrar que habria
        hecho el experto tambien en esos estados.

    Args:
        accion_base (int): Accion que ejecutaria el ejecutor sin ruido.
        mascara (Sequence[bool]): Acciones validas en este estado.
        epsilon (float): Probabilidad de sustituirla por una valida al azar, en [0, 1].
        azar (random.Random): Generador del episodio.

    Returns:
        int: accion_base o, con probabilidad epsilon, una accion valida al azar.

    Raises:
        ValueError: Si epsilon esta fuera de [0, 1] o la mascara no deja ninguna accion.

    Ejemplo:
        ruido_dart(17, mascara, 0.1, random.Random(0))
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon fuera de [0, 1]: {epsilon}")
    validas = [i for i, ok in enumerate(mascara) if ok]
    if not validas:
        raise ValueError("la mascara no admite ninguna accion")
    if epsilon > 0.0 and azar.random() < epsilon:
        return validas[azar.randrange(len(validas))]
    return accion_base


def particionar_episodios(episodios: NDArray[np.int64], fraccion_validacion: float,
                          semilla: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Reparte los pasos entre entrenamiento y validacion por episodios completos.

    Contexto:
        Antes de cada ajuste. Los estados de un mismo episodio son casi identicos entre si:
        repartir pasos sueltos pondria en validacion estados casi copia de los de entrenamiento
        y ocultaria el sobreajuste. El sorteo de cada episodio depende solo de su identificador y
        de la semilla, no del resto del conjunto: al sumar rondas de DAgger un episodio nunca
        cambia de lado, asi que las curvas de rondas distintas son comparables y ninguna ronda
        entrena con lo que una anterior uso para validar.

    Args:
        episodios (NDArray[np.int64]): Identificador de episodio de cada paso.
        fraccion_validacion (float): Fraccion esperada de episodios para validar, en (0, 1).
        semilla (int): Semilla del sorteo.

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: Mascaras de entrenamiento y de validacion;
            son complementarias y ningun episodio aparece en ambas. Ambas tienen al menos un
            episodio.

    Raises:
        ValueError: Si la fraccion esta fuera de (0, 1) o hay menos de dos episodios.

    Ejemplo:
        entrenamiento, validacion = particionar_episodios(datos.episodios, 0.15, 0)
    """
    if not 0.0 < fraccion_validacion < 1.0:
        raise ValueError(f"fraccion_validacion fuera de (0, 1): {fraccion_validacion}")
    unicos = np.unique(episodios)
    if unicos.size < 2:
        raise ValueError("se necesitan al menos dos episodios para particionar")
    sorteo = np.asarray([np.random.default_rng((semilla, int(e))).random() for e in unicos])
    es_validacion = sorteo < fraccion_validacion
    if not es_validacion.any():
        es_validacion[int(sorteo.argmin())] = True
    if es_validacion.all():
        es_validacion[int(sorteo.argmax())] = False
    validacion = np.isin(episodios, unicos[es_validacion])
    return ~validacion, validacion


def pesos_de_clase(etiquetas: NDArray[np.int16], n_clases: int, potencia: float,
                   tope: float) -> NDArray[np.float32]:
    """
    Pesos de la perdida que compensan las acciones raras (parry, caguama, especial).

    Contexto:
        Ajuste supervisado. La accion "nada" y el movimiento dominan el conjunto; sin balance la
        red tiende a no emitir las acciones raras, que son las que deciden la pelea.

    Args:
        etiquetas (NDArray[np.int16]): Etiqueta de cada paso.
        n_clases (int): Numero de acciones.
        potencia (float): Exponente de la frecuencia inversa; 0 da pesos uniformes y 1 equilibra
            las clases por completo.
        tope (float): Cota (>= 1) de la razon entre un peso y el de una clase tipica.

    Returns:
        NDArray[np.float32]: Peso por clase; 1.0 para las que no aparecen.

    Raises:
        ValueError: Si potencia es negativa o tope es menor que 1.

    Ejemplo:
        pesos = pesos_de_clase(datos.etiquetas, 22, 0.5, 3.0)
    """
    if potencia < 0.0 or tope < 1.0:
        raise ValueError("potencia debe ser >= 0 y tope >= 1")
    cuentas = np.bincount(etiquetas.astype(np.int64), minlength=n_clases).astype(np.float64)
    pesos = np.ones(n_clases, dtype=np.float64)
    presentes = cuentas > 0
    if potencia > 0.0 and presentes.any():
        tipica = float(np.median(cuentas[presentes]))
        pesos[presentes] = np.clip((tipica / cuentas[presentes]) ** potencia, 1.0 / tope, tope)
    return pesos.astype(np.float32)


@dataclass(frozen=True)
class MetricaAccion:
    """
    Calidad de la imitacion de una accion.

    Contexto:
        metricas_por_accion; se reporta por accion porque la exactitud global la dominan las
        acciones frecuentes.

    Atributos:
        accion: indice de accion.
        soporte: cuantas veces el experto la eligio.
        recall: fraccion de esas veces que la red tambien la elige.
        precision: fraccion de las veces que la red la elige que coinciden con el experto.
    """

    accion: int
    soporte: int
    recall: float
    precision: float


def metricas_por_accion(prediccion: NDArray[np.int64], etiqueta: NDArray[np.int64],
                        n_clases: int) -> list[MetricaAccion]:
    """
    Recall y precision de cada accion.

    Contexto:
        Informe de validacion de cada ronda.

    Args:
        prediccion (NDArray[np.int64]): Accion elegida por la red en cada paso.
        etiqueta (NDArray[np.int64]): Accion del experto en cada paso.
        n_clases (int): Numero de acciones.

    Returns:
        list[MetricaAccion]: Una entrada por accion; recall/precision 0 si no hay base para medirla.

    Raises:
        ValueError: Si las longitudes difieren.

    Ejemplo:
        metricas_por_accion(pred, etiquetas, 22)[19].recall
    """
    if prediccion.shape != etiqueta.shape:
        raise ValueError("prediccion y etiqueta deben tener la misma forma")
    resultado: list[MetricaAccion] = []
    for accion in range(n_clases):
        real = etiqueta == accion
        predicha = prediccion == accion
        aciertos = int(np.count_nonzero(real & predicha))
        soporte = int(np.count_nonzero(real))
        emitidas = int(np.count_nonzero(predicha))
        resultado.append(MetricaAccion(
            accion, soporte, aciertos / soporte if soporte else 0.0, aciertos / emitidas if emitidas else 0.0,
        ))
    return resultado


def retornos_descontados(recompensas: NDArray[np.float32], episodios: NDArray[np.int64],
                         gamma: float) -> NDArray[np.float32]:
    """
    Retorno descontado de cada paso dentro de su episodio.

    Contexto:
        Ajuste de la cabeza de valor, para que PPO parta de un valor razonable y no de uno al
        azar que estropearia la politica clonada en las primeras actualizaciones. Un episodio
        truncado por tiempo se trata como terminado (aproximacion aceptada: son pocos).

    Args:
        recompensas (NDArray[np.float32]): Recompensa de cada paso.
        episodios (NDArray[np.int64]): Episodio de cada paso (filas contiguas por episodio).
        gamma (float): Descuento en [0, 1].

    Returns:
        NDArray[np.float32]: Retorno de cada paso.

    Raises:
        ValueError: Si las longitudes difieren o gamma esta fuera de [0, 1].

    Ejemplo:
        retornos_descontados(np.array([1, 1], np.float32), np.array([0, 0]), 0.5)  # [1.5, 1.0]
    """
    if recompensas.shape != episodios.shape:
        raise ValueError("recompensas y episodios deben tener la misma forma")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma fuera de [0, 1]: {gamma}")
    retornos = np.zeros(recompensas.shape[0], dtype=np.float64)
    acumulado = 0.0
    for i in range(recompensas.shape[0] - 1, -1, -1):
        if i == recompensas.shape[0] - 1 or episodios[i] != episodios[i + 1]:
            acumulado = 0.0
        acumulado = float(recompensas[i]) + gamma * acumulado
        retornos[i] = acumulado
    return retornos.astype(np.float32)


def brecha_sobreajuste(perdida_entrenamiento: float, perdida_validacion: float) -> float:
    """
    Diferencia relativa entre la perdida de validacion y la de entrenamiento.

    Contexto:
        Criterio de sobreajuste del reporte: valores cercanos a cero indican que la red se
        comporta igual en episodios que nunca vio.

    Args:
        perdida_entrenamiento (float): Perdida medida sobre entrenamiento (> 0).
        perdida_validacion (float): Perdida medida sobre validacion.

    Returns:
        float: (validacion - entrenamiento) / entrenamiento.

    Raises:
        ValueError: Si la perdida de entrenamiento no es positiva.

    Ejemplo:
        brecha_sobreajuste(0.20, 0.25)  # 0.25
    """
    if perdida_entrenamiento <= 0.0:
        raise ValueError("la perdida de entrenamiento debe ser positiva")
    return (perdida_validacion - perdida_entrenamiento) / perdida_entrenamiento


def intervalo_wilson(exitos: int, total: int) -> tuple[float, float]:
    """
    Intervalo de confianza del 95% de una proporcion (Wilson).

    Contexto:
        Con 50 peleas la tasa de victoria tiene +-12 puntos de incertidumbre: comparar dos
        modelos sin el intervalo seria comparar ruido.

    Args:
        exitos (int): Victorias.
        total (int): Peleas.

    Returns:
        tuple[float, float]: Limites inferior y superior; (0, 1) si no hay peleas.

    Raises:
        ValueError: Si exitos no esta entre 0 y total.

    Ejemplo:
        intervalo_wilson(36, 50)  # (0.58, 0.83) aproximadamente
    """
    if not 0 <= exitos <= total:
        raise ValueError("exitos debe estar entre 0 y total")
    if total == 0:
        return 0.0, 1.0
    p = exitos / total
    z2 = Z_NORMAL_95**2
    centro = (p + z2 / (2 * total)) / (1 + z2 / total)
    radio = Z_NORMAL_95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return max(0.0, centro - radio), min(1.0, centro + radio)


def resumen_de_juego(episodios: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """
    Metricas de una evaluacion en juego pensadas para comparar modelos.

    Contexto:
        Seleccion del modelo y tablas del reporte. Complementa evaluacion.resumir con la causa de
        las derrotas y el intervalo de confianza de la tasa de victoria.

    Args:
        episodios (Sequence[Mapping[str, Any]]): Resumenes de episodios del entorno.

    Returns:
        dict[str, float]: peleas, victorias, tasa_victoria con su intervalo, dano, parries,
            caguamas interrumpidas y muertes por fase; vacio si no hay episodios.

    Raises:
        Ninguna.

    Ejemplo:
        resumen_de_juego([{"victoria": True, "danio_hecho": 2700.0}])["tasa_victoria"]
    """
    validos = [e for e in episodios if e.get("resultado") != "error"]
    n = len(validos)
    if n == 0:
        return {}
    victorias = sum(1 for e in validos if e.get("victoria"))
    bajo, alto = intervalo_wilson(victorias, n)

    def media(clave: str) -> float:
        """
        Media de una clave numerica entre los episodios validos.

        Contexto:
            resumen_de_juego.

        Args:
            clave (str): Campo del resumen.

        Returns:
            float: Media; 0.0 si ninguno lo trae.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        valores = [float(e[clave]) for e in validos if clave in e]
        return float(np.mean(valores)) if valores else 0.0

    return {
        "peleas": float(n), "errores": float(len(episodios) - n), "victorias": float(victorias),
        "tasa_victoria": victorias / n, "tasa_victoria_inferior": bajo, "tasa_victoria_superior": alto,
        "tasa_fase2": sum(1 for e in validos if e.get("fase_max", 1) >= 2) / n,
        "danio_hecho": media("danio_hecho"), "danio_recibido": media("danio_recibido"),
        "progreso_jefe": media("progreso_jefe"), "parries": media("parries"),
        "caguamas_usadas": media("caguamas_usadas"), "caguamas_interrumpidas": media("caguamas_interrumpidas"),
        "duracion_s": media("duracion_s"),
        "muertes_fase1": float(sum(1 for e in validos if e.get("muerte") and e.get("fase_max", 1) < 2)),
        "muertes_fase2": float(sum(1 for e in validos if e.get("muerte") and e.get("fase_max", 1) >= 2)),
        "tiempos_agotados": float(sum(1 for e in validos if e.get("resultado") == "tiempo")),
    }


def clave_de_seleccion(resumen: Mapping[str, float]) -> tuple[float, float, float]:
    """
    Orden para elegir el mejor modelo por desempeno en juego.

    Contexto:
        elegir_mejor. Primero la tasa de victoria, luego el progreso contra el jefe y por ultimo
        el dano recibido (menor es mejor).

    Args:
        resumen (Mapping[str, float]): Salida de resumen_de_juego.

    Returns:
        tuple[float, float, float]: Clave comparable; mayor es mejor.

    Raises:
        Ninguna.

    Ejemplo:
        clave_de_seleccion({"tasa_victoria": 0.5})
    """
    return (resumen.get("tasa_victoria", 0.0), resumen.get("progreso_jefe", 0.0),
            -resumen.get("danio_recibido", 0.0))


def elegir_mejor(resumenes: Mapping[int, Mapping[str, float]]) -> int:
    """
    Ronda cuyo modelo rinde mejor en juego.

    Contexto:
        Fin del ciclo: el criterio de seleccion es el desempeno en juego, no la perdida de
        validacion.

    Args:
        resumenes (Mapping[int, Mapping[str, float]]): Resumen de juego de cada ronda.

    Returns:
        int: Numero de ronda ganadora; en empate gana la mas reciente (mas datos).

    Raises:
        ValueError: Si no hay ninguna ronda evaluada.

    Ejemplo:
        elegir_mejor({0: {"tasa_victoria": 0.4}, 1: {"tasa_victoria": 0.6}})  # 1
    """
    if not resumenes:
        raise ValueError("no hay rondas evaluadas")
    return max(resumenes, key=lambda ronda: (clave_de_seleccion(resumenes[ronda]), ronda))
