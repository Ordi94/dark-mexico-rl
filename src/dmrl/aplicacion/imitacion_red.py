"""
Ajuste supervisado de la politica MaskablePPO con las demostraciones del bot.

Se entrena el MISMO objeto que luego usara PPO: la cabeza de politica (mlp_extractor.policy_net mas
action_net) de un MaskablePPO creado con los hiperparametros y la arquitectura de la corrida
principal. Asi el archivo resultante se carga con MaskablePPO.load y se puede continuar con PPO sin
traducir pesos. La politica de SB3 no tiene capas de dropout, y anadirlas rompe esa compatibilidad,
asi que la regularizacion es weight decay (AdamW), parada temprana por perdida de validacion y
balance moderado de clases.

La perdida es entropia cruzada sobre logits enmascarados: una accion que la mascara del estado
prohibe recibe probabilidad cero, no aporta a la perdida ni recibe gradiente. Es lo mismo que hace
MaskablePPO al muestrear, de modo que lo entrenado coincide con lo que se ejecuta.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import functional

from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento, crear_modelo
from dmrl.dominio import acciones as acc
from dmrl.dominio.imitacion import ConfigAprendizaje, Demostraciones, MetricaAccion, metricas_por_accion, pesos_de_clase
from dmrl.dominio.observacion import TAMANO_SIMPLE

# Valor al que se lleva el logit de una accion prohibida (el mismo orden de magnitud que SB3).
LOGIT_INVALIDO = -1.0e8
# Tamano maximo (bytes) del conjunto que se deja residente en el dispositivo de calculo.
BYTES_RESIDENTES_MAX = 6 * 1024**3
MUESTRA_ENTRENAMIENTO = 100_000
MEJORA_MINIMA = 1e-4


class EntornoEspacios(gym.Env[NDArray[np.float32], int]):
    """
    Entorno sin juego con los mismos espacios que DarkMexicoEnv.

    Contexto:
        Permite construir un MaskablePPO con la forma exacta de la politica de la corrida
        principal sin lanzar el juego. Nunca se ejecuta una pelea con el.
    """

    def __init__(self, apilado: int = 4, tamano_simple: int = TAMANO_SIMPLE) -> None:
        """
        Crea el entorno vacio.

        Contexto:
            crear_modelo_base.

        Args:
            apilado (int): Observaciones apiladas; debe coincidir con ConfigEntorno.apilado.
            tamano_simple (int): Valores de una observacion sin apilar (la v1 por defecto, la v2 en
                el entorno multi-jefe).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoEspacios(4)
        """
        super().__init__()
        self._forma = (tamano_simple * apilado,)
        self.observation_space = spaces.Box(-1.0, 1.0, self._forma, np.float32)
        self.action_space: spaces.Discrete = spaces.Discrete(acc.N_ACCIONES)  # type: ignore[assignment]

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Observacion nula.

        Contexto:
            Requerido por la API de Gymnasium; no se usa para aprender.

        Args:
            seed (int, opcional): Ignorada.
            options (dict, opcional): Ignoradas.

        Returns:
            tuple[NDArray[np.float32], dict[str, Any]]: Ceros e info vacia.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoEspacios().reset()
        """
        super().reset(seed=seed)
        return np.zeros(self._forma, dtype=np.float32), {}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Paso nulo que termina el episodio.

        Contexto:
            Requerido por la API de Gymnasium; no se usa para aprender.

        Args:
            action (int): Ignorada.

        Returns:
            tuple: Ceros, recompensa 0, terminado.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoEspacios().step(0)
        """
        return np.zeros(self._forma, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self) -> NDArray[np.bool_]:
        """
        Mascara con todas las acciones validas.

        Contexto:
            Requerida por MaskablePPO al construirse.

        Args:
            Ninguno.

        Returns:
            NDArray[np.bool_]: Unos.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoEspacios().action_masks()
        """
        return np.ones(acc.N_ACCIONES, dtype=np.bool_)


def crear_modelo_base(config: ConfigEntrenamiento, tamano_simple: int = TAMANO_SIMPLE) -> Any:
    """
    MaskablePPO nuevo con la arquitectura e hiperparametros de la corrida principal.

    Contexto:
        Inicio de cada ajuste. Reutiliza crear_modelo del entrenamiento para que cualquier cambio
        de arquitectura alli se refleje aqui.

    Args:
        config (ConfigEntrenamiento): Configuracion de la corrida a la que se alimentara.
        tamano_simple (int): Valores de una observacion sin apilar; la v2 del entorno multi-jefe usa
            tamano_simple_v2(catalogo).

    Returns:
        MaskablePPO: Modelo en CPU con pesos iniciales al azar.

    Raises:
        ValueError: Si los hiperparametros no son validos.

    Ejemplo:
        modelo = crear_modelo_base(ConfigEntrenamiento(nombre="imitacion"))
    """
    entorno = DummyVecEnv([lambda: EntornoEspacios(config.entorno.apilado, tamano_simple)])
    return crear_modelo(config, entorno, "cpu")


def logits_actor(politica: Any, obs: torch.Tensor) -> torch.Tensor:
    """
    Logits de la cabeza de politica (sin mascara).

    Contexto:
        Perdida y evaluacion. El extractor de caracteristicas de MlpPolicy es un aplanado, por lo
        que la observacion entra directa a la red de la politica.

    Args:
        politica (Any): MaskableActorCriticPolicy.
        obs (torch.Tensor): Lote de observaciones apiladas, forma (n, dimension).

    Returns:
        torch.Tensor: Logits de forma (n, N_ACCIONES).

    Raises:
        Ninguna.

    Ejemplo:
        logits = logits_actor(modelo.policy, obs)
    """
    resultado: torch.Tensor = politica.action_net(politica.mlp_extractor.forward_actor(obs))
    return resultado


def valor_critico(politica: Any, obs: torch.Tensor) -> torch.Tensor:
    """
    Valor estimado de cada observacion.

    Contexto:
        Ajuste de la cabeza de valor.

    Args:
        politica (Any): MaskableActorCriticPolicy.
        obs (torch.Tensor): Lote de observaciones apiladas.

    Returns:
        torch.Tensor: Valores de forma (n,).

    Raises:
        Ninguna.

    Ejemplo:
        valores = valor_critico(modelo.policy, obs)
    """
    resultado: torch.Tensor = politica.value_net(politica.mlp_extractor.forward_critic(obs)).squeeze(-1)
    return resultado


def perdida_enmascarada(logits: torch.Tensor, etiquetas: torch.Tensor, mascaras: torch.Tensor,
                        pesos: torch.Tensor | None = None) -> torch.Tensor:
    """
    Entropia cruzada donde las acciones prohibidas no cuentan.

    Contexto:
        Ajuste de la politica. Las acciones prohibidas pierden toda su probabilidad, de modo que
        los logits que la red asigne a acciones que el juego rechazaria no afectan la perdida ni
        reciben gradiente.

    Args:
        logits (torch.Tensor): Logits (n, N_ACCIONES).
        etiquetas (torch.Tensor): Accion del experto (n,), valida segun su mascara.
        mascaras (torch.Tensor): Acciones validas (n, N_ACCIONES), booleana.
        pesos (torch.Tensor, opcional): Peso por clase (N_ACCIONES,).

    Returns:
        torch.Tensor: Perdida escalar (media ponderada).

    Raises:
        Ninguna.

    Ejemplo:
        perdida = perdida_enmascarada(logits, etiquetas, mascaras, pesos)
    """
    enmascarados = logits.masked_fill(~mascaras, LOGIT_INVALIDO)
    return functional.cross_entropy(enmascarados, etiquetas, weight=pesos)


def predecir_enmascarado(logits: torch.Tensor, mascaras: torch.Tensor) -> torch.Tensor:
    """
    Accion de mayor probabilidad entre las validas (lo que hace la politica determinista).

    Contexto:
        Evaluacion de precision.

    Args:
        logits (torch.Tensor): Logits (n, N_ACCIONES).
        mascaras (torch.Tensor): Acciones validas (n, N_ACCIONES).

    Returns:
        torch.Tensor: Indices de accion (n,).

    Raises:
        Ninguna.

    Ejemplo:
        acciones = predecir_enmascarado(logits, mascaras)
    """
    resultado: torch.Tensor = logits.masked_fill(~mascaras, LOGIT_INVALIDO).argmax(dim=1)
    return resultado


@dataclass
class TensoresDatos:
    """
    Demostraciones convertidas a tensores en el dispositivo de calculo.

    Contexto:
        Ajuste de politica y de valor.

    Atributos:
        obs: observaciones (n, dimension).
        mascaras: acciones validas (n, N_ACCIONES).
        etiquetas: accion del experto (n,).
        retornos: retorno descontado de cada paso (n,), o None si no se ajusta el valor.
    """

    obs: torch.Tensor
    mascaras: torch.Tensor
    etiquetas: torch.Tensor
    retornos: torch.Tensor | None = None

    def __len__(self) -> int:
        """
        Cantidad de pasos.

        Contexto:
            Bucles de lotes.

        Args:
            Ninguno.

        Returns:
            int: Filas.

        Raises:
            Ninguna.

        Ejemplo:
            len(tensores)
        """
        return int(self.etiquetas.shape[0])

    def lote(self, indices: torch.Tensor, dispositivo: torch.device) -> TensoresDatos:
        """
        Extrae un lote y lo lleva al dispositivo de calculo.

        Contexto:
            Cada paso de optimizacion.

        Args:
            indices (torch.Tensor): Filas del lote.
            dispositivo (torch.device): Dispositivo de destino.

        Returns:
            TensoresDatos: Lote en el dispositivo.

        Raises:
            Ninguna.

        Ejemplo:
            lote = tensores.lote(indices, dispositivo)
        """
        retornos = None if self.retornos is None else self.retornos[indices].to(dispositivo)
        return TensoresDatos(self.obs[indices].to(dispositivo), self.mascaras[indices].to(dispositivo),
                             self.etiquetas[indices].to(dispositivo), retornos)


def a_tensores(datos: Demostraciones, dispositivo: torch.device,
               retornos: NDArray[np.float32] | None = None) -> TensoresDatos:
    """
    Convierte las demostraciones a tensores.

    Contexto:
        Antes del ajuste. Si el conjunto cabe (BYTES_RESIDENTES_MAX) queda residente en el
        dispositivo; si no, se queda en CPU y cada lote viaja por separado.

    Args:
        datos (Demostraciones): Conjunto a convertir.
        dispositivo (torch.device): Dispositivo de calculo.
        retornos (NDArray[np.float32], opcional): Retornos descontados de cada paso.

    Returns:
        TensoresDatos: Tensores listos para los lotes.

    Raises:
        ValueError: Si los retornos no tienen una entrada por paso.

    Ejemplo:
        tensores = a_tensores(entrenamiento, torch.device("cpu"))
    """
    if retornos is not None and retornos.shape[0] != len(datos):
        raise ValueError("debe haber un retorno por paso")
    residente = datos.obs.nbytes <= BYTES_RESIDENTES_MAX
    destino = dispositivo if residente else torch.device("cpu")
    return TensoresDatos(
        torch.from_numpy(datos.obs).to(destino), torch.from_numpy(datos.mascaras).to(destino),
        torch.from_numpy(datos.etiquetas.astype(np.int64)).to(destino),
        None if retornos is None else torch.from_numpy(retornos).to(destino),
    )


@dataclass(frozen=True)
class EvaluacionConjunto:
    """
    Calidad de la politica sobre un conjunto.

    Contexto:
        Resultado de evaluar_politica.

    Atributos:
        perdida: perdida ponderada (la que se optimiza).
        perdida_plana: entropia cruzada sin pesos de clase (comparable entre rondas).
        precision: fraccion de pasos donde la accion de mayor probabilidad coincide con el experto.
        predicciones: accion elegida en cada paso.
    """

    perdida: float
    perdida_plana: float
    precision: float
    predicciones: NDArray[np.int64]


def evaluar_politica(politica: Any, datos: TensoresDatos, pesos: torch.Tensor, dispositivo: torch.device,
                     tamano_lote: int = 8192) -> EvaluacionConjunto:
    """
    Mide la perdida y la precision sin tocar los pesos.

    Contexto:
        Validacion de cada epoca y metricas finales.

    Args:
        politica (Any): MaskableActorCriticPolicy.
        datos (TensoresDatos): Conjunto a medir.
        pesos (torch.Tensor): Pesos de clase de la perdida ponderada.
        dispositivo (torch.device): Dispositivo de calculo.
        tamano_lote (int): Filas por lote de inferencia.

    Returns:
        EvaluacionConjunto: Perdidas, precision y predicciones.

    Raises:
        ValueError: Si el conjunto esta vacio.

    Ejemplo:
        evaluacion = evaluar_politica(modelo.policy, validacion, pesos, dispositivo)
    """
    n = len(datos)
    if n == 0:
        raise ValueError("no se puede evaluar un conjunto vacio")
    suma_ponderada = 0.0
    suma_pesos = 0.0
    suma_plana = 0.0
    predicciones: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for inicio in range(0, n, tamano_lote):
            indices = torch.arange(inicio, min(n, inicio + tamano_lote), device=datos.etiquetas.device)
            lote = datos.lote(indices, dispositivo)
            enmascarados = logits_actor(politica, lote.obs).masked_fill(~lote.mascaras, LOGIT_INVALIDO)
            por_paso = functional.cross_entropy(enmascarados, lote.etiquetas, reduction="none")
            peso_paso = pesos[lote.etiquetas]
            suma_ponderada += float((por_paso * peso_paso).sum())
            suma_pesos += float(peso_paso.sum())
            suma_plana += float(por_paso.sum())
            predicciones.append(enmascarados.argmax(dim=1).cpu().numpy())
    todas = np.concatenate(predicciones)
    etiquetas = datos.etiquetas.cpu().numpy()
    return EvaluacionConjunto(suma_ponderada / suma_pesos, suma_plana / n, float((todas == etiquetas).mean()), todas)


@dataclass(frozen=True)
class PuntoCurva:
    """
    Una epoca de las curvas de aprendizaje.

    Contexto:
        ResultadoAjuste.curva; se vuelca al reporte.

    Atributos:
        epoca: numero de epoca (desde 1).
        perdida_entrenamiento: perdida ponderada sobre una muestra fija de entrenamiento.
        perdida_validacion: perdida ponderada sobre validacion.
        precision_entrenamiento: precision sobre esa muestra.
        precision_validacion: precision sobre validacion.
    """

    epoca: int
    perdida_entrenamiento: float
    perdida_validacion: float
    precision_entrenamiento: float
    precision_validacion: float


@dataclass
class ResultadoAjuste:
    """
    Resumen de un ajuste de politica.

    Contexto:
        Salida de entrenar_politica; se guarda junto al modelo.

    Atributos:
        curva: una entrada por epoca.
        mejor_epoca: epoca cuyos pesos se conservan.
        pasos_entrenamiento: pasos de entrenamiento.
        pasos_validacion: pasos de validacion.
        episodios_entrenamiento: episodios de entrenamiento.
        episodios_validacion: episodios de validacion.
        entrenamiento: metricas finales sobre la muestra de entrenamiento.
        validacion: metricas finales sobre validacion.
        por_accion: recall y precision de cada accion en validacion.
        pesos_clase: peso de cada accion en la perdida.
        detenido_por: "paciencia" o "epocas".
    """

    curva: list[PuntoCurva] = field(default_factory=list)
    mejor_epoca: int = 0
    pasos_entrenamiento: int = 0
    pasos_validacion: int = 0
    episodios_entrenamiento: int = 0
    episodios_validacion: int = 0
    entrenamiento: dict[str, float] = field(default_factory=dict)
    validacion: dict[str, float] = field(default_factory=dict)
    por_accion: list[MetricaAccion] = field(default_factory=list)
    pesos_clase: list[float] = field(default_factory=list)
    detenido_por: str = "epocas"


def _parametros_politica(politica: Any) -> list[torch.nn.Parameter]:
    """
    Parametros que el ajuste de politica modifica.

    Contexto:
        entrenar_politica. Solo la cabeza del actor: la del critico queda intacta.

    Args:
        politica (Any): MaskableActorCriticPolicy.

    Returns:
        list[torch.nn.Parameter]: Pesos de policy_net y action_net.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    return [*politica.mlp_extractor.policy_net.parameters(), *politica.action_net.parameters()]


def _parametros_valor(politica: Any) -> list[torch.nn.Parameter]:
    """
    Parametros que el ajuste de valor modifica.

    Contexto:
        entrenar_valor. Solo la cabeza del critico: el actor queda intacto.

    Args:
        politica (Any): MaskableActorCriticPolicy.

    Returns:
        list[torch.nn.Parameter]: Pesos de value_net (del extractor) y value_net final.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    return [*politica.mlp_extractor.value_net.parameters(), *politica.value_net.parameters()]


def _copiar_estado(parametros: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """
    Copia independiente de los pesos actuales.

    Contexto:
        Guardar el mejor punto de la parada temprana.

    Args:
        parametros (list[torch.nn.Parameter]): Parametros a copiar.

    Returns:
        list[torch.Tensor]: Copias en el mismo dispositivo.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    return [copy.deepcopy(p.detach()) for p in parametros]


def _restaurar_estado(parametros: list[torch.nn.Parameter], estado: list[torch.Tensor]) -> None:
    """
    Devuelve los parametros a una copia guardada.

    Contexto:
        Fin de la parada temprana.

    Args:
        parametros (list[torch.nn.Parameter]): Parametros a restaurar.
        estado (list[torch.Tensor]): Copia tomada con _copiar_estado.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    with torch.no_grad():
        for parametro, guardado in zip(parametros, estado, strict=True):
            parametro.copy_(guardado)


def entrenar_politica(modelo: Any, entrenamiento: Demostraciones, validacion: Demostraciones,
                      config: ConfigAprendizaje, semilla: int, dispositivo: torch.device,
                      al_epoca: Callable[[PuntoCurva], None] | None = None) -> ResultadoAjuste:
    """
    Ajusta la cabeza de politica a las demostraciones con parada temprana.

    Contexto:
        Cada ronda del ciclo de imitacion. Conserva los pesos de la epoca con menor perdida de
        validacion: con episodios de validacion que la red nunca vio, es la medida directa de
        generalizacion dentro de este jefe.

    Args:
        modelo (Any): MaskablePPO cuya politica se ajusta (se modifica in situ).
        entrenamiento (Demostraciones): Episodios de entrenamiento.
        validacion (Demostraciones): Episodios de validacion (disjuntos de los anteriores).
        config (ConfigAprendizaje): Hiperparametros.
        semilla (int): Semilla del orden de los lotes.
        dispositivo (torch.device): Dispositivo de calculo.
        al_epoca (Callable, opcional): Se llama con cada punto de la curva (progreso en consola).

    Returns:
        ResultadoAjuste: Curvas, metricas finales y precision por accion.

    Raises:
        ValueError: Si el entrenamiento o la validacion estan vacios.

    Ejemplo:
        resultado = entrenar_politica(modelo, train, val, ConfigAprendizaje(), 0, torch.device("cpu"))
    """
    if len(entrenamiento) == 0 or len(validacion) == 0:
        raise ValueError("el entrenamiento y la validacion no pueden estar vacios")
    politica = modelo.policy
    politica.to(dispositivo)
    pesos = torch.from_numpy(pesos_de_clase(entrenamiento.etiquetas, acc.N_ACCIONES, config.potencia_pesos,
                                            config.tope_peso)).to(dispositivo)
    datos_train = a_tensores(entrenamiento, dispositivo)
    datos_val = a_tensores(validacion, dispositivo)
    azar = torch.Generator(device="cpu").manual_seed(semilla)
    n = len(datos_train)
    muestra = torch.randperm(n, generator=azar)[: min(n, MUESTRA_ENTRENAMIENTO)].to(datos_train.etiquetas.device)
    muestra_train = datos_train.lote(muestra, dispositivo)
    parametros = _parametros_politica(politica)
    optimizador = torch.optim.AdamW(parametros, lr=config.tasa_aprendizaje, weight_decay=config.decaimiento_pesos)
    programa = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, factor=0.5, patience=max(1, config.paciencia // 3))
    resultado = ResultadoAjuste(
        pasos_entrenamiento=n, pasos_validacion=len(datos_val),
        episodios_entrenamiento=int(np.unique(entrenamiento.episodios).size),
        episodios_validacion=int(np.unique(validacion.episodios).size),
        pesos_clase=[float(p) for p in pesos.cpu().tolist()],
    )
    mejor_perdida = float("inf")
    mejor_estado = _copiar_estado(parametros)
    sin_mejora = 0
    for epoca in range(1, config.epocas_maximas + 1):
        orden = torch.randperm(n, generator=azar).to(datos_train.etiquetas.device)
        for inicio in range(0, n, config.tamano_lote):
            lote = datos_train.lote(orden[inicio: inicio + config.tamano_lote], dispositivo)
            perdida = perdida_enmascarada(logits_actor(politica, lote.obs), lote.etiquetas, lote.mascaras, pesos)
            optimizador.zero_grad(set_to_none=True)
            torch.autograd.backward(perdida)
            torch.nn.utils.clip_grad_norm_(parametros, 1.0)
            optimizador.step()
        eval_train = evaluar_politica(politica, muestra_train, pesos, dispositivo)
        eval_val = evaluar_politica(politica, datos_val, pesos, dispositivo)
        punto = PuntoCurva(epoca, eval_train.perdida, eval_val.perdida, eval_train.precision, eval_val.precision)
        resultado.curva.append(punto)
        if al_epoca:
            al_epoca(punto)
        programa.step(eval_val.perdida)
        if eval_val.perdida < mejor_perdida - MEJORA_MINIMA:
            mejor_perdida, mejor_estado, sin_mejora = eval_val.perdida, _copiar_estado(parametros), 0
            resultado.mejor_epoca = epoca
        else:
            sin_mejora += 1
            if sin_mejora >= config.paciencia:
                resultado.detenido_por = "paciencia"
                break
    _restaurar_estado(parametros, mejor_estado)
    final_train = evaluar_politica(politica, muestra_train, pesos, dispositivo)
    final_val = evaluar_politica(politica, datos_val, pesos, dispositivo)
    resultado.entrenamiento = {"perdida": final_train.perdida, "perdida_plana": final_train.perdida_plana,
                               "precision": final_train.precision}
    resultado.validacion = {"perdida": final_val.perdida, "perdida_plana": final_val.perdida_plana,
                            "precision": final_val.precision}
    resultado.por_accion = metricas_por_accion(final_val.predicciones, validacion.etiquetas.astype(np.int64),
                                               acc.N_ACCIONES)
    return resultado


def entrenar_valor(modelo: Any, entrenamiento: Demostraciones, validacion: Demostraciones,
                   retornos_entrenamiento: NDArray[np.float32], retornos_validacion: NDArray[np.float32],
                   config: ConfigAprendizaje, semilla: int, dispositivo: torch.device) -> dict[str, float]:
    """
    Ajusta la cabeza de valor a los retornos descontados de las demostraciones.

    Contexto:
        Despues del ajuste de politica. Un critico al azar produce ventajas sin sentido en las
        primeras actualizaciones de PPO y deshace la politica clonada; con el critico ya ajustado
        PPO arranca desde una estimacion razonable. No modifica la politica.

    Args:
        modelo (Any): MaskablePPO cuya cabeza de valor se ajusta.
        entrenamiento (Demostraciones): Episodios de entrenamiento.
        validacion (Demostraciones): Episodios de validacion.
        retornos_entrenamiento (NDArray[np.float32]): Retorno de cada paso de entrenamiento.
        retornos_validacion (NDArray[np.float32]): Retorno de cada paso de validacion.
        config (ConfigAprendizaje): Hiperparametros (epocas_valor, lote, tasa, decaimiento).
        semilla (int): Semilla del orden de los lotes.
        dispositivo (torch.device): Dispositivo de calculo.

    Returns:
        dict[str, float]: Error cuadratico medio y varianza explicada en entrenamiento y validacion.

    Raises:
        ValueError: Si algun conjunto esta vacio.

    Ejemplo:
        metricas = entrenar_valor(modelo, train, val, r_train, r_val, ConfigAprendizaje(), 0, dispositivo)
    """
    if len(entrenamiento) == 0 or len(validacion) == 0:
        raise ValueError("el entrenamiento y la validacion no pueden estar vacios")
    politica = modelo.policy
    politica.to(dispositivo)
    datos_train = a_tensores(entrenamiento, dispositivo, retornos_entrenamiento)
    datos_val = a_tensores(validacion, dispositivo, retornos_validacion)
    parametros = _parametros_valor(politica)
    optimizador = torch.optim.AdamW(parametros, lr=config.tasa_aprendizaje, weight_decay=config.decaimiento_pesos)
    azar = torch.Generator(device="cpu").manual_seed(semilla)
    n = len(datos_train)

    def medir(datos: TensoresDatos) -> tuple[float, float]:
        """
        Error cuadratico medio y varianza explicada del critico sobre un conjunto.

        Contexto:
            entrenar_valor, tras cada epoca y al final.

        Args:
            datos (TensoresDatos): Conjunto con retornos.

        Returns:
            tuple[float, float]: (ecm, varianza explicada).

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        predichos: list[torch.Tensor] = []
        with torch.no_grad():
            for inicio in range(0, len(datos), 8192):
                indices = torch.arange(inicio, min(len(datos), inicio + 8192), device=datos.etiquetas.device)
                lote = datos.lote(indices, dispositivo)
                predichos.append(valor_critico(politica, lote.obs).cpu())
        prediccion = torch.cat(predichos)
        real = datos.retornos.cpu() if datos.retornos is not None else prediccion
        ecm = float(((prediccion - real) ** 2).mean())
        varianza = float(real.var())
        return ecm, (1.0 - ecm / varianza) if varianza > 0 else 0.0

    mejor = float("inf")
    mejor_estado = _copiar_estado(parametros)
    sin_mejora = 0
    for _ in range(config.epocas_valor):
        orden = torch.randperm(n, generator=azar).to(datos_train.etiquetas.device)
        for inicio in range(0, n, config.tamano_lote):
            lote = datos_train.lote(orden[inicio: inicio + config.tamano_lote], dispositivo)
            assert lote.retornos is not None
            perdida = functional.mse_loss(valor_critico(politica, lote.obs), lote.retornos)
            optimizador.zero_grad(set_to_none=True)
            torch.autograd.backward(perdida)
            torch.nn.utils.clip_grad_norm_(parametros, 1.0)
            optimizador.step()
        ecm_val, _ = medir(datos_val)
        if ecm_val < mejor - MEJORA_MINIMA:
            mejor, mejor_estado, sin_mejora = ecm_val, _copiar_estado(parametros), 0
        else:
            sin_mejora += 1
            if sin_mejora >= max(1, config.paciencia // 2):
                break
    _restaurar_estado(parametros, mejor_estado)
    ecm_train, var_train = medir(datos_train)
    ecm_val, var_val = medir(datos_val)
    return {"ecm_entrenamiento": ecm_train, "ecm_validacion": ecm_val,
            "varianza_explicada_entrenamiento": var_train, "varianza_explicada_validacion": var_val}


def guardar_modelo(modelo: Any, ruta: Path) -> None:
    """
    Guarda el modelo en formato cargable por MaskablePPO.load.

    Contexto:
        Fin de cada ajuste. La politica vuelve a CPU antes de guardar para que el archivo se cargue
        igual en el entrenamiento y en la evaluacion.

    Args:
        modelo (Any): MaskablePPO.
        ruta (Path): Archivo .zip de destino.

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        guardar_modelo(modelo, Path("runs/imitacion/mejor.zip"))
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    modelo.policy.to("cpu")
    modelo.save(str(ruta))


def coincidencia_tras_cargar(ruta: Path, obs: NDArray[np.float32], mascaras: NDArray[np.bool_],
                             esperadas: NDArray[np.int64]) -> float:
    """
    Fraccion de decisiones que el modelo recargado desde disco reproduce.

    Contexto:
        Verificacion de que el archivo guardado es cargable por MaskablePPO y decide igual que el
        modelo en memoria, con la misma API que usa el juego (predict con action_masks).

    Args:
        ruta (Path): Archivo .zip guardado.
        obs (NDArray[np.float32]): Observaciones de la muestra.
        mascaras (NDArray[np.bool_]): Mascaras de la muestra.
        esperadas (NDArray[np.int64]): Acciones que el modelo en memoria elige en la muestra.

    Returns:
        float: Fraccion de coincidencias, en [0, 1].

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la muestra esta vacia.

    Ejemplo:
        coincidencia_tras_cargar(Path("mejor.zip"), obs, mascaras, esperadas)  # 1.0
    """
    from sb3_contrib import MaskablePPO

    if obs.shape[0] == 0:
        raise ValueError("la muestra de verificacion esta vacia")
    cargado = MaskablePPO.load(str(ruta), device="cpu")
    elegidas = np.asarray([
        int(cargado.predict(obs[i], action_masks=mascaras[i], deterministic=True)[0]) for i in range(obs.shape[0])
    ])
    return float((elegidas == esperadas).mean())
