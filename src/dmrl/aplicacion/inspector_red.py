"""
Inspector de la red de una MaskablePPO: activaciones, distribucion de acciones, valor y atribucion.

Observa la politica cargada sin modificarla: engancha hooks de solo lectura en las capas ocultas
durante una pasada hacia adelante y calcula el gradiente del logit con torch.autograd.grad, que no
escribe en los .grad de los parametros ni cambia sus pesos. La estructura (capas, activaciones,
extractor compartido o no) se lee del propio objeto, no se supone.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from dmrl.dominio.panel import atribucion_por_caracteristica, distribucion_enmascarada

# Tanh acota las activaciones a [-1, 1]. Otras activaciones (ReLU, etc.) no tienen cota natural: usan la
# misma escala fija con recorte, para que el color de una neurona signifique lo mismo en todo momento.
ESCALA_FIJA_COLOR = 1.0


@dataclass(frozen=True)
class EstructuraRed:
    """
    Forma de la red tal como esta cargada.

    Contexto:
        Mensaje de configuracion del panel.

    Atributos:
        capas_politica: neuronas de cada capa oculta de la red de politica.
        capas_valor: neuronas de cada capa oculta de la red de valor.
        activacion: nombre en minusculas de la funcion de activacion de la politica.
        extractor_compartido: True si politica y valor comparten el extractor de caracteristicas.
        tamano_entrada: largo de la observacion apilada.
        n_acciones: acciones de la cabeza de politica.
    """

    capas_politica: tuple[int, ...]
    capas_valor: tuple[int, ...]
    activacion: str
    extractor_compartido: bool
    tamano_entrada: int
    n_acciones: int


@dataclass(frozen=True)
class Inspeccion:
    """
    Resultado de inspeccionar una observacion.

    Contexto:
        PoliticaPanel lo convierte en un cuadro del panel.

    Atributos:
        activaciones_politica: salida de cada capa oculta de la politica.
        activaciones_valor: salida de cada capa oculta del valor.
        logits: logits de las acciones sin enmascarar.
        probabilidades: distribucion con la mascara aplicada.
        valor: estimacion V(s).
        atribucion: gradiente x entrada del logit elegido, agregado por caracteristica.
    """

    activaciones_politica: tuple[NDArray[np.float32], ...]
    activaciones_valor: tuple[NDArray[np.float32], ...]
    logits: NDArray[np.float32]
    probabilidades: NDArray[np.float64]
    valor: float
    atribucion: NDArray[np.float64]


def _salidas_de_capas(red: nn.Sequential) -> list[nn.Module]:
    """
    Modulos cuya salida es la de cada capa oculta de una red secuencial.

    Contexto:
        InspectorRed. Cada Linear cuenta como una capa; su salida visible es la de la activacion que
        le sigue, o la suya propia si no la tiene.

    Args:
        red (nn.Sequential): Red de politica o de valor del extractor.

    Returns:
        list[nn.Module]: Un modulo por capa, en orden.

    Raises:
        Ninguna.

    Ejemplo:
        _salidas_de_capas(nn.Sequential(nn.Linear(4, 3), nn.Tanh()))  # [Tanh()]
    """
    modulos = list(red)
    salidas: list[nn.Module] = []
    for i, modulo in enumerate(modulos):
        if not isinstance(modulo, nn.Linear):
            continue
        siguiente = modulos[i + 1] if i + 1 < len(modulos) else None
        salidas.append(siguiente if siguiente is not None and not isinstance(siguiente, nn.Linear) else modulo)
    return salidas


def _neuronas_de(red: nn.Sequential) -> tuple[int, ...]:
    """
    Neuronas de cada capa lineal de una red secuencial.

    Contexto:
        InspectorRed.estructura.

    Args:
        red (nn.Sequential): Red de politica o de valor del extractor.

    Returns:
        tuple[int, ...]: out_features de cada Linear.

    Raises:
        Ninguna.

    Ejemplo:
        _neuronas_de(nn.Sequential(nn.Linear(4, 3)))  # (3,)
    """
    return tuple(m.out_features for m in red if isinstance(m, nn.Linear))


class InspectorRed:
    """
    Observador de solo lectura de la politica de una MaskablePPO.

    Contexto:
        Lo crea panel_vivo y lo consulta PoliticaPanel en cada decision.
    """

    def __init__(self, politica: Any, tamano_simple: int) -> None:
        """
        Prepara el inspector sobre una politica cargada.

        Contexto:
            Arranque del panel; la politica es model.policy.

        Args:
            politica (Any): MaskableActorCriticPolicy con mlp_extractor, action_net y value_net.
            tamano_simple (int): Longitud de una observacion simple (las copias apiladas se agregan
                con ella).

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si la observacion apilada no es multiplo de tamano_simple, o si la red de
                politica o la de valor no tienen capas lineales.

        Ejemplo:
            inspector = InspectorRed(modelo.policy, 160)
        """
        self._politica = politica
        self._tamano_simple = tamano_simple
        extractor = politica.mlp_extractor
        self._red_politica: nn.Sequential = extractor.policy_net
        self._red_valor: nn.Sequential = extractor.value_net
        self._salidas_politica = _salidas_de_capas(self._red_politica)
        self._salidas_valor = _salidas_de_capas(self._red_valor)
        if not self._salidas_politica or not self._salidas_valor:
            raise ValueError("la red de politica y la de valor deben tener al menos una capa lineal")
        entrada = int(politica.observation_space.shape[0])
        if tamano_simple <= 0 or entrada % tamano_simple != 0:
            raise ValueError(f"la observacion de {entrada} valores no es multiplo de {tamano_simple}")
        self._tamano_entrada = entrada

    def estructura(self) -> EstructuraRed:
        """
        Describe la red cargada.

        Contexto:
            Mensaje de configuracion del panel.

        Args:
            Ninguno.

        Returns:
            EstructuraRed: Capas, activacion y entrada leidas del objeto.

        Raises:
            Ninguna.

        Ejemplo:
            inspector.estructura().capas_politica  # (256, 256)
        """
        salida = self._salidas_politica[-1]
        activacion = type(salida).__name__.lower() if not isinstance(salida, nn.Linear) else "lineal"
        return EstructuraRed(
            capas_politica=_neuronas_de(self._red_politica), capas_valor=_neuronas_de(self._red_valor),
            activacion=activacion, extractor_compartido=bool(self._politica.share_features_extractor),
            tamano_entrada=self._tamano_entrada, n_acciones=int(self._politica.action_net.out_features))

    def escala_color(self) -> float:
        """
        Valor absoluto que corresponde al color extremo de las cuadriculas.

        Contexto:
            Mensaje de configuracion del panel. Es constante: no se renormaliza por cuadro.

        Args:
            Ninguno.

        Returns:
            float: ESCALA_FIJA_COLOR.

        Raises:
            Ninguna.

        Ejemplo:
            inspector.escala_color()  # 1.0
        """
        return ESCALA_FIJA_COLOR

    @contextmanager
    def _capturar(self) -> Iterator[tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]]:
        """
        Engancha hooks de lectura en las capas ocultas mientras dura el bloque.

        Contexto:
            inspeccionar. Los hooks se retiran siempre al salir: el modelo queda como estaba.

        Args:
            Ninguno.

        Returns:
            Iterator: Dos diccionarios (politica, valor) indice de capa -> activacion desacoplada.

        Raises:
            Ninguna.

        Ejemplo:
            with self._capturar() as (pi, vf): ...
        """
        capturas_pi: dict[int, torch.Tensor] = {}
        capturas_vf: dict[int, torch.Tensor] = {}
        manejadores = []

        def enganchar(modulo: nn.Module, destino: dict[int, torch.Tensor], indice: int) -> None:
            """
            Registra el hook de una capa.

            Contexto:
                Cuerpo de _capturar.

            Args:
                modulo (nn.Module): Capa a observar.
                destino (dict): Diccionario donde se guarda la salida.
                indice (int): Posicion de la capa.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            def guardar(_modulo: nn.Module, _entrada: Any, salida: torch.Tensor) -> None:
                """
                Guarda una copia sin gradiente de la salida de la capa.

                Contexto:
                    Hook de pasada hacia adelante.

                Args:
                    _modulo (nn.Module): Capa observada.
                    _entrada (Any): Entrada de la capa (no usada).
                    salida (torch.Tensor): Salida de la capa.

                Returns:
                    None: No retorna valor.

                Raises:
                    Ninguna.

                Ejemplo:
                    (llamado por torch)
                """
                destino[indice] = salida.detach().clone()

            manejadores.append(modulo.register_forward_hook(guardar))

        for i, modulo in enumerate(self._salidas_politica):
            enganchar(modulo, capturas_pi, i)
        for i, modulo in enumerate(self._salidas_valor):
            enganchar(modulo, capturas_vf, i)
        try:
            yield capturas_pi, capturas_vf
        finally:
            for manejador in manejadores:
                manejador.remove()

    def inspeccionar(self, observacion: NDArray[np.float32], mascara: NDArray[np.bool_], accion: int) -> Inspeccion:
        """
        Calcula activaciones, probabilidades, valor y atribucion para una observacion.

        Contexto:
            PoliticaPanel.decidir, una vez por decision, con la accion que ya eligio la politica.

        Args:
            observacion (NDArray[np.float32]): Observacion apilada que vio la red.
            mascara (NDArray[np.bool_]): Acciones validas.
            accion (int): Accion elegida; su logit es el que se deriva.

        Returns:
            Inspeccion: Activaciones por capa, probabilidades, valor y atribucion por caracteristica.

        Raises:
            ValueError: Si la observacion, la mascara o la accion no corresponden a la red.

        Ejemplo:
            inspector.inspeccionar(obs, env.action_masks(), 19)
        """
        n_acciones = int(self._politica.action_net.out_features)
        vector = np.asarray(observacion, dtype=np.float32).reshape(-1)
        if vector.size != self._tamano_entrada:
            raise ValueError(f"la observacion tiene {vector.size} valores y la red espera {self._tamano_entrada}")
        if not 0 <= int(accion) < n_acciones:
            raise ValueError(f"accion fuera de 0..{n_acciones - 1}: {accion}")
        entrada = torch.from_numpy(vector.copy()).reshape(1, -1).requires_grad_(True)
        with self._capturar() as (capturas_pi, capturas_vf):
            caracteristicas = self._politica.extract_features(entrada)
            if isinstance(caracteristicas, tuple):
                para_politica, para_valor = caracteristicas
            else:
                para_politica = para_valor = caracteristicas
            extractor = self._politica.mlp_extractor
            logits = self._politica.action_net(extractor.forward_actor(para_politica))
            valor = self._politica.value_net(extractor.forward_critic(para_valor))
        # autograd.grad evita escribir en los .grad de los parametros del modelo.
        (gradiente,) = torch.autograd.grad(logits[0, int(accion)], entrada)
        logits_np = logits.detach().numpy()[0].astype(np.float32)
        return Inspeccion(
            activaciones_politica=tuple(capturas_pi[i].numpy()[0] for i in range(len(self._salidas_politica))),
            activaciones_valor=tuple(capturas_vf[i].numpy()[0] for i in range(len(self._salidas_valor))),
            logits=logits_np,
            probabilidades=distribucion_enmascarada(logits_np, mascara),
            valor=float(valor.detach().numpy().reshape(-1)[0]),
            atribucion=atribucion_por_caracteristica(gradiente.numpy()[0], vector, self._tamano_simple))
