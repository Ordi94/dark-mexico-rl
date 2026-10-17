"""InspectorRed: estructura leida del modelo, fidelidad con la politica y ausencia de efectos sobre ella."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from torch import nn

from dmrl.aplicacion.inspector_red import ESCALA_FIJA_COLOR, InspectorRed
from dmrl.dominio.acciones import N_ACCIONES
from tests.support_ppo import CAPAS_PRUEBA, TAMANO_SIMPLE_PRUEBA, guardar_modelo_pequeno

APILADO = 4


@pytest.fixture
def modelo(tmp_path: Path) -> MaskablePPO:
    """
    Modelo pequeno (observacion de 4 x 6, capas 8 y 8) cargado como lo hace ver-jefes.

    Contexto:
        Pruebas del inspector.

    Args:
        tmp_path (Path): Carpeta temporal.

    Returns:
        MaskablePPO: Modelo en CPU.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    return MaskablePPO.load(str(guardar_modelo_pequeno(tmp_path / "m.zip")), device="cpu")


def observacion(semilla: int = 0) -> np.ndarray[Any, Any]:
    """
    Observacion apilada aleatoria del tamano del modelo de prueba.

    Contexto:
        Pruebas del inspector.

    Args:
        semilla (int): Semilla del generador.

    Returns:
        np.ndarray: Vector float32.

    Raises:
        Ninguna.

    Ejemplo:
        observacion(1)
    """
    return np.random.default_rng(semilla).random(APILADO * TAMANO_SIMPLE_PRUEBA).astype(np.float32)


def mascara() -> np.ndarray[Any, Any]:
    """
    Mascara con dos acciones prohibidas.

    Contexto:
        Pruebas del inspector.

    Args:
        Ninguno.

    Returns:
        np.ndarray: 22 booleanos.

    Raises:
        Ninguna.

    Ejemplo:
        mascara()[5]  # False
    """
    valida = np.ones(N_ACCIONES, dtype=bool)
    valida[[5, 21]] = False
    return valida


def test_la_estructura_se_lee_del_modelo_cargado(modelo: MaskablePPO) -> None:
    estructura = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA).estructura()
    assert estructura.capas_politica == CAPAS_PRUEBA and estructura.capas_valor == CAPAS_PRUEBA
    assert estructura.activacion == "tanh" and estructura.tamano_entrada == APILADO * TAMANO_SIMPLE_PRUEBA
    assert estructura.n_acciones == N_ACCIONES
    assert estructura.extractor_compartido is bool(modelo.policy.share_features_extractor)


def test_la_escala_de_color_es_fija(modelo: MaskablePPO) -> None:
    assert InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA).escala_color() == ESCALA_FIJA_COLOR == 1.0


def test_las_activaciones_coinciden_con_una_pasada_manual(modelo: MaskablePPO) -> None:
    obs = observacion()
    resultado = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA).inspeccionar(obs, mascara(), 0)
    entrada = torch.from_numpy(obs).reshape(1, -1)
    with torch.no_grad():
        esperado = modelo.policy.mlp_extractor.policy_net[1](modelo.policy.mlp_extractor.policy_net[0](entrada))
    assert len(resultado.activaciones_politica) == len(resultado.activaciones_valor) == 2
    assert resultado.activaciones_politica[0].shape == (CAPAS_PRUEBA[0],)
    np.testing.assert_allclose(resultado.activaciones_politica[0], esperado.numpy()[0], atol=1e-6)
    assert np.abs(resultado.activaciones_politica[1]).max() <= 1.0


def test_probabilidades_valor_y_accion_coinciden_con_predict(modelo: MaskablePPO) -> None:
    obs, valida = observacion(3), mascara()
    accion, _ = modelo.predict(obs, action_masks=valida, deterministic=True)
    resultado = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA).inspeccionar(obs, valida, int(accion))
    assert int(resultado.probabilidades.argmax()) == int(accion)
    assert resultado.probabilidades.sum() == pytest.approx(1.0)
    assert resultado.probabilidades[~valida].sum() == 0.0
    esperado = modelo.policy.predict_values(torch.from_numpy(obs).reshape(1, -1)).item()
    assert resultado.valor == pytest.approx(esperado, abs=1e-5)


def test_la_atribucion_es_gradiente_por_entrada_del_logit_elegido(modelo: MaskablePPO) -> None:
    obs, accion = observacion(4), 7
    resultado = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA).inspeccionar(obs, mascara(), accion)
    x = torch.from_numpy(obs).reshape(1, -1).requires_grad_(True)
    latente, _ = modelo.policy.mlp_extractor(x)
    (gradiente,) = torch.autograd.grad(modelo.policy.action_net(latente)[0, accion], x)
    esperado = (gradiente * x).detach().numpy()[0].reshape(APILADO, TAMANO_SIMPLE_PRUEBA).sum(axis=0)
    assert resultado.atribucion.shape == (TAMANO_SIMPLE_PRUEBA,)
    np.testing.assert_allclose(resultado.atribucion, esperado, atol=1e-5)


def test_inspeccionar_no_cambia_pesos_ni_deja_gradientes_ni_hooks(modelo: MaskablePPO) -> None:
    antes = [p.detach().clone() for p in modelo.policy.parameters()]
    inspector = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA)
    for _ in range(3):
        inspector.inspeccionar(observacion(), mascara(), 2)
    assert all(torch.equal(a, p.detach()) for a, p in zip(antes, modelo.policy.parameters(), strict=True))
    assert all(p.grad is None for p in modelo.policy.parameters())
    assert sum(len(m._forward_hooks) for m in modelo.policy.modules()) == 0


def test_los_hooks_se_retiran_aunque_la_pasada_falle(modelo: MaskablePPO) -> None:
    inspector = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA)
    with pytest.raises(ValueError, match="accion fuera"):
        inspector.inspeccionar(observacion(), mascara(), N_ACCIONES)
    assert sum(len(m._forward_hooks) for m in modelo.policy.modules()) == 0


def test_entradas_invalidas_se_rechazan_antes_de_calcular(modelo: MaskablePPO) -> None:
    inspector = InspectorRed(modelo.policy, TAMANO_SIMPLE_PRUEBA)
    with pytest.raises(ValueError, match="espera 24"):
        inspector.inspeccionar(np.zeros(10, dtype=np.float32), mascara(), 0)
    with pytest.raises(ValueError, match="accion fuera"):
        inspector.inspeccionar(observacion(), mascara(), -1)


def test_tamano_simple_incompatible_con_la_entrada_se_rechaza(modelo: MaskablePPO) -> None:
    with pytest.raises(ValueError, match="no es multiplo"):
        InspectorRed(modelo.policy, 5)
    with pytest.raises(ValueError, match="no es multiplo"):
        InspectorRed(modelo.policy, 0)


class MlpFalso(nn.Module):
    """Extractor de politica y valor separados con redes configurables."""

    def __init__(self, politica: nn.Sequential, valor: nn.Sequential) -> None:
        """
        Guarda las dos redes.

        Contexto:
            Politica falsa de las pruebas del inspector.

        Args:
            politica (nn.Sequential): Red de politica.
            valor (nn.Sequential): Red de valor.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            MlpFalso(nn.Sequential(nn.Linear(4, 3)), nn.Sequential(nn.Linear(4, 2)))
        """
        super().__init__()
        self.policy_net, self.value_net = politica, valor

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasada de la red de politica.

        Contexto:
            Politica falsa de las pruebas.

        Args:
            x (torch.Tensor): Caracteristicas.

        Returns:
            torch.Tensor: Latente de la politica.

        Raises:
            Ninguna.

        Ejemplo:
            extractor.forward_actor(x)
        """
        return self.policy_net(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasada de la red de valor.

        Contexto:
            Politica falsa de las pruebas.

        Args:
            x (torch.Tensor): Caracteristicas.

        Returns:
            torch.Tensor: Latente del valor.

        Raises:
            Ninguna.

        Ejemplo:
            extractor.forward_critic(x)
        """
        return self.value_net(x)


def politica_falsa(politica: nn.Sequential, valor: nn.Sequential, compartido: bool = True) -> SimpleNamespace:
    """
    Politica con la interfaz que usa el inspector y redes a medida.

    Contexto:
        Cubre arquitecturas distintas a la real (ReLU, sin activacion, extractor no compartido).

    Args:
        politica (nn.Sequential): Red de politica con salida de 3 neuronas.
        valor (nn.Sequential): Red de valor con salida de 3 neuronas.
        compartido (bool): Si es False el extractor devuelve una tupla (politica, valor).

    Returns:
        SimpleNamespace: Politica falsa con observacion de 4 valores y 2 acciones.

    Raises:
        Ninguna.

    Ejemplo:
        politica_falsa(nn.Sequential(nn.Linear(4, 3)), nn.Sequential(nn.Linear(4, 3)))
    """
    return SimpleNamespace(
        mlp_extractor=MlpFalso(politica, valor), action_net=nn.Linear(3, 2), value_net=nn.Linear(3, 1),
        observation_space=spaces.Box(-1, 1, (4,), np.float32), share_features_extractor=compartido,
        extract_features=(lambda x: x) if compartido else (lambda x: (x, x * 2)))


def test_una_red_lineal_sin_activacion_muestra_la_salida_de_cada_capa() -> None:
    politica = politica_falsa(nn.Sequential(nn.Linear(4, 3)), nn.Sequential(nn.Linear(4, 3), nn.ReLU()))
    inspector = InspectorRed(politica, 2)
    assert inspector.estructura().activacion == "lineal"
    resultado = inspector.inspeccionar(np.ones(4, dtype=np.float32), np.array([True, True]), 1)
    assert resultado.activaciones_politica[0].shape == (3,)
    assert (resultado.activaciones_valor[0] >= 0).all()


def test_un_extractor_no_compartido_alimenta_cada_red_con_su_propia_entrada() -> None:
    politica = politica_falsa(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
                              compartido=False)
    estructura = InspectorRed(politica, 4).estructura()
    assert estructura.extractor_compartido is False and estructura.activacion == "relu"
    resultado = InspectorRed(politica, 4).inspeccionar(np.ones(4, dtype=np.float32), np.array([True, False]), 0)
    assert resultado.probabilidades.tolist() == [1.0, 0.0]


def test_una_red_sin_capas_lineales_se_rechaza() -> None:
    vacia = politica_falsa(nn.Sequential(nn.Tanh()), nn.Sequential(nn.Linear(4, 3)))
    with pytest.raises(ValueError, match="al menos una capa"):
        InspectorRed(vacia, 2)
