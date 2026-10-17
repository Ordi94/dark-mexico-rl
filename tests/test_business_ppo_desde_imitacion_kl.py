"""
Reglas de negocio del PPO que parte de la politica clonada: el critico se calienta con la politica
congelada y la penalizacion KL la mantiene cerca de la imitacion.
"""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3.common.logger import configure

from dmrl.aplicacion.entrenamiento_jefes import cargar_modelo_ppo
from dmrl.aplicacion.ppo_kl import MaskablePPOKL, divergencia_kl
from dmrl.dominio.acciones import N_ACCIONES
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from tests.support_ppo import (
    config_ppo,
    guardar_modelo_pequeno,
    observaciones_fijas,
    pesos_de,
    son_iguales,
    vec_prueba,
)


def preparar(tmp_path: Path, config: ConfigPPOJefes, nombre: str = "inicial") -> tuple[MaskablePPOKL, Path]:
    """
    Carga el modelo clonado de partida con su referencia y un logger sin salidas.

    Contexto:
        Atajo de las pruebas de este archivo; todas parten de los mismos pesos.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        config (ConfigPPOJefes): Regla de la corrida.
        nombre (str): Nombre del zip de partida.

    Returns:
        tuple[MaskablePPOKL, Path]: Modelo y zip de partida.

    Raises:
        Ninguna.

    Ejemplo:
        modelo, inicial = preparar(tmp_path, config_ppo())
    """
    inicial = tmp_path / f"{nombre}.zip"
    if not inicial.exists():
        guardar_modelo_pequeno(inicial, semilla=11)
    modelo = cargar_modelo_ppo(inicial, vec_prueba(), config, tmp_path / "corrida")
    modelo.configurar_kl(config, inicial)
    modelo.set_logger(configure(None, []))
    return modelo, inicial


def parametros_politica(modelo: MaskablePPOKL) -> list[torch.Tensor]:
    """
    Pesos de la cabeza de politica.

    Contexto:
        Comparar antes y despues de entrenar.

    Args:
        modelo (MaskablePPOKL): Modelo a leer.

    Returns:
        list[torch.Tensor]: Copias de policy_net y action_net.

    Raises:
        Ninguna.

    Ejemplo:
        antes = parametros_politica(modelo)
    """
    politica = modelo.policy
    return pesos_de([*politica.mlp_extractor.policy_net.parameters(), *politica.action_net.parameters()])


def parametros_valor(modelo: MaskablePPOKL) -> list[torch.Tensor]:
    """
    Pesos de la cabeza de valor.

    Contexto:
        Comparar antes y despues de entrenar.

    Args:
        modelo (MaskablePPOKL): Modelo a leer.

    Returns:
        list[torch.Tensor]: Copias de value_net del extractor y la capa final.

    Raises:
        Ninguna.

    Ejemplo:
        antes = parametros_valor(modelo)
    """
    politica = modelo.policy
    return pesos_de([*politica.mlp_extractor.value_net.parameters(), *politica.value_net.parameters()])


def kl_a_la_referencia(modelo: MaskablePPOKL) -> float:
    """
    KL medio entre la politica clonada y la actual sobre un lote fijo de estados.

    Contexto:
        Medir cuanto se alejo PPO de lo clonado, con la misma distribucion enmascarada del juego.

    Args:
        modelo (MaskablePPOKL): Modelo con referencia.

    Returns:
        float: KL(referencia || actual).

    Raises:
        Ninguna.

    Ejemplo:
        kl = kl_a_la_referencia(modelo)
    """
    obs = observaciones_fijas()
    mascaras = torch.ones((obs.shape[0], N_ACCIONES), dtype=torch.bool).numpy()
    with torch.no_grad():
        referencia = modelo.politica_referencia.get_distribution(obs, action_masks=mascaras)
        actual = modelo.policy.get_distribution(obs, action_masks=mascaras)
    return float(divergencia_kl(referencia, actual))


def test_durante_el_calentamiento_los_pesos_de_la_politica_no_cambian_y_los_del_valor_si(tmp_path: Path) -> None:
    modelo, _ = preparar(tmp_path, config_ppo(pasos_calentamiento=10_000))
    politica_antes, valor_antes = parametros_politica(modelo), parametros_valor(modelo)
    torch.manual_seed(2)
    modelo.learn(96, log_interval=1000)
    assert modelo.num_timesteps == 96
    assert son_iguales(politica_antes, parametros_politica(modelo))
    assert not son_iguales(valor_antes, parametros_valor(modelo))


def test_el_calentamiento_no_se_corta_por_target_kl_porque_la_politica_no_se_mueve(tmp_path: Path) -> None:
    modelo, _ = preparar(tmp_path, config_ppo(pasos_calentamiento=10_000, target_kl=1e-12))
    torch.manual_seed(2)
    modelo.learn(32, log_interval=1000)
    assert modelo._n_updates == 2


def test_tras_el_calentamiento_la_politica_cambia(tmp_path: Path) -> None:
    modelo, _ = preparar(tmp_path, config_ppo(pasos_calentamiento=40))
    torch.manual_seed(2)
    modelo.learn(32, log_interval=1000)
    politica_calentada = parametros_politica(modelo)
    modelo.learn(64, log_interval=1000, reset_num_timesteps=False)
    assert not son_iguales(politica_calentada, parametros_politica(modelo))


def corrida_con_beta(tmp_path: Path, beta: float) -> float:
    """
    Entrena con una semilla fija y devuelve cuanto se alejo la politica de la referencia.

    Contexto:
        Comparar dos valores de beta con todo lo demas igual.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        beta (float): Coeficiente inicial de la penalizacion.

    Returns:
        float: KL final a la referencia.

    Raises:
        Ninguna.

    Ejemplo:
        corrida_con_beta(tmp_path, 0.0)
    """
    config = config_ppo(beta_kl=beta, pasos_decaimiento_kl=1_000_000, learning_rate=3e-3, lr_minimo=1e-4,
                        n_epochs=4)
    modelo, _ = preparar(tmp_path, config)
    torch.manual_seed(7)
    modelo.learn(160, log_interval=1000)
    return kl_a_la_referencia(modelo)


def test_la_penalizacion_kl_mantiene_la_politica_mas_cerca_de_la_referencia_que_sin_ella(tmp_path: Path) -> None:
    sin_penalizacion = corrida_con_beta(tmp_path, 0.0)
    con_penalizacion = corrida_con_beta(tmp_path, 50.0)
    assert sin_penalizacion > 1e-4
    assert con_penalizacion < sin_penalizacion


def test_la_penalizacion_decae_a_cero_y_deja_libre_a_ppo(tmp_path: Path) -> None:
    config = config_ppo(beta_kl=5.0, pasos_decaimiento_kl=16)
    modelo, _ = preparar(tmp_path, config)
    torch.manual_seed(2)
    modelo.learn(64, log_interval=1000)
    assert modelo.logger.name_to_value["dm/beta_kl"] == 0.0
    assert "dm/kl_referencia" not in modelo.logger.name_to_value
