"""MaskablePPOKL: divergencia KL, configuracion de la referencia, guardado y registro de metricas."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    vec_prueba,
)


def modelo_configurado(tmp_path: Path, config: ConfigPPOJefes, con_referencia: bool = True,
                       semilla: int = 0) -> tuple[MaskablePPOKL, Path]:
    """
    Carga un modelo pequeno como MaskablePPOKL y le fija la regla de la corrida.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        config (ConfigPPOJefes): Regla de calentamiento y penalizacion.
        con_referencia (bool): Si es False se desactiva la penalizacion.
        semilla (int): Semilla de los pesos del zip de partida.

    Returns:
        tuple[MaskablePPOKL, Path]: Modelo con logger propio y zip del que parte.

    Raises:
        Ninguna.

    Ejemplo:
        modelo, zip_inicial = modelo_configurado(tmp_path, config_ppo())
    """
    inicial = guardar_modelo_pequeno(tmp_path / f"inicial_{semilla}.zip", semilla)
    modelo = cargar_modelo_ppo(inicial, vec_prueba(), config, tmp_path / "corrida")
    modelo.configurar_kl(config, inicial if con_referencia else None)
    modelo.set_logger(configure(None, []))
    return modelo, inicial


def logits_de(*logits: float) -> SimpleNamespace:
    """
    Distribucion falsa que solo expone sus logits.

    Contexto:
        divergencia_kl solo lee distribucion.distribution.logits.

    Args:
        *logits (float): Logits de una sola fila.

    Returns:
        SimpleNamespace: Objeto con .distribution.logits de forma (1, n).

    Raises:
        Ninguna.

    Ejemplo:
        logits_de(0.0, 0.0)
    """
    return SimpleNamespace(distribution=SimpleNamespace(logits=torch.tensor([list(logits)])))


def test_divergencia_kl_coincide_con_el_valor_analitico() -> None:
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.9, 0.1])
    esperado = float((p * (p / q).log()).sum())
    assert float(divergencia_kl(logits_de(*p.log().tolist()), logits_de(*q.log().tolist()))) == pytest.approx(
        esperado, abs=1e-6)


def test_divergencia_kl_ignora_las_acciones_enmascaradas() -> None:
    prohibido = -1.0e8
    referencia = logits_de(0.0, 0.0, prohibido)
    actual = logits_de(0.0, math.log(3.0), prohibido)
    esperado = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert float(divergencia_kl(referencia, actual)) == pytest.approx(esperado, abs=1e-6)


def test_divergencia_kl_es_cero_entre_distribuciones_iguales_y_positiva_si_difieren(tmp_path: Path) -> None:
    config = config_ppo()
    modelo, inicial = modelo_configurado(tmp_path, config)
    otro, _ = modelo_configurado(tmp_path, config, semilla=5)
    obs = observaciones_fijas()
    mascaras = torch.ones((obs.shape[0], N_ACCIONES), dtype=torch.bool).numpy()
    with torch.no_grad():
        propia = modelo.policy.get_distribution(obs, action_masks=mascaras)
        copia = modelo.politica_referencia.get_distribution(obs, action_masks=mascaras)
        ajena = otro.policy.get_distribution(obs, action_masks=mascaras)
    assert float(divergencia_kl(copia, propia)) == pytest.approx(0.0, abs=1e-9)
    assert float(divergencia_kl(copia, ajena)) > 1e-7
    assert inicial.exists()


def test_configurar_kl_sin_ruta_desactiva_la_penalizacion_pero_guarda_la_regla(tmp_path: Path) -> None:
    config = config_ppo()
    modelo, _ = modelo_configurado(tmp_path, config, con_referencia=False)
    assert modelo.politica_referencia is None and modelo.config_ppo is config


def test_configurar_kl_con_ruta_inexistente_falla(tmp_path: Path) -> None:
    modelo, _ = modelo_configurado(tmp_path, config_ppo(), con_referencia=False)
    with pytest.raises(FileNotFoundError, match="politica de referencia"):
        modelo.configurar_kl(config_ppo(), tmp_path / "no_existe.zip")


def test_configurar_kl_congela_la_referencia_y_la_separa_del_modelo(tmp_path: Path) -> None:
    modelo, _ = modelo_configurado(tmp_path, config_ppo())
    referencia = modelo.politica_referencia
    assert referencia is not modelo.policy
    assert not referencia.training
    assert all(not p.requires_grad for p in referencia.parameters())
    assert all(p.requires_grad for p in modelo.policy.parameters())


def test_configurar_kl_una_segunda_vez_sin_ruta_descarta_la_referencia_anterior(tmp_path: Path) -> None:
    modelo, _ = modelo_configurado(tmp_path, config_ppo())
    modelo.configurar_kl(config_ppo(), None)
    assert modelo.politica_referencia is None


def test_train_sin_configurar_falla(tmp_path: Path) -> None:
    inicial = guardar_modelo_pequeno(tmp_path / "m.zip")
    modelo = cargar_modelo_ppo(inicial, vec_prueba(), config_ppo(), tmp_path / "corrida")
    with pytest.raises(RuntimeError, match="configurar_kl"):
        modelo.train()


def test_el_zip_no_guarda_la_referencia_ni_la_config(tmp_path: Path) -> None:
    modelo, _ = modelo_configurado(tmp_path, config_ppo(beta_kl=1.0))
    assert "politica_referencia" in modelo._excluded_save_params()
    assert "config_ppo" in modelo._excluded_save_params()
    destino = tmp_path / "guardado.zip"
    modelo.save(destino)
    with zipfile.ZipFile(destino) as zip_modelo:
        datos = json.loads(zip_modelo.read("data"))
        nombres = zip_modelo.namelist()
    assert "politica_referencia" not in datos and "config_ppo" not in datos
    assert not any("referencia" in n for n in nombres)
    recargado = MaskablePPOKL.load(str(destino), device="cpu")
    assert recargado.politica_referencia is None and recargado.config_ppo is None


def test_train_registra_las_metricas_de_la_regla(tmp_path: Path) -> None:
    config = config_ppo(beta_kl=5.0, pasos_calentamiento=0, pasos_decaimiento_kl=1000)
    modelo, _ = modelo_configurado(tmp_path, config)
    torch.manual_seed(1)
    modelo.learn(32, log_interval=1000)
    valores = modelo.logger.name_to_value
    assert valores["dm/calentamiento_critico"] == 0.0
    assert valores["dm/beta_kl"] == pytest.approx(5.0 * (1.0 - 32 / 1000))
    assert valores["dm/kl_referencia"] >= 0.0
    assert valores["dm/tasa_aprendizaje"] == pytest.approx(config.learning_rate)
    assert {"train/loss", "train/value_loss", "train/approx_kl", "train/explained_variance"} <= set(valores)


def test_sin_referencia_no_hay_penalizacion_ni_metrica_kl(tmp_path: Path) -> None:
    config = config_ppo(beta_kl=5.0, pasos_decaimiento_kl=1000)
    modelo, _ = modelo_configurado(tmp_path, config, con_referencia=False)
    torch.manual_seed(1)
    modelo.learn(32, log_interval=1000)
    valores = modelo.logger.name_to_value
    assert valores["dm/beta_kl"] == 0.0
    assert "dm/kl_referencia" not in valores


def test_durante_el_calentamiento_se_registra_el_calentamiento_y_beta_cero(tmp_path: Path) -> None:
    config = config_ppo(beta_kl=5.0, pasos_calentamiento=1000)
    modelo, _ = modelo_configurado(tmp_path, config)
    torch.manual_seed(1)
    modelo.learn(32, log_interval=1000)
    valores = modelo.logger.name_to_value
    assert valores["dm/calentamiento_critico"] == 1.0
    assert valores["dm/beta_kl"] == 0.0
    assert "dm/kl_referencia" not in valores


def test_target_kl_corta_las_epocas_de_la_actualizacion_fuera_del_calentamiento(tmp_path: Path) -> None:
    tolerante = config_ppo(target_kl=100.0, batch_size=16)
    estricto = config_ppo(target_kl=1e-12, batch_size=16)
    modelo_a, _ = modelo_configurado(tmp_path, tolerante, semilla=1)
    modelo_b, _ = modelo_configurado(tmp_path, estricto, semilla=1)
    for modelo in (modelo_a, modelo_b):
        torch.manual_seed(3)
        modelo.learn(64, log_interval=1000)
    # Dos rollouts de 32 pasos en minibatches de 16 y dos epocas: sin corte son 4 actualizaciones; con
    # corte, el segundo minibatch ya supera el umbral y solo cuenta una por rollout.
    assert modelo_a._n_updates == 4
    assert modelo_b._n_updates == 2
