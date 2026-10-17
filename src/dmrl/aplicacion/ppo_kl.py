"""
MaskablePPO con calentamiento del critico y penalizacion KL hacia la politica clonada.

La politica de imitacion ya juega al nivel del bot, pero su critico se ajusto a los retornos de las
demostraciones y no a lo que hara la politica mientras PPO la cambia. Dos defensas evitan que las
primeras actualizaciones deshagan lo clonado:

* Calentamiento: durante los primeros pasos solo se entrena la cabeza de valor. La politica y el
  critico son redes separadas (net_arch con pi y vf), de modo que anular la perdida de politica y de
  entropia deja a la politica con gradiente nulo: queda congelada sin tocar el optimizador.
* Penalizacion KL: despues del calentamiento la perdida suma beta * KL(politica clonada || politica
  actual) sobre los estados del rollout. beta baja linealmente a cero (coeficiente_kl): PPO queda
  libre cuando el critico ya es fiable. La divergencia se mide sobre la distribucion enmascarada, la
  misma con que se juega.

El resto de train() es el de MaskablePPO (sb3-contrib 2.9.0) sin cambios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional

from dmrl.dominio.ppo_jefes import ConfigPPOJefes, coeficiente_kl, en_calentamiento


def divergencia_kl(distribucion_referencia: Any, distribucion_actual: Any) -> torch.Tensor:
    """
    KL(referencia || actual) media sobre un lote, entre distribuciones categoricas enmascaradas.

    Contexto:
        MaskablePPOKL.train. Las acciones prohibidas llevan un logit muy negativo en ambas
        distribuciones, de modo que su probabilidad es cero y no aportan.

    Args:
        distribucion_referencia (Any): Distribucion de la politica clonada (get_distribution).
        distribucion_actual (Any): Distribucion de la politica que se entrena.

    Returns:
        torch.Tensor: Escalar con el KL medio del lote; cero si las distribuciones coinciden.

    Raises:
        Ninguna.

    Ejemplo:
        kl = divergencia_kl(referencia.get_distribution(obs, action_masks=m),
                            actual.get_distribution(obs, action_masks=m))
    """
    logp_ref = functional.log_softmax(distribucion_referencia.distribution.logits, dim=-1)
    logp_act = functional.log_softmax(distribucion_actual.distribution.logits, dim=-1)
    p_ref = logp_ref.exp()
    return (p_ref * (logp_ref - logp_act)).sum(dim=-1).mean()


class MaskablePPOKL(MaskablePPO):
    """
    MaskablePPO que calienta el critico y se mantiene cerca de una politica de referencia.

    Contexto:
        Entrenamiento nocturno multi-jefe. Los parametros de la regla se fijan con configurar_kl
        despues de crear o cargar el modelo; no se guardan en el zip, asi que un modelo reanudado
        los recibe otra vez de la configuracion de la corrida.
    """

    config_ppo: ConfigPPOJefes | None = None
    politica_referencia: Any = None

    def _excluded_save_params(self) -> list[str]:
        """
        Atributos que no se serializan en el zip del modelo.

        Contexto:
            MaskablePPO.save. La politica de referencia y la regla de la corrida se vuelven a
            asignar con configurar_kl al reanudar; guardarlas inflaria el archivo y lo ataria a
            clases que pueden cambiar.

        Args:
            Ninguno.

        Returns:
            list[str]: Los excluidos por SB3 mas politica_referencia y config_ppo.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por save)
        """
        return [*super()._excluded_save_params(), "politica_referencia", "config_ppo"]

    def configurar_kl(self, config: ConfigPPOJefes, ruta_referencia: Path | None) -> None:
        """
        Fija la regla de calentamiento y penalizacion y carga la politica de referencia.

        Contexto:
            entrenamiento_jefes, tras crear o cargar el modelo.

        Args:
            config (ConfigPPOJefes): Parametros de la corrida (calentamiento, beta, decaimiento).
            ruta_referencia (Path, opcional): Modelo clonado cuya politica se usa como referencia;
                None desactiva la penalizacion (el calentamiento sigue aplicando).

        Returns:
            None: No retorna valor.

        Raises:
            FileNotFoundError: Si se da una ruta que no existe.

        Ejemplo:
            modelo.configurar_kl(config, Path("runs/multijefe_imitacion/mejor.zip"))
        """
        self.config_ppo = config
        self.politica_referencia = None
        if ruta_referencia is None:
            return
        if not ruta_referencia.exists():
            raise FileNotFoundError(f"no existe la politica de referencia {ruta_referencia}")
        referencia = MaskablePPO.load(str(ruta_referencia), device=self.device)
        politica = referencia.policy
        politica.set_training_mode(False)
        for parametro in politica.parameters():
            parametro.requires_grad_(False)
        self.politica_referencia = politica

    def train(self) -> None:
        """
        Actualiza la politica con el rollout reunido: PPO con calentamiento y penalizacion KL.

        Contexto:
            MaskablePPO.learn, tras cada rollout. Copia el cuerpo de MaskablePPO.train con tres
            diferencias: en calentamiento la perdida es solo la del valor; despues se suma
            beta * KL a la referencia; y se registran kl_referencia, beta_kl y calentamiento.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            RuntimeError: Si el modelo se uso sin configurar_kl.

        Ejemplo:
            (llamado por learn)
        """
        if self.config_ppo is None:
            raise RuntimeError("configurar_kl debe llamarse antes de entrenar")
        config = self.config_ppo
        calentando = en_calentamiento(self.num_timesteps, config)
        beta = 0.0 if self.politica_referencia is None else coeficiente_kl(self.num_timesteps, config)
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        kls_referencia: list[float] = []
        approx_kl_divs: list[Any] = []
        continue_training = True
        loss = torch.zeros(())
        for _epoca in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio,
                                         advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                value_loss = functional.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                if calentando:
                    loss = self.vf_coef * value_loss
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    if beta > 0.0:
                        actual = self.policy.get_distribution(
                            rollout_data.observations,
                            action_masks=rollout_data.action_masks)  # type: ignore[arg-type]
                        with torch.no_grad():
                            referencia = self.politica_referencia.get_distribution(
                                rollout_data.observations, action_masks=rollout_data.action_masks)
                        kl_ref = divergencia_kl(referencia, actual)
                        kls_referencia.append(kl_ref.item())
                        loss = loss + beta * kl_ref
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy())
                if not calentando and self.target_kl is not None and approx_kl_divs[-1] > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("dm/calentamiento_critico", float(calentando))
        self.logger.record("dm/beta_kl", beta)
        self.logger.record("dm/tasa_aprendizaje", float(self.policy.optimizer.param_groups[0]["lr"]))
        if kls_referencia:
            self.logger.record("dm/kl_referencia", float(np.mean(kls_referencia)))
