"""
Politicas de referencia sobre la observacion del puente.

Sirven para validar el entorno (que el parry siga siendo acertable, que los golpes conecten)
y como linea base contra la que comparar al agente. Leen solo la observacion publica del
protocolo, igual que lo haria el agente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dmbridge.protocol import Action

# Impactos parriables por montage del jefe (posicion en segundos de montage), medidos con
# hooks de TakeDamage. Los golpes no parriables no estan: contra ellos el bloqueo no sirve.
PARRY_HITS: dict[str, tuple[float, ...]] = {
    "Boss_Attack_Uppercut_InP_Montage": (0.895,),
    "Boss_Attack_SwingAndSlam_InP_Montage": (1.844,),
    "Boss_Attack_ChargedCombo_InP_Montage": (0.713, 1.19, 1.87),
    "Boss_Attack_ChargedCombo_InP_Montage1": (0.713, 1.18, 1.90),
}

# La ventana de parry se abre ~0.17 s de juego despues de pulsar LB y dura ~0.12 s: pulsar
# 0.235 s (escalados por el play rate del montage del jefe) antes del impacto la centra.
PARRY_LEAD = 0.235


@dataclass
class ParryBot:
    """
    Bot de parry con ataque oportunista, para pasos de un frame.

    Contexto:
        scripts/bench_combat.py: validacion del combate en cada modo de simulacion.

    Atributos:
        lead: segundos de montage antes del impacto en que se pulsa LB (por play rate).
        hold: segundos de juego que se mantiene LB antes de soltarlo.
        attack_every: segundos de juego entre ataques cuando el jefe esta cerca y quieto.
        approach_dist: distancia a partir de la cual el bot camina hacia el jefe.
        attack_dist: distancia maxima para atacar.
    """

    lead: float = PARRY_LEAD
    hold: float = 0.3
    attack_every: float = 0.5
    approach_dist: float = 230.0
    attack_dist: float = 200.0
    attack: bool = True

    def __post_init__(self) -> None:
        """
        Inicializa el estado interno del bot.

        Contexto:
            Construccion del dataclass.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot = ParryBot()
        """
        self.reset()

    def reset(self) -> None:
        """
        Olvida el estado del episodio anterior.

        Contexto:
            Tras cada reset del entorno.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            bot.reset()
        """
        self.prev_montage: str | None = None
        self.prev_pos = 0.0
        self.release_at = -1.0
        self.next_attack = 0.0
        self.parry_presses = 0

    def act(self, obs: dict[str, Any]) -> Action:
        """
        Decide la accion del siguiente paso a partir de la observacion.

        Contexto:
            Bucle de scripts/bench_combat.py con pasos de 1 frame.

        Args:
            obs (dict[str, Any]): Observacion del puente.

        Returns:
            Action: Accion del paso.

        Raises:
            Ninguna.

        Ejemplo:
            accion = bot.act(obs)
        """
        boss = obs.get("boss")
        now = float(obs.get("t", 0.0))
        lb = now < self.release_at
        if boss is None or obs.get("player") is None:
            return Action(lb=lb)
        montage = boss.get("montage") or {}
        name = montage.get("name")
        pos = float(montage.get("pos", 0.0))
        rate = float(montage.get("rate", 1.0))
        hits = PARRY_HITS.get(name or "")
        if hits and name == self.prev_montage:
            for impact in hits:
                trigger = impact - self.lead * rate
                if self.prev_pos < trigger <= pos:
                    lb = True
                    self.release_at = now + self.hold
                    self.parry_presses += 1
        self.prev_montage, self.prev_pos = name, pos
        dist = float(obs.get("dist", 0.0))
        my = 1.0 if dist > self.approach_dist and not hits else 0.0
        btn = None
        if self.attack and not lb and dist < self.attack_dist and not boss.get("attacking") and now >= self.next_attack:
            btn = "rb"
            self.next_attack = now + self.attack_every
        return Action(btn=btn, lb=lb, my=my)
