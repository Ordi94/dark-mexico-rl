"""Reglas de negocio del bot de parry de referencia."""

from __future__ import annotations

from typing import Any

import pytest

from dmbridge.reference_bots import PARRY_HITS, PARRY_LEAD, ParryBot

UPPERCUT = "Boss_Attack_Uppercut_InP_Montage"
COMBO = "Boss_Attack_ChargedCombo_InP_Montage"


def make_obs(
    t: float = 0.0,
    name: str | None = None,
    pos: float = 0.0,
    rate: float = 1.0,
    dist: float = 100.0,
    attacking: bool = False,
) -> dict[str, Any]:
    """
    Construye una observacion minima del puente.

    Contexto:
        Entrada de ParryBot.act en las pruebas.

    Args:
        t (float): Tiempo de juego.
        name (str, opcional): Montage del jefe; None = sin montage.
        pos (float): Posicion en el montage.
        rate (float): Play rate del montage.
        dist (float): Distancia al jefe.
        attacking (bool): Si el jefe esta atacando.

    Returns:
        dict[str, Any]: Observacion con jugador y jefe presentes.

    Raises:
        Ninguna.

    Ejemplo:
        make_obs(t=1.0, name=UPPERCUT, pos=0.7)
    """
    montage = {"name": name, "pos": pos, "rate": rate} if name else None
    return {"t": t, "player": {}, "dist": dist, "boss": {"montage": montage, "attacking": attacking}}


def test_business_parry_bot_pulsa_lb_antes_del_impacto_escalado_por_play_rate() -> None:
    """Con play rate 1 el disparo es 0.235 s de montage antes del impacto en 0.895."""
    bot = ParryBot(attack=False)
    assert not bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.60)).lb
    assert not bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.65)).lb
    assert bot.act(make_obs(t=1.2, name=UPPERCUT, pos=0.67)).lb
    assert bot.parry_presses == 1


def test_business_parry_bot_adelanta_el_disparo_con_play_rate_mayor() -> None:
    """Con play rate 2 el montage avanza al doble: el disparo se adelanta lead*rate."""
    bot = ParryBot(attack=False)
    bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.30, rate=2.0))
    # trigger = 0.895 - 0.235*2 = 0.425; con rate 1 (0.66) este salto no dispararia.
    assert bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.45, rate=2.0)).lb


def test_business_parry_bot_con_play_rate_menor_dispara_mas_tarde() -> None:
    """Con play rate 0.5 el disparo se retrasa: 0.895 - 0.1175 = 0.7775."""
    bot = ParryBot(attack=False)
    bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.60, rate=0.5))
    assert not bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.70, rate=0.5)).lb
    assert bot.act(make_obs(t=1.2, name=UPPERCUT, pos=0.80, rate=0.5)).lb


def test_business_parry_bot_suelta_lb_tras_hold() -> None:
    """LB se mantiene hold segundos de juego y luego se suelta."""
    bot = ParryBot(attack=False, hold=0.3)
    bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.60))
    assert bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.70)).lb
    assert bot.act(make_obs(t=1.3, name=UPPERCUT, pos=0.80)).lb
    assert not bot.act(make_obs(t=1.45, name=UPPERCUT, pos=0.90)).lb


def test_business_parry_bot_no_ataca_si_el_jefe_ataca() -> None:
    """Mientras el jefe ataca el bot no lanza golpes aunque este en rango."""
    bot = ParryBot()
    assert bot.act(make_obs(t=1.0, dist=100.0, attacking=True)).btn is None
    assert bot.act(make_obs(t=1.0, dist=100.0, attacking=False)).btn == "rb"


def test_business_parry_bot_espacia_los_ataques_attack_every() -> None:
    """Tras un ataque no vuelve a atacar hasta pasados attack_every segundos."""
    bot = ParryBot(attack_every=0.5)
    assert bot.act(make_obs(t=1.0)).btn == "rb"
    assert bot.act(make_obs(t=1.2)).btn is None
    assert bot.act(make_obs(t=1.5)).btn == "rb"


def test_business_parry_bot_no_ataca_si_esta_bloqueando() -> None:
    """Con LB activo no se ataca: el bloqueo tiene prioridad."""
    bot = ParryBot()
    bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.60))
    action = bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.70))
    assert action.lb
    assert action.btn is None


def test_business_parry_bot_no_ataca_fuera_de_rango_ni_si_attack_es_false() -> None:
    """Fuera de attack_dist o con attack=False el bot no golpea."""
    assert ParryBot().act(make_obs(dist=250.0)).btn is None
    assert ParryBot(attack=False).act(make_obs(dist=50.0)).btn is None


def test_business_parry_bot_camina_hacia_el_jefe_si_esta_lejos() -> None:
    """Mas alla de approach_dist avanza (my=1); cerca se queda quieto."""
    assert ParryBot().act(make_obs(dist=300.0)).my == 1.0
    assert ParryBot().act(make_obs(dist=100.0)).my == 0.0


def test_business_parry_bot_no_camina_durante_un_ataque_parriable() -> None:
    """Si el jefe esta en un montage parriable el bot no avanza aunque este lejos."""
    bot = ParryBot()
    assert bot.act(make_obs(dist=300.0, name=UPPERCUT, pos=0.1)).my == 0.0


def test_business_parry_bot_sin_jefe_o_sin_jugador_solo_conserva_el_bloqueo() -> None:
    """Sin jefe o sin jugador no hay decision: solo se mantiene LB si falta soltarlo."""
    bot = ParryBot()
    assert bot.act({"t": 0.0, "boss": None, "player": {}}) == bot.act({"t": 0.0, "player": {}})
    bot.release_at = 5.0
    assert bot.act({"t": 1.0, "boss": {}, "player": None}).lb
    assert not bot.act({"t": 6.0, "boss": None}).lb


def test_business_parry_bot_montage_desconocido_no_dispara_parry() -> None:
    """Un montage fuera de PARRY_HITS (golpe no parriable) no provoca bloqueo."""
    bot = ParryBot(attack=False)
    bot.act(make_obs(t=1.0, name="Boss_Attack_Otro", pos=0.0))
    assert not bot.act(make_obs(t=1.1, name="Boss_Attack_Otro", pos=5.0)).lb


def test_business_parry_bot_cambio_de_montage_no_dispara_con_posicion_vieja() -> None:
    """Al cambiar de montage la posicion previa no es comparable y no se dispara."""
    bot = ParryBot(attack=False)
    bot.act(make_obs(t=1.0, name=COMBO, pos=0.0))
    assert not bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.80)).lb


def test_business_parry_bot_combo_con_varios_impactos_pulsa_una_vez_por_impacto() -> None:
    """El combo cargado tiene tres impactos: cada cruce de disparo cuenta una pulsacion."""
    bot = ParryBot(attack=False, hold=0.05)
    t = 0.0
    for pos in (0.0, 0.40, 0.50, 0.95, 1.00, 1.60, 1.70):
        bot.act(make_obs(t=t, name=COMBO, pos=pos))
        t += 0.1
    assert bot.parry_presses == 3


def test_business_parry_bot_reset_olvida_el_episodio_anterior() -> None:
    """reset limpia montage previo, temporizadores y contador."""
    bot = ParryBot(attack=False)
    bot.act(make_obs(t=1.0, name=UPPERCUT, pos=0.60))
    bot.act(make_obs(t=1.1, name=UPPERCUT, pos=0.70))
    bot.reset()
    assert (bot.prev_montage, bot.prev_pos, bot.release_at, bot.next_attack, bot.parry_presses) == (
        None,
        0.0,
        -1.0,
        0.0,
        0,
    )


def test_business_parry_bot_montage_ausente_o_sin_nombre_se_tolera() -> None:
    """Un jefe sin montage o con montage vacio no rompe la decision."""
    bot = ParryBot()
    assert bot.act({"t": 0.0, "player": {}, "dist": 100.0, "boss": {"montage": None}}).btn == "rb"
    assert bot.act({"t": 9.0, "player": {}, "dist": 100.0, "boss": {"montage": {}}}).btn == "rb"


def test_business_tabla_de_impactos_es_coherente() -> None:
    """Cada montage parriable tiene impactos positivos, ordenados y el disparo cae dentro del montage."""
    assert pytest.approx(0.235) == PARRY_LEAD
    for hits in PARRY_HITS.values():
        assert list(hits) == sorted(hits)
        assert all(h - PARRY_LEAD > 0 for h in hits)
