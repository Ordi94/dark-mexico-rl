"""Reglas de negocio del bot de reglas de referencia."""

from __future__ import annotations

from dmrl.dominio import acciones as acc
from dmrl.dominio.bot_reglas import BotReglas
from dmrl.dominio.tabla_golpes import Golpe, TablaGolpes
from tests.support_dmrl import jefe, jugador, montage, obs

PARABLE = "Boss_Attack_Uppercut_InP_Montage"
NO_PARABLE = "Anim_Sword_Air_Attack1_InPlace_Montage"
TABLA = TablaGolpes({
    PARABLE: (Golpe(0.9, True, 0.9, 0.9, 10.0),),
    NO_PARABLE: (Golpe(0.55, False, 0.55, 0.55, 35.0),),
    "Combo": (Golpe(0.5, True, 0.5, 0.5, 25.0), Golpe(1.0, True, 1.0, 1.0, 25.0)),
})


def decidir(bot: BotReglas, jug: dict[str, object] | None = None, jf: dict[str, object] | None = None,
            dist: float = 150.0) -> int:
    """
    Una decision del bot con la mascara que corresponde al jugador.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        bot (BotReglas): Bot bajo prueba.
        jug (dict, opcional): Bloque del jugador.
        jf (dict, opcional): Bloque del jefe.
        dist (float): Distancia al jefe.

    Returns:
        int: Accion elegida.

    Raises:
        Ninguna.

    Ejemplo:
        decidir(BotReglas(TABLA))
    """
    o = obs(jug if jug is not None else jugador(), jf if jf is not None else jefe(), dist)
    return bot.decidir(o, acc.mascara_acciones(o["player"]))


def jefe_en(nombre: str, pos: float, rate: float = 1.0, **extra: object) -> dict[str, object]:
    """
    Bloque del jefe con un montage en una posicion.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        nombre (str): Montage del jefe.
        pos (float): Posicion en segundos.
        rate (float): Play rate.
        **extra (object): Campos extra del jefe.

    Returns:
        dict[str, object]: Jefe atacando con ese montage.

    Raises:
        Ninguna.

    Ejemplo:
        jefe_en(PARABLE, 0.6)
    """
    return jefe(montage=montage(nombre, pos, rate, 4.0), attacking=True, **extra)


def test_hace_parry_cuando_faltan_menos_de_0_26_segundos_a_un_golpe_parable() -> None:
    """Golpe a 0.9 con el montage en 0.7: faltan 0.2 s, dentro del disparo de parry."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY


def test_espera_sin_atacar_mientras_el_golpe_parable_aun_esta_lejos() -> None:
    """A 0.4 s del golpe aun no pulsa LB, pero tampoco se compromete con un ataque."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(PARABLE, 0.5)) == acc.ACCION_NADA


def test_el_play_rate_acorta_el_tiempo_hasta_el_golpe() -> None:
    """Con rate 2.0 los 0.4 s de montage son 0.2 s de juego: ya toca el parry."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(PARABLE, 0.5, rate=2.0)) == acc.ACCION_PARRY


def test_no_repite_el_parry_para_el_mismo_golpe() -> None:
    """Tras pulsar LB para un golpe no vuelve a pulsarlo en el paso siguiente."""
    bot = BotReglas(TABLA)
    assert decidir(bot, jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY
    assert decidir(bot, jf=jefe_en(PARABLE, 0.77)) == acc.ACCION_NADA


def test_atiende_cada_golpe_de_un_combo_por_separado() -> None:
    """El segundo golpe de un montage tiene su propio parry."""
    bot = BotReglas(TABLA)
    assert decidir(bot, jf=jefe_en("Combo", 0.3)) == acc.ACCION_PARRY
    assert decidir(bot, jf=jefe_en("Combo", 0.45)) == acc.ACCION_NADA
    assert decidir(bot, jf=jefe_en("Combo", 0.8)) == acc.ACCION_PARRY


def test_un_montage_que_reinicia_vuelve_a_tener_golpes_por_atender() -> None:
    """Si la posicion retrocede es un ataque nuevo del mismo montage."""
    bot = BotReglas(TABLA)
    assert decidir(bot, jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY
    assert decidir(bot, jf=jefe_en(PARABLE, 0.1)) == acc.ACCION_NADA
    assert decidir(bot, jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY


def test_rueda_ante_un_golpe_no_parable_inminente() -> None:
    """Golpe no parable a 0.55 con el montage en 0.4: faltan 0.15 s, momento del roll lateral."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(NO_PARABLE, 0.4)) == acc.ACCION_ROLL_PRIMERA + 2


def test_no_rueda_demasiado_pronto_porque_la_invulnerabilidad_ya_habria_terminado() -> None:
    """A 0.3 s del golpe (> 0.20) todavia espera: rodar antes lo deja sin invulnerabilidad."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(NO_PARABLE, 0.25)) == acc.ACCION_NADA


def test_sin_estamina_no_puede_rodar_y_espera() -> None:
    """Con Resistencia 0 el roll no esta en la mascara: el bot no lo intenta."""
    assert decidir(BotReglas(TABLA), jugador(stamina=0.0), jefe_en(NO_PARABLE, 0.4)) == acc.ACCION_NADA


def test_contraataca_con_especial_si_el_jefe_esta_aturdido_y_hay_barra() -> None:
    """Aturdido y cerca: especial."""
    assert decidir(BotReglas(TABLA), jf=jefe(stunned=True)) == acc.ACCION_ESPECIAL


def test_contraataca_con_ataque_si_no_hay_barra() -> None:
    """Aturdido, cerca y sin barra llena: ataque normal."""
    j = jugador(power=0.3, power_ready=False)
    assert decidir(BotReglas(TABLA), j, jefe(stunned=True)) == acc.ACCION_ATAQUE


def test_se_acerca_al_jefe_aturdido_si_esta_lejos() -> None:
    """Aturdido pero a 400 uu: primero se acerca."""
    assert decidir(BotReglas(TABLA), jf=jefe(stunned=True), dist=400.0) == acc.ACCION_MOVER_PRIMERA


def test_bebe_con_el_jefe_aturdido_aunque_este_cerca() -> None:
    """El aturdimiento (~2.3 s) dura mas que la bebida (2.15 s): beber es seguro."""
    assert decidir(BotReglas(TABLA), jugador(hp=30.0), jefe(stunned=True), dist=100.0) == acc.ACCION_CAGUAMA


def test_bebe_con_vida_baja_sin_golpes_pendientes_y_lejos() -> None:
    """Vida < 50, sin montage del jefe y a 600 uu: bebe."""
    assert decidir(BotReglas(TABLA), jugador(hp=40.0), dist=600.0) == acc.ACCION_CAGUAMA


def test_no_bebe_si_hay_un_golpe_pendiente() -> None:
    """Medido: beber con un golpe pendiente en el montage del jefe se interrumpe siempre."""
    accion = decidir(BotReglas(TABLA), jugador(hp=40.0), jefe_en("Combo", 0.0), dist=600.0)
    assert accion != acc.ACCION_CAGUAMA


def test_no_bebe_si_el_jefe_esta_a_menos_de_450_y_se_aleja_para_crear_distancia() -> None:
    """Con vida baja y el jefe a 300 uu retrocede (direccion 180 grados)."""
    assert decidir(BotReglas(TABLA), jugador(hp=40.0), dist=300.0) == acc.ACCION_MOVER_PRIMERA + 4


def test_no_bebe_con_vida_de_50_o_mas() -> None:
    """El umbral es estrictamente menor que 50."""
    assert decidir(BotReglas(TABLA), jugador(hp=50.0), dist=600.0) != acc.ACCION_CAGUAMA


def test_no_bebe_sin_caguamas() -> None:
    """Sin caguamas no intenta curarse ni se aleja por ello."""
    assert decidir(BotReglas(TABLA), jugador(hp=10.0, caguamas=0), dist=100.0) == acc.ACCION_ESPECIAL


def test_mientras_bebe_se_aleja() -> None:
    """Durante la bebida solo se puede caminar: retrocede."""
    j = jugador(hp=40.0, drinking=True, montage=montage("MM_Idle1_Montage", 0.5))
    assert decidir(BotReglas(TABLA), j) == acc.ACCION_MOVER_PRIMERA + 4


def test_ataca_en_ventana_segura_con_reserva_de_estamina() -> None:
    """Sin golpes pendientes y cerca ataca (especial si hay barra)."""
    assert decidir(BotReglas(TABLA)) == acc.ACCION_ESPECIAL
    assert decidir(BotReglas(TABLA), jugador(power=0.0, power_ready=False)) == acc.ACCION_ATAQUE


def test_no_ataca_si_la_estamina_esta_por_debajo_de_la_reserva_del_roll() -> None:
    """Con Resistencia 0.2 guarda lo que queda para esquivar."""
    assert decidir(BotReglas(TABLA), jugador(stamina=0.2)) == acc.ACCION_NADA


def test_no_ataca_si_el_proximo_golpe_esta_a_menos_de_una_ventana_segura() -> None:
    """Un golpe a 0.8 s hace demasiado arriesgado empezar un combo."""
    assert decidir(BotReglas(TABLA), jf=jefe_en(PARABLE, 0.1)) == acc.ACCION_NADA


def test_se_acerca_al_jefe_si_esta_lejos_y_no_hay_golpes() -> None:
    """A 800 uu camina hacia el jefe."""
    assert decidir(BotReglas(TABLA), dist=800.0) == acc.ACCION_MOVER_PRIMERA


def test_no_hace_nada_si_el_jugador_esta_muerto_o_falta_el_jefe() -> None:
    """Sin jugador vivo o sin jefe el bot no actua."""
    bot = BotReglas(TABLA)
    assert decidir(bot, jugador(dead=True)) == acc.ACCION_NADA
    assert bot.decidir({"player": jugador(), "ev": {}}, acc.mascara_acciones(jugador())) == acc.ACCION_NADA
    assert bot.decidir({}, [True] * acc.N_ACCIONES) == acc.ACCION_NADA


def test_espera_si_no_puede_atacar_en_ventana_segura() -> None:
    """Sin ataque ni especial validos en la mascara, espera."""
    assert decidir(BotReglas(TABLA), jugador(can_attack=False)) == acc.ACCION_NADA


def test_parry_imposible_cae_en_espera() -> None:
    """Durante una reaccion fuerte la mascara no ofrece LB y el bot espera el golpe."""
    j = jugador(montage=montage(acc.MONTAGE_REACCION_FUERTE, 0.1))
    assert decidir(BotReglas(TABLA), j, jefe_en(PARABLE, 0.7)) == acc.ACCION_NADA


def test_contraataque_sin_ninguna_opcion_valida_espera() -> None:
    """Aturdido y cerca pero sin ataque valido: espera."""
    j = jugador(can_attack=False)
    assert decidir(BotReglas(TABLA), j, jefe(stunned=True)) == acc.ACCION_NADA


def test_reiniciar_olvida_los_golpes_atendidos() -> None:
    """Tras reiniciar el episodio el bot vuelve a atender el mismo golpe."""
    bot = BotReglas(TABLA)
    assert decidir(bot, jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY
    bot.reiniciar()
    assert decidir(bot, jf=jefe_en(PARABLE, 0.7)) == acc.ACCION_PARRY
