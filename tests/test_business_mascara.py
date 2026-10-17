"""Reglas de negocio de la mascara de acciones y de la traduccion accion -> orden del juego."""

from __future__ import annotations

import pytest

from dmrl.dominio import acciones as acc
from tests.support_dmrl import jugador, montage


def validas(**cambios: object) -> list[bool]:
    """
    Mascara para un jugador en reposo con los cambios dados.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        **cambios (object): Campos del jugador a sobrescribir.

    Returns:
        list[bool]: Mascara de 22 valores.

    Raises:
        Ninguna.

    Ejemplo:
        validas(hp=40)
    """
    return acc.mascara_acciones(jugador(**cambios))


def test_caguama_invalida_con_vida_llena() -> None:
    """Con la vida al maximo el juego ignora la bebida: no se ofrece."""
    assert not validas()[acc.ACCION_CAGUAMA]


def test_caguama_valida_con_vida_baja_y_caguamas() -> None:
    """Con vida baja, caguamas y Can Heal la bebida arranca."""
    assert validas(hp=40.0)[acc.ACCION_CAGUAMA]


def test_caguama_invalida_sin_caguamas() -> None:
    """Sin caguamas la bebida no hace nada."""
    assert not validas(hp=40.0, caguamas=0)[acc.ACCION_CAGUAMA]


def test_caguama_invalida_si_ya_esta_bebiendo() -> None:
    """Durante la bebida solo se puede mover: ni otra caguama ni nada de combate."""
    mascara = validas(hp=40.0, drinking=True, montage=montage("MM_Idle1_Montage", 0.5))
    assert not mascara[acc.ACCION_CAGUAMA]
    assert mascara[acc.ACCION_NADA]
    assert all(mascara[acc.ACCION_MOVER_PRIMERA:acc.ACCION_ROLL_PRIMERA])
    assert not any(mascara[acc.ACCION_ROLL_PRIMERA:])


def test_caguama_invalida_en_reaccion_fuerte_por_can_heal_falso() -> None:
    """En una reaccion de golpe fuerte Can Heal es falso y la bebida se ignora."""
    assert not validas(hp=40.0, can_heal=False)[acc.ACCION_CAGUAMA]


def test_especial_invalido_sin_barra_llena() -> None:
    """El especial exige la barra llena."""
    assert not validas(power=0.5, power_ready=False)[acc.ACCION_ESPECIAL]
    assert validas()[acc.ACCION_ESPECIAL]


def test_especial_invalido_con_barra_en_0_9999999999999999() -> None:
    """Diez golpes dan 0.9999999999999999: el juego lo rechaza aunque el JSON lo muestre como 1."""
    assert not validas(power=1.0, power_ready=False)[acc.ACCION_ESPECIAL]


def test_especial_usa_el_valor_numerico_si_no_hay_booleano() -> None:
    """Sin power_ready (puente antiguo) se compara el valor con tolerancia minima."""
    base = jugador(power=1.0)
    del base["power_ready"]
    assert acc.mascara_acciones(base)[acc.ACCION_ESPECIAL]
    base["power"] = 0.5
    assert not acc.mascara_acciones(base)[acc.ACCION_ESPECIAL]


def test_especial_invalido_durante_su_propio_montage() -> None:
    """Repetir el especial sobre su montage no tiene efecto (medido 0 de 307)."""
    mascara = validas(montage=montage(acc.MONTAGE_ESPECIAL, 0.3))
    assert not mascara[acc.ACCION_ESPECIAL]


def test_nada_siempre_es_valida_aunque_el_jugador_este_muerto() -> None:
    """La mascara nunca queda vacia; muerto solo queda "nada"."""
    mascara = validas(dead=True)
    assert mascara[acc.ACCION_NADA]
    assert sum(mascara) == 1
    assert acc.mascara_acciones(None) == [True] + [False] * (acc.N_ACCIONES - 1)


def test_sin_estamina_no_se_ofrece_roll_ni_ataque_ni_especial() -> None:
    """Con Resistencia en cero el juego ignora roll y ataques (medido 3% de efecto)."""
    mascara = validas(stamina=0.0)
    assert not any(mascara[acc.ACCION_ROLL_PRIMERA:acc.ACCION_ESPECIAL + 1])
    assert mascara[acc.ACCION_PARRY] and mascara[acc.ACCION_BLOQUEO]


def test_roll_invalido_si_el_juego_marca_can_roll_falso() -> None:
    """CanRoll falso bloquea las ocho direcciones de la maniobra."""
    mascara = validas(can_roll=False)
    assert not any(mascara[acc.ACCION_ROLL_PRIMERA:acc.ACCION_ATAQUE])


def test_roll_invalido_mientras_dura_otro_roll() -> None:
    """Rodar sobre un roll en curso no hace nada."""
    mascara = validas(montage=montage(acc.MONTAGE_ROLL, 0.2))
    assert not any(mascara[acc.ACCION_ROLL_PRIMERA:acc.ACCION_ATAQUE])


def test_roll_valido_cancelando_un_ataque_o_un_bloqueo() -> None:
    """El roll interrumpe los montages de ataque y bloqueo (efecto medido 100%)."""
    for nombre in (acc.MONTAGES_ATAQUE[0], "NinjaBlockIntro_Montage1"):
        assert all(validas(montage=montage(nombre, 0.2))[acc.ACCION_ROLL_PRIMERA:acc.ACCION_ATAQUE])


def test_ataque_invalido_al_inicio_de_un_combo_y_valido_pasada_la_ventana() -> None:
    """Dentro de un montage de ataque el siguiente golpe entra solo pasado el 25% del montage."""
    nombre = acc.MONTAGES_ATAQUE[0]
    assert not validas(montage=montage(nombre, 0.2, largo=2.0))[acc.ACCION_ATAQUE]
    assert validas(montage=montage(nombre, 1.0, largo=2.0))[acc.ACCION_ATAQUE]


def test_ataque_valido_sin_montage_y_con_largo_desconocido() -> None:
    """Sin montage (o con largo cero) la fraccion se toma como completa."""
    assert validas()[acc.ACCION_ATAQUE]
    assert validas(montage=montage(acc.MONTAGES_ATAQUE[0], 0.0, largo=0.0))[acc.ACCION_ATAQUE]


def test_ataque_invalido_si_can_attack_es_falso() -> None:
    """CanAttack falso bloquea ataque y especial."""
    mascara = validas(can_attack=False)
    assert not mascara[acc.ACCION_ATAQUE] and not mascara[acc.ACCION_ESPECIAL]


def test_parry_y_bloqueo_invalidos_en_reaccion_fuerte_y_en_el_especial() -> None:
    """Durante Hurt_Big y el especial el juego ignora LB."""
    for nombre in (acc.MONTAGE_REACCION_FUERTE, acc.MONTAGE_ESPECIAL):
        mascara = validas(montage=montage(nombre, 0.1))
        assert not mascara[acc.ACCION_PARRY] and not mascara[acc.ACCION_BLOQUEO]


def test_parry_valido_repetido_sobre_su_propio_montage() -> None:
    """Pulsar LB otra vez durante el bloqueo reinicia la ventana de parry: sigue siendo valido."""
    mascara = validas(montage=montage("NinjaBlockIntro_Montage1", 0.4))
    assert mascara[acc.ACCION_PARRY] and mascara[acc.ACCION_BLOQUEO]


@pytest.mark.parametrize("accion", range(acc.N_ACCIONES))
def test_toda_accion_tiene_nombre_y_orden_valida(accion: int) -> None:
    """Las 22 acciones se traducen a una orden y tienen nombre."""
    assert isinstance(acc.comando_de_accion(accion), acc.ComandoJuego)
    assert len(acc.NOMBRES_ACCIONES) == acc.N_ACCIONES


def test_movimiento_cero_grados_va_hacia_el_jefe_y_noventa_a_su_derecha() -> None:
    """Direccion 0 = hacia el jefe (my=1); 90 grados = a su derecha (mx=1)."""
    adelante = acc.comando_de_accion(acc.ACCION_MOVER_PRIMERA)
    derecha = acc.comando_de_accion(acc.ACCION_MOVER_PRIMERA + 2)
    atras = acc.comando_de_accion(acc.ACCION_MOVER_PRIMERA + 4)
    assert (adelante.mx, adelante.my) == (0.0, 1.0)
    assert (derecha.mx, derecha.my) == (1.0, 0.0)
    assert (atras.mx, atras.my) == (0.0, -1.0)


def test_roll_lleva_el_boton_b_y_la_direccion_del_movimiento() -> None:
    """La maniobra es el boton B con la misma direccion que el movimiento correspondiente."""
    orden = acc.comando_de_accion(acc.ACCION_ROLL_PRIMERA + 2)
    assert orden.btn == "b" and (orden.mx, orden.my) == (1.0, 0.0)


def test_parry_es_una_pulsacion_corta_y_bloqueo_es_sostenido() -> None:
    """Dos parries seguidos alternan pulsar/soltar; el bloqueo mantiene LB."""
    assert acc.comando_de_accion(acc.ACCION_PARRY, acc.ACCION_NADA).lb
    assert not acc.comando_de_accion(acc.ACCION_PARRY, acc.ACCION_PARRY).lb
    assert acc.comando_de_accion(acc.ACCION_BLOQUEO, acc.ACCION_BLOQUEO).lb


def test_botones_de_ataque_especial_y_caguama() -> None:
    """Ataque = RB, especial = Y, caguama = X."""
    assert acc.comando_de_accion(acc.ACCION_ATAQUE).btn == "rb"
    assert acc.comando_de_accion(acc.ACCION_ESPECIAL).btn == "y"
    assert acc.comando_de_accion(acc.ACCION_CAGUAMA).btn == "x"


def test_acciones_y_direcciones_fuera_de_rango_se_rechazan() -> None:
    """Los indices invalidos fallan en lugar de traducirse a otra cosa."""
    with pytest.raises(ValueError):
        acc.comando_de_accion(acc.N_ACCIONES)
    with pytest.raises(ValueError):
        acc.comando_de_accion(-1)
    with pytest.raises(ValueError):
        acc.vector_direccion(8)
