"""Reglas de negocio de la recompensa y del final de episodio."""

from __future__ import annotations

import pytest

from dmrl.dominio.configuracion import ConfigRecompensa
from dmrl.dominio.recompensa import (
    RESULTADO_MUERTE,
    RESULTADO_VICTORIA,
    CalculadoraRecompensa,
    fase_de,
    resultado_final,
)
from tests.support_dmrl import jefe, jugador, obs

CFG = ConfigRecompensa()


def calculadora(**jug: object) -> CalculadoraRecompensa:
    """
    Calculadora iniciada con un jugador dado.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        **jug (object): Campos del jugador inicial.

    Returns:
        CalculadoraRecompensa: Calculadora en el punto de partida del episodio.

    Raises:
        Ninguna.

    Ejemplo:
        calculadora(hp=40.0)
    """
    c = CalculadoraRecompensa(CFG)
    c.reiniciar(obs(jugador(**jug)))
    return c


def test_cada_paso_cuesta_la_presion_de_tiempo() -> None:
    """Sin nada que premiar o castigar la recompensa es -0.001."""
    assert calculadora().calcular(obs()).recompensa == pytest.approx(-0.001)


def test_danio_al_jefe_se_premia_con_un_centesimo_por_punto() -> None:
    """40 de dano al jefe suman 0.4."""
    r = calculadora().calcular(obs(dmg_dealt=40.0))
    assert r.terminos["danio_hecho"] == pytest.approx(0.4)


def test_danio_recibido_pesa_el_doble_que_el_hecho() -> None:
    """Perder 50 de vida resta 1.0; hacer 50 de dano suma 0.5."""
    c = calculadora()
    perdido = c.calcular(obs(jugador(hp=50.0))).terminos["danio_recibido"]
    assert perdido == pytest.approx(-1.0)
    assert CalculadoraRecompensa(CFG).config.danio_hecho * 50 == pytest.approx(0.5)


def test_un_golpe_parado_no_cuenta_como_danio_recibido() -> None:
    """Si la vida no baja (parry, bloqueo o roll) no hay castigo aunque el golpe se intentara."""
    r = calculadora().calcular(obs(jugador(hp=100.0), dmg_taken=25.0, hits_taken=1))
    assert "danio_recibido" not in r.terminos


def test_el_dano_mortal_no_cuenta_mas_alla_de_la_vida_restante() -> None:
    """Con 10 de vida y un golpe de 40 solo se castigan los 10 que quedaban."""
    c = calculadora(hp=10.0)
    r = c.calcular(obs(jugador(hp=-30.0, dead=True), player_death=True))
    assert r.terminos["danio_recibido"] == pytest.approx(-0.2)
    assert c.estadisticas.danio_recibido == pytest.approx(10.0)


def test_parry_exitoso_suma_medio_punto() -> None:
    """Cada contraataque de parry premia 0.5."""
    r = calculadora().calcular(obs(parries=2))
    assert r.terminos["parry"] == pytest.approx(1.0)


def test_curacion_se_premia_si_la_vida_previa_era_menor_de_60() -> None:
    """Curar de 40 a 90 premia 0.5 (proporcional a lo recuperado)."""
    r = calculadora(hp=40.0).calcular(obs(jugador(hp=90.0)))
    assert r.terminos["curacion"] == pytest.approx(0.5)


def test_curacion_no_se_premia_con_la_vida_previa_en_60_o_mas() -> None:
    """Beber con 70 de vida no da recompensa: evita dejarse golpear para curar."""
    c = calculadora(hp=70.0)
    r = c.calcular(obs(jugador(hp=100.0)))
    assert "curacion" not in r.terminos
    assert c.estadisticas.curado == pytest.approx(30.0)


def test_curacion_es_proporcional_a_la_vida_efectivamente_recuperada() -> None:
    """De 55 a 100 se recuperan 45, no 50."""
    r = calculadora(hp=55.0).calcular(obs(jugador(hp=100.0)))
    assert r.terminos["curacion"] == pytest.approx(0.45)


def test_bebida_completada_cuenta_como_caguama_usada() -> None:
    """Si la caguama se gasto antes de terminar el montage, la bebida fue completa."""
    c = calculadora(hp=40.0)
    c.calcular(obs(jugador(hp=40.0, drinking=True, caguamas=3)))
    c.calcular(obs(jugador(hp=90.0, drinking=True, caguamas=2)))
    r = c.calcular(obs(jugador(hp=90.0, drinking=False, caguamas=2)))
    assert c.estadisticas.caguamas_usadas == 1
    assert c.estadisticas.caguamas_interrumpidas == 0
    assert "bebida_interrumpida" not in r.terminos


def test_bebida_interrumpida_sin_gastar_caguama_se_castiga_poco() -> None:
    """Un golpe antes del notify cancela la bebida sin gastar la caguama: castigo de 0.05."""
    c = calculadora(hp=60.0)
    c.calcular(obs(jugador(hp=60.0, drinking=True, caguamas=3)))
    r = c.calcular(obs(jugador(hp=35.0, drinking=False, caguamas=3)))
    assert r.terminos["bebida_interrumpida"] == pytest.approx(-0.05)
    assert c.estadisticas.caguamas_interrumpidas == 1


def test_pasar_a_fase_2_da_dos_puntos_una_sola_vez() -> None:
    """El premio se da en la transicion y no se repite mientras sigue la fase 2."""
    c = calculadora()
    fase2 = obs(jf=jefe(phase=1, cls="BP_Enemy_JuicePhase2_C"))
    assert c.calcular(fase2).terminos["fase2"] == pytest.approx(2.0)
    assert "fase2" not in c.calcular(fase2).terminos
    assert c.estadisticas.fase_max == 2


def test_empezar_en_fase_2_no_da_el_premio_de_transicion() -> None:
    """Un episodio que arranca en la fase 2 no cobra el premio por llegar a ella."""
    c = CalculadoraRecompensa(CFG)
    inicial = obs(jf=jefe(phase=1, cls="BP_Enemy_JuicePhase2_C"))
    c.reiniciar(inicial)
    assert "fase2" not in c.calcular(inicial).terminos


def test_victoria_da_diez_puntos() -> None:
    """Matar al jefe de fase 2 suma +10."""
    r = calculadora().calcular(obs(victory=True))
    assert r.terminos["victoria"] == pytest.approx(10.0)


def test_muerte_resta_cinco_puntos() -> None:
    """Morir resta 5 ademas del dano recibido."""
    r = calculadora().calcular(obs(jugador(dead=True, hp=0.0), player_death=True))
    assert r.terminos["muerte"] == pytest.approx(-5.0)


def test_resultado_final_prefiere_muerte_si_coinciden() -> None:
    """Si el jugador muere en el mismo paso en que cae el jefe se cuenta la muerte."""
    assert resultado_final(obs(jugador(dead=True), victory=True)) == RESULTADO_MUERTE
    assert resultado_final(obs(victory=True)) == RESULTADO_VICTORIA


def test_resultado_final_detecta_el_jefe_de_fase_2_muerto_sin_evento() -> None:
    """Si falta el evento pero el jefe de fase 2 aparece muerto, es victoria."""
    assert resultado_final(obs(jf=jefe(cls="BP_Enemy_JuicePhase2_C", dead=True))) == RESULTADO_VICTORIA


def test_resultado_final_none_mientras_la_pelea_sigue() -> None:
    """Sin muerte ni victoria no hay final."""
    assert resultado_final(obs()) is None
    assert resultado_final({}) is None


def test_fase_de_usa_el_valor_por_defecto_sin_jefe() -> None:
    """Sin bloque de jefe se conserva la fase por defecto."""
    assert fase_de({}, 2) == 2
    assert fase_de(obs(jf=jefe(phase=1))) == 2
    assert fase_de(obs()) == 1


def test_estadisticas_acumulan_dano_hecho_y_pasos() -> None:
    """Las estadisticas del episodio suman lo que se premio y castigo."""
    c = calculadora()
    c.calcular(obs(dmg_dealt=30.0))
    c.calcular(obs(dmg_dealt=20.0))
    assert c.estadisticas.danio_hecho == pytest.approx(50.0)
    assert c.estadisticas.pasos == 2


def test_observacion_sin_jugador_conserva_la_vida_anterior() -> None:
    """Si el bloque del jugador falta (carga de mapa) no se inventa dano ni curacion."""
    c = calculadora(hp=80.0)
    r = c.calcular({"ev": {}})
    assert set(r.terminos) == {"paso"}


def test_reiniciar_a_mitad_de_una_bebida_toma_la_caguama_actual() -> None:
    """Reiniciar con la bebida en curso no cuenta una bebida falsa al terminar."""
    c = CalculadoraRecompensa(CFG)
    c.reiniciar(obs(jugador(drinking=True, caguamas=2)))
    c.calcular(obs(jugador(drinking=False, caguamas=2)))
    assert c.estadisticas.caguamas_interrumpidas == 1
