"""Reglas de negocio del PPO que parte de la politica clonada: promocion de checkpoints y correccion por regresion."""

from __future__ import annotations

import pytest

from dmrl.dominio import ppo_jefes as pj

REFERENCIA = {"juice": 0.8, "amlo": 0.7, "nahual": 0.9}


def evaluar(tasas: dict[str, float], mejor_media: float = 0.8, config: pj.ConfigPPOJefes | None = None
            ) -> pj.DecisionEvaluacion:
    """
    Evalua un checkpoint contra la referencia fija de estas pruebas.

    Contexto:
        Atajo de las reglas de promocion.

    Args:
        tasas (dict[str, float]): Tasa de victoria del checkpoint por jefe.
        mejor_media (float): Media del mejor modelo vigente.
        config (ConfigPPOJefes, opcional): Tolerancias; por defecto las de la configuracion estandar.

    Returns:
        DecisionEvaluacion: Veredicto.

    Raises:
        Ninguna.

    Ejemplo:
        evaluar({"juice": 0.9, "amlo": 0.7, "nahual": 0.9}).promover
    """
    return pj.evaluar_checkpoint(tasas, REFERENCIA, mejor_media, config or pj.ConfigPPOJefes())


def test_checkpoint_no_se_promueve_si_un_jefe_cae_mas_de_diez_puntos() -> None:
    decision = evaluar({"juice": 1.0, "amlo": 0.55, "nahual": 1.0}, mejor_media=0.5)
    assert decision.media > 0.5 and not decision.promover
    assert decision.jefes_que_caen == ("amlo",)


def test_checkpoint_se_promueve_si_mejora_la_media_sin_bajar_ningun_jefe() -> None:
    decision = evaluar({"juice": 0.85, "amlo": 0.75, "nahual": 0.95}, mejor_media=0.8)
    assert decision.promover and decision.jefes_que_caen == () and not decision.regresion


def test_checkpoint_no_se_promueve_si_la_media_no_supera_al_mejor_vigente() -> None:
    empate = evaluar({"juice": 0.8, "amlo": 0.7, "nahual": 0.9}, mejor_media=pj.tasa_media(REFERENCIA))
    assert not empate.promover and empate.jefes_que_caen == ()


def test_caida_exacta_de_diez_puntos_se_tolera() -> None:
    decision = evaluar({"juice": 0.7, "amlo": 0.7, "nahual": 0.9}, mejor_media=0.0)
    assert decision.jefes_que_caen == () and decision.promover


def test_caida_apenas_mayor_a_diez_puntos_no_se_tolera() -> None:
    decision = evaluar({"juice": 0.69, "amlo": 0.7, "nahual": 0.9}, mejor_media=0.0)
    assert decision.jefes_que_caen == ("juice",) and not decision.promover


def test_la_tolerancia_por_jefe_la_define_la_configuracion() -> None:
    estricta = pj.ConfigPPOJefes(caida_maxima_por_jefe=0.02)
    assert evaluar({"juice": 0.75, "amlo": 0.7, "nahual": 0.9}, 0.0, estricta).jefes_que_caen == ("juice",)


def test_mejorar_a_un_jefe_a_costa_de_otro_no_es_una_mejora() -> None:
    decision = evaluar({"juice": 1.0, "amlo": 0.5, "nahual": 1.0}, mejor_media=0.7)
    assert decision.media > 0.7 and not decision.promover


def test_una_caida_media_exacta_de_diez_puntos_no_cuenta_como_regresion() -> None:
    media = pj.tasa_media(REFERENCIA)
    en_el_limite = {j: t - 0.10 for j, t in REFERENCIA.items()}
    assert pj.tasa_media(en_el_limite) == pytest.approx(media - 0.10)
    assert not evaluar(en_el_limite, mejor_media=0.0).regresion


def test_una_caida_media_mayor_a_diez_puntos_cuenta_como_regresion() -> None:
    decision = evaluar({j: t - 0.11 for j, t in REFERENCIA.items()}, mejor_media=0.0)
    assert decision.regresion


def test_dos_regresiones_seguidas_disparan_la_correccion() -> None:
    control = pj.ControlRegresion(pj.ConfigPPOJefes())
    regresion = evaluar({j: 0.0 for j in REFERENCIA})
    assert regresion.regresion
    assert control.registrar(regresion) is False
    assert control.registrar(regresion) is True


def test_una_evaluacion_buena_entre_dos_malas_reinicia_el_conteo() -> None:
    control = pj.ControlRegresion(pj.ConfigPPOJefes())
    mala = evaluar({j: 0.0 for j in REFERENCIA})
    buena = evaluar(dict(REFERENCIA))
    assert [control.registrar(d) for d in (mala, buena, mala)] == [False, False, False]
    assert control.seguidas == 1


def test_tras_corregir_el_contador_empieza_de_nuevo() -> None:
    control = pj.ControlRegresion(pj.ConfigPPOJefes())
    mala = evaluar({j: 0.0 for j in REFERENCIA})
    resultados = [control.registrar(mala) for _ in range(4)]
    assert resultados == [False, True, False, True]


def test_el_numero_de_regresiones_para_corregir_es_configurable() -> None:
    control = pj.ControlRegresion(pj.ConfigPPOJefes(evaluaciones_para_revertir=3))
    mala = evaluar({j: 0.0 for j in REFERENCIA})
    assert [control.registrar(mala) for _ in range(3)] == [False, False, True]


def test_la_tasa_de_aprendizaje_corregida_se_reduce_por_el_factor() -> None:
    config = pj.ConfigPPOJefes(factor_reduccion_lr=0.5)
    assert pj.tasa_aprendizaje_corregida(1e-4, config) == pytest.approx(5e-5)


def test_la_tasa_de_aprendizaje_corregida_nunca_baja_del_minimo() -> None:
    config = pj.ConfigPPOJefes(lr_minimo=1e-5, factor_reduccion_lr=0.5)
    tasa = 1e-4
    for _ in range(20):
        tasa = pj.tasa_aprendizaje_corregida(tasa, config)
    assert tasa == config.lr_minimo


def test_el_coeficiente_kl_es_cero_en_calentamiento_y_decae_linealmente_a_cero() -> None:
    config = pj.ConfigPPOJefes(pasos_calentamiento=200, pasos_decaimiento_kl=800, beta_kl=0.3)
    assert all(pj.coeficiente_kl(p, config) == 0.0 for p in (0, 100, 199))
    valores = [pj.coeficiente_kl(200 + k * 100, config) for k in range(9)]
    assert valores[0] == pytest.approx(0.3) and valores[-1] == 0.0
    pasos = [a - b for a, b in zip(valores, valores[1:], strict=False)]
    assert all(p == pytest.approx(0.3 / 8) for p in pasos)


def test_estado_danado_se_reemplaza_por_el_de_defecto() -> None:
    por_defecto = pj.EstadoEvaluacion({"amlo": 0.8}, 0.8, learning_rate=1e-4)
    assert pj.EstadoEvaluacion.desde_dict({"referencia": "corrupto"}, por_defecto) is por_defecto
    assert pj.EstadoEvaluacion.desde_dict({}, por_defecto) is por_defecto


def test_reanudar_conserva_la_tasa_corregida_el_mejor_modelo_y_las_reversiones() -> None:
    guardado = pj.EstadoEvaluacion({"amlo": 0.8}, 0.85, mejor_paso=900_000, learning_rate=2.5e-5, seguidas=1,
                                   promociones=4, reversiones=2)
    por_defecto = pj.EstadoEvaluacion({"amlo": 0.8}, 0.8)
    restaurado = pj.EstadoEvaluacion.desde_dict(guardado.a_dict(), por_defecto)
    assert (restaurado.learning_rate, restaurado.mejor_paso, restaurado.reversiones) == (2.5e-5, 900_000, 2)
    assert restaurado.mejor_media == 0.85 and restaurado.promociones == 4


def test_metricas_de_ventana_publica_solo_los_ataques_mortales_mas_frecuentes() -> None:
    episodios: list[dict[str, object]] = (
        [{"muerte": True, "golpe_mortal": "rayo"}] * 5
        + [{"muerte": True, "golpe_mortal": "corte"}] * 3
        + [{"muerte": True, "golpe_mortal": "salto"}] * 1
    )
    m = pj.metricas_de_ventana(episodios, 1, maximo_ataques=2)
    assert m["muerte_por/rayo"] == pytest.approx(5 / 9) and m["muerte_por/corte"] == pytest.approx(3 / 9)
    assert "muerte_por/salto" not in m


def test_los_nombres_de_ataque_se_limpian_para_ser_claves_validas() -> None:
    m = pj.metricas_de_ventana([{"muerte": True, "golpe_mortal": "BP Boss/Slam-01"}], 1)
    assert "muerte_por/BP_Boss_Slam_01" in m
