"""Configuracion, calendario KL, compuerta de promocion, estado persistente y metricas del PPO multi-jefe."""

from __future__ import annotations

from typing import Any

import pytest

from dmrl.dominio import ppo_jefes as pj


@pytest.mark.parametrize("cambios", [
    {"learning_rate": 0.0},
    {"lr_minimo": 0.0},
    {"learning_rate": 1e-5, "lr_minimo": 1e-4},
    {"factor_reduccion_lr": 0.0},
    {"factor_reduccion_lr": 1.0},
    {"beta_kl": -0.1},
    {"pasos_calentamiento": -1},
    {"pasos_decaimiento_kl": -1},
    {"caida_maxima_por_jefe": -0.01},
    {"caida_maxima_por_jefe": 1.01},
    {"caida_clara_media": 1.5},
    {"caida_clara_media": -0.5},
    {"evaluaciones_para_revertir": 0},
    {"episodios_evaluacion": 0},
    {"minutos_evaluacion": 0.0},
    {"minutos_primera_evaluacion": 0.0},
])
def test_la_configuracion_rechaza_cada_valor_absurdo(cambios: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        pj.ConfigPPOJefes(**cambios)


def test_la_configuracion_por_defecto_es_valida_y_admite_los_limites_cerrados() -> None:
    pj.ConfigPPOJefes()
    pj.ConfigPPOJefes(caida_maxima_por_jefe=0.0, caida_clara_media=1.0, beta_kl=0.0, pasos_calentamiento=0,
                      pasos_decaimiento_kl=0, learning_rate=1e-4, lr_minimo=1e-4)


def test_en_calentamiento_solo_mientras_no_se_alcanzan_los_pasos_de_calentamiento() -> None:
    config = pj.ConfigPPOJefes(pasos_calentamiento=100)
    assert pj.en_calentamiento(99, config) and not pj.en_calentamiento(100, config)


def test_coeficiente_kl_en_los_puntos_clave_del_calendario() -> None:
    config = pj.ConfigPPOJefes(pasos_calentamiento=100, pasos_decaimiento_kl=1000, beta_kl=0.4)
    assert pj.coeficiente_kl(99, config) == 0.0
    assert pj.coeficiente_kl(100, config) == pytest.approx(0.4)
    assert pj.coeficiente_kl(600, config) == pytest.approx(0.2)
    assert pj.coeficiente_kl(1100, config) == 0.0
    assert pj.coeficiente_kl(10_000, config) == 0.0
    sin_decaimiento = pj.ConfigPPOJefes(pasos_calentamiento=0, pasos_decaimiento_kl=0)
    assert pj.coeficiente_kl(5, sin_decaimiento) == 0.0


def test_tasa_media_pesa_igual_a_cada_jefe_y_exige_al_menos_uno() -> None:
    assert pj.tasa_media({"amlo": 0.9, "nahual": 1.0}) == pytest.approx(0.95)
    with pytest.raises(ValueError, match="tasas"):
        pj.tasa_media({})


def test_evaluar_checkpoint_exige_todos_los_jefes_de_la_referencia() -> None:
    with pytest.raises(ValueError, match="nahual"):
        pj.evaluar_checkpoint({"amlo": 0.9}, {"amlo": 0.9, "nahual": 0.8}, 0.5, pj.ConfigPPOJefes())


def test_evaluar_checkpoint_ignora_jefes_que_no_estan_en_la_referencia() -> None:
    decision = pj.evaluar_checkpoint({"amlo": 0.9, "extra": 0.0}, {"amlo": 0.9}, 0.5, pj.ConfigPPOJefes())
    assert decision.media == pytest.approx(0.9) and set(decision.caidas) == {"amlo"}


def test_el_motivo_del_veredicto_describe_la_causa() -> None:
    config = pj.ConfigPPOJefes()
    cae = pj.evaluar_checkpoint({"a": 0.5, "b": 1.0}, {"a": 0.9, "b": 0.9}, 0.0, config)
    assert "cae mas de 10%" in cae.motivo and "a" in cae.motivo
    igual = pj.evaluar_checkpoint({"a": 0.9}, {"a": 0.9}, 0.9, config)
    assert "no supera" in igual.motivo
    mejor = pj.evaluar_checkpoint({"a": 0.95}, {"a": 0.9}, 0.9, config)
    assert "mejora la media" in mejor.motivo


def test_control_regresion_registra_el_historial_en_orden() -> None:
    control = pj.ControlRegresion(pj.ConfigPPOJefes(evaluaciones_para_revertir=3))
    malo = pj.evaluar_checkpoint({"a": 0.5}, {"a": 0.9}, 0.9, control.config)
    bueno = pj.evaluar_checkpoint({"a": 0.9}, {"a": 0.9}, 0.9, control.config)
    for decision in (malo, bueno, malo):
        control.registrar(decision)
    assert control.historial == [True, False, True] and control.seguidas == 1


def test_estado_a_dict_y_desde_dict_son_inversos() -> None:
    estado = pj.EstadoEvaluacion({"amlo": 0.9}, 0.9, mejor_paso=5, learning_rate=5e-5, seguidas=1, promociones=2,
                                 reversiones=3, evaluaciones=[{"pasos": 5}])
    base = pj.EstadoEvaluacion({"otro": 0.1}, 0.1)
    assert pj.EstadoEvaluacion.desde_dict(estado.a_dict(), base) == estado


def test_estado_incompleto_conserva_lo_leido_y_usa_defecto_para_lo_demas() -> None:
    base = pj.EstadoEvaluacion({"otro": 0.1}, 0.1, learning_rate=3e-5)
    leido = pj.EstadoEvaluacion.desde_dict({"referencia": {"amlo": "0.8"}, "mejor_media": "0.8"}, base)
    assert leido.referencia == {"amlo": 0.8} and leido.mejor_media == 0.8
    assert leido.learning_rate == 3e-5 and leido.mejor_paso == 0 and leido.evaluaciones == []


@pytest.mark.parametrize("datos", [
    {},
    {"referencia": {"amlo": 0.9}},
    {"referencia": {"amlo": "no es numero"}, "mejor_media": 0.5},
    {"referencia": 5, "mejor_media": 0.5},
    {"referencia": {"amlo": 0.9}, "mejor_media": 0.5, "seguidas": "x"},
])
def test_estado_danado_se_reemplaza_por_el_de_defecto(datos: dict[str, object]) -> None:
    base = pj.EstadoEvaluacion({"otro": 0.1}, 0.1)
    assert pj.EstadoEvaluacion.desde_dict(datos, base) is base


def test_metricas_de_ventana_sin_episodios_esta_vacia() -> None:
    assert pj.metricas_de_ventana([], 1) == {}


def test_metricas_de_ventana_calcula_tasas_y_medias() -> None:
    episodios = [
        {"victoria": True, "fase_inicio": 1, "danio_hecho": 100.0, "danio_recibido": 10.0, "parries": 2.0,
         "duracion_s": 30.0, "progreso_jefe": 1.0, "recompensa": 5.0, "muerte": False},
        {"victoria": False, "fase_inicio": 2, "danio_hecho": 50.0, "muerte": True},
    ]
    m = pj.metricas_de_ventana(episodios, fase_normal=1)
    assert m["tasa_victoria"] == 0.5 and m["tasa_victoria_inicio_normal"] == 1.0
    assert m["danio_al_jefe"] == pytest.approx(75.0) and m["danio_recibido"] == pytest.approx(10.0)
    assert m["muertes"] == 0.5 and m["recompensa"] == pytest.approx(5.0)
    assert not any(k.startswith("muerte_por/") for k in m)


def test_metricas_de_ventana_sin_peleas_de_inicio_normal_reporta_cero() -> None:
    m = pj.metricas_de_ventana([{"victoria": True, "fase_inicio": 2}], fase_normal=1)
    assert m["tasa_victoria_inicio_normal"] == 0.0 and m["danio_al_jefe"] == 0.0


def test_metricas_de_ventana_publica_solo_los_ataques_mas_frecuentes_con_nombres_limpios() -> None:
    muertes = (["Golpe Fuerte!"] * 3 + ["rayo/azul"] * 2 + ["a"] + ["b"])
    episodios: list[dict[str, object]] = [{"muerte": True, "golpe_mortal": g} for g in muertes]
    episodios.append({"muerte": True})
    episodios.append({"muerte": False, "golpe_mortal": "ignorado"})
    m = pj.metricas_de_ventana(episodios, 1, maximo_ataques=3)
    ataques = {k: v for k, v in m.items() if k.startswith("muerte_por/")}
    assert set(ataques) == {"muerte_por/Golpe_Fuerte_", "muerte_por/rayo_azul", "muerte_por/a"}
    assert ataques["muerte_por/Golpe_Fuerte_"] == pytest.approx(3 / len(episodios))
