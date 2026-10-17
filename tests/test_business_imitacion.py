"""Reglas de negocio de la clonacion de comportamiento: lo que debe cumplirse aunque cambie la implementacion."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from dmrl.aplicacion import imitacion_red as red
from dmrl.aplicacion import recoleccion as rec
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento, crear_modelo
from dmrl.dominio import acciones as acc
from dmrl.dominio import imitacion as im
from dmrl.dominio.tabla_golpes import TablaGolpes
from tests.support_dmrl import JuegoSintetico
from tests.support_imitacion import DIMENSION, EjecutorFijo, ExpertoFijo, demostraciones_sinteticas


def test_acciones_invalidas_no_cuentan_en_la_perdida() -> None:
    """Cambiar los logits de acciones prohibidas no altera la perdida ni recibe gradiente."""
    azar = torch.Generator().manual_seed(0)
    logits = torch.randn(16, acc.N_ACCIONES, generator=azar, requires_grad=True)
    mascaras = torch.rand(16, acc.N_ACCIONES, generator=azar) > 0.4
    etiquetas = torch.arange(16) % acc.N_ACCIONES
    mascaras[torch.arange(16), etiquetas] = True
    base = red.perdida_enmascarada(logits, etiquetas, mascaras)
    alterados = logits.detach().clone()
    alterados[~mascaras] += 40.0
    assert torch.isclose(base, red.perdida_enmascarada(alterados, etiquetas, mascaras))
    base.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert float(logits.grad[~mascaras].abs().sum()) == 0.0
    assert float(logits.grad[mascaras].abs().sum()) > 0.0


def test_la_perdida_enmascarada_equivale_a_la_probabilidad_entre_las_validas() -> None:
    """Con dos acciones validas e iguales la perdida es ln 2, sin importar las prohibidas."""
    logits = torch.zeros(1, acc.N_ACCIONES)
    mascaras = torch.zeros(1, acc.N_ACCIONES, dtype=torch.bool)
    mascaras[0, [3, 5]] = True
    perdida = red.perdida_enmascarada(logits, torch.tensor([3]), mascaras)
    assert float(perdida) == pytest.approx(float(np.log(2.0)), abs=1e-6)


def test_particion_por_episodios_sin_fuga() -> None:
    """Ningun episodio aparece en entrenamiento y validacion a la vez; entre ambos cubren todo."""
    datos = demostraciones_sinteticas(episodios=40, pasos=15)
    entrenamiento, validacion = im.particionar_episodios(datos.episodios, 0.15, 3)
    assert not np.any(entrenamiento & validacion) and np.all(entrenamiento | validacion)
    assert not set(datos.episodios[entrenamiento]) & set(datos.episodios[validacion])
    assert 0 < validacion.sum() < len(datos)


def test_un_episodio_no_cambia_de_lado_al_agregar_rondas() -> None:
    """Sumar episodios de DAgger no mueve los anteriores: las curvas de cada ronda son comparables."""
    ronda0 = demostraciones_sinteticas(episodios=30, pasos=5, ronda=0)
    ronda1 = demostraciones_sinteticas(episodios=30, pasos=5, ronda=1, semilla=1)
    _, val0 = im.particionar_episodios(ronda0.episodios, 0.2, 7)
    todo = im.concatenar([ronda0, ronda1])
    _, val_todo = im.particionar_episodios(todo.episodios, 0.2, 7)
    assert set(ronda0.episodios[val0]) == set(todo.episodios[val_todo]) & set(ronda0.episodios)


def test_la_validacion_nunca_queda_vacia_ni_se_come_el_entrenamiento() -> None:
    """Con muy pocos episodios se fuerza al menos uno de cada lado."""
    ids = np.repeat(np.array([1, 2], dtype=np.int64), 3)
    for fraccion in (0.001, 0.999):
        entrenamiento, validacion = im.particionar_episodios(ids, fraccion, 0)
        assert entrenamiento.any() and validacion.any()


def test_el_ruido_nunca_ejecuta_una_accion_prohibida() -> None:
    """Aunque el ruido sea total, solo salen acciones que la mascara admite."""
    mascara = [False] * acc.N_ACCIONES
    for permitida in (0, 4, 17, 19):
        mascara[permitida] = True
    azar = random.Random(1)
    elegidas = {im.ruido_dart(17, mascara, 1.0, azar) for _ in range(300)}
    assert elegidas <= {0, 4, 17, 19} and len(elegidas) > 1


def test_sin_ruido_se_ejecuta_la_accion_del_ejecutor() -> None:
    """Con epsilon cero la accion no se altera nunca."""
    mascara = [True] * acc.N_ACCIONES
    azar = random.Random(0)
    assert {im.ruido_dart(19, mascara, 0.0, azar) for _ in range(100)} == {19}


def test_el_plan_cubre_ambas_fases_y_ambos_regimenes_con_conteos_exactos() -> None:
    """Con 50 episodios, 20% en fase 2 son exactamente 10 y la mitad lleva ruido en el rango pedido."""
    plan = im.planificar_episodios(50, 0.2, 0.5, 0.05, 0.2, 11)
    assert sum(1 for p in plan if p.fase_inicio == 2) == 10
    ruidosos = [p for p in plan if p.epsilon > 0]
    assert len(ruidosos) == 25 and all(0.05 <= p.epsilon <= 0.2 for p in ruidosos)
    assert [p.indice for p in plan] == list(range(50))
    assert plan == im.planificar_episodios(50, 0.2, 0.5, 0.05, 0.2, 11)


def test_la_etiqueta_es_la_del_experto_aunque_se_ejecute_otra_accion() -> None:
    """DART: con ruido total lo ejecutado se desvia, pero lo guardado sigue siendo la decision del experto."""
    env = DarkMexicoEnv(lambda: JuegoSintetico(periodo=50), TablaGolpes())
    plan = im.PlanEpisodio(indice=3, fase_inicio=1, epsilon=1.0, semilla=5)
    demostraciones, meta = rec.recolectar_episodio(env, ExpertoFijo(), None, plan, ronda=2)  # type: ignore[arg-type]
    assert demostraciones is not None
    assert set(demostraciones.etiquetas.tolist()) == {acc.ACCION_ATAQUE}
    assert np.any(demostraciones.ejecutadas != demostraciones.etiquetas)
    assert set(demostraciones.episodios.tolist()) == {im.identificador_episodio(2, 3)}
    assert meta["ronda"] == 2 and meta["epsilon"] == 1.0


def test_en_dagger_ejecuta_la_red_pero_etiqueta_el_experto() -> None:
    """Cuando juega la politica aprendida, las acciones ejecutadas son las suyas y las etiquetas las del bot."""
    env = DarkMexicoEnv(lambda: JuegoSintetico(periodo=4), TablaGolpes())
    plan = im.PlanEpisodio(indice=0, fase_inicio=1, epsilon=0.0, semilla=0)
    ejecutor = EjecutorFijo(acc.ACCION_PARRY)
    demostraciones, meta = rec.recolectar_episodio(env, ExpertoFijo(), ejecutor, plan, ronda=1)  # type: ignore[arg-type]
    assert demostraciones is not None and meta["ejecutor"] == "modelo"
    assert set(demostraciones.ejecutadas.tolist()) == {acc.ACCION_PARRY}
    assert set(demostraciones.etiquetas.tolist()) == {acc.ACCION_ATAQUE}
    assert ejecutor.reinicios == 1


def test_una_etiqueta_que_la_mascara_prohibe_se_rechaza_al_guardar() -> None:
    """Entrenar con una etiqueta invalida ensenaria una accion que el entorno sustituye por nada."""
    datos = demostraciones_sinteticas(episodios=2, pasos=4)
    datos.mascaras[0, datos.etiquetas[0]] = False
    with pytest.raises(ValueError, match="mascara"):
        datos.validar()


def test_los_pesos_favorecen_las_acciones_raras_con_tope() -> None:
    """El parry (raro) pesa mas que el movimiento (frecuente), sin pasar del tope pedido."""
    etiquetas = np.array([1] * 900 + [acc.ACCION_PARRY] * 9 + [acc.ACCION_CAGUAMA] * 1, dtype=np.int16)
    pesos = im.pesos_de_clase(etiquetas, acc.N_ACCIONES, 0.5, 3.0)
    assert pesos[acc.ACCION_PARRY] > pesos[1]
    assert pesos.max() <= 3.0 + 1e-6 and pesos.min() >= 1.0 / 3.0 - 1e-6
    assert pesos[acc.ACCION_ESPECIAL] == 1.0  # una accion ausente queda neutra
    assert np.all(im.pesos_de_clase(etiquetas, acc.N_ACCIONES, 0.0, 3.0) == 1.0)


def test_los_retornos_no_cruzan_episodios() -> None:
    """El retorno de un episodio no hereda recompensas del siguiente."""
    recompensas = np.array([1.0, 1.0, 10.0, 10.0], dtype=np.float32)
    episodios = np.array([0, 0, 1, 1], dtype=np.int64)
    retornos = im.retornos_descontados(recompensas, episodios, 0.5)
    assert retornos.tolist() == pytest.approx([1.5, 1.0, 15.0, 10.0])


def test_se_elige_por_desempeno_en_juego_y_no_por_perdida() -> None:
    """La ronda con mas victorias gana aunque otra tenga menor perdida; el empate lo gana la mas reciente."""
    resumenes = {0: {"tasa_victoria": 0.40, "progreso_jefe": 0.8}, 1: {"tasa_victoria": 0.62, "progreso_jefe": 0.9},
                 2: {"tasa_victoria": 0.62, "progreso_jefe": 0.9}}
    assert im.elegir_mejor(resumenes) == 2
    resumenes[2]["progreso_jefe"] = 0.7
    assert im.elegir_mejor(resumenes) == 1


def test_el_intervalo_de_confianza_refleja_lo_poco_que_dicen_50_peleas() -> None:
    """36 de 50 (72%) no distingue de 60% ni de 80%: el intervalo es ancho."""
    bajo, alto = im.intervalo_wilson(36, 50)
    assert bajo < 0.60 and alto > 0.80 and bajo < 0.72 < alto


def test_el_modelo_guardado_es_cargable_por_maskableppo_y_compatible_con_la_politica_principal(
        tmp_path: Path) -> None:
    """Mismos nombres y formas de pesos que la politica de la corrida principal; predict respeta la mascara."""
    from sb3_contrib import MaskablePPO

    configuracion = ConfigEntrenamiento(nombre="prueba_imitacion")
    modelo = red.crear_modelo_base(configuracion)
    principal = crear_modelo(configuracion, DummyVecEnv([lambda: red.EntornoEspacios()]), "cpu")
    forma = {k: tuple(v.shape) for k, v in principal.policy.state_dict().items()}
    assert {k: tuple(v.shape) for k, v in modelo.policy.state_dict().items()} == forma
    datos = demostraciones_sinteticas(episodios=6, pasos=30)
    entrenamiento, validacion = im.particionar_episodios(datos.episodios, 0.3, 0)
    ajuste_config = im.ConfigAprendizaje(epocas_maximas=3, paciencia=3, tamano_lote=64)
    red.entrenar_politica(modelo, datos.seleccionar(entrenamiento), datos.seleccionar(validacion), ajuste_config,
                          0, torch.device("cpu"))
    ruta = tmp_path / "mejor.zip"
    red.guardar_modelo(modelo, ruta)
    cargado = MaskablePPO.load(str(ruta), device="cpu")
    for i in range(10):
        accion, _ = cargado.predict(datos.obs[i], action_masks=datos.mascaras[i], deterministic=True)
        assert datos.mascaras[i][int(accion)]
    assert cargado.observation_space.shape == (DIMENSION,)


def test_la_red_aprende_una_regla_y_la_generaliza_a_episodios_que_no_vio() -> None:
    """Con una regla aprendible, la precision en episodios de validacion es alta: no hay memorizacion ciega."""
    datos = demostraciones_sinteticas(episodios=60, pasos=60, semilla=2)
    es_train, es_val = im.particionar_episodios(datos.episodios, 0.2, 0)
    modelo = red.crear_modelo_base(ConfigEntrenamiento(nombre="prueba_imitacion"))
    ajuste = red.entrenar_politica(
        modelo, datos.seleccionar(es_train), datos.seleccionar(es_val),
        im.ConfigAprendizaje(epocas_maximas=25, paciencia=4, tamano_lote=128, tasa_aprendizaje=2e-3),
        0, torch.device("cpu"),
    )
    assert ajuste.validacion["precision"] > 0.75
    assert ajuste.entrenamiento["precision"] >= ajuste.validacion["precision"] - 0.05


def test_ajustar_la_politica_no_toca_el_critico_y_ajustar_el_valor_no_toca_la_politica() -> None:
    """Cada ajuste modifica solo su cabeza: PPO hereda una politica clonada y un critico ajustado por separado."""
    datos = demostraciones_sinteticas(episodios=8, pasos=30)
    es_train, es_val = im.particionar_episodios(datos.episodios, 0.25, 0)
    entrenamiento, validacion = datos.seleccionar(es_train), datos.seleccionar(es_val)
    modelo = red.crear_modelo_base(ConfigEntrenamiento(nombre="prueba_imitacion"))
    config = im.ConfigAprendizaje(epocas_maximas=2, paciencia=2, epocas_valor=2, tamano_lote=64)
    critico_antes = [p.detach().clone() for p in red._parametros_valor(modelo.policy)]
    red.entrenar_politica(modelo, entrenamiento, validacion, config, 0, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(critico_antes, red._parametros_valor(modelo.policy), strict=True))
    actor_antes = [p.detach().clone() for p in red._parametros_politica(modelo.policy)]
    retornos_train = im.retornos_descontados(entrenamiento.recompensas, entrenamiento.episodios, 0.995)
    retornos_val = im.retornos_descontados(validacion.recompensas, validacion.episodios, 0.995)
    red.entrenar_valor(modelo, entrenamiento, validacion, retornos_train, retornos_val, config, 0,
                       torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(actor_antes, red._parametros_politica(modelo.policy), strict=True))
