"""Cobertura del ajuste supervisado: entorno de espacios, tensores, parada temprana y verificacion de recarga."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dmrl.aplicacion import imitacion_red as red
from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento
from dmrl.dominio import acciones as acc
from dmrl.dominio import imitacion as im
from dmrl.dominio.observacion import TAMANO_SIMPLE
from tests.support_imitacion import DIMENSION, demostraciones_sinteticas

CPU = torch.device("cpu")


def modelo_nuevo() -> Any:
    """
    Modelo base para las pruebas.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        Ninguno.

    Returns:
        MaskablePPO: Modelo con pesos iniciales.

    Raises:
        Ninguna.

    Ejemplo:
        modelo = modelo_nuevo()
    """
    return red.crear_modelo_base(ConfigEntrenamiento(nombre="prueba_imitacion"))


def particion(datos: im.Demostraciones) -> tuple[im.Demostraciones, im.Demostraciones]:
    """
    Entrenamiento y validacion por episodios.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        datos (im.Demostraciones): Conjunto completo.

    Returns:
        tuple: (entrenamiento, validacion).

    Raises:
        Ninguna.

    Ejemplo:
        train, val = particion(datos)
    """
    train, val = im.particionar_episodios(datos.episodios, 0.25, 0)
    return datos.seleccionar(train), datos.seleccionar(val)


def test_el_entorno_de_espacios_cumple_la_api_sin_juego() -> None:
    """Mismos espacios que DarkMexicoEnv; reset y step nulos; mascara con todo valido."""
    entorno = red.EntornoEspacios()
    obs, info = entorno.reset(seed=1)
    assert obs.shape == (TAMANO_SIMPLE * 4,) and info == {}
    siguiente, recompensa, terminado, truncado, _ = entorno.step(0)
    assert siguiente.shape == obs.shape and (recompensa, terminado, truncado) == (0.0, True, False)
    assert entorno.action_masks().all() and entorno.action_space.n == acc.N_ACCIONES


def test_logits_valor_y_prediccion_enmascarada_tienen_las_formas_y_respetan_la_mascara() -> None:
    """La prediccion enmascarada nunca elige una accion prohibida."""
    modelo = modelo_nuevo()
    obs = torch.randn(5, DIMENSION)
    logits = red.logits_actor(modelo.policy, obs)
    assert logits.shape == (5, acc.N_ACCIONES) and red.valor_critico(modelo.policy, obs).shape == (5,)
    mascaras = torch.zeros(5, acc.N_ACCIONES, dtype=torch.bool)
    mascaras[:, 7] = True
    assert red.predecir_enmascarado(logits, mascaras).tolist() == [7] * 5


def test_la_prediccion_enmascarada_coincide_con_predict_de_maskableppo() -> None:
    """Lo que se evalua en validacion es lo que el juego ejecutaria con el modelo."""
    modelo = modelo_nuevo()
    datos = demostraciones_sinteticas(episodios=3, pasos=10)
    logits = red.logits_actor(modelo.policy, torch.from_numpy(datos.obs))
    propias = red.predecir_enmascarado(logits, torch.from_numpy(datos.mascaras)).numpy()
    de_sb3 = [int(modelo.predict(datos.obs[i], action_masks=datos.mascaras[i], deterministic=True)[0])
              for i in range(len(datos))]
    assert propias.tolist() == de_sb3


def test_a_tensores_valida_retornos_y_deja_en_cpu_lo_que_no_cabe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un retorno por paso; un conjunto demasiado grande no se deja residente en el dispositivo."""
    datos = demostraciones_sinteticas(episodios=2, pasos=4)
    with pytest.raises(ValueError):
        red.a_tensores(datos, CPU, np.zeros(3, np.float32))
    monkeypatch.setattr(red, "BYTES_RESIDENTES_MAX", 0)
    tensores = red.a_tensores(datos, CPU, np.zeros(8, np.float32))
    lote = tensores.lote(torch.arange(3), CPU)
    assert len(tensores) == 8 and len(lote) == 3 and lote.retornos is not None
    assert red.a_tensores(datos, CPU).lote(torch.arange(2), CPU).retornos is None


def test_evaluar_politica_rechaza_un_conjunto_vacio_y_mide_el_resto() -> None:
    """Perdidas, precision y una prediccion por paso."""
    modelo = modelo_nuevo()
    datos = demostraciones_sinteticas(episodios=2, pasos=6)
    pesos = torch.ones(acc.N_ACCIONES)
    resultado = red.evaluar_politica(modelo.policy, red.a_tensores(datos, CPU), pesos, CPU, tamano_lote=5)
    assert resultado.predicciones.shape == (12,) and 0.0 <= resultado.precision <= 1.0
    assert resultado.perdida == pytest.approx(resultado.perdida_plana)
    vacio = datos.seleccionar(np.zeros(12, dtype=bool))
    with pytest.raises(ValueError):
        red.evaluar_politica(modelo.policy, red.a_tensores(vacio, CPU), pesos, CPU)


def test_entrenar_politica_rechaza_conjuntos_vacios() -> None:
    """Sin entrenamiento o sin validacion no se ajusta nada."""
    datos = demostraciones_sinteticas(episodios=2, pasos=4)
    vacio = datos.seleccionar(np.zeros(8, dtype=bool))
    with pytest.raises(ValueError):
        red.entrenar_politica(modelo_nuevo(), vacio, datos, im.ConfigAprendizaje(), 0, CPU)
    with pytest.raises(ValueError):
        red.entrenar_politica(modelo_nuevo(), datos, vacio, im.ConfigAprendizaje(), 0, CPU)
    with pytest.raises(ValueError):
        red.entrenar_valor(modelo_nuevo(), vacio, datos, np.zeros(0, np.float32), np.zeros(8, np.float32),
                           im.ConfigAprendizaje(), 0, CPU)


def test_la_parada_temprana_corta_cuando_la_validacion_no_mejora_y_restaura_el_mejor_punto() -> None:
    """Con etiquetas de validacion al azar la perdida de validacion no baja y el ajuste se detiene por paciencia."""
    datos = demostraciones_sinteticas(episodios=16, pasos=40, semilla=3)
    entrenamiento, validacion = particion(datos)
    azar = np.random.default_rng(0)
    ruido = azar.integers(1, 5, len(validacion)).astype(np.int16)
    validacion.etiquetas = ruido
    validacion.mascaras[np.arange(len(validacion)), ruido] = True
    puntos: list[red.PuntoCurva] = []
    config = im.ConfigAprendizaje(epocas_maximas=40, paciencia=2, tamano_lote=64, tasa_aprendizaje=5e-3)
    resultado = red.entrenar_politica(modelo_nuevo(), entrenamiento, validacion, config, 0, CPU, puntos.append)
    assert resultado.detenido_por == "paciencia" and len(resultado.curva) < 40 and puntos == resultado.curva
    assert resultado.mejor_epoca <= len(resultado.curva) - 2
    assert resultado.episodios_entrenamiento + resultado.episodios_validacion == 16
    assert len(resultado.por_accion) == acc.N_ACCIONES and len(resultado.pesos_clase) == acc.N_ACCIONES


def test_sin_paciencia_agotada_el_ajuste_corre_las_epocas_pedidas() -> None:
    """Con pocas epocas y paciencia amplia termina por epocas."""
    entrenamiento, validacion = particion(demostraciones_sinteticas(episodios=8, pasos=30))
    config = im.ConfigAprendizaje(epocas_maximas=2, paciencia=10, tamano_lote=64)
    resultado = red.entrenar_politica(modelo_nuevo(), entrenamiento, validacion, config, 0, CPU)
    assert resultado.detenido_por == "epocas" and len(resultado.curva) == 2


def test_el_ajuste_de_valor_reduce_el_error_y_para_por_paciencia() -> None:
    """El critico aprende retornos que dependen de la observacion; con paciencia 2 puede cortar antes."""
    datos = demostraciones_sinteticas(episodios=12, pasos=40, semilla=4)
    entrenamiento, validacion = particion(datos)
    r_train = entrenamiento.obs[:, 0] * 3.0
    r_val = validacion.obs[:, 0] * 3.0
    modelo = modelo_nuevo()
    config = im.ConfigAprendizaje(epocas_valor=40, paciencia=2, tamano_lote=64, tasa_aprendizaje=3e-3)
    metricas = red.entrenar_valor(modelo, entrenamiento, validacion, r_train, r_val, config, 0, CPU)
    assert metricas["varianza_explicada_validacion"] > 0.1
    assert metricas["ecm_entrenamiento"] < float((r_train**2).mean())
    constantes = red.entrenar_valor(modelo_nuevo(), entrenamiento, validacion, np.zeros(len(entrenamiento), np.float32),
                                    np.zeros(len(validacion), np.float32), config, 0, CPU)
    assert constantes["varianza_explicada_validacion"] == 0.0


def test_guardar_y_recargar_reproduce_las_decisiones(tmp_path: Path) -> None:
    """El archivo guardado decide igual que el modelo en memoria; una muestra vacia es un error."""
    modelo = modelo_nuevo()
    datos = demostraciones_sinteticas(episodios=2, pasos=10)
    ruta = tmp_path / "carpeta" / "m.zip"
    red.guardar_modelo(modelo, ruta)
    esperadas = np.asarray([int(modelo.predict(datos.obs[i], action_masks=datos.mascaras[i],
                                               deterministic=True)[0]) for i in range(len(datos))])
    assert red.coincidencia_tras_cargar(ruta, datos.obs, datos.mascaras, esperadas) == 1.0
    with pytest.raises(ValueError):
        red.coincidencia_tras_cargar(ruta, datos.obs[:0], datos.mascaras[:0], esperadas[:0])


def test_el_entorno_de_espacios_acepta_un_tamano_de_observacion_distinto_al_de_la_v1() -> None:
    """La observacion v2 del entorno multi-jefe tiene otra dimension que la v1."""
    entorno = red.EntornoEspacios(apilado=3, tamano_simple=50)
    obs, _ = entorno.reset()
    assert obs.shape == (150,) and entorno.observation_space.shape == (150,)


def test_crear_modelo_base_con_tamano_simple_produce_una_politica_con_esa_entrada() -> None:
    """La primera capa de la politica recibe tamano_simple por observaciones apiladas; sin tamano usa la v1."""
    config = ConfigEntrenamiento(nombre="prueba_v2")
    modelo = red.crear_modelo_base(config, 50)
    apilado = config.entorno.apilado
    assert modelo.observation_space.shape == (50 * apilado,)
    assert red.logits_actor(modelo.policy, torch.zeros(2, 50 * apilado)).shape == (2, acc.N_ACCIONES)
    assert red.crear_modelo_base(config).observation_space.shape == (TAMANO_SIMPLE * apilado,)
