"""Cobertura de las reglas puras de la imitacion: validaciones de entrada y casos limite."""

from __future__ import annotations

import random

import numpy as np
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio import imitacion as im
from tests.support_imitacion import demostraciones_sinteticas


def test_la_configuracion_por_defecto_es_la_documentada() -> None:
    """Valores de partida del ciclo."""
    c = im.ConfigImitacion()
    assert (c.instancias, c.rondas_dagger, c.prob_fase2, c.fraccion_validacion) == (4, 3, 0.2, 0.15)
    assert c.aprendizaje.decaimiento_pesos > 0 and c.aprendizaje.gamma == 0.995


def test_len_y_seleccionar_conservan_el_orden() -> None:
    """Una mascara de filas devuelve el subconjunto en el mismo orden."""
    datos = demostraciones_sinteticas(episodios=2, pasos=3)
    filas = np.array([True, False, True, False, False, True])
    sub = datos.seleccionar(filas)
    assert len(datos) == 6 and len(sub) == 3
    assert np.array_equal(sub.obs, datos.obs[filas])
    with pytest.raises(ValueError):
        datos.seleccionar(np.array([True]))


def test_validar_detecta_cada_incoherencia() -> None:
    """Longitudes, columnas de mascara, valores no finitos y etiquetas fuera de rango se rechazan."""
    datos = demostraciones_sinteticas(episodios=2, pasos=3)
    datos.validar()
    corto = demostraciones_sinteticas(episodios=2, pasos=3)
    corto.ejecutadas = corto.ejecutadas[:-1]
    with pytest.raises(ValueError, match="longitudes"):
        corto.validar()
    columnas = demostraciones_sinteticas(episodios=2, pasos=3)
    columnas.mascaras = columnas.mascaras[:, :5]
    with pytest.raises(ValueError, match="columnas"):
        columnas.validar()
    infinito = demostraciones_sinteticas(episodios=2, pasos=3)
    infinito.obs[0, 0] = np.inf
    with pytest.raises(ValueError, match="finitos"):
        infinito.validar()
    rango = demostraciones_sinteticas(episodios=2, pasos=3)
    rango.etiquetas[0] = acc.N_ACCIONES
    with pytest.raises(ValueError, match="fuera"):
        rango.validar()


def test_un_conjunto_vacio_es_valido() -> None:
    """Sin filas no hay nada que comprobar."""
    vacio = demostraciones_sinteticas(episodios=1, pasos=2).seleccionar(np.array([False, False]))
    vacio.validar()
    assert len(vacio) == 0


def test_concatenar_une_y_exige_al_menos_una_parte() -> None:
    """Las rondas se suman en orden; sin partes es un error."""
    a, b = demostraciones_sinteticas(2, 3), demostraciones_sinteticas(1, 3, semilla=1, ronda=1)
    assert len(im.concatenar([a, b])) == 9
    with pytest.raises(ValueError):
        im.concatenar([])


def test_identificador_de_episodio_codifica_la_ronda() -> None:
    """ronda * 100000 + indice, con rangos validados."""
    assert im.identificador_episodio(2, 7) == 200007
    assert im.ronda_de_episodio(200007) == 2
    for malo in ((-1, 0), (0, -1), (0, im.EPISODIOS_POR_RONDA_MAX)):
        with pytest.raises(ValueError):
            im.identificador_episodio(*malo)


def test_planificar_rechaza_parametros_fuera_de_rango() -> None:
    """Cantidad, probabilidades y orden de los epsilon se validan."""
    for kwargs in (dict(cantidad=-1), dict(cantidad=im.EPISODIOS_POR_RONDA_MAX + 1), dict(prob_fase2=1.5),
                   dict(fraccion_ruido=-0.1), dict(epsilon_min=0.3, epsilon_max=0.1), dict(epsilon_max=2.0)):
        base: dict[str, float] = dict(cantidad=4, prob_fase2=0.2, fraccion_ruido=0.5, epsilon_min=0.05,
                                      epsilon_max=0.2, semilla=0)
        base.update(kwargs)
        with pytest.raises(ValueError):
            im.planificar_episodios(int(base["cantidad"]), base["prob_fase2"], base["fraccion_ruido"],
                                    base["epsilon_min"], base["epsilon_max"], int(base["semilla"]))
    assert im.planificar_episodios(0, 0.2, 0.5, 0.05, 0.2, 0) == []


def test_ruido_dart_rechaza_epsilon_invalido_y_mascara_vacia() -> None:
    """Los dos errores de entrada del ruido."""
    azar = random.Random(0)
    with pytest.raises(ValueError):
        im.ruido_dart(0, [True], 1.5, azar)
    with pytest.raises(ValueError):
        im.ruido_dart(0, [False, False], 0.1, azar)


def test_particionar_rechaza_fracciones_y_conjuntos_pequenos() -> None:
    """Fraccion fuera de (0, 1) o un solo episodio no se pueden particionar."""
    ids = np.array([1, 1, 2, 2], dtype=np.int64)
    for fraccion in (0.0, 1.0):
        with pytest.raises(ValueError):
            im.particionar_episodios(ids, fraccion, 0)
    with pytest.raises(ValueError):
        im.particionar_episodios(np.array([5, 5], dtype=np.int64), 0.2, 0)


def test_pesos_de_clase_rechaza_parametros_invalidos() -> None:
    """Potencia negativa o tope menor que uno."""
    etiquetas = np.array([1, 2], dtype=np.int16)
    with pytest.raises(ValueError):
        im.pesos_de_clase(etiquetas, 22, -1.0, 3.0)
    with pytest.raises(ValueError):
        im.pesos_de_clase(etiquetas, 22, 0.5, 0.5)


def test_metricas_por_accion_calcula_recall_y_precision() -> None:
    """Recall = aciertos / veces del experto; precision = aciertos / veces de la red."""
    etiqueta = np.array([19, 19, 19, 1, 1, 0], dtype=np.int64)
    prediccion = np.array([19, 19, 1, 1, 19, 0], dtype=np.int64)
    por = im.metricas_por_accion(prediccion, etiqueta, acc.N_ACCIONES)
    parry = por[19]
    assert (parry.soporte, parry.recall) == (3, pytest.approx(2 / 3)) and parry.precision == pytest.approx(2 / 3)
    assert por[5].soporte == 0 and por[5].recall == 0.0 and por[5].precision == 0.0
    with pytest.raises(ValueError):
        im.metricas_por_accion(prediccion[:-1], etiqueta, 22)


def test_retornos_rechaza_formas_y_gamma_invalidos() -> None:
    """Errores de entrada de los retornos descontados."""
    r, e = np.zeros(3, np.float32), np.zeros(3, np.int64)
    with pytest.raises(ValueError):
        im.retornos_descontados(r, e[:2], 0.9)
    with pytest.raises(ValueError):
        im.retornos_descontados(r, e, 1.5)


def test_brecha_de_sobreajuste_es_relativa_y_exige_perdida_positiva() -> None:
    """(val - train) / train."""
    assert im.brecha_sobreajuste(0.2, 0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        im.brecha_sobreajuste(0.0, 0.1)


def test_intervalo_wilson_casos_limite() -> None:
    """Sin peleas el intervalo es total; con todo victorias el techo es 1; datos incoherentes fallan."""
    assert im.intervalo_wilson(0, 0) == (0.0, 1.0)
    bajo, alto = im.intervalo_wilson(10, 10)
    assert alto == pytest.approx(1.0) and bajo > 0.6
    with pytest.raises(ValueError):
        im.intervalo_wilson(5, 3)


def test_resumen_de_juego_cuenta_derrotas_por_fase_y_excluye_errores() -> None:
    """Las muertes se separan por fase alcanzada; los episodios con error no cuentan como peleas."""
    episodios = [
        {"victoria": True, "fase_max": 2, "danio_hecho": 2700.0, "resultado": "victoria"},
        {"victoria": False, "muerte": True, "fase_max": 1, "danio_hecho": 500.0, "resultado": "muerte"},
        {"victoria": False, "muerte": True, "fase_max": 2, "danio_hecho": 2000.0, "resultado": "muerte"},
        {"victoria": False, "fase_max": 1, "danio_hecho": 100.0, "resultado": "tiempo"},
        {"resultado": "error"},
    ]
    r = im.resumen_de_juego(episodios)
    assert (r["peleas"], r["errores"], r["victorias"]) == (4.0, 1.0, 1.0)
    assert (r["muertes_fase1"], r["muertes_fase2"], r["tiempos_agotados"]) == (1.0, 1.0, 1.0)
    assert r["tasa_victoria"] == 0.25 and r["tasa_fase2"] == 0.5
    assert r["tasa_victoria_inferior"] < 0.25 < r["tasa_victoria_superior"]
    assert r["parries"] == 0.0
    assert im.resumen_de_juego([]) == {} and im.resumen_de_juego([{"resultado": "error"}]) == {}


def test_clave_de_seleccion_y_elegir_mejor_sin_rondas() -> None:
    """Orden por victorias, progreso y menor dano recibido; sin rondas es un error."""
    assert im.clave_de_seleccion({}) == (0.0, 0.0, -0.0)
    a = {"tasa_victoria": 0.5, "progreso_jefe": 0.9, "danio_recibido": 100.0}
    b = {"tasa_victoria": 0.5, "progreso_jefe": 0.9, "danio_recibido": 50.0}
    assert im.clave_de_seleccion(b) > im.clave_de_seleccion(a)
    with pytest.raises(ValueError):
        im.elegir_mejor({})
