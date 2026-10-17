"""Reglas puras del panel: distribucion, atribucion, lectura de valores, eventos y mensajes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio import panel as pn
from dmrl.dominio.observacion_v2 import campos_v2
from tests.support_jefes import catalogo_prueba


def test_hay_una_etiqueta_en_espanol_por_accion() -> None:
    assert len(pn.ETIQUETAS_ACCIONES) == acc.N_ACCIONES
    assert pn.ETIQUETAS_ACCIONES[0] == "Nada" and pn.ETIQUETAS_ACCIONES[1] == "Moverse N"
    assert pn.ETIQUETAS_ACCIONES[acc.ACCION_ROLL_PRIMERA + 4] == "Rodar S"
    assert pn.ETIQUETAS_ACCIONES[acc.ACCION_PARRY] == "Parry"


def test_distribucion_enmascarada_suma_uno_y_anula_las_acciones_invalidas() -> None:
    p = pn.distribucion_enmascarada([1.0, 2.0, 3.0], [True, True, False])
    assert p[2] == 0.0 and p.sum() == pytest.approx(1.0)
    assert p[1] == pytest.approx(np.exp(2) / (np.exp(1) + np.exp(2)))


def test_distribucion_rechaza_formas_distintas_y_mascara_vacia() -> None:
    with pytest.raises(ValueError, match="mismo largo"):
        pn.distribucion_enmascarada([1.0, 2.0], [True])
    with pytest.raises(ValueError, match="ninguna accion"):
        pn.distribucion_enmascarada([1.0, 2.0], [False, False])


def test_agregar_copias_suma_por_caracteristica_y_valida_el_largo() -> None:
    assert pn.agregar_copias([1, 2, 3, 4], 2).tolist() == [4.0, 6.0]
    for malo in ([1, 2, 3], []):
        with pytest.raises(ValueError, match="no se reparten"):
            pn.agregar_copias(malo, 2)
    with pytest.raises(ValueError, match="no se reparten"):
        pn.agregar_copias([1, 2], 0)


def test_atribucion_es_gradiente_por_entrada_agregada() -> None:
    assert pn.atribucion_por_caracteristica([1, 1, 2, 2], [1, 0, 1, 1], 2).tolist() == [3.0, 2.0]
    with pytest.raises(ValueError, match="mismo largo"):
        pn.atribucion_por_caracteristica([1, 1], [1], 1)


def test_indices_mas_influyentes_ordena_por_magnitud_y_omite_ceros() -> None:
    assert pn.indices_mas_influyentes([0.1, -0.5, 0.0, 0.3], 3) == [1, 3, 0]
    assert pn.indices_mas_influyentes([0.0, 0.0], 5) == []
    with pytest.raises(ValueError, match="positiva"):
        pn.indices_mas_influyentes([1.0], 0)


@pytest.mark.parametrize(("nombre", "valor", "esperado"), [
    ("amenaza_golpe_tiempo", 0.4, ("Tiempo al proximo golpe", "0.6 s")),
    ("amenaza_golpe_tiempo", 1.0, ("Tiempo al proximo golpe", "sin golpe")),
    ("jefe_montage_restante", 1.0, ("Tiempo restante del ataque del jefe", "4 s")),
    ("jugador_vida", 0.755, ("Vida del jugador", "76%")),
    ("jefe_aturdido", 1.0, ("Jefe aturdido", "si")),
    ("jefe_aturdido", 0.0, ("Jefe aturdido", "no")),
    ("rel_distancia", 0.25, ("Distancia al jefe", "1000 uu (10 m)")),
    ("jugador_velocidad", 0.5, ("Velocidad del jugador", "400 uu/s")),
    ("jefe_montage_rate", 0.5, ("Velocidad de la animacion del jefe", "x1")),
    ("rel_jefe_sen", -0.25, ("Jefe a un lado", "-0.25")),
    ("jugador_caguamas", 2 / 3, ("Caguamas restantes", "2")),
    ("amenaza_golpe_danio", 0.5, ("Dano del proximo golpe", "25")),
    ("accion_previa_parry", 1.0, ("Accion previa: Parry", "si")),
    ("accion_previa_inexistente", 0.0, ("Accion previa: inexistente", "no")),
    ("jefe_montage_NinjaWheelKick_Montage", 1.0, ("Animacion del jefe: NinjaWheelKick_Montage", "si")),
    ("jugador_montage_ninguno", 0.0, ("Animacion del jugador: ninguno", "no")),
    ("jefe_id_amlo", 1.0, ("Jefe activo: amlo", "si")),
    ("campo_desconocido", 0.5, ("campo_desconocido", "0.50")),
])
def test_formatear_caracteristica_traduce_a_unidades_del_juego(
        nombre: str, valor: float, esperado: tuple[str, str]) -> None:
    assert pn.formatear_caracteristica(nombre, valor) == esperado


def test_todas_las_caracteristicas_de_la_observacion_v2_tienen_lectura_legible() -> None:
    for nombre, _ in campos_v2(catalogo_prueba()):
        etiqueta, texto = pn.formatear_caracteristica(nombre, 0.5)
        assert etiqueta != nombre, nombre
        assert texto


def test_construir_atencion_trae_nombre_valor_legible_y_atribucion() -> None:
    nombres = ["jugador_vida", "amenaza_golpe_tiempo", "jefe_aturdido"]
    fila = pn.construir_atencion(np.array([0.1, -0.9, 0.0]), [0.5, 0.2, 1.0], nombres, cantidad=5)
    assert [f["indice"] for f in fila] == [1, 0]
    assert fila[0] == {"indice": 1, "nombre": "amenaza_golpe_tiempo", "etiqueta": "Tiempo al proximo golpe",
                       "valor": 0.2, "valorTexto": "0.3 s", "atribucion": -0.9}
    with pytest.raises(ValueError, match="no coinciden"):
        pn.construir_atencion(np.array([0.1]), [0.5, 0.2], nombres)


def test_encabezado_toma_jefe_vidas_caguamas_y_tiempo() -> None:
    cruda = {"player": {"hp": 40.04, "hp_max": 100.0, "caguamas": 2},
             "boss": {"cls": "BP_X_C", "hp": -5.0, "hp_max": 3000.0}}
    e = pn.encabezado_de(cruda, lambda c: "AMLO" if c == "BP_X_C" else None, 2, 12.345, "amlo")
    assert e == {"jefe": "AMLO", "fase": 2, "vidaJugador": 40.0, "vidaJugadorMax": 100.0, "vidaJefe": 0.0,
                 "vidaJefeMax": 3000.0, "caguamas": 2, "tiempoPelea": 12.35}


def test_encabezado_sin_jefe_usa_el_de_la_sesion_y_ceros() -> None:
    e = pn.encabezado_de({}, lambda c: None, 1, 0.0, "nahual")
    assert e["jefe"] == "nahual" and e["vidaJugador"] == 0.0 and e["caguamas"] == 0


def cruda(hp: float = 100.0, caguamas: int = 3, parries: int = 0, fase_jefe: int = 1) -> dict[str, Any]:
    """
    Observacion cruda minima para el detector.

    Contexto:
        Pruebas de DetectorEventos.

    Args:
        hp (float): Vida del jugador.
        caguamas (int): Caguamas restantes.
        parries (int): Parries del paso.
        fase_jefe (int): Fase que el detector leera.

    Returns:
        dict[str, Any]: Observacion cruda.

    Raises:
        Ninguna.

    Ejemplo:
        cruda(hp=50)
    """
    return {"player": {"hp": hp, "caguamas": caguamas}, "ev": {"parries": parries}, "fase": fase_jefe}


def detector() -> pn.DetectorEventos:
    """
    Detector cuya fase sale del campo "fase" de la observacion de prueba.

    Contexto:
        Pruebas de DetectorEventos.

    Args:
        Ninguno.

    Returns:
        pn.DetectorEventos: Detector listo.

    Raises:
        Ninguna.

    Ejemplo:
        detector().reiniciar(cruda())
    """
    return pn.DetectorEventos(lambda obs, por_defecto: int(obs.get("fase", por_defecto)))


def test_detector_sin_referencia_solo_ve_parries_y_fase() -> None:
    d = detector()
    assert d.observar(cruda(parries=2)) == [{"tipo": "parry", "detalle": 2}]


def test_detector_reconoce_parry_golpe_caguama_y_fase() -> None:
    d = detector()
    d.reiniciar(cruda())
    assert d.observar(cruda()) == []
    assert d.observar(cruda(hp=88.0, parries=1)) == [
        {"tipo": "parry", "detalle": 1}, {"tipo": "golpe", "detalle": 12.0}]
    assert d.observar(cruda(hp=100.0, caguamas=2)) == [{"tipo": "caguama", "detalle": 1}]
    assert d.observar(cruda(hp=100.0, caguamas=2, fase_jefe=2)) == [{"tipo": "fase", "detalle": 2}]
    assert d.observar(cruda(hp=100.0, caguamas=2, fase_jefe=2)) == []


def test_detector_no_marca_curacion_ni_vida_por_debajo_de_cero_como_dano_extra() -> None:
    d = detector()
    d.reiniciar(cruda(hp=10.0))
    assert d.observar(cruda(hp=-5.0)) == [{"tipo": "golpe", "detalle": 10.0}]


def datos(**cambios: Any) -> pn.DatosCuadro:
    """
    Datos de cuadro de ejemplo.

    Contexto:
        Pruebas de construir_cuadro.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        pn.DatosCuadro: Datos con dos capas de tres neuronas.

    Raises:
        Ninguna.

    Ejemplo:
        datos(accion=3)
    """
    base: dict[str, Any] = {
        "paso": 3, "encabezado": {"jefe": "AMLO"}, "activaciones_politica": [np.array([0.123456, -1.0, np.nan])],
        "activaciones_valor": [np.array([0.5])], "probabilidades": np.array([0.25, 0.75]), "mascara": [True, False],
        "accion": 1, "valor": 1.23456, "eventos": [{"tipo": "parry", "detalle": 1}],
        "atencion": [{"indice": 0}], "milisegundos": 1.2345}
    base.update(cambios)
    return pn.DatosCuadro(**base)


def test_cuadro_es_json_en_camel_case_con_valores_finitos() -> None:
    cuadro = pn.construir_cuadro(datos())
    assert cuadro["tipo"] == "cuadro" and cuadro["accionElegida"] == 1
    assert cuadro["activacionesPolitica"] == [[0.12, -1.0, 0.0]]
    assert cuadro["activacionesValor"] == [[0.5]]
    assert cuadro["probabilidades"] == [0.25, 0.75] and cuadro["mascara"] == [True, False]
    assert cuadro["valorEstimado"] == 1.2346 and cuadro["milisegundosCalculo"] == 1.23
    json.dumps(cuadro, allow_nan=False)
    assert not any("_" in clave for clave in cuadro)


def test_cuadro_con_valor_no_finito_se_publica_como_cero() -> None:
    assert pn.construir_cuadro(datos(valor=float("nan")))["valorEstimado"] == 0.0


def test_configuracion_describe_la_red_y_los_nombres_de_las_acciones() -> None:
    c = pn.construir_configuracion([256, 256], [256, 128], "tanh", 1.0, "amlo", "ppo", 15.0)
    assert c["capasPolitica"] == [256, 256] and c["capasValor"] == [256, 128]
    assert c["acciones"] == list(pn.ETIQUETAS_ACCIONES) and c["escalaColor"] == 1.0
    assert c["frecuenciaHz"] == 15.0 and c["segundosHistorial"] == pn.SEGUNDOS_HISTORIAL
