"""Observacion generica multi-jefe: forma, rangos, identidad y rasgos de amenaza."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenaza, Amenazas, SeguidorAmenazas
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.observacion import TAMANO_JUGADOR, ApiladorObservaciones, bloque_jugador, campos_jugador
from dmrl.dominio.observacion_v2 import (
    ConstructorObservacionV2,
    campos_v2,
    especificacion_markdown_v2,
    tamano_simple_v2,
)
from dmrl.infraestructura.jefes import cargar_catalogo
from tests.support_dmrl import jefe, jugador, montage
from tests.support_jefes import catalogo_prueba


def indice(catalogo: Any, nombre: str) -> int:
    return [n for n, _ in campos_v2(catalogo)].index(nombre)


def crudo(clase: str = "BP_AMLOBOSS_C", nombre: str | None = None, pos: float = 0.0,
          **jefe_extra: Any) -> dict[str, Any]:
    b = jefe(cls=clase, pos=[500.0, 0.0, 0.0], yaw=180.0, **jefe_extra)
    if nombre:
        b["montage"] = montage(nombre, pos, 1.0, 4.0)
    return {"t": 1.0, "dist": 500.0, "player": jugador(), "boss": b, "ev": {}}


def test_el_bloque_del_jugador_es_el_de_la_v1_y_los_campos_son_unicos() -> None:
    catalogo = catalogo_prueba()
    nombres = [n for n, _ in campos_v2(catalogo)]
    assert len(set(nombres)) == len(nombres)
    assert nombres[:TAMANO_JUGADOR] == [n for n, _ in campos_jugador()]
    assert len(bloque_jugador({})) == TAMANO_JUGADOR
    assert tamano_simple_v2(catalogo) == len(nombres)


def test_vector_tiene_el_tamano_y_el_rango_declarados() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    seguimiento = SeguidorAmenazas(catalogo)
    obs = crudo(nombre="amlo_rafaga", pos=0.1)
    vector = constructor.construir(obs, acc.ACCION_ATAQUE, 12.0, seguimiento.actualizar(obs))
    assert vector.shape == (tamano_simple_v2(catalogo),) and vector.dtype == np.float32
    assert np.isfinite(vector).all() and vector.min() >= -1.0 and vector.max() <= 1.0


def test_identidad_del_jefe_sale_de_su_clase_y_se_distingue_la_segunda_fase() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    for clase, nombre, avanzada in (("BP_AMLOBOSS_C", "jefe_id_amlo", 0.0), ("BP_Chamuco_C", "jefe_id_chamuco", 1.0),
                                    ("BP_CHUPACABRAS_C", "jefe_id_nahual", 0.0)):
        obs = crudo(clase=clase)
        vector = constructor.construir(obs, 0, 0.0, Amenazas())
        ids = [vector[indice(catalogo, f"jefe_id_{p.id}")] for p in catalogo.perfiles]
        assert sum(ids) == 1.0 and vector[indice(catalogo, nombre)] == 1.0
        assert vector[indice(catalogo, "jefe_fase_avanzada")] == avanzada


def test_clase_desconocida_deja_la_identidad_en_cero() -> None:
    catalogo = catalogo_prueba()
    vector = ConstructorObservacionV2(catalogo).construir(crudo(clase="BP_EnemyNPCFRIENDLY_C"), 0, 0.0, Amenazas())
    assert sum(vector[indice(catalogo, f"jefe_id_{p.id}")] for p in catalogo.perfiles) == 0.0


def test_montage_del_jefe_usa_el_vocabulario_union() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    # Un montage propio de otro jefe sigue siendo reconocible con el vocabulario union.
    vector = constructor.construir(crudo(nombre="nahual_corte"), 0, 0.0, Amenazas())
    assert vector[indice(catalogo, "jefe_montage_nahual_corte")] == 1.0
    desconocido = constructor.construir(crudo(nombre="montage_nuevo"), 0, 0.0, Amenazas())
    assert desconocido[indice(catalogo, "jefe_montage_otro")] == 1.0
    sin = constructor.construir(crudo(), 0, 0.0, Amenazas())
    assert sin[indice(catalogo, "jefe_montage_ninguno")] == 1.0


def test_rasgos_genericos_de_amenaza_con_la_tabla_del_jefe() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    obs = crudo(nombre="amlo_corte", pos=0.25)
    amenazas = SeguidorAmenazas(catalogo).actualizar(obs)
    v = constructor.construir(obs, 0, 0.0, amenazas)
    assert v[indice(catalogo, "amenaza_hay")] == 1.0
    assert v[indice(catalogo, "amenaza_golpe_tiempo")] == pytest.approx(0.75 / 1.5)
    assert v[indice(catalogo, "amenaza_golpe_parable")] == 1.0
    assert v[indice(catalogo, "amenaza_golpe_danio")] == pytest.approx(20.0 / 50.0)
    assert v[indice(catalogo, "amenaza_golpes_pendientes")] == pytest.approx(1 / 5)
    assert v[indice(catalogo, "amenaza_parry_tiempo")] == pytest.approx(0.5)
    assert v[indice(catalogo, "amenaza_esquivar_tiempo")] == 1.0
    assert v[indice(catalogo, "remota_activa")] == 0.0 and v[indice(catalogo, "remota_tiempo")] == 1.0


def test_sin_amenazas_los_tiempos_valen_uno_y_los_flags_cero() -> None:
    catalogo = catalogo_prueba()
    v = ConstructorObservacionV2(catalogo).construir(crudo(), 0, 0.0, Amenazas())
    for nombre in ("amenaza_golpe_tiempo", "amenaza_parry_tiempo", "amenaza_esquivar_tiempo", "remota_tiempo"):
        assert v[indice(catalogo, nombre)] == 1.0
    for nombre in ("amenaza_hay", "amenaza_golpe_parable", "amenaza_en_alcance", "remota_activa"):
        assert v[indice(catalogo, nombre)] == 0.0


def test_amenaza_a_distancia_expone_tiempo_origen_y_si_se_evita() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    obs = crudo()
    obs["player"]["pos"] = [0.0, 0.0, 0.0]
    obs["player"]["yaw"] = 0.0
    remota = Amenaza(("e", 1, 0), 1.25, False, 35.0, 6000.0, "proyectil", True, "evento", 0, (500.0, 0.0), "jefe")
    v = constructor.construir(obs, 0, 0.0, Amenazas((remota,), 500.0))
    assert v[indice(catalogo, "remota_activa")] == 1.0
    assert v[indice(catalogo, "remota_tiempo")] == pytest.approx(0.5)
    assert v[indice(catalogo, "remota_esquivable")] == 1.0 and v[indice(catalogo, "remota_parable")] == 0.0
    assert v[indice(catalogo, "remota_origen_distancia")] == pytest.approx(0.5)
    assert v[indice(catalogo, "remota_origen_cos")] == pytest.approx(1.0)
    assert v[indice(catalogo, "remota_sobre_jugador")] == 0.0
    sobre = Amenaza(("e", 2, 0), 1.0, False, 30.0, 6000.0, "area", True, "evento", 0, (0.0, 0.0), "jugador")
    w = constructor.construir(obs, 0, 0.0, Amenazas((sobre,), 500.0))
    assert w[indice(catalogo, "remota_sobre_jugador")] == 1.0 and w[indice(catalogo, "remota_origen_distancia")] == 0.0


def test_teletransporte_y_aturdimiento_del_jefe() -> None:
    catalogo = catalogo_prueba()
    obs = crudo(teleporting=True, stunned=True)
    v = ConstructorObservacionV2(catalogo).construir(obs, 0, 0.0, Amenazas())
    assert v[indice(catalogo, "jefe_teletransportando")] == 1.0 and v[indice(catalogo, "jefe_aturdido")] == 1.0


def test_relativo_usa_escala_de_4000_y_ceros_sin_jefe() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo, ConfigEntorno())
    v = constructor.construir(crudo(), 0, 0.0, Amenazas())
    assert v[indice(catalogo, "rel_distancia")] == pytest.approx(500.0 / 4000.0)
    vacio = constructor.construir({"player": jugador(), "boss": None, "ev": {}}, 0, 0.0, Amenazas())
    assert vacio[indice(catalogo, "rel_distancia")] == 0.0 and vacio[indice(catalogo, "jefe_montage_ninguno")] == 1.0


def test_accion_previa_y_tiempo_de_episodio() -> None:
    catalogo = catalogo_prueba()
    v = ConstructorObservacionV2(catalogo).construir(crudo(), acc.ACCION_PARRY, 180.0, Amenazas())
    assert v[indice(catalogo, "accion_previa_parry")] == 1.0
    assert v[indice(catalogo, "tiempo_episodio")] == pytest.approx(0.5)


def test_la_observacion_apilada_tiene_cuatro_instantes() -> None:
    catalogo = catalogo_prueba()
    constructor = ConstructorObservacionV2(catalogo)
    apilador = ApiladorObservaciones(constructor.tamano, 4)
    vector = constructor.construir(crudo(), 0, 0.0, Amenazas())
    assert apilador.iniciar(vector).shape == (constructor.tamano * 4,)


def test_especificacion_documenta_cada_indice_e_identidades() -> None:
    catalogo = catalogo_prueba()
    texto = especificacion_markdown_v2(catalogo, 4)
    assert f"| {tamano_simple_v2(catalogo) - 1} | tiempo_episodio |" in texto
    assert "0 juice, 1 juice_fase2, 2 amlo, 3 chamuco, 4 nahual" in texto
    assert "juice_corte" in texto


def test_con_los_registros_reales_el_tamano_es_estable_y_los_indices_no_chocan() -> None:
    catalogo = cargar_catalogo()
    campos = campos_v2(catalogo)
    assert len({n for n, _ in campos}) == len(campos)
    assert tamano_simple_v2(catalogo) == len(campos) == 35 + 5 + 9 + len(catalogo.vocabulario) + 2 + 8 + 9 + 5 + 22 + 1
