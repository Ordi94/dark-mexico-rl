"""Seguimiento de amenazas: golpes por montage, ataques a distancia y tiempos de vuelo."""

from __future__ import annotations

from typing import Any

import pytest

from dmrl.dominio.amenazas import Amenaza, Amenazas, SeguidorAmenazas
from tests.support_dmrl import jefe, jugador, montage
from tests.support_jefes import catalogo_prueba

CLASE = "BP_AMLOBOSS_C"


def observar(t: float, nombre: str | None = None, pos: float = 0.0, rate: float = 1.0, dist: float = 300.0,
             clase: str = CLASE, remoto: list[dict[str, Any]] | None = None, muerto: bool = False,
             con_jefe: bool = True) -> dict[str, Any]:
    """Observacion cruda minima para el seguidor."""
    b = jefe(cls=clase, dead=muerto, pos=[1000.0, 0.0, 0.0])
    if nombre:
        b["montage"] = montage(nombre, pos, rate, 4.0)
    ev: dict[str, Any] = {"remote": remoto} if remoto else {}
    return {"t": t, "dist": dist, "player": jugador(), "boss": b if con_jefe else None, "ev": ev}


def seguidor() -> SeguidorAmenazas:
    return SeguidorAmenazas(catalogo_prueba())


def test_golpe_de_montage_se_programa_con_su_posicion_y_rate() -> None:
    s = seguidor()
    a = s.actualizar(observar(10.0, "amlo_corte", 0.2))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(0.8)
    assert a.proximo.parry and a.proximo.tipo == "melee" and a.proximo.origen == "montage"
    a = SeguidorAmenazas(catalogo_prueba()).actualizar(observar(10.0, "amlo_corte", 0.0, rate=2.0))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(0.5)


def test_la_pausa_de_la_animacion_no_desfasa_el_golpe() -> None:
    s = seguidor()
    s.actualizar(observar(10.0, "amlo_corte", 0.2))
    # El juego detiene la animacion 0.3 s (golpe dado): el reloj avanza y la posicion no.
    a = s.actualizar(observar(10.3, "amlo_corte", 0.2))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(0.8)


def test_golpe_ya_pasado_del_montage_no_se_programa() -> None:
    a = seguidor().actualizar(observar(5.0, "amlo_corte", 1.2))
    assert a.proximo is None and a.pendientes == ()


def test_montage_interrumpido_cancela_sus_golpes() -> None:
    s = seguidor()
    s.actualizar(observar(1.0, "amlo_corte", 0.1))
    a = s.actualizar(observar(1.1, "Boss_Parried_RM_Montage", 0.0))
    assert a.proximo is None
    a = s.actualizar(observar(1.2, None))
    assert a.proximo is None


def test_montage_que_reinicia_programa_de_nuevo() -> None:
    s = seguidor()
    s.actualizar(observar(1.0, "amlo_corte", 0.9))
    a = s.actualizar(observar(2.0, "amlo_corte", 0.1))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(0.9)


def test_amenaza_a_distancia_activa_tras_fin_del_montage() -> None:
    """El rayo cae despues de terminar su montage: debe seguir pendiente sin montage activo."""
    s = seguidor()
    s.actualizar(observar(0.0, "amlo_rafaga", 0.0))
    a = s.actualizar(observar(1.8, None))
    assert a.proximo is not None and a.proximo.tipo == "rayo"
    assert a.proximo.segundos == pytest.approx(0.2)
    assert a.remota is a.proximo
    a = s.actualizar(observar(2.5, None))
    assert a.proximo is None


def test_proyectil_vuela_distancia_sobre_velocidad_y_se_fija_al_lanzarse() -> None:
    s = seguidor()
    a = s.actualizar(observar(0.0, "amlo_rafaga", 0.0, dist=500.0))
    proyectil = a.pendientes[0]
    assert proyectil.tipo == "proyectil" and proyectil.segundos == pytest.approx(0.5 + 0.5)
    # El jugador se acerca antes del lanzamiento: el vuelo se acorta.
    a = s.actualizar(observar(0.2, "amlo_rafaga", 0.2, dist=300.0))
    assert a.pendientes[0].segundos == pytest.approx(0.3 + 0.3)
    # Al pasar el lanzamiento la distancia queda fijada (200 uu = 0.2 s de vuelo).
    a = s.actualizar(observar(0.6, "amlo_rafaga", 0.6, dist=200.0))
    assert a.pendientes[0].segundos == pytest.approx(0.1)
    a = s.actualizar(observar(0.65, "amlo_rafaga", 0.65, dist=900.0))
    assert a.pendientes[0].segundos == pytest.approx(0.05)


def test_evento_de_proyectil_se_programa_con_su_vuelo_y_origen_en_el_jefe() -> None:
    s = seguidor()
    registro = {"id": "amlo_throw", "t": 5.0, "px": 0.0, "py": 0.0, "bx": 1000.0, "by": 0.0}
    a = s.actualizar(observar(5.0, remoto=[registro]))
    assert a.proximo is not None
    assert a.proximo.segundos == pytest.approx(0.2 + 1000.0 / 5000.0)
    assert a.proximo.origen == "evento" and a.proximo.origen_xy == (1000.0, 0.0) and a.proximo.centro == "jefe"
    assert not a.proximo.parry and a.proximo.esquivable


def test_evento_de_area_cae_sobre_el_jugador() -> None:
    s = seguidor()
    registro = {"id": "amlo_floor", "t": 3.0, "px": 10.0, "py": 20.0, "bx": 500.0, "by": 0.0}
    a = s.actualizar(observar(3.0, remoto=[registro]))
    assert a.proximo is not None
    assert a.proximo.segundos == pytest.approx(1.9) and a.proximo.tipo == "area"
    assert a.proximo.centro == "jugador" and a.proximo.origen_xy == (10.0, 20.0)


def test_evento_sin_posiciones_usa_la_distancia_actual() -> None:
    s = seguidor()
    a = s.actualizar(observar(3.0, dist=2500.0, remoto=[{"id": "amlo_throw", "t": 3.0}]))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(0.2 + 0.5)
    assert a.proximo.origen_xy is None


def test_evento_desconocido_se_ignora() -> None:
    a = seguidor().actualizar(observar(1.0, remoto=[{"id": "otro", "t": 1.0}]))
    assert a.pendientes == ()


def test_evento_sobrevive_al_cambio_de_montage_y_expira() -> None:
    s = seguidor()
    s.actualizar(observar(3.0, "amlo_corte", 0.0, remoto=[{"id": "amlo_floor", "t": 3.0}]))
    a = s.actualizar(observar(3.5, "Boss_Parried_RM_Montage", 0.0))
    assert [x.tipo for x in a.pendientes] == ["area"]
    assert s.actualizar(observar(6.0, None)).pendientes == ()


def test_cambio_de_actor_cancela_los_golpes_de_montage_y_conserva_los_eventos() -> None:
    s = seguidor()
    s.actualizar(observar(1.0, "amlo_rafaga", 0.0, remoto=[{"id": "amlo_floor", "t": 1.0}]))
    a = s.actualizar(observar(1.1, None, clase="BP_Chamuco_C"))
    assert [x.origen for x in a.pendientes] == ["evento"]


def test_jefe_muerto_o_ausente_no_programa_golpes() -> None:
    assert seguidor().actualizar(observar(1.0, "amlo_corte", 0.0, muerto=True)).pendientes == ()
    assert seguidor().actualizar(observar(1.0, con_jefe=False)).pendientes == ()


def test_clase_desconocida_no_programa_golpes() -> None:
    assert seguidor().actualizar(observar(1.0, "amlo_corte", 0.0, clase="BP_EnemyNPCFRIENDLY_C")).pendientes == ()


def test_reiniciar_olvida_todo() -> None:
    s = seguidor()
    s.actualizar(observar(1.0, "amlo_corte", 0.0, remoto=[{"id": "amlo_floor", "t": 1.0}]))
    s.reiniciar()
    assert s.actualizar(observar(1.1, None)).pendientes == ()


def test_rate_invalido_se_trata_como_uno() -> None:
    a = seguidor().actualizar(observar(1.0, "amlo_corte", 0.0, rate=0.0))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(1.0)


def test_propiedades_de_las_amenazas() -> None:
    s = seguidor()
    a = s.actualizar(observar(0.0, "amlo_rafaga", 0.0))
    assert a.del_ataque == 2 and a.proximo_parry is None
    assert a.remota is not None and a.remota.tipo == "proyectil"
    b = SeguidorAmenazas(catalogo_prueba()).actualizar(observar(0.0, "amlo_corte", 0.0))
    assert b.proximo_parry is b.proximo and b.remota is None
    assert Amenazas().proximo is None and Amenazas().del_ataque == 0


def test_alcance_solo_limita_a_los_golpes_melee() -> None:
    melee = Amenaza(("m", 1, 0), 0.5, True, 20.0, 200.0, "melee", False, "montage", 0)
    assert melee.en_alcance(400.0) and not melee.en_alcance(401.0)
    area = Amenaza(("m", 1, 1), 0.5, False, 20.0, 6000.0, "area", True, "montage", 1)
    assert area.en_alcance(5000.0)
