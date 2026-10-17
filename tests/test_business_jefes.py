"""Reglas de negocio del entorno multi-jefe, comprobadas con los registros reales de data/jefes."""

from __future__ import annotations

from typing import Any

import lupa
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenazas, SeguidorAmenazas
from dmrl.dominio.bot_jefes import BotJefes
from dmrl.dominio.jefes import ALCANCE_GLOBAL, CatalogoJefes
from dmrl.dominio.recompensa import CalculadoraRecompensa
from dmrl.dominio.reglas_jefes import resolver_fase_de, resolver_final_de
from dmrl.infraestructura.jefes import cargar_catalogo
from tests.support_dmrl import jefe, jugador, montage
from tests.test_lua_modulos import SCRIPTS, modulo, tabla


@pytest.fixture(scope="module")
def catalogo() -> CatalogoJefes:
    return cargar_catalogo()


def crudo(clase: str, t: float = 0.0, nombre: str | None = None, pos: float = 0.0, dist: float = 400.0,
          remoto: list[dict[str, Any]] | None = None, **jefe_extra: Any) -> dict[str, Any]:
    b = jefe(cls=clase, pos=[dist, 0.0, 0.0], **jefe_extra)
    if nombre:
        b["montage"] = montage(nombre, pos, 1.0, 8.0)
    return {"t": t, "dist": dist, "player": jugador(), "boss": b, "ev": {"remote": remoto} if remoto else {}}


def test_jefe_se_elige_por_clase_no_por_bandera_is_boss() -> None:
    """Un NPC amistoso con IS BOSS? en true y controlador nunca es el rival."""
    lua = lupa.LuaRuntime(unpack_returned_tuples=True)
    lua.execute(f'package.path = "{(SCRIPTS / "?.lua").as_posix()};" .. package.path')
    seleccion = modulo(lua, "dmb.selection")
    aliado = {"cls": "BP_EnemyNPCFRIENDLY_C", "controlled": True, "dead": False, "is_boss": True}
    rival = {"cls": "BP_CHUPACABRAS_C", "controlled": True, "dead": False, "is_boss": False}
    assert seleccion.pick(tabla(lua, [aliado, rival]), tabla(lua, ["BP_CHUPACABRAS_C"])) == 2
    assert seleccion.pick(tabla(lua, [aliado]), tabla(lua, ["BP_CHUPACABRAS_C"])) is None


def test_amenaza_a_distancia_activa_tras_fin_del_montage(catalogo: CatalogoJefes) -> None:
    """El rayo del Nahual cae 1.216 s despues de iniciar un montage de 1.18 s: sigue pendiente sin montage."""
    seguidor = SeguidorAmenazas(catalogo)
    montage_rayo = "AS_Mage_Ability_Spell_HighRtLight_02_Montage"
    seguidor.actualizar(crudo("BP_CHUPACABRAS_C", 10.0, montage_rayo, 0.0))
    pendientes = seguidor.actualizar(crudo("BP_CHUPACABRAS_C", 11.19, None)).pendientes
    assert [a.tipo for a in pendientes] == ["rayo"]
    assert pendientes[0].segundos == pytest.approx(0.026, abs=0.01) and not pendientes[0].parry
    assert seguidor.actualizar(crudo("BP_CHUPACABRAS_C", 11.7, None)).pendientes == ()


def test_chamuco_directo_inicia_con_vida_completa(catalogo: CatalogoJefes) -> None:
    """El Chamuco generado a mano nace con 20 HP de clase: el registro lo fija a su vida real y cuenta la fase 2."""
    perfil = catalogo.perfil("chamuco")
    inicio = perfil.episodio.mod["start"] if perfil.episodio.mod else {}
    assert inicio["kind"] == "spawn_class" and inicio["hp"] == perfil.vida_max == 1500.0
    assert perfil.episodio.start_phase == 2 and perfil.fases_inicio == (2,)
    calculadora = CalculadoraRecompensa(resolver_final=resolver_final_de(perfil.episodio.clases_victoria),
                                        resolver_fase=resolver_fase_de(catalogo))
    primero = crudo("BP_Chamuco_C", hp=1500.0, hp_max=1500.0)
    calculadora.reiniciar(primero)
    assert "fase2" not in calculadora.calcular(primero).terminos  # arrancar en su fase no premia


def test_en_amlo_pasar_al_chamuco_es_un_hito_y_solo_la_muerte_del_chamuco_gana(catalogo: CatalogoJefes) -> None:
    perfil = catalogo.perfil("amlo")
    final = resolver_final_de(perfil.episodio.clases_victoria)
    calculadora = CalculadoraRecompensa(resolver_final=final, resolver_fase=resolver_fase_de(catalogo))
    calculadora.reiniciar(crudo("BP_AMLOBOSS_C"))
    hito = calculadora.calcular(crudo("BP_Chamuco_C"))
    assert hito.terminos["fase2"] == calculadora.config.fase2 and "victoria" not in hito.terminos
    assert final(crudo("BP_AMLOBOSS_C", dead=True)) is None
    assert final(crudo("BP_Chamuco_C", dead=True)) == "victoria"


def test_nahual_gana_con_la_muerte_del_chupacabras_y_no_con_la_de_otro(catalogo: CatalogoJefes) -> None:
    final = resolver_final_de(catalogo.perfil("nahual").episodio.clases_victoria)
    assert final(crudo("BP_CHUPACABRAS_C", dead=True)) == "victoria"
    assert final(crudo("BP_Chamuco_C", dead=True)) is None


def test_juice_gana_con_la_muerte_de_su_actor_de_fase_2(catalogo: CatalogoJefes) -> None:
    final = resolver_final_de(catalogo.perfil("juice").episodio.clases_victoria)
    assert final(crudo("BP_Enemy_Juice_C", dead=True)) is None
    assert final(crudo("BP_Enemy_JuicePhase2_C", dead=True)) == "victoria"


def test_la_vida_efectiva_de_cada_pelea_descarta_lo_que_se_pierde_al_transformarse(catalogo: CatalogoJefes) -> None:
    assert catalogo.vida_efectiva("juice", 1.0, 1) == 1700.0 + 1000.0
    assert catalogo.vida_efectiva("amlo", 1.0, 1) == 1600.0 + 1500.0
    assert catalogo.vida_efectiva("amlo", 1.0, 2) == 1500.0
    assert catalogo.vida_efectiva("nahual", 0.5, 1) == 1000.0


def test_el_proyectil_del_nahual_llega_mas_tarde_cuanto_mas_lejos(catalogo: CatalogoJefes) -> None:
    """Rafaga de fuego medida: posicion = base + distancia / 3050 uu/s."""
    montage_fuego = "AS_Mage_Elemental_Fire_Volley_01_Montage"
    cerca = SeguidorAmenazas(catalogo).actualizar(crudo("BP_CHUPACABRAS_C", 5.0, montage_fuego, 0.0, dist=200.0))
    lejos = SeguidorAmenazas(catalogo).actualizar(crudo("BP_CHUPACABRAS_C", 5.0, montage_fuego, 0.0, dist=2000.0))
    assert lejos.pendientes[0].segundos - cerca.pendientes[0].segundos == pytest.approx(1800.0 / 3050.0)
    assert [a.parry for a in cerca.pendientes] == [False, False, False]


def test_la_katana_lanzada_de_amlo_sigue_pendiente_al_terminar_su_montage(catalogo: CatalogoJefes) -> None:
    """El montage dura 0.4 s y la katana llega 0.156 s + distancia / 4975 uu/s despues de empezar."""
    seguidor = SeguidorAmenazas(catalogo)
    lanzamiento = "A_spear_1hand04_ThrowMid01_Montage"
    inicio = crudo("BP_AMLOBOSS_C", 8.0, lanzamiento, 0.0, dist=2000.0)
    inicio["boss"]["montage"]["rate"] = 2.0
    seguidor.actualizar(inicio)
    a = seguidor.actualizar(crudo("BP_AMLOBOSS_C", 8.45, None, dist=2000.0))
    assert a.proximo is not None and a.proximo.tipo == "proyectil"
    assert a.proximo.segundos == pytest.approx(0.156 + 2000.0 / 4975.0 - 0.45)
    assert not a.proximo.parry and a.proximo.esquivable


def test_el_agarre_a_distancia_de_amlo_es_un_golpe_no_parable_a_los_4_2_segundos(catalogo: CatalogoJefes) -> None:
    """Force Choke: arrastra al jugador y lo mata al final del montage si no esquiva."""
    golpe = catalogo.perfil("amlo").golpes_de("Paired_ForceChoke_Att_Montage")[0]
    assert golpe.pos == 4.2 and not golpe.parry and golpe.alcance == ALCANCE_GLOBAL
    a = SeguidorAmenazas(catalogo).actualizar(crudo("BP_AMLOBOSS_C", 10.0, "Paired_ForceChoke_Att_Montage", 3.0))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(1.2)


def test_los_eventos_de_las_katanas_del_piso_se_programan_por_evento(catalogo: CatalogoJefes) -> None:
    evento = {"id": "amlo_katanas_floor", "t": 8.0, "px": 0.0, "py": 0.0, "bx": 2000.0, "by": 0.0}
    a = SeguidorAmenazas(catalogo).actualizar(crudo("BP_AMLOBOSS_C", 8.0, remoto=[evento]))
    assert a.proximo is not None and a.proximo.segundos == pytest.approx(1.93)
    assert a.proximo.esquivable and not a.proximo.parry and a.proximo.centro == "jugador"


def test_golpes_que_llegan_a_cualquier_distancia_se_marcan_como_globales(catalogo: CatalogoJefes) -> None:
    lava = catalogo.perfil("chamuco").golpes_de("Anim_Spear_JumpAttackA_RM_Montage")[0]
    assert lava.tipo == "area" and lava.alcance == ALCANCE_GLOBAL and not lava.parry


def test_el_bot_esquiva_con_maniobra_el_golpe_de_piso_del_chamuco(catalogo: CatalogoJefes) -> None:
    seguidor = SeguidorAmenazas(catalogo)
    bot = BotJefes.desde_catalogo(catalogo)
    obs = crudo("BP_Chamuco_C", 3.0, "Anim_Spear_JumpAttackA_RM_Montage", 0.65)
    accion = bot.decidir(obs, [True] * acc.N_ACCIONES, seguidor.actualizar(obs))
    assert acc.ACCION_ROLL_PRIMERA <= accion < acc.ACCION_ATAQUE


def test_el_bot_no_marca_un_golpe_como_atendido_si_su_accion_no_se_ejecuto(catalogo: CatalogoJefes) -> None:
    """Defecto de etiquetas corregido: el parry no ejecutado se vuelve a pedir en el paso siguiente."""
    seguidor = SeguidorAmenazas(catalogo)
    bot = BotJefes.desde_catalogo(catalogo)
    mascara = [True] * acc.N_ACCIONES
    obs = crudo("BP_CHUPACABRAS_C", 1.0, "Anim_Whisper_Attack1_Montage", 0.5)
    assert bot.decidir(obs, mascara, seguidor.actualizar(obs)) == acc.ACCION_PARRY
    bot.confirmar(acc.ACCION_ROLL_PRIMERA)  # el ruido de ejecucion ejecuto otra cosa
    obs = crudo("BP_CHUPACABRAS_C", 1.067, "Anim_Whisper_Attack1_Montage", 0.567)
    assert bot.decidir(obs, mascara, seguidor.actualizar(obs)) == acc.ACCION_PARRY


def test_el_bot_no_bebe_con_un_impacto_a_distancia_pendiente(catalogo: CatalogoJefes) -> None:
    seguidor = SeguidorAmenazas(catalogo)
    bot = BotJefes.desde_catalogo(catalogo)
    evento = {"id": "amlo_katanas_floor", "t": 5.0, "px": 0.0, "py": 0.0, "bx": 900.0, "by": 0.0}
    obs = crudo("BP_AMLOBOSS_C", 5.0, dist=900.0, remoto=[evento])
    obs["player"]["hp"] = 30.0
    assert bot.decidir(obs, [True] * acc.N_ACCIONES, seguidor.actualizar(obs)) != acc.ACCION_CAGUAMA
    despejado = crudo("BP_AMLOBOSS_C", 9.0, dist=900.0)
    despejado["player"]["hp"] = 30.0
    assert bot.decidir(despejado, [True] * acc.N_ACCIONES, seguidor.actualizar(despejado)) == acc.ACCION_CAGUAMA


def test_cada_clase_de_jefe_pertenece_a_un_solo_perfil_y_los_aliados_a_ninguno(catalogo: CatalogoJefes) -> None:
    for perfil in catalogo.perfiles:
        for clase in perfil.clases:
            assert catalogo.por_clase(clase) is perfil
    assert catalogo.por_clase("BP_EnemyNPCFRIENDLY_C") is None
    assert Amenazas().pendientes == ()


def test_los_hooks_de_victoria_de_cada_jefe_apuntan_a_sus_clases_de_victoria(catalogo: CatalogoJefes) -> None:
    for identificador in ("amlo", "chamuco", "nahual"):
        episodio = catalogo.perfil(identificador).episodio
        victorias = [h for h in (episodio.mod or {})["hooks"] if h["event"] == "victory"]
        assert victorias, identificador
        for gancho in victorias:
            clase = gancho["path"].split(":")[0].rsplit(".", 1)[1]
            assert clase in episodio.clases_victoria
