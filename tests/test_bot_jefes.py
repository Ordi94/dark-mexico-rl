"""Bot de reglas multi-jefe: reglas de reaccion, contraataque, curacion y ofensiva."""

from __future__ import annotations

from typing import Any

import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio.amenazas import Amenaza, Amenazas
from dmrl.dominio.bot_jefes import BotJefes, ParametrosBot
from dmrl.dominio.jefes import PerfilJefe
from tests.support_dmrl import jefe, jugador
from tests.support_jefes import catalogo_prueba, registro

MASCARA_TODA = [True] * acc.N_ACCIONES


def amenaza(segundos: float, parry: bool = True, tipo: str = "melee", clave: str = "a", danio: float = 25.0,
            origen: str = "montage") -> Amenaza:
    """Amenaza de prueba con clave propia."""
    return Amenaza((clave,), segundos, parry, danio, 200.0 if tipo == "melee" else 6000.0, tipo, not parry, origen, 0)


def crudo(dist: float = 100.0, **jug: Any) -> dict[str, Any]:
    return {"player": jugador(**jug), "boss": jefe(), "dist": dist, "ev": {}}


def mascara(**apagadas: bool) -> list[bool]:
    m = list(MASCARA_TODA)
    for nombre, valor in apagadas.items():
        m[acc.NOMBRES_ACCIONES.index(nombre)] = valor
    return m


def test_parry_ante_golpe_parable_inminente_y_espera_si_aun_falta() -> None:
    bot = BotJefes()
    assert bot.decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.40),), 100.0)) == acc.ACCION_NADA
    assert bot.decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.25),), 100.0)) == acc.ACCION_PARRY


def test_golpe_lejano_no_provoca_reaccion() -> None:
    bot = BotJefes()
    accion = bot.decidir(crudo(dist=100.0, stamina=1.0), MASCARA_TODA, Amenazas((amenaza(3.0),), 100.0))
    assert accion == acc.ACCION_ATAQUE or accion == acc.ACCION_ESPECIAL


def test_roll_lateral_ante_golpe_no_parable() -> None:
    bot = BotJefes()
    amenazas = Amenazas((amenaza(0.15, parry=False),), 100.0)
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_ROLL_PRIMERA + 2
    bot2 = BotJefes()
    assert bot2.decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.30, parry=False),), 100.0)) == acc.ACCION_NADA


def test_direccion_de_la_maniobra_depende_del_tipo_de_amenaza() -> None:
    parametros = ParametrosBot(direccion_roll_melee=0, direccion_roll_remoto=4)
    melee = BotJefes(parametros).decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.1, False),), 100.0))
    remoto = BotJefes(parametros).decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.1, False, "rayo"),), 100.0))
    assert melee == acc.ACCION_ROLL_PRIMERA + 0 and remoto == acc.ACCION_ROLL_PRIMERA + 4


def test_golpe_parable_con_parry_bloqueado_se_esquiva() -> None:
    bot = BotJefes()
    accion = bot.decidir(crudo(), mascara(parry=False), Amenazas((amenaza(0.1, parry=True),), 100.0))
    assert accion == acc.ACCION_ROLL_PRIMERA + 2


def test_sin_maniobra_valida_espera() -> None:
    bot = BotJefes()
    m = mascara(**{f"roll_{i * 45}": False for i in range(8)})
    assert bot.decidir(crudo(), m, Amenazas((amenaza(0.1, False),), 100.0)) == acc.ACCION_NADA


def test_ignora_golpes_melee_fuera_de_alcance_si_el_jefe_lo_pide() -> None:
    lejos = Amenazas((amenaza(0.1, parry=True),), 3000.0)
    ignorante = BotJefes(ParametrosBot(ignora_fuera_de_alcance=1.0))
    assert ignorante.decidir(crudo(3000.0), MASCARA_TODA, lejos) != acc.ACCION_PARRY
    assert BotJefes().decidir(crudo(3000.0), MASCARA_TODA, lejos) == acc.ACCION_PARRY


def test_marca_de_atendido_solo_se_confirma_si_la_accion_se_ejecuto() -> None:
    bot = BotJefes()
    amenazas = Amenazas((amenaza(0.2, clave="g1"),), 100.0)
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_PARRY
    bot.confirmar(acc.ACCION_ATAQUE)  # el ruido de ejecucion cambio la accion
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_PARRY  # sigue sin atender
    bot.confirmar(acc.ACCION_PARRY)
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_NADA  # ya atendido


def test_sin_confirmar_se_da_por_ejecutada_la_accion_pedida() -> None:
    bot = BotJefes()
    amenazas = Amenazas((amenaza(0.2, clave="g1"),), 100.0)
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_PARRY
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_NADA


def test_las_marcas_se_podan_cuando_el_impacto_desaparece() -> None:
    bot = BotJefes()
    bot.decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.2, clave="g1"),), 100.0))
    bot.confirmar(acc.ACCION_PARRY)
    bot.decidir(crudo(), MASCARA_TODA, Amenazas((), 100.0))
    # Una amenaza nueva con la misma clave (otro ataque) vuelve a atenderse.
    assert bot.decidir(crudo(), MASCARA_TODA, Amenazas((amenaza(0.2, clave="g1"),), 100.0)) == acc.ACCION_PARRY


def test_reiniciar_olvida_las_marcas() -> None:
    bot = BotJefes()
    amenazas = Amenazas((amenaza(0.2, clave="g1"),), 100.0)
    bot.decidir(crudo(), MASCARA_TODA, amenazas)
    bot.confirmar(acc.ACCION_PARRY)
    bot.reiniciar()
    assert bot.decidir(crudo(), MASCARA_TODA, amenazas) == acc.ACCION_PARRY


def test_jugador_muerto_o_sin_jefe_no_hace_nada_y_bebiendo_se_aleja() -> None:
    bot = BotJefes()
    assert bot.decidir(crudo(dead=True), MASCARA_TODA, Amenazas()) == acc.ACCION_NADA
    assert bot.decidir({"player": jugador(), "boss": None, "dist": 0.0, "ev": {}}, MASCARA_TODA, Amenazas()) == 0
    assert bot.decidir(crudo(drinking=True), MASCARA_TODA, Amenazas()) == acc.ACCION_MOVER_PRIMERA + 4


def test_contraataque_con_el_jefe_aturdido() -> None:
    bot = BotJefes()
    obs = crudo(dist=100.0)
    obs["boss"]["stunned"] = True
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) == acc.ACCION_ESPECIAL
    assert bot.decidir(obs, mascara(especial=False), Amenazas()) == acc.ACCION_ATAQUE
    assert bot.decidir(obs, mascara(especial=False, ataque=False), Amenazas()) == acc.ACCION_NADA
    lejos = crudo(dist=600.0)
    lejos["boss"]["stunned"] = True
    assert bot.decidir(lejos, MASCARA_TODA, Amenazas()) == acc.ACCION_MOVER_PRIMERA
    herido = crudo(dist=100.0, hp=30.0)
    herido["boss"]["stunned"] = True
    assert bot.decidir(herido, MASCARA_TODA, Amenazas()) == acc.ACCION_CAGUAMA


def test_curacion_solo_sin_impactos_en_lo_que_dura_la_bebida() -> None:
    bot = BotJefes()
    herido = crudo(dist=600.0, hp=30.0)
    assert bot.decidir(herido, MASCARA_TODA, Amenazas()) == acc.ACCION_CAGUAMA
    con_amenaza = Amenazas((amenaza(1.5, parry=False, tipo="rayo", origen="evento"),), 600.0)
    assert bot.decidir(herido, MASCARA_TODA, con_amenaza) != acc.ACCION_CAGUAMA
    cerca = crudo(dist=200.0, hp=30.0)
    assert bot.decidir(cerca, MASCARA_TODA, Amenazas()) == acc.ACCION_MOVER_PRIMERA + 4
    atacando = crudo(dist=600.0, hp=30.0)
    atacando["boss"]["attacking"] = True
    assert bot.decidir(atacando, MASCARA_TODA, Amenazas()) != acc.ACCION_CAGUAMA
    assert bot.decidir(crudo(dist=600.0, hp=30.0, caguamas=0), MASCARA_TODA, Amenazas()) != acc.ACCION_CAGUAMA
    assert bot.decidir(herido, mascara(caguama=False), Amenazas()) != acc.ACCION_CAGUAMA


def test_ofensiva_se_acerca_ataca_y_respeta_la_reserva_de_resistencia() -> None:
    bot = BotJefes()
    assert bot.decidir(crudo(dist=500.0), MASCARA_TODA, Amenazas()) == acc.ACCION_MOVER_PRIMERA
    assert bot.decidir(crudo(dist=100.0), MASCARA_TODA, Amenazas()) == acc.ACCION_ESPECIAL
    assert bot.decidir(crudo(dist=100.0), mascara(especial=False), Amenazas()) == acc.ACCION_ATAQUE
    assert bot.decidir(crudo(dist=100.0, stamina=0.2), MASCARA_TODA, Amenazas()) == acc.ACCION_NADA
    assert bot.decidir(crudo(dist=100.0), mascara(especial=False, ataque=False), Amenazas()) == acc.ACCION_NADA
    assert bot.decidir(crudo(dist=100.0), MASCARA_TODA, Amenazas((amenaza(1.0),), 100.0)) == acc.ACCION_NADA


def test_especial_seguro_evita_el_especial_si_un_impacto_llega_antes() -> None:
    bot = BotJefes(ParametrosBot(especial_seguro=2.7, ventana_segura=0.5))
    pronto = Amenazas((amenaza(1.5),), 100.0)
    assert bot.decidir(crudo(dist=100.0), MASCARA_TODA, pronto) == acc.ACCION_ATAQUE
    lejos = Amenazas((amenaza(3.5),), 100.0)
    assert bot.decidir(crudo(dist=100.0), MASCARA_TODA, lejos) == acc.ACCION_ESPECIAL


def test_parametros_desde_el_registro_del_jefe() -> None:
    datos = registro("amlo", 2)
    datos["bot"] = {"parry_disparo": 0.3, "direccion_roll_remoto": 4, "especial_seguro": 2.7}
    parametros = ParametrosBot.desde_perfil(PerfilJefe.desde_dict(datos))
    assert parametros.parry_disparo == 0.3 and parametros.direccion_roll_remoto == 4
    assert isinstance(parametros.direccion_roll_remoto, int) and parametros.roll_disparo == 0.20
    datos["bot"] = {"inventado": 1.0}
    with pytest.raises(ValueError, match="desconocidos"):
        ParametrosBot.desde_perfil(PerfilJefe.desde_dict(datos))


def test_el_bot_usa_los_parametros_del_actor_que_tiene_delante() -> None:
    catalogo = catalogo_prueba()
    bot = BotJefes.desde_catalogo(catalogo)
    obs = crudo()
    obs["boss"]["cls"] = "BP_AMLOBOSS_C"
    bot.decidir(obs, MASCARA_TODA, Amenazas())
    assert bot.parametros == ParametrosBot()
    por_clase = {"BP_Chamuco_C": ParametrosBot(parry_disparo=0.1)}
    especial = BotJefes(por_clase=por_clase)
    obs["boss"]["cls"] = "BP_Chamuco_C"
    especial.decidir(obs, MASCARA_TODA, Amenazas())
    assert especial.parametros.parry_disparo == 0.1


def test_pausa_ofensiva_con_poca_vida_del_jefe_y_fin_de_la_pausa() -> None:
    """La IA del jefe no cambia de fase si lo aturdes sin parar: bajo la fraccion el bot deja de atacar."""
    bot = BotJefes(ParametrosBot(pausa_bajo_vida=0.2, pausa_maxima=10.0))
    obs = crudo(dist=100.0)
    obs["boss"].update(hp=300.0, hp_max=2000.0)
    obs["t"] = 100.0
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) == acc.ACCION_NADA
    obs["t"] = 105.0
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) == acc.ACCION_NADA
    obs["t"] = 111.0  # la pausa dura como mucho pausa_maxima
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) in (acc.ACCION_ATAQUE, acc.ACCION_ESPECIAL)


def test_la_pausa_tampoco_contraataca_y_se_reinicia_si_la_vida_sube() -> None:
    bot = BotJefes(ParametrosBot(pausa_bajo_vida=0.2))
    obs = crudo(dist=100.0)
    obs["boss"].update(hp=300.0, hp_max=2000.0, stunned=True)
    obs["t"] = 1.0
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) == acc.ACCION_NADA
    obs["boss"].update(hp=1800.0, stunned=False)
    obs["t"] = 2.0
    assert bot.decidir(obs, MASCARA_TODA, Amenazas()) in (acc.ACCION_ATAQUE, acc.ACCION_ESPECIAL)
    assert bot._t_pausa is None


def test_en_pausa_sigue_reaccionando_a_los_golpes() -> None:
    bot = BotJefes(ParametrosBot(pausa_bajo_vida=0.2))
    obs = crudo(dist=100.0)
    obs["boss"].update(hp=100.0, hp_max=2000.0)
    assert bot.decidir(obs, MASCARA_TODA, Amenazas((amenaza(0.2),), 100.0)) == acc.ACCION_PARRY


def test_sin_pausa_configurada_no_se_pausa_nunca() -> None:
    obs = crudo(dist=100.0)
    obs["boss"].update(hp=1.0, hp_max=2000.0)
    assert BotJefes().decidir(obs, MASCARA_TODA, Amenazas()) in (acc.ACCION_ATAQUE, acc.ACCION_ESPECIAL)
