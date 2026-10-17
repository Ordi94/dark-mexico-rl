"""Victoria, derrota y fase por jefe de episodio."""

from __future__ import annotations

from dmrl.dominio.reglas_jefes import resolver_fase_de, resolver_final_de
from tests.support_dmrl import jefe, jugador
from tests.support_jefes import catalogo_prueba


def obs(cls: str = "BP_Chamuco_C", dead: bool = False, jug_muerto: bool = False, **ev: object) -> dict[str, object]:
    return {"player": jugador(dead=jug_muerto), "boss": jefe(cls=cls, dead=dead), "ev": dict(ev)}


def test_la_muerte_de_una_clase_de_victoria_gana() -> None:
    final = resolver_final_de(("BP_Chamuco_C",))
    assert final(obs(dead=True)) == "victoria"
    assert final(obs(victory=True)) == "victoria"
    assert final(obs()) is None


def test_la_muerte_de_otra_clase_no_gana() -> None:
    final = resolver_final_de(("BP_Chamuco_C",))
    assert final(obs(cls="BP_AMLOBOSS_C", dead=True)) is None


def test_si_el_jugador_muere_en_el_mismo_paso_cuenta_la_muerte() -> None:
    final = resolver_final_de(("BP_Chamuco_C",))
    assert final(obs(dead=True, jug_muerto=True)) == "muerte"
    assert final(obs(player_death=True, victory=True)) == "muerte"


def test_fase_de_la_pelea_sale_de_la_clase_del_actor() -> None:
    fase = resolver_fase_de(catalogo_prueba())
    assert fase({"boss": jefe(cls="BP_AMLOBOSS_C")}, 1) == 1
    assert fase({"boss": jefe(cls="BP_Chamuco_C")}, 1) == 2
    assert fase({"boss": jefe(cls="BP_Enemy_JuicePhase2_C")}, 1) == 2
    assert fase({"boss": jefe(cls="BP_Desconocido_C")}, 1) == 1
    assert fase({}, 2) == 2
