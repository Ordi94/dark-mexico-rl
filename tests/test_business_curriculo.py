"""Reglas de negocio del curriculo de dificultad."""

from __future__ import annotations

import pytest

from dmrl.dominio.configuracion import ConfigCurriculo
from dmrl.dominio.curriculo import Curriculo, vida_efectiva_jefe

RAPIDO = ConfigCurriculo(niveles=(0.3, 0.5, 1.0), tasa_victoria_subida=0.6, ventana=10, minimo_episodios=10)


def jugar(curriculo: Curriculo, victorias: int, derrotas: int, fase: int = 1) -> bool:
    """
    Registra victorias y derrotas y dice si hubo subida de nivel.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        curriculo (Curriculo): Curriculo bajo prueba.
        victorias (int): Victorias a registrar.
        derrotas (int): Derrotas a registrar.
        fase (int): Fase de inicio de los episodios.

    Returns:
        bool: True si alguno de los episodios provoco una subida.

    Raises:
        Ninguna.

    Ejemplo:
        jugar(Curriculo(RAPIDO), 6, 4)
    """
    subio = False
    for _ in range(victorias):
        subio |= curriculo.registrar_episodio(True, fase)
    for _ in range(derrotas):
        subio |= curriculo.registrar_episodio(False, fase)
    return subio


def test_el_curriculo_empieza_con_el_jefe_al_30_por_ciento() -> None:
    """El primer nivel es 0.3 y el ultimo 1.0."""
    c = Curriculo()
    assert c.multiplicador == 0.3
    assert c.config.niveles[-1] == 1.0


def test_sube_de_nivel_al_alcanzar_la_tasa_de_victoria_en_la_ventana() -> None:
    """Con 6 victorias de 10 (60%) en la ventana llena se sube."""
    c = Curriculo(RAPIDO)
    assert jugar(c, 6, 4)
    assert c.nivel == 1 and c.multiplicador == 0.5


def test_no_sube_si_la_tasa_no_llega_al_umbral() -> None:
    """Con 5 victorias de 10 no se alcanza el 60%."""
    c = Curriculo(RAPIDO)
    assert not jugar(c, 5, 5)
    assert c.nivel == 0


def test_no_sube_antes_del_minimo_de_episodios_aunque_gane_todo() -> None:
    """Con menos de minimo_episodios no se evalua el umbral."""
    c = Curriculo(ConfigCurriculo(niveles=(0.5, 1.0), ventana=5, minimo_episodios=20))
    assert not jugar(c, 10, 0)


def test_al_subir_la_ventana_empieza_de_cero() -> None:
    """La tasa del nivel nuevo no hereda las victorias del anterior."""
    c = Curriculo(RAPIDO)
    jugar(c, 10, 0)
    assert c.nivel == 1
    assert c.tasa_victoria == 0.0 and c.episodios_en_nivel == 0


def test_los_episodios_que_empiezan_en_fase_2_no_cuentan_para_la_tasa() -> None:
    """Son un complemento de entrenamiento: no suben ni bajan el nivel."""
    c = Curriculo(RAPIDO)
    assert not jugar(c, 20, 0, fase=2)
    assert c.episodios_en_nivel == 0


def test_en_el_ultimo_nivel_no_hay_mas_subidas() -> None:
    """Con el jefe al 100% el curriculo esta completo y no avanza."""
    c = Curriculo(RAPIDO, nivel=2)
    assert c.completo
    assert not jugar(c, 30, 0)
    assert c.nivel == 2


def test_la_tasa_de_victoria_es_cero_con_la_ventana_vacia() -> None:
    """Sin episodios la tasa es 0.0."""
    assert Curriculo(RAPIDO).tasa_victoria == 0.0


def test_estado_guardado_y_restaurado_conserva_nivel_y_ventana() -> None:
    """Reanudar desde un checkpoint recupera el nivel y la ventana de resultados."""
    c = Curriculo(RAPIDO)
    jugar(c, 10, 0)
    jugar(c, 3, 2)
    d = Curriculo(RAPIDO)
    d.cargar(c.a_dict())
    assert d.nivel == c.nivel and d.episodios_en_nivel == 5
    assert d.tasa_victoria == pytest.approx(0.6)


def test_cargar_un_nivel_inexistente_falla() -> None:
    """Un checkpoint de una configuracion con mas niveles no se acepta en silencio."""
    with pytest.raises(ValueError):
        Curriculo(RAPIDO).cargar({"nivel": 7})


@pytest.mark.parametrize(
    "niveles", [(), (0.3, 0.5), (0.5, 0.3, 1.0), (0.0, 1.0), (0.3, 0.3, 1.0)],
)
def test_configuraciones_de_niveles_invalidas_se_rechazan(niveles: tuple[float, ...]) -> None:
    """Los niveles deben ser positivos, crecientes y terminar en 1.0."""
    with pytest.raises(ValueError):
        Curriculo(ConfigCurriculo(niveles=niveles))


def test_nivel_inicial_fuera_de_rango_se_rechaza() -> None:
    """No se puede arrancar en un nivel que no existe."""
    with pytest.raises(ValueError):
        Curriculo(RAPIDO, nivel=3)


def test_la_vida_efectiva_del_jefe_escala_con_el_multiplicador() -> None:
    """Fase 1 completa = 1700 + 1000 de vida util; empezar en fase 2 = 1000."""
    assert vida_efectiva_jefe(1.0, 1) == pytest.approx(2700.0)
    assert vida_efectiva_jefe(0.3, 1) == pytest.approx(810.0)
    assert vida_efectiva_jefe(0.5, 2) == pytest.approx(500.0)


def test_la_probabilidad_de_fase_2_por_defecto_es_del_20_por_ciento() -> None:
    """El curriculo por defecto mezcla un 20% de episodios que empiezan en la fase 2."""
    assert ConfigCurriculo().prob_fase2 == pytest.approx(0.2)


def test_no_sube_con_la_ventana_incompleta_aunque_se_cumpla_el_minimo() -> None:
    """Con ventana 10 y minimo 5, seis victorias seguidas no bastan: la ventana aun no se llena."""
    c = Curriculo(ConfigCurriculo(niveles=(0.5, 1.0), ventana=10, minimo_episodios=5))
    assert not jugar(c, 6, 0)
    assert c.nivel == 0
