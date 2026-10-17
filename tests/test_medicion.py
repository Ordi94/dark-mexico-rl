"""Agregacion de mediciones de combate: golpes por montage, retardos remotos y fusion con el registro."""

from __future__ import annotations

import pytest

from dmrl.dominio import medicion as med


def golpe(am: str, ap: float, amount: float = 30.0, parry: bool = True, d: float = 100.0, att: str = "B",
          t: float = 0.0, **extra: object) -> dict[str, object]:
    return {"att": att, "am": am, "ap": ap, "amount": amount, "parry": parry, "d": d, "t": t, **extra}


def test_percentil_interpola_y_valida() -> None:
    assert med.percentil([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert med.percentil([5.0], 95.0) == 5.0
    with pytest.raises(ValueError):
        med.percentil([], 50.0)
    with pytest.raises(ValueError):
        med.percentil([1.0], 101.0)


def test_agrupar_junta_impactos_cercanos_y_separa_los_lejanos() -> None:
    golpes = [golpe("M", 0.50, d=100), golpe("M", 0.52, d=120), golpe("M", 1.20, parry=False, amount=40.0, d=300),
              golpe("M", 1.22, parry=False, amount=40.0, d=310), golpe("M", 1.21, parry=True, d=90)]
    grupos = med.agrupar_golpes(golpes, ["B"])["M"]
    assert [g["n"] for g in grupos] == [2, 3]
    assert grupos[0]["pos"] == pytest.approx(0.51) and grupos[0]["parry"] is True
    assert grupos[1]["parry"] is False and grupos[1]["dano"] == 40.0
    assert grupos[1]["alcance"] == 310.0


def test_agrupar_ignora_otras_clases_y_golpes_sin_montage() -> None:
    golpes = [golpe("M", 0.5, att="OTRO"), {"att": "B", "am": None, "ap": 0.0, "amount": 10, "parry": False},
              golpe("M", 0.5, att="B")]
    assert list(med.agrupar_golpes(golpes, ["B"])) == ["M"]
    assert med.agrupar_golpes(golpes, ["Z"]) == {}


def test_agrupar_sin_distancia_omite_el_alcance() -> None:
    sin_d = {"att": "B", "am": "M", "ap": 1.0, "amount": 5.0, "parry": True}
    assert "alcance" not in med.agrupar_golpes([sin_d], ["B"])["M"][0]


def test_retardos_miden_evento_a_impacto_incluso_con_causante_nulo() -> None:
    eventos = [{"id": "kat", "t": 10.0, "px": 0.0, "py": 0.0, "bx": 300.0, "by": 400.0}, {"id": "kat", "t": 20.0}]
    golpes = [{"t": 10.9, "att": "B", "am": None, "d": 800.0}, {"t": 22.0, "att": None, "am": None},
              {"t": 30.0, "att": "B", "am": None}, {"t": 10.5, "att": "B", "am": "M"},
              {"t": 10.6, "att": "OTRO", "am": None}]
    r = med.retardos_remotos(eventos, golpes, ["B"])
    assert r["kat"][0]["retardo"] == pytest.approx(0.9) and r["kat"][0]["distancia_lanzamiento"] == 500.0
    assert r["kat"][1]["retardo"] == pytest.approx(2.0) and "distancia_lanzamiento" not in r["kat"][1]
    assert len(r["kat"]) == 2


def test_resumen_de_retardos() -> None:
    r = med.resumir_retardos({"e": [{"retardo": 1.0}, {"retardo": 3.0}], "vacio": []})
    assert r["e"]["mediana"] == 2.0 and r["e"]["minimo"] == 1.0 and r["e"]["maximo"] == 3.0 and "vacio" not in r


def test_fusion_reemplaza_el_golpe_cercano_y_conserva_tipo_y_post_montage() -> None:
    registro = {"M": [{"pos": 1.0, "parry": False, "tipo": "melee", "alcance": 350.0, "dano": 10.0},
                      {"pos": 3.0, "parry": True, "tipo": "melee"}]}
    medidos = {"M": [{"pos": 1.05, "parry": True, "dano": 30.0, "n": 5, "alcance": 140.0}]}
    fusion = med.fusionar_golpes(registro, medidos)
    assert fusion["M"][0]["pos"] == 1.05 and fusion["M"][0]["parry"] is True and fusion["M"][0]["alcance"] == 140.0
    assert fusion["M"][1]["pos"] == 3.0  # lo no cubierto se conserva


def test_fusion_agrega_golpes_nuevos_solo_con_observaciones_suficientes() -> None:
    medidos = {"N": [{"pos": 0.5, "parry": True, "dano": 20.0, "n": 4}, {"pos": 0.9, "parry": True, "n": 1}]}
    fusion = med.fusionar_golpes({}, medidos)
    assert [g["pos"] for g in fusion["N"]] == [0.5] and fusion["N"][0]["tipo"] == "melee"
    assert med.fusionar_golpes({}, medidos, agregar=False) == {}


def test_fusion_no_toca_los_montages_curados_que_no_son_melee() -> None:
    registro = {"R": [{"pos": 1.2, "parry": False, "tipo": "rayo", "alcance": 6000.0, "post_montage": True}]}
    medidos = {"R": [{"pos": 1.216, "parry": False, "n": 40, "alcance": 800.0}]}
    assert med.fusionar_golpes(registro, medidos) == registro


def test_fusion_conserva_el_alcance_de_un_golpe_de_area_dentro_de_un_montage_mixto() -> None:
    registro = {"M": [{"pos": 0.8, "parry": False, "tipo": "area", "alcance": 6000.0, "post_montage": False},
                      {"pos": 0.2, "parry": True, "tipo": "melee"}]}
    medidos = {"M": [{"pos": 0.81, "parry": False, "n": 9, "alcance": 400.0}]}
    fusion = med.fusionar_golpes(registro, medidos)
    area = next(g for g in fusion["M"] if g["tipo"] == "area")
    assert area["alcance"] == 6000.0 and area["pos"] == 0.81
