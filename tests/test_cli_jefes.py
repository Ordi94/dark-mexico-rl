"""Comandos de la linea de comandos del entorno multi-jefe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dmrl.aplicacion import entrenamiento as ent
from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion import recoleccion_jefes as recj
from dmrl.entrada import cli
from tests.support_jefes import JuegoJefesSintetico, catalogo_prueba


def test_el_parser_conoce_los_comandos_de_jefes() -> None:
    p = cli.construir_parser()
    j = p.parse_args(["evaluar-jefes", "--jefe", "amlo"])
    assert (j.jefe, j.episodios, j.instancias, j.fase_inicio, j.epsilon_ruido) == ("amlo", 50, 4, None, 0.0)
    assert p.parse_args(["verificar-jefes", "--jefe", "nahual"]).jefe == "nahual"
    r = p.parse_args(["recolectar-jefes", "--jefes", "amlo", "nahual"])
    assert (r.jefes, r.episodios, r.nombre, r.ronda) == (["amlo", "nahual"], 5, "imitacion_v2", 0)
    assert p.parse_args(["especificacion-v2"]).comando == "especificacion-v2"
    with pytest.raises(SystemExit):
        p.parse_args(["evaluar-jefes"])


def test_evaluar_jefes_imprime_el_resumen_y_guarda_el_detalle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    llamadas: list[tuple[Any, ...]] = []

    def evaluar(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        llamadas.append((args, kwargs))
        return [{"victoria": True, "muerte": False, "danio_hecho": 5.0}]

    monkeypatch.setattr(evj, "evaluar_jefe_en_paralelo", evaluar)
    monkeypatch.setattr(cli, "cargar_catalogo", catalogo_prueba)
    salida = tmp_path / "x" / "r.json"
    assert cli.main(["evaluar-jefes", "--jefe", "amlo", "--episodios", "3", "--instancias", "2", "--fase-inicio", "2",
                     "--epsilon-ruido", "0.05", "--salida", str(salida)]) == 0
    assert llamadas[0][0] == ("amlo", 3, 2, 0, {"start_phase": 2}) and llamadas[0][1] == {"epsilon_ruido": 0.05}
    assert json.loads(salida.read_text())["resumen"]["tasa_victoria"] == 1.0
    assert "tasa_victoria" in capsys.readouterr().out
    cli.main(["evaluar-jefes", "--jefe", "nahual"])
    with pytest.raises(KeyError):
        cli.main(["evaluar-jefes", "--jefe", "no_existe"])


def test_verificar_jefes_corre_check_env_y_cierra(monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "cargar_catalogo", catalogo_prueba)
    monkeypatch.setattr(evj, "crear_fabrica_puerto", lambda params: (lambda: JuegoJefesSintetico()))
    assert cli.main(["verificar-jefes", "--jefe", "nahual"]) == 0
    assert "check_env (nahual): OK" in capsys.readouterr().out


def test_especificacion_v2_escribe_el_documento(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destino = tmp_path / "docs" / "ESPECIFICACION_OBSERVACION_V2.md"
    monkeypatch.setattr(cli, "ARCHIVO_ESPECIFICACION_V2", destino)
    monkeypatch.setattr(cli, "cargar_catalogo", catalogo_prueba)
    assert cli.main(["especificacion-v2"]) == 0
    assert "tiempo_episodio" in destino.read_text(encoding="utf-8")


def test_recolectar_jefes_informa_lo_que_falta_y_el_resumen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "cargar_catalogo", catalogo_prueba)
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path)
    monkeypatch.setattr(cli, "ajustar_proceso_actual", lambda *a, **k: None)
    faltan = {"n": 0}
    monkeypatch.setattr(recj, "recolectar_jefes", lambda *a, **k: faltan["n"])
    monkeypatch.setattr(recj, "resumen_de_ronda", lambda *a, **k: {"amlo": {"episodios": 5.0}})
    assert cli.main(["recolectar-jefes", "--jefes", "amlo", "--episodios", "5"]) == 0
    salida = json.loads(capsys.readouterr().out)
    assert salida["faltan"] == 0 and salida["resumen"]["amlo"]["episodios"] == 5.0
    faltan["n"] = 2
    assert cli.main(["recolectar-jefes", "--jefes", "amlo"]) == 2
    with pytest.raises(KeyError):
        cli.main(["recolectar-jefes", "--jefes", "no_existe"])
