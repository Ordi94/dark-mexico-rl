"""Recoleccion de demostraciones sobre un juego sintetico: lotes, reanudacion, errores y parada."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dmrl.aplicacion import recoleccion as rec
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio import acciones as acc
from dmrl.dominio.imitacion import PlanEpisodio, planificar_episodios
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.demostraciones import cargar_lote, lotes_de_ronda, metadatos_de_ronda
from tests.support_dmrl import JuegoSintetico
from tests.support_imitacion import ExpertoFijo


class ExpertoProhibido:
    """Experto de mentira cuya accion la mascara siempre rechaza (caguama con la vida llena)."""

    def reiniciar(self) -> None:
        """
        Sin estado.

        Contexto:
            Cumple la interfaz del bot.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ExpertoProhibido().reiniciar()
        """

    def decidir(self, obs: dict[str, Any], mascara: list[bool]) -> int:
        """
        Pide tomar caguama.

        Contexto:
            Bucle de recoleccion.

        Args:
            obs (dict[str, Any]): Observacion cruda (no usada).
            mascara (list[bool]): Acciones validas (no usada).

        Returns:
            int: La accion de caguama.

        Raises:
            Ninguna.

        Ejemplo:
            ExpertoProhibido().decidir({}, [])
        """
        return acc.ACCION_CAGUAMA


class ModeloFijo:
    """Modelo de mentira que siempre propone el parry."""

    def predict(self, obs: Any, action_masks: Any = None, deterministic: bool = True) -> tuple[Any, None]:
        """
        Accion fija.

        Contexto:
            Sustituye a MaskablePPO.predict.

        Args:
            obs (Any): Observacion (no usada).
            action_masks (Any): Mascara (no usada).
            deterministic (bool): Ignorado.

        Returns:
            tuple[Any, None]: Parry y None.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return np.asarray(acc.ACCION_PARRY), None


def parchear_modelo(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Sustituye la carga de MaskablePPO por un modelo fijo y evita tocar los hilos de torch.

    Contexto:
        Pruebas de recoleccion con DAgger.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        parchear_modelo(monkeypatch)
    """
    import sb3_contrib
    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(sb3_contrib.MaskablePPO, "load", lambda ruta, device: ModeloFijo())


def preparar(monkeypatch: pytest.MonkeyPatch, falla_en: int | None = None) -> list[str]:
    """
    Sustituye lo que toca el sistema en ejecutar_recoleccion.

    Contexto:
        Pruebas de ejecutar_recoleccion y recolectar_ronda.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        falla_en (int, opcional): Paso en el que cada instancia sintetica falla una vez.

    Returns:
        list[str]: Ids de instancia pedidos a la fabrica.

    Raises:
        Ninguna.

    Ejemplo:
        ids = preparar(monkeypatch)
    """
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        """
        Fabrica de juegos sinteticos que registra el id.

        Contexto:
            Sustituye a crear_fabrica_puerto.

        Args:
            params (Any): ParametrosInstancia.

        Returns:
            Callable: Constructor de juegos sinteticos.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ids.append(params.id_instancia)
        return lambda: JuegoSintetico(periodo=3, falla_en=falla_en)

    monkeypatch.setattr(rec, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(rec, "cargar_tabla_golpes", lambda: TablaGolpes())
    monkeypatch.setattr(rec, "ajustar_proceso_actual", lambda *a, **k: None)
    return ids


def test_un_episodio_con_error_del_juego_no_produce_datos() -> None:
    """Si la instancia se cae a mitad del episodio los pasos se descartan."""
    env = DarkMexicoEnv(lambda: JuegoSintetico(periodo=3, falla_en=2), TablaGolpes())
    plan = PlanEpisodio(0, 1, 0.0, 0)
    demostraciones, meta = rec.recolectar_episodio(env, ExpertoFijo(), None, plan, 0)  # type: ignore[arg-type]
    assert demostraciones is None and meta["resultado"] == "error"


def test_una_etiqueta_prohibida_del_experto_se_guarda_como_nada() -> None:
    """Si la mascara rechaza la accion del experto, se guarda "nada" (lo que el entorno ejecutaria)."""
    env = DarkMexicoEnv(lambda: JuegoSintetico(periodo=3), TablaGolpes())
    demostraciones, _ = rec.recolectar_episodio(env, ExpertoProhibido(), None, PlanEpisodio(0, 1, 0.0, 0), 0)  # type: ignore[arg-type]
    assert demostraciones is not None
    assert acc.ACCION_NADA in set(demostraciones.etiquetas.tolist())
    demostraciones.validar()


def test_repartir_planes_alterna_y_no_deja_grupos_vacios() -> None:
    """Round-robin entre trabajadores; trabajadores < 1 es un error."""
    planes = planificar_episodios(5, 0.2, 0.5, 0.05, 0.2, 0)
    grupos = rec.repartir_planes(planes, 3)
    assert [len(g) for g in grupos] == [2, 2, 1] and grupos[0][1].indice == 3
    assert len(rec.repartir_planes(planes[:2], 4)) == 2
    with pytest.raises(ValueError):
        rec.repartir_planes(planes, 0)


def test_ejecutar_recoleccion_guarda_lotes_con_el_id_de_su_instancia(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    """5 episodios con lotes de 2 dejan 3 archivos y devuelven los 5 metadatos."""
    ids = preparar(monkeypatch)
    planes = tuple(planificar_episodios(5, 0.2, 0.5, 0.05, 0.2, 0))
    trabajo = rec.TrabajoRecoleccion(indice=1, ronda=0, planes=planes, carpeta_datos=str(tmp_path), prefijo_id="x",
                                     lote_episodios=2)
    guardados = rec.ejecutar_recoleccion(trabajo)
    assert ids == ["x1"] and len(guardados) == 5
    assert len(lotes_de_ronda(tmp_path, 0)) == 3
    assert sorted(m["indice"] for m in metadatos_de_ronda(tmp_path, 0)) == [0, 1, 2, 3, 4]
    cargar_lote(lotes_de_ronda(tmp_path, 0)[0])[0].validar()


def test_ejecutar_recoleccion_con_modelo_ejecuta_sus_acciones(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    """Con ruta de modelo, las acciones ejecutadas son las del modelo y las etiquetas siguen siendo del bot."""
    preparar(monkeypatch)
    parchear_modelo(monkeypatch)
    plan = tuple(planificar_episodios(2, 0.0, 0.0, 0.05, 0.2, 0))
    rec.ejecutar_recoleccion(rec.TrabajoRecoleccion(0, 1, plan, str(tmp_path), ruta_modelo="m.zip"))
    datos = cargar_lote(lotes_de_ronda(tmp_path, 1)[0])[0]
    assert set(datos.ejecutadas.tolist()) <= {acc.ACCION_PARRY, acc.ACCION_NADA}
    assert acc.ACCION_PARRY in set(datos.ejecutadas.tolist())


def test_el_archivo_de_parada_detiene_antes_del_siguiente_episodio(monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    """Con la parada pedida no se juega ningun episodio y el juego se cierra igual."""
    preparar(monkeypatch)
    parada = tmp_path / "PARAR"
    parada.write_text("", encoding="utf-8")
    planes = tuple(planificar_episodios(3, 0.2, 0.5, 0.05, 0.2, 0))
    trabajo = rec.TrabajoRecoleccion(0, 0, planes, str(tmp_path / "d"), archivo_parada=str(parada))
    assert rec.ejecutar_recoleccion(trabajo) == []


def test_un_episodio_con_error_no_se_guarda_y_el_trabajador_sigue(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    """Todos los episodios caen por error: no hay lotes, pero el trabajador termina limpio."""
    preparar(monkeypatch, falla_en=2)
    planes = tuple(planificar_episodios(2, 0.2, 0.5, 0.05, 0.2, 0))
    assert rec.ejecutar_recoleccion(rec.TrabajoRecoleccion(0, 0, planes, str(tmp_path))) == []
    assert lotes_de_ronda(tmp_path, 0) == []


def test_recolectar_ronda_reparte_y_es_reanudable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Una segunda llamada no repite los episodios ya guardados."""
    ids = preparar(monkeypatch)
    planes = planificar_episodios(6, 0.2, 0.5, 0.05, 0.2, 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert rec.recolectar_ronda(tmp_path, 0, planes[:4], 2, ejecutor=pool) == 0
        antes = len(ids)
        assert rec.recolectar_ronda(tmp_path, 0, planes[:4], 2, ejecutor=pool) == 0
        assert len(ids) == antes
        assert rec.recolectar_ronda(tmp_path, 0, planes, 2, ejecutor=pool) == 0
    assert sorted(m["indice"] for m in metadatos_de_ronda(tmp_path, 0)) == list(range(6))


def test_recolectar_ronda_reintenta_y_reporta_lo_que_falta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Si los episodios siguen fallando se agotan las pasadas y se informa cuantos faltan."""
    ids = preparar(monkeypatch, falla_en=2)
    planes = planificar_episodios(2, 0.2, 0.5, 0.05, 0.2, 0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert rec.recolectar_ronda(tmp_path, 0, planes, 2, ejecutor=pool) == 2
    assert len(ids) == 2 * rec.MAX_PASADAS


def test_recolectar_ronda_respeta_la_parada_y_crea_su_ejecutor(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    """Con la parada ya pedida no lanza nada; sin ejecutor usa procesos spawn (sustituidos por hilos aqui)."""
    preparar(monkeypatch)
    planes = planificar_episodios(2, 0.2, 0.5, 0.05, 0.2, 0)
    parada = tmp_path / "PARAR"
    parada.write_text("", encoding="utf-8")
    assert rec.recolectar_ronda(tmp_path / "d", 0, planes, 2, archivo_parada=parada) == 2
    parada.unlink()
    creados: list[Any] = []

    class EjecutorHilos(ThreadPoolExecutor):
        """Ejecutor de hilos con la firma de ProcessPoolExecutor."""

        def __init__(self, max_workers: int, mp_context: Any) -> None:
            """
            Crea el ejecutor registrando el contexto.

            Contexto:
                Sustituye a ProcessPoolExecutor.

            Args:
                max_workers (int): Trabajadores.
                mp_context (Any): Contexto de multiproceso pedido.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            super().__init__(max_workers=max_workers)
            creados.append(mp_context)

    monkeypatch.setattr(rec, "ProcessPoolExecutor", EjecutorHilos)
    parchear_modelo(monkeypatch)
    assert rec.recolectar_ronda(tmp_path / "d", 0, planes, 2, ruta_modelo=Path("m.zip"), archivo_parada=parada) == 0
    assert len(creados) == 1
