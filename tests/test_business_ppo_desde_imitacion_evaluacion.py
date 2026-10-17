"""
Reglas de negocio de la evaluacion automatica del PPO: compuerta de promocion, correccion por
regresion, descarte de evaluaciones fallidas y reanudacion con el estado guardado.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from sb3_contrib import MaskablePPO

from dmrl.aplicacion import callbacks_jefes as cj
from dmrl.aplicacion.callbacks_jefes import CallbackEvaluacionPeriodica
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from tests.support_ppo import (
    LanzadorFalso,
    ModeloFalso,
    RelojFalso,
    cargar_modelo_prueba,
    config_ppo,
    pesos_de,
    son_iguales,
)

REFERENCIA = {"amlo": 0.5, "nahual": 0.5}
CLARAMENTE_PEOR = {"amlo": 0.2, "nahual": 0.2}


def conectar(tmp_path: Path, modelo: Any, config: ConfigPPOJefes | None = None, lanzador: LanzadorFalso | None = None,
             reloj: RelojFalso | None = None) -> CallbackEvaluacionPeriodica:
    """
    Callback de evaluacion conectado a un modelo, con el plazo de una evaluacion por minuto.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        tmp_path (Path): Carpeta temporal; la corrida vive en tmp_path/corrida.
        modelo (Any): Modelo real o falso.
        config (ConfigPPOJefes, opcional): Configuracion de la corrida.
        lanzador (LanzadorFalso, opcional): Lanzador inyectado.
        reloj (RelojFalso, opcional): Reloj inyectado.

    Returns:
        CallbackEvaluacionPeriodica: Callback inicializado y con el entrenamiento iniciado.

    Raises:
        Ninguna.

    Ejemplo:
        callback = conectar(tmp_path, ModeloFalso())
    """
    callback = CallbackEvaluacionPeriodica(
        config or config_ppo(minutos_evaluacion=1.0), tmp_path / "corrida", REFERENCIA,
        lanzador or LanzadorFalso(), reloj or RelojFalso())
    callback.init_callback(modelo)
    callback.on_training_start({}, {})
    return callback


def instalar_mejor(tmp_path: Path, contenido: bytes) -> Path:
    """
    Crea el mejor.zip vigente de la corrida con un contenido reconocible.

    Contexto:
        Precondicion de las pruebas de promocion.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        contenido (bytes): Contenido del archivo.

    Returns:
        Path: Ruta de mejor.zip.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        mejor = instalar_mejor(tmp_path, b"imitacion")
    """
    mejor = tmp_path / "corrida" / "mejor.zip"
    mejor.parent.mkdir(parents=True, exist_ok=True)
    mejor.write_bytes(contenido)
    return mejor


def copia_candidata(tmp_path: Path, nombre: str = "snap.zip") -> Path:
    """
    Crea la copia del checkpoint evaluado.

    Contexto:
        Es lo que el callback guarda antes de lanzar la evaluacion.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        nombre (str): Nombre del archivo.

    Returns:
        Path: Ruta de la copia.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        snap = copia_candidata(tmp_path)
    """
    copia = tmp_path / nombre
    copia.write_bytes(b"candidato")
    return copia


def test_un_checkpoint_que_cae_mas_de_diez_puntos_en_un_jefe_no_reemplaza_a_mejor_zip(tmp_path: Path) -> None:
    mejor = instalar_mejor(tmp_path, b"imitacion")
    callback = conectar(tmp_path, ModeloFalso())
    snapshot = copia_candidata(tmp_path)
    # La media sube de 0.50 a 0.675, pero amlo pierde 15 puntos respecto a la imitacion.
    callback.aplicar_resultado({"amlo": 0.35, "nahual": 1.0}, snapshot, 1000)
    assert mejor.read_bytes() == b"imitacion"
    assert not snapshot.exists()
    assert callback.estado.promociones == 0
    assert callback.estado.mejor_media == pytest.approx(0.5)
    registro = json.loads((tmp_path / "corrida" / "evaluaciones.jsonl").read_text(encoding="utf-8"))
    assert registro["promovido"] is False and "amlo" in registro["motivo"]


def test_un_checkpoint_que_baja_exactamente_diez_puntos_aun_puede_ser_promovido(tmp_path: Path) -> None:
    mejor = instalar_mejor(tmp_path, b"imitacion")
    callback = conectar(tmp_path, ModeloFalso())
    callback.aplicar_resultado({"amlo": 0.4, "nahual": 0.9}, copia_candidata(tmp_path), 1000)
    assert mejor.read_bytes() == b"candidato"


def test_un_checkpoint_que_mejora_la_media_reemplaza_a_mejor_zip_de_forma_atomica(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mejor = instalar_mejor(tmp_path, b"imitacion")
    visto_durante_la_copia: dict[str, Any] = {}
    copiar = shutil.copyfile

    def copiar_observando(origen: Path, destino: Path) -> Any:
        """
        Copia y anota lo que veria un lector de mejor.zip en ese instante.

        Contexto:
            Comprobar que el destino de la copia es un temporal y no mejor.zip.

        Args:
            origen (Path): Archivo a copiar.
            destino (Path): Destino de la copia.

        Returns:
            Any: Lo que devuelve shutil.copyfile.

        Raises:
            OSError: Si falla la copia.

        Ejemplo:
            (sustituye a shutil.copyfile)
        """
        visto_durante_la_copia["destino"] = Path(destino)
        visto_durante_la_copia["mejor"] = mejor.read_bytes()
        return copiar(origen, destino)

    monkeypatch.setattr(cj.shutil, "copyfile", copiar_observando)
    callback = conectar(tmp_path, ModeloFalso())
    callback.aplicar_resultado({"amlo": 0.7, "nahual": 0.7}, copia_candidata(tmp_path), 5000)
    assert visto_durante_la_copia["destino"] != mejor
    assert visto_durante_la_copia["mejor"] == b"imitacion"
    assert mejor.read_bytes() == b"candidato"
    assert sorted(p.name for p in mejor.parent.glob("mejor*")) == ["mejor.zip"]
    assert callback.estado.promociones == 1 and callback.estado.mejor_paso == 5000
    assert callback.estado.mejor_media == pytest.approx(0.7)


def test_un_checkpoint_que_no_supera_la_media_vigente_no_se_promueve(tmp_path: Path) -> None:
    mejor = instalar_mejor(tmp_path, b"imitacion")
    callback = conectar(tmp_path, ModeloFalso())
    snapshot = copia_candidata(tmp_path)
    callback.aplicar_resultado({"amlo": 0.5, "nahual": 0.5}, snapshot, 100)
    assert mejor.read_bytes() == b"imitacion" and not snapshot.exists()


@pytest.mark.parametrize(("tasa_inicial", "tasa_minima", "esperada"), [(1e-3, 1e-4, 5e-4), (2e-4, 1.5e-4, 1.5e-4)])
def test_dos_evaluaciones_seguidas_claramente_por_debajo_de_la_imitacion_restauran_los_pesos_del_mejor_y_bajan_la_tasa_de_aprendizaje(  # noqa: E501
        tmp_path: Path, tasa_inicial: float, tasa_minima: float, esperada: float) -> None:
    config = config_ppo(learning_rate=tasa_inicial, lr_minimo=tasa_minima, minutos_evaluacion=1.0)
    modelo, inicial = cargar_modelo_prueba(tmp_path, config)
    mejor = instalar_mejor(tmp_path, b"")
    shutil.copyfile(inicial, mejor)
    callback = conectar(tmp_path, modelo, config)
    torch.manual_seed(0)
    modelo.learn(32, log_interval=1000)
    guardado = MaskablePPO.load(str(mejor), device="cpu").policy
    assert not son_iguales(pesos_de(modelo.policy.parameters()), pesos_de(guardado.parameters()))
    pesos_del_mejor = pesos_de(guardado.parameters())

    callback.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "a.zip"), 100)
    assert callback.estado.reversiones == 0 and modelo.learning_rate == tasa_inicial
    assert not son_iguales(pesos_de(modelo.policy.parameters()), pesos_del_mejor)

    callback.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "b.zip"), 200)
    assert son_iguales(pesos_de(modelo.policy.parameters()), pesos_del_mejor)
    assert len(modelo.policy.optimizer.state) == 0
    assert modelo.learning_rate == pytest.approx(esperada)
    assert all(g["lr"] == pytest.approx(esperada) for g in modelo.policy.optimizer.param_groups)
    assert callback.estado.reversiones == 1 and callback.control.seguidas == 0
    assert mejor.exists()


def test_una_regresion_aislada_no_revierte_si_la_siguiente_evaluacion_se_recupera(tmp_path: Path) -> None:
    config = config_ppo()
    modelo, inicial = cargar_modelo_prueba(tmp_path, config)
    shutil.copyfile(inicial, instalar_mejor(tmp_path, b""))
    callback = conectar(tmp_path, modelo, config)
    callback.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "a.zip"), 100)
    callback.aplicar_resultado({"amlo": 0.5, "nahual": 0.5}, copia_candidata(tmp_path, "b.zip"), 200)
    callback.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "c.zip"), 300)
    assert callback.estado.reversiones == 0 and modelo.learning_rate == config.learning_rate


def test_una_evaluacion_fallida_no_cambia_el_modelo_ni_el_conteo_de_regresiones(tmp_path: Path) -> None:
    config = config_ppo(minutos_evaluacion=1.0)
    modelo, _ = cargar_modelo_prueba(tmp_path, config)
    reloj, lanzador = RelojFalso(), LanzadorFalso(CLARAMENTE_PEOR, codigo=1, terminar_al_lanzar=True)
    callback = conectar(tmp_path, modelo, config, lanzador, reloj)
    callback.control.seguidas = 1
    antes = pesos_de(modelo.policy.parameters())
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    copia = lanzador.llamadas[0][0]
    callback.on_rollout_start()
    assert son_iguales(antes, pesos_de(modelo.policy.parameters()))
    assert callback.control.seguidas == 1 and callback.estado.reversiones == 0
    assert callback.estado.evaluaciones == [] and not copia.exists()
    assert modelo.learning_rate == config.learning_rate
    assert not (tmp_path / "corrida" / "evaluaciones.jsonl").exists()
    reloj.avanzar(30.0)
    callback.on_rollout_start()
    assert len(lanzador.llamadas) == 1  # la siguiente evaluacion se reprogramo a un plazo completo


def test_una_evaluacion_exitosa_pero_sin_tasas_tampoco_cambia_nada(tmp_path: Path) -> None:
    reloj, lanzador = RelojFalso(), LanzadorFalso(tasas={}, codigo=0, terminar_al_lanzar=True)
    callback = conectar(tmp_path, ModeloFalso(), lanzador=lanzador, reloj=reloj)
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    callback.on_rollout_start()
    assert callback.estado.evaluaciones == [] and callback.estado.promociones == 0


def test_no_se_lanza_otra_evaluacion_mientras_una_sigue_corriendo(tmp_path: Path) -> None:
    reloj, lanzador = RelojFalso(), LanzadorFalso({"amlo": 0.6, "nahual": 0.6})
    callback = conectar(tmp_path, ModeloFalso(), lanzador=lanzador, reloj=reloj)
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    for _ in range(5):
        reloj.avanzar(3600.0)
        callback.on_rollout_start()
    assert len(lanzador.llamadas) == 1
    assert callback.estado.evaluaciones == []
    lanzador.procesos[0].terminar_ya()
    callback.on_rollout_start()
    assert len(callback.estado.evaluaciones) == 1
    assert len(lanzador.llamadas) == 1  # el resultado se aplica primero; la siguiente espera su plazo
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    assert len(lanzador.llamadas) == 2


def test_el_estado_se_restaura_al_reanudar(tmp_path: Path) -> None:
    config = config_ppo(learning_rate=1e-3, lr_minimo=1e-4, minutos_evaluacion=1.0)
    modelo, inicial = cargar_modelo_prueba(tmp_path, config)
    mejor = instalar_mejor(tmp_path, b"")
    shutil.copyfile(inicial, mejor)
    primera = conectar(tmp_path, modelo, config)
    promovida = tmp_path / "m.zip"
    shutil.copyfile(inicial, promovida)  # el mejor promovido debe ser un modelo cargable
    primera.aplicar_resultado({"amlo": 0.6, "nahual": 0.6}, promovida, 100)
    primera.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "a.zip"), 200)
    primera.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "b.zip"), 300)
    primera.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "c.zip"), 400)
    guardado = json.loads((tmp_path / "corrida" / "checkpoints" / "estado_ppo.json").read_text(encoding="utf-8"))
    assert guardado["learning_rate"] == pytest.approx(5e-4) and guardado["mejor_media"] == pytest.approx(0.6)
    assert guardado["reversiones"] == 1 and guardado["seguidas"] == 1 and guardado["promociones"] == 1

    # Proceso nuevo: el modelo se carga con la tasa de la configuracion, no con la corregida.
    reanudado, _ = cargar_modelo_prueba(tmp_path, config, semilla=1)
    assert reanudado.learning_rate == pytest.approx(1e-3)
    segunda = conectar(tmp_path, reanudado, config)
    assert reanudado.learning_rate == pytest.approx(5e-4)
    assert all(g["lr"] == pytest.approx(5e-4) for g in reanudado.policy.optimizer.param_groups)
    assert segunda.estado.mejor_media == pytest.approx(0.6) and segunda.estado.mejor_paso == 100
    assert segunda.estado.reversiones == 1 and segunda.estado.promociones == 1
    # Queda una regresion pendiente: una sola mas basta para volver a corregir.
    assert segunda.control.seguidas == 1
    segunda.aplicar_resultado(CLARAMENTE_PEOR, copia_candidata(tmp_path, "d.zip"), 500)
    assert segunda.estado.reversiones == 2
    assert reanudado.learning_rate == pytest.approx(2.5e-4)


def test_al_terminar_el_entrenamiento_se_termina_el_proceso_en_marcha(tmp_path: Path) -> None:
    reloj, lanzador = RelojFalso(), LanzadorFalso()
    callback = conectar(tmp_path, ModeloFalso(), lanzador=lanzador, reloj=reloj)
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    assert lanzador.procesos[0].poll() is None
    callback.on_training_end()
    assert lanzador.procesos[0].terminaciones == 1
