"""Callbacks del PPO multi-jefe: lanzador, metricas, checkpoints y evaluacion periodica."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sb3_contrib import MaskablePPO

from dmrl.aplicacion import callbacks_jefes as cj
from dmrl.aplicacion.callbacks import CallbackParada
from dmrl.aplicacion.callbacks_jefes import (
    CallbackCheckpointJefes,
    CallbackEvaluacionPeriodica,
    CallbackMetricasJefes,
    aplicar_tasa_aprendizaje,
    lanzar_evaluacion,
    restaurar_pesos,
)
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from tests.support_ppo import (
    LanzadorFalso,
    ModeloFalso,
    RelojFalso,
    cargar_modelo_prueba,
    config_ppo,
    pesos_de,
    resumen_jefe,
    son_iguales,
)

REFERENCIA = {"amlo": 0.9, "nahual": 0.9}


class PopenFalso:
    """Sustituye a subprocess.Popen: anota los argumentos y devuelve un marcador."""

    def __init__(self, falla: bool = False) -> None:
        """
        Crea el registro vacio.

        Contexto:
            Pruebas de lanzar_evaluacion.

        Args:
            falla (bool): Si es True el lanzamiento lanza OSError.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            PopenFalso(falla=True)
        """
        self.falla = falla
        self.args: list[str] = []
        self.kwargs: dict[str, Any] = {}

    def __call__(self, args: list[str], **kwargs: Any) -> str:
        """
        Anota la llamada.

        Contexto:
            Sustituye a Popen(...).

        Args:
            args (list[str]): Linea de comandos.
            **kwargs (Any): Opciones de Popen.

        Returns:
            str: Marcador del proceso.

        Raises:
            OSError: Si se configuro para fallar.

        Ejemplo:
            PopenFalso()(["python"])
        """
        self.args, self.kwargs = args, kwargs
        if self.falla:
            raise OSError("no se pudo crear el proceso")
        return "proceso"


def crear_callback(tmp_path: Path, modelo: Any, lanzador: LanzadorFalso | None = None,
                   reloj: RelojFalso | None = None, config: ConfigPPOJefes | None = None,
                   referencia: dict[str, float] | None = None) -> CallbackEvaluacionPeriodica:
    """
    Callback de evaluacion conectado a un modelo.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba (la corrida vive en tmp_path/corrida).
        modelo (Any): Modelo real o falso.
        lanzador (LanzadorFalso, opcional): Lanzador inyectado.
        reloj (RelojFalso, opcional): Reloj inyectado.
        config (ConfigPPOJefes, opcional): Configuracion de la corrida.
        referencia (dict, opcional): Tasas de la imitacion.

    Returns:
        CallbackEvaluacionPeriodica: Callback inicializado.

    Raises:
        Ninguna.

    Ejemplo:
        callback = crear_callback(tmp_path, ModeloFalso())
    """
    callback = CallbackEvaluacionPeriodica(
        config or config_ppo(minutos_evaluacion=1.0), tmp_path / "corrida", referencia or REFERENCIA,
        lanzador or LanzadorFalso(), reloj or RelojFalso())
    callback.init_callback(modelo)
    return callback


def test_lanzar_evaluacion_arma_el_comando_sin_ventana_y_con_prioridad_baja(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    falso = PopenFalso()
    monkeypatch.setattr(cj, "subprocess", SimpleNamespace(Popen=falso, STDOUT=subprocess.STDOUT))
    modelo, salida = tmp_path / "snap.zip", tmp_path / "anidada" / "eval.json"
    proceso = lanzar_evaluacion(modelo, salida, ConfigPPOJefes(episodios_evaluacion=7, semilla=3))
    assert proceso == "proceso"
    assert falso.args == [sys.executable, "-m", "dmrl", "evaluar-modelo-jefes", "--modelo", str(modelo),
                          "--episodios", "7", "--instancias", "4", "--semilla", "7003", "--prefijo", "evppo",
                          "--salida", str(salida)]
    assert falso.kwargs["creationflags"] == cj.BELOW_NORMAL_PRIORITY_CLASS | cj.CREATE_NO_WINDOW
    assert falso.kwargs["stderr"] == subprocess.STDOUT
    registro = falso.kwargs["stdout"]
    assert registro.closed and str(registro.name).endswith("eval.log")


def test_lanzar_evaluacion_cierra_el_registro_aunque_falle_el_proceso(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    falso = PopenFalso(falla=True)
    monkeypatch.setattr(cj, "subprocess", SimpleNamespace(Popen=falso, STDOUT=subprocess.STDOUT))
    with pytest.raises(OSError, match="proceso"):
        lanzar_evaluacion(tmp_path / "s.zip", tmp_path / "e.json", ConfigPPOJefes())
    assert (tmp_path / "e.log").exists()


def test_aplicar_tasa_aprendizaje_cambia_el_atributo_el_programa_y_el_optimizador(tmp_path: Path) -> None:
    modelo, _ = cargar_modelo_prueba(tmp_path, config_ppo(learning_rate=1e-3))
    aplicar_tasa_aprendizaje(modelo, 2.5e-4)
    assert modelo.learning_rate == 2.5e-4
    assert modelo.lr_schedule(1.0) == 2.5e-4 and modelo.lr_schedule(0.1) == 2.5e-4
    assert all(g["lr"] == 2.5e-4 for g in modelo.policy.optimizer.param_groups)


def test_restaurar_pesos_con_archivo_inexistente_falla(tmp_path: Path) -> None:
    modelo, _ = cargar_modelo_prueba(tmp_path, config_ppo())
    with pytest.raises(FileNotFoundError, match="no existe el modelo"):
        restaurar_pesos(modelo, tmp_path / "no_existe.zip")


def test_restaurar_pesos_devuelve_la_politica_al_zip_y_descarta_los_momentos_de_adam(tmp_path: Path) -> None:
    modelo, inicial = cargar_modelo_prueba(tmp_path, config_ppo())
    torch.manual_seed(0)
    modelo.learn(32, log_interval=1000)
    assert len(modelo.policy.optimizer.state) > 0
    guardado = MaskablePPO.load(str(inicial), device="cpu").policy
    assert not son_iguales(pesos_de(modelo.policy.parameters()), pesos_de(guardado.parameters()))
    restaurar_pesos(modelo, inicial)
    assert son_iguales(pesos_de(modelo.policy.parameters()), pesos_de(guardado.parameters()))
    assert len(modelo.policy.optimizer.state) == 0


def conectar_metricas(**opciones: Any) -> tuple[CallbackMetricasJefes, ModeloFalso]:
    """
    Callback de metricas conectado a un modelo falso.

    Contexto:
        Atajo de las pruebas de metricas.

    Args:
        **opciones (Any): Argumentos del constructor (fase_normal, ventana, imprimir).

    Returns:
        tuple[CallbackMetricasJefes, ModeloFalso]: Callback y su modelo.

    Raises:
        Ninguna.

    Ejemplo:
        callback, modelo = conectar_metricas(imprimir=False)
    """
    opciones.setdefault("fase_normal", {"amlo": 1, "nahual": 1})
    callback = CallbackMetricasJefes(**opciones)
    modelo = ModeloFalso()
    callback.init_callback(modelo)
    callback.on_training_start({}, {})
    return callback, modelo


def avanzar_pasos(callback: CallbackMetricasJefes, *resumenes: dict[str, Any]) -> None:
    """
    Simula un paso de los entornos con episodios terminados.

    Contexto:
        Atajo de las pruebas de metricas.

    Args:
        callback (CallbackMetricasJefes): Callback inicializado.
        *resumenes (dict[str, Any]): Episodios que terminan en el paso.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        avanzar_pasos(callback, resumen_jefe("amlo", True))
    """
    callback.update_locals({"infos": [{"episodio": r} for r in resumenes] + [{}]})
    assert callback.on_step()


def test_metricas_publican_una_serie_por_jefe_y_la_media_de_las_tasas(capsys: pytest.CaptureFixture[str]) -> None:
    callback, modelo = conectar_metricas()
    modelo.num_timesteps = 500
    avanzar_pasos(callback, resumen_jefe("amlo", True), resumen_jefe("amlo", True),
                  resumen_jefe("amlo", False, golpe_mortal="Corte rapido!"), resumen_jefe("nahual", False))
    callback.on_rollout_end()
    v = modelo.logger.valores
    assert v["dm/amlo/tasa_victoria"] == pytest.approx(2 / 3)
    assert v["dm/nahual/tasa_victoria"] == 0.0
    assert v["dm/tasa_victoria_media"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert v["dm/amlo/muertes"] == pytest.approx(1 / 3)
    assert v["dm/amlo/muerte_por/Corte_rapido_"] == pytest.approx(1 / 3)
    assert v["dm/amlo/danio_al_jefe"] == pytest.approx(100.0)
    assert v["dm/episodios_total"] == 4 and v["dm/errores_instancia_total"] == 0
    assert v["dm/pasos_por_segundo"] > 0
    linea = capsys.readouterr().out
    assert "pasos=500" in linea and "episodios=4" in linea and "amlo=0.67" in linea and "nahual=0.00" in linea


def test_metricas_ignoran_los_episodios_con_error_y_los_cuentan_aparte() -> None:
    callback, modelo = conectar_metricas(imprimir=False)
    avanzar_pasos(callback, resumen_jefe("amlo", False, resultado="error"), resumen_jefe("amlo", True))
    callback.on_rollout_end()
    v = modelo.logger.valores
    assert v["dm/errores_instancia_total"] == 1 and v["dm/episodios_total"] == 1
    assert v["dm/amlo/tasa_victoria"] == 1.0


def test_metricas_sin_episodios_no_publican_tasas_y_con_imprimir_falso_no_escriben(
        capsys: pytest.CaptureFixture[str]) -> None:
    callback, modelo = conectar_metricas(imprimir=False)
    callback.on_rollout_end()
    assert "dm/tasa_victoria_media" not in modelo.logger.valores
    assert "dm/pasos_por_segundo" in modelo.logger.valores
    assert capsys.readouterr().out == ""


def test_metricas_aceptan_jefes_no_declarados_y_episodios_sin_jefe() -> None:
    callback, modelo = conectar_metricas(imprimir=False)
    sin_jefe = resumen_jefe("amlo", True)
    del sin_jefe["jefe"]
    avanzar_pasos(callback, resumen_jefe("juice", True), sin_jefe)
    callback.on_rollout_end()
    v = modelo.logger.valores
    assert v["dm/juice/tasa_victoria"] == 1.0 and v["dm/?/tasa_victoria"] == 1.0
    assert v["dm/tasa_victoria_media"] == 1.0


def test_metricas_solo_promedian_la_ventana_reciente_de_cada_jefe() -> None:
    callback, modelo = conectar_metricas(ventana=2, imprimir=False)
    avanzar_pasos(callback, resumen_jefe("amlo", False), resumen_jefe("amlo", True), resumen_jefe("amlo", True))
    callback.on_rollout_end()
    assert modelo.logger.valores["dm/amlo/tasa_victoria"] == 1.0
    assert modelo.logger.valores["dm/episodios_total"] == 3


def test_checkpoint_guarda_ultimo_de_forma_atomica_y_devuelve_su_ruta(tmp_path: Path) -> None:
    callback = CallbackCheckpointJefes(tmp_path / "ck", cada=100)
    callback.init_callback(ModeloFalso())
    ruta = callback.guardar()
    assert ruta == tmp_path / "ck" / "ultimo.zip" and ruta.read_bytes() == b"modelo"
    assert sorted(p.name for p in (tmp_path / "ck").iterdir()) == ["ultimo.zip"]


def marcar_antiguedad(carpeta: Path) -> None:
    """
    Fija la fecha de cada checkpoint numerado a su numero de pasos.

    Contexto:
        El orden de borrado usa la fecha de modificacion; en pruebas rapidas dos archivos pueden
        compartirla, asi que se vuelve determinista.

    Args:
        carpeta (Path): Carpeta de checkpoints.

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede cambiar la fecha.

    Ejemplo:
        marcar_antiguedad(tmp_path)
    """
    for archivo in carpeta.glob("modelo_*.zip"):
        pasos = int(archivo.stem.split("_")[1])
        os.utime(archivo, (pasos, pasos))


def test_checkpoint_numera_cada_n_guardados_y_conserva_solo_los_ultimos(tmp_path: Path) -> None:
    callback = CallbackCheckpointJefes(tmp_path / "ck", cada=100, cada_numerado=2, conservar=2)
    modelo = ModeloFalso()
    callback.init_callback(modelo)
    for pasos in (50, 100, 200, 300, 400, 500, 600):
        modelo.num_timesteps = pasos
        assert callback.on_step()
        marcar_antiguedad(tmp_path / "ck")
    nombres = sorted(p.name for p in (tmp_path / "ck").iterdir())
    assert nombres == ["modelo_400.zip", "modelo_600.zip", "ultimo.zip"]


def test_checkpoint_guarda_una_sola_vez_al_saltar_varios_intervalos(tmp_path: Path) -> None:
    callback = CallbackCheckpointJefes(tmp_path / "ck", cada=100, cada_numerado=1)
    modelo = ModeloFalso()
    callback.init_callback(modelo)
    modelo.num_timesteps = 99
    callback.on_step()
    assert list((tmp_path / "ck").iterdir()) == []
    modelo.num_timesteps = 350
    callback.on_step()
    modelo.num_timesteps = 399
    callback.on_step()
    assert sorted(p.name for p in (tmp_path / "ck").iterdir()) == ["modelo_350.zip", "ultimo.zip"]
    modelo.num_timesteps = 400
    callback.on_step()
    assert (tmp_path / "ck" / "modelo_400.zip").exists()


def test_la_parada_ordenada_convive_con_los_demas_callbacks(tmp_path: Path) -> None:
    archivo = tmp_path / "PARAR"
    parada = CallbackParada(archivo, cada=1)
    parada.init_callback(ModeloFalso())
    assert parada.on_step()
    archivo.write_text("", encoding="utf-8")
    assert not parada.on_step()


def test_el_constructor_toma_el_estado_guardado_o_el_inicial_si_esta_danado(tmp_path: Path) -> None:
    modelo = ModeloFalso()
    inicial = crear_callback(tmp_path, modelo)
    assert inicial.estado.mejor_media == pytest.approx(0.9)
    assert inicial.estado.learning_rate == 1e-3
    ruta = tmp_path / "corrida" / "checkpoints" / "estado_ppo.json"
    ruta.parent.mkdir(parents=True)
    ruta.write_text("{ no es json", encoding="utf-8")
    assert crear_callback(tmp_path, modelo).estado.mejor_media == pytest.approx(0.9)
    ruta.write_text(json.dumps({"referencia": {"amlo": 0.1}, "mejor_media": 0.3, "learning_rate": 2e-4,
                                "seguidas": 1}), encoding="utf-8")
    restaurado = crear_callback(tmp_path, modelo)
    assert restaurado.estado.mejor_media == 0.3 and restaurado.estado.learning_rate == 2e-4
    assert restaurado.estado.referencia == REFERENCIA  # la referencia vigente manda sobre la guardada
    assert restaurado.control.seguidas == 1


def test_on_step_de_la_evaluacion_no_interviene(tmp_path: Path) -> None:
    assert crear_callback(tmp_path, ModeloFalso()).on_step() is True


def test_al_empezar_se_programa_la_primera_evaluacion_a_los_minutos_configurados(tmp_path: Path) -> None:
    modelo, _ = cargar_modelo_prueba(tmp_path, config_ppo())
    reloj, lanzador = RelojFalso(), LanzadorFalso(REFERENCIA)
    reloj.ahora = 100.0
    callback = crear_callback(tmp_path, modelo, lanzador, reloj, config_ppo(minutos_evaluacion=2.0))
    callback.on_training_start({}, {})
    reloj.avanzar(119.0)
    callback.on_rollout_start()
    assert lanzador.llamadas == []
    reloj.avanzar(1.0)
    callback.on_rollout_start()
    assert len(lanzador.llamadas) == 1


def test_una_corrida_reanudada_con_evaluaciones_previas_usa_el_intervalo_normal(tmp_path: Path) -> None:
    modelo, _ = cargar_modelo_prueba(tmp_path, config_ppo())
    reloj, lanzador = RelojFalso(), LanzadorFalso(REFERENCIA)
    config = config_ppo(minutos_evaluacion=5.0, minutos_primera_evaluacion=1.0)
    callback = crear_callback(tmp_path, modelo, lanzador, reloj, config)
    callback.estado.evaluaciones.append({"pasos": 1})
    callback.on_training_start({}, {})
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    assert lanzador.llamadas == []
    reloj.avanzar(240.0)
    callback.on_rollout_start()
    assert len(lanzador.llamadas) == 1


def test_iniciar_evaluacion_guarda_una_copia_con_los_pasos_y_lanza_el_proceso(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    modelo = ModeloFalso()
    modelo.num_timesteps = 4096
    reloj, lanzador = RelojFalso(), LanzadorFalso()
    callback = crear_callback(tmp_path, modelo, lanzador, reloj)
    callback.on_training_start({}, {})
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    copia, salida = lanzador.llamadas[0]
    carpeta = tmp_path / "corrida" / "evaluaciones"
    assert copia == carpeta / "modelo_4096.zip" and salida == carpeta / "eval_4096.json"
    assert copia.read_bytes() == b"modelo"
    assert "lanzada con el modelo de 4096 pasos" in capsys.readouterr().out


def test_una_evaluacion_terminada_sin_resultado_se_descarta_y_borra_la_copia(tmp_path: Path) -> None:
    modelo = ModeloFalso()
    reloj, lanzador = RelojFalso(), LanzadorFalso(tasas=None, terminar_al_lanzar=True)
    callback = crear_callback(tmp_path, modelo, lanzador, reloj)
    callback.on_training_start({}, {})
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    copia = lanzador.llamadas[0][0]
    assert copia.exists()
    callback.on_rollout_start()
    assert not copia.exists()
    assert callback.proceso is None
    assert callback.estado.evaluaciones == []
    assert not (tmp_path / "corrida" / "evaluaciones.jsonl").exists()


def test_terminar_evaluacion_sin_proceso_ni_copia_solo_reprograma(tmp_path: Path) -> None:
    reloj = RelojFalso()
    callback = crear_callback(tmp_path, ModeloFalso(), reloj=reloj)
    callback.on_training_start({}, {})
    reloj.avanzar(500.0)
    callback._terminar_evaluacion()
    assert callback.estado.evaluaciones == []
    assert callback._proxima == 560.0


def test_aplicar_resultado_escribe_el_historial_el_estado_y_las_metricas(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    modelo = ModeloFalso()
    callback = crear_callback(tmp_path, modelo)
    snapshot = tmp_path / "snap.zip"
    snapshot.write_bytes(b"candidato")
    callback.aplicar_resultado({"amlo": 1.0, "nahual": 0.95}, snapshot, 2048, {"episodios": 3})
    v = modelo.logger.valores
    assert v["dm/eval/tasa_media"] == pytest.approx(0.975) and v["dm/eval/promovido"] == 1.0
    assert v["dm/eval/revertido"] == 0.0 and v["dm/eval/amlo/tasa_victoria"] == 1.0
    linea = json.loads((tmp_path / "corrida" / "evaluaciones.jsonl").read_text(encoding="utf-8"))
    assert linea["pasos"] == 2048 and linea["promovido"] is True and linea["detalle"] == {"episodios": 3}
    guardado = json.loads((tmp_path / "corrida" / "checkpoints" / "estado_ppo.json").read_text(encoding="utf-8"))
    assert guardado["promociones"] == 1 and guardado["mejor_paso"] == 2048
    assert "PROMOVIDO" in capsys.readouterr().out


def test_aplicar_resultado_sin_detalle_deja_el_detalle_vacio(tmp_path: Path) -> None:
    callback = crear_callback(tmp_path, ModeloFalso())
    snapshot = tmp_path / "snap.zip"
    snapshot.write_bytes(b"x")
    callback.aplicar_resultado({"amlo": 0.9, "nahual": 0.9}, snapshot, 10)
    linea = json.loads((tmp_path / "corrida" / "evaluaciones.jsonl").read_text(encoding="utf-8"))
    assert linea["detalle"] == {} and linea["promovido"] is False
    assert not snapshot.exists()


def test_aplicar_resultado_con_jefes_faltantes_falla_sin_tocar_nada(tmp_path: Path) -> None:
    callback = crear_callback(tmp_path, ModeloFalso())
    snapshot = tmp_path / "snap.zip"
    snapshot.write_bytes(b"x")
    with pytest.raises(ValueError, match="no incluye los jefes"):
        callback.aplicar_resultado({"amlo": 0.9}, snapshot, 10)
    assert snapshot.exists() and callback.estado.evaluaciones == []


def test_al_terminar_se_pide_cerrar_la_evaluacion_en_marcha_y_se_guarda_el_estado(tmp_path: Path) -> None:
    reloj, lanzador = RelojFalso(), LanzadorFalso()
    callback = crear_callback(tmp_path, ModeloFalso(), lanzador, reloj)
    callback.on_training_start({}, {})
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    callback.on_training_end()
    assert lanzador.procesos[0].terminaciones == 1
    assert (tmp_path / "corrida" / "checkpoints" / "estado_ppo.json").exists()


def test_al_terminar_no_se_toca_un_proceso_que_ya_salio_ni_se_falla_sin_proceso(tmp_path: Path) -> None:
    reloj, lanzador = RelojFalso(), LanzadorFalso(terminar_al_lanzar=True)
    callback = crear_callback(tmp_path, ModeloFalso(), lanzador, reloj)
    callback.on_training_end()
    callback.on_training_start({}, {})
    reloj.avanzar(61.0)
    callback.on_rollout_start()
    callback.on_training_end()
    assert lanzador.procesos[0].terminaciones == 0


def test_integrada_con_learn_evalua_promueve_y_deja_historial(tmp_path: Path) -> None:
    config = config_ppo(minutos_evaluacion=0.5)
    modelo, _ = cargar_modelo_prueba(tmp_path, config, jefes=("amlo", "nahual"))
    lanzador = LanzadorFalso({"amlo": 0.95, "nahual": 0.95}, terminar_al_lanzar=True)
    reloj = RelojFalso()

    def reloj_que_avanza() -> float:
        """
        Hora que sube 100 s en cada consulta, para que cada rollout encuentre vencido el plazo.

        Contexto:
            Entrenamiento integrado con learn.

        Args:
            Ninguno.

        Returns:
            float: Segundos acumulados.

        Raises:
            Ninguna.

        Ejemplo:
            reloj_que_avanza()
        """
        reloj.avanzar(100.0)
        return reloj()

    carpeta = tmp_path / "corrida"
    callbacks = [CallbackMetricasJefes({"amlo": 1, "nahual": 1}, imprimir=False),
                 CallbackCheckpointJefes(carpeta / "checkpoints", cada=32),
                 CallbackEvaluacionPeriodica(config, carpeta, REFERENCIA, lanzador, reloj_que_avanza)]
    torch.manual_seed(0)
    modelo.learn(160, callback=callbacks, log_interval=1000)
    assert (carpeta / "checkpoints" / "ultimo.zip").exists()
    assert len(lanzador.llamadas) >= 1
    assert (carpeta / "mejor.zip").exists()
    historial = (carpeta / "evaluaciones.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(historial[0])["promovido"] is True
    assert modelo.logger.name_to_value["dm/amlo/tasa_victoria"] == 1.0


def test_protocolo_proceso_evaluacion_expone_sus_metodos() -> None:
    assert cj.ProcesoEvaluacion.poll(SimpleNamespace()) is None  # type: ignore[arg-type]
    assert cj.ProcesoEvaluacion.terminate(SimpleNamespace()) is None  # type: ignore[arg-type]
