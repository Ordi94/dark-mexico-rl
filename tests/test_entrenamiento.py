"""Entrenamiento MaskablePPO sobre entornos con juego sintetico (sin lanzar el juego real)."""

from __future__ import annotations

import functools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from stable_baselines3.common.monitor import Monitor

from dmbridge import Action
from dmrl.aplicacion import entrenamiento as ent
from dmrl.aplicacion.callbacks import CallbackParada
from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.dominio.configuracion import ConfigCurriculo, ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from tests.support_dmrl import JuegoSintetico

ENTORNO_RAPIDO = replace(
    ConfigEntorno(), curriculo=ConfigCurriculo(niveles=(0.1, 1.0), ventana=5, minimo_episodios=5, prob_fase2=0.3),
)


@pytest.fixture(autouse=True)
def aislar_sistema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """
    Evita que las pruebas toquen runs/, la prioridad del proceso o procesos ajenos.

    Contexto:
        Todas las pruebas de este archivo.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        tmp_path (Path): Carpeta temporal de la prueba.

    Returns:
        list[Any]: Registro de las llamadas a ajustar_proceso_actual.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    llamadas: list[Any] = []
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path / "runs")
    monkeypatch.setattr(ent, "ARCHIVO_PID", tmp_path / "runs" / "entrenamiento.pid")
    monkeypatch.setattr(ent, "ajustar_proceso_actual", lambda afinidad, *a, **k: llamadas.append(afinidad))
    monkeypatch.setattr(ent, "apagar_descendientes", lambda pid, *a, **k: llamadas.append(("hijos", pid)))
    return llamadas


def fabrica_entorno(periodo: int = 6, juegos: list[JuegoSintetico] | None = None) -> Any:
    """
    Fabrica de un entorno monitoreado sobre un juego sintetico.

    Contexto:
        Sustituye a crear_entorno en los VecEnv de prueba.

    Args:
        periodo (int): Pasos entre golpes del jefe sintetico.
        juegos (list, opcional): Lista donde se registran los juegos creados.

    Returns:
        Callable: Funcion sin argumentos que crea el entorno.

    Raises:
        Ninguna.

    Ejemplo:
        fabrica_entorno()()
    """

    def crear() -> Monitor:
        """
        Crea el entorno.

        Contexto:
            Cuerpo de la fabrica.

        Args:
            Ninguno.

        Returns:
            Monitor: Entorno envuelto.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        def lanzar() -> JuegoSintetico:
            """
            Lanza el juego sintetico.

            Contexto:
                Hace de FabricaPuerto.

            Args:
                Ninguno.

            Returns:
                JuegoSintetico: Juego nuevo.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            juego = JuegoSintetico(periodo=periodo)
            if juegos is not None:
                juegos.append(juego)
            return juego

        return Monitor(DarkMexicoEnv(lanzar, TablaGolpes(), ENTORNO_RAPIDO))

    return crear


def config(**cambios: Any) -> ent.ConfigEntrenamiento:
    """
    Configuracion de entrenamiento pequena para pruebas.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        ent.ConfigEntrenamiento: Configuracion con rollouts de 64 pasos.

    Raises:
        Ninguna.

    Ejemplo:
        config(pasos_totales=512)
    """
    base = ent.ConfigEntrenamiento(
        nombre="t", instancias=2, pasos_totales=512, n_steps=64, batch_size=64, n_epochs=1, capas=(16,),
        dispositivo="cpu", hilos_torch=1, checkpoint_cada=256, entorno=ENTORNO_RAPIDO,
    )
    return replace(base, **cambios)


def test_entrenar_crea_checkpoints_configuracion_y_registro_de_tensorboard(aislar_sistema: list[Any]) -> None:
    """Una corrida corta deja ultimo.zip, curriculo.json, config.json, el PID y el log de TensorBoard."""
    venv = ent.crear_vec_env(2, ENTORNO_RAPIDO, 0, fabricas=[fabrica_entorno(), fabrica_entorno()])
    cfg = config()
    ultimo = ent.entrenar(cfg, venv)
    assert ultimo.exists() and (ultimo.parent / "curriculo.json").exists()
    assert (cfg.carpeta / "config.json").exists() and any((cfg.carpeta / "tb").rglob("events.out*"))
    assert json.loads((cfg.carpeta / "config.json").read_text())["gamma"] == 0.995
    assert (ent.CARPETA_RUNS / "entrenamiento.pid").exists()
    assert any(p.name.startswith("modelo_") for p in ultimo.parent.iterdir())
    assert aislar_sistema[0] == ent.AFINIDAD_ENTRENAMIENTO
    venv.close()


def test_entrenar_reanuda_desde_el_ultimo_checkpoint_y_conserva_los_pasos() -> None:
    """Una segunda corrida carga ultimo.zip y sigue contando desde donde quedo."""
    venv = ent.crear_vec_env(2, ENTORNO_RAPIDO, 0, fabricas=[fabrica_entorno(), fabrica_entorno()])
    ent.entrenar(config(pasos_totales=256), venv)
    modelo1 = ent.cargar_modelo(ent.CARPETA_RUNS / "t" / "checkpoints" / "ultimo.zip", venv, "cpu")
    primeros = modelo1.num_timesteps
    ent.entrenar(config(pasos_totales=primeros + 256), venv)
    modelo2 = ent.cargar_modelo(ent.CARPETA_RUNS / "t" / "checkpoints" / "ultimo.zip", venv, "cpu")
    assert modelo2.num_timesteps >= primeros + 256 - 128 and modelo2.num_timesteps > primeros
    venv.close()


def test_cargar_un_modelo_ajeno_redirige_tensorboard_a_la_corrida_actual() -> None:
    """Un modelo creado en otra corrida (la de imitacion) no debe escribir sus curvas en la carpeta de origen."""
    venv = ent.crear_vec_env(2, ENTORNO_RAPIDO, 0, fabricas=[fabrica_entorno(), fabrica_entorno()])
    ent.entrenar(config(pasos_totales=128), venv)
    ruta = ent.CARPETA_RUNS / "t" / "checkpoints" / "ultimo.zip"
    conservado = ent.cargar_modelo(ruta, venv, "cpu")
    redirigido = ent.cargar_modelo(ruta, venv, "cpu", str(ent.CARPETA_RUNS / "otra" / "tb"))
    assert conservado.tensorboard_log == str(ent.CARPETA_RUNS / "t" / "tb")
    assert redirigido.tensorboard_log == str(ent.CARPETA_RUNS / "otra" / "tb")
    venv.close()


def test_entrenar_desde_cero_ignora_el_checkpoint_existente() -> None:
    """Con reanudar=False el modelo nuevo empieza en cero pasos."""
    venv = ent.crear_vec_env(2, ENTORNO_RAPIDO, 0, fabricas=[fabrica_entorno(), fabrica_entorno()])
    ent.entrenar(config(pasos_totales=256), venv)
    ent.entrenar(config(pasos_totales=128, reanudar=False), venv)
    modelo = ent.cargar_modelo(ent.CARPETA_RUNS / "t" / "checkpoints" / "ultimo.zip", venv, "cpu")
    assert modelo.num_timesteps <= 256
    venv.close()


def test_archivo_de_parada_detiene_el_entrenamiento_y_guarda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si aparece PARAR el entrenamiento se detiene mucho antes del techo de pasos y guarda ultimo.zip."""
    parada = ent.CARPETA_RUNS / "t" / "PARAR"

    class ParaAlDecimoPaso(JuegoSintetico):
        """Juego que crea el archivo de parada al llegar a su paso 10."""

        def step(self, action: Action, n_frames: int = 1) -> dict[str, Any]:
            """
            Avanza y, en el paso 10, crea el archivo PARAR.

            Contexto:
                Escenario de la prueba.

            Args:
                action (Action): Accion recibida.
                n_frames (int): Frames del paso.

            Returns:
                dict[str, Any]: Observacion del paso.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            if self.pasos_totales == 10:
                parada.write_text("", encoding="utf-8")
            return super().step(action, n_frames)

    monkeypatch.setattr(ent, "CallbackParada", lambda archivo: CallbackParada(archivo, cada=1))

    def lanzar() -> Monitor:
        """
        Crea un entorno sobre el juego que se detiene a si mismo.

        Contexto:
            Fabrica de la prueba.

        Args:
            Ninguno.

        Returns:
            Monitor: Entorno envuelto.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return Monitor(DarkMexicoEnv(ParaAlDecimoPaso, TablaGolpes(), ENTORNO_RAPIDO))

    venv = ent.crear_vec_env(1, ENTORNO_RAPIDO, 0, fabricas=[lanzar])
    ultimo = ent.entrenar(config(instancias=1, pasos_totales=100_000), venv)
    modelo = ent.cargar_modelo(ultimo, venv, "cpu")
    assert 0 < modelo.num_timesteps < 1000
    assert not parada.exists()
    venv.close()


def test_entrenar_con_entorno_propio_lo_cierra_y_mata_a_los_hijos(monkeypatch: pytest.MonkeyPatch,
                                                                   aislar_sistema: list[Any]) -> None:
    """Sin venv dado, entrenar crea el suyo, lo cierra y ejecuta la red de seguridad de procesos."""
    juegos: list[JuegoSintetico] = []
    creado = ent.crear_vec_env(1, ENTORNO_RAPIDO, 0, fabricas=[fabrica_entorno(juegos=juegos)])
    monkeypatch.setattr(ent, "crear_vec_env", lambda *a, **k: creado)
    ent.entrenar(config(instancias=1, pasos_totales=128))
    assert juegos and juegos[0].cerrado
    assert any(isinstance(x, tuple) and x[0] == "hijos" for x in aislar_sistema)


def test_crear_vec_env_rechaza_cero_instancias_y_usa_subprocesos_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin fabricas se crea un SubprocVecEnv con un entorno por instancia y arranque spawn."""
    with pytest.raises(ValueError):
        ent.crear_vec_env(0, ENTORNO_RAPIDO, 0)
    recibido: dict[str, Any] = {}

    class FalsoSubproc:
        """Registra lo que recibe SubprocVecEnv sin lanzar procesos."""

        def __init__(self, fabricas: list[Any], start_method: str) -> None:
            """
            Guarda los argumentos.

            Contexto:
                Sustituye a SubprocVecEnv en la prueba.

            Args:
                fabricas (list[Any]): Funciones de creacion de entornos.
                start_method (str): Metodo de arranque de procesos.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            recibido["fabricas"], recibido["metodo"] = fabricas, start_method

    monkeypatch.setattr(ent, "SubprocVecEnv", FalsoSubproc)
    ent.crear_vec_env(4, ENTORNO_RAPIDO, 5)
    assert recibido["metodo"] == "spawn" and len(recibido["fabricas"]) == 4
    assert all(isinstance(f, functools.partial) and f.func is ent.crear_entorno for f in recibido["fabricas"])
    assert [f.args[0] for f in recibido["fabricas"]] == [0, 1, 2, 3]


def test_crear_entorno_ajusta_el_proceso_y_envuelve_con_monitor(monkeypatch: pytest.MonkeyPatch,
                                                                aislar_sistema: list[Any]) -> None:
    """Cada subproceso queda con prioridad baja y sin las CPUs 0-3; el id de instancia lleva el indice."""
    ids: list[str] = []

    def fabrica_de(params: Any) -> Any:
        """
        Sustituye a crear_fabrica_puerto registrando el id.

        Contexto:
            Prueba de crear_entorno.

        Args:
            params (Any): ParametrosInstancia recibidos.

        Returns:
            Callable: Fabrica de juegos sinteticos.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        ids.append(params.id_instancia)
        return JuegoSintetico

    monkeypatch.setattr(ent, "crear_fabrica_puerto", fabrica_de)
    monkeypatch.setattr(ent, "cargar_tabla_golpes", lambda: TablaGolpes())
    env = ent.crear_entorno(3, ENTORNO_RAPIDO, 0)
    assert isinstance(env, Monitor) and ids == ["tr3"]
    assert aislar_sistema == [ent.AFINIDAD_ENTRENAMIENTO]
    explicito = ent.crear_entorno(1, ENTORNO_RAPIDO, 0, fabrica=JuegoSintetico, tabla=TablaGolpes())
    assert isinstance(explicito, Monitor)


def test_elegir_dispositivo_prefiere_la_cpu_en_auto() -> None:
    """auto da cpu (medido mas rapido para esta red); cuda solo si se pide; un valor desconocido falla."""
    assert ent.elegir_dispositivo("cpu") == "cpu" and ent.elegir_dispositivo("cuda") == "cuda"
    assert ent.elegir_dispositivo("auto") == "cpu"
    with pytest.raises(ValueError):
        ent.elegir_dispositivo("tpu")


def test_la_configuracion_documenta_los_hiperparametros_por_defecto() -> None:
    """Los hiperparametros iniciales son los documentados."""
    cfg = ent.ConfigEntrenamiento()
    assert (cfg.gamma, cfg.gae_lambda, cfg.ent_coef, cfg.capas) == (0.995, 0.95, 0.01, (256, 256))
    assert cfg.carpeta == ent.CARPETA_RUNS / "principal"
    assert cfg.a_dict()["entorno"]["frames_por_paso"] == 2
