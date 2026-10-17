"""PPO multi-jefe: rutas, reparto de instancias, carga de modelos y el ciclo completo de entrenamiento."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from dmrl.aplicacion import entrenamiento_jefes as entj
from dmrl.aplicacion.imitacion_red import EntornoEspacios
from dmrl.aplicacion.ppo_kl import MaskablePPOKL
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.ppo_jefes import ConfigPPOJefes
from dmrl.infraestructura.jefes import cargar_catalogo
from dmrl.infraestructura.juego import AFINIDAD_ENTRENAMIENTO
from tests.support_jefes import catalogo_prueba
from tests.support_ppo import (
    EntornoAleatorio,
    LanzadorFalso,
    config_ppo,
    guardar_modelo_pequeno,
    vec_prueba,
)

TASAS_IMITACION = {"juice": 0.5, "amlo": 0.5, "chamuco": 0.5, "nahual": 0.5}


@pytest.fixture(autouse=True)
def aislar_sistema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    """
    Redirige runs/ y el archivo de PID a la carpeta temporal y neutraliza los procesos del sistema.

    Contexto:
        Todas las pruebas de este archivo: el runs/ real y su entrenamiento.pid pertenecen a una
        corrida en curso y no deben tocarse; tampoco se ajusta la prioridad ni se apagan procesos.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        tmp_path (Path): Carpeta temporal de la prueba.

    Returns:
        Iterator[dict[str, Any]]: Registro de las llamadas a ajustar_proceso_actual y apagar_descendientes.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    registro: dict[str, Any] = {"afinidad": [], "apagados": []}
    monkeypatch.setattr(entj, "CARPETA_RUNS", tmp_path / "runs")
    monkeypatch.setattr(entj, "ARCHIVO_PID", tmp_path / "runs" / "entrenamiento.pid")
    monkeypatch.setattr(entj, "CARPETA_IMITACION", tmp_path / "runs" / "multijefe_imitacion")
    monkeypatch.setattr(entj, "ajustar_proceso_actual", lambda afinidad, *a, **k: registro["afinidad"].append(afinidad))
    monkeypatch.setattr(entj, "apagar_descendientes", lambda pid, *a, **k: registro["apagados"].append(pid))
    hilos = torch.get_num_threads()
    yield registro
    torch.set_num_threads(hilos)


def preparar_imitacion(tmp_path: Path, tasas: dict[str, float] | None = None) -> Path:
    """
    Crea la carpeta de la imitacion con mejor.zip y referencia.json.

    Contexto:
        Precondicion de entrenar_jefes.

    Args:
        tmp_path (Path): Carpeta temporal de la prueba.
        tasas (dict, opcional): Tasas de victoria por jefe; por defecto TASAS_IMITACION.

    Returns:
        Path: Ruta del mejor.zip de la imitacion.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        imitacion = preparar_imitacion(tmp_path)
    """
    carpeta = tmp_path / "imitacion"
    carpeta.mkdir(exist_ok=True)
    (carpeta / "referencia.json").write_text(json.dumps({"tasas": tasas or TASAS_IMITACION}), encoding="utf-8")
    return guardar_modelo_pequeno(carpeta / "mejor.zip", semilla=4)


def config_corrida(**cambios: Any) -> ConfigPPOJefes:
    """
    Configuracion de una corrida diminuta de 64 pasos.

    Contexto:
        Pruebas de entrenar_jefes.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        ConfigPPOJefes: Calentamiento de un rollout y penalizacion KL activa despues.

    Raises:
        ValueError: Si los cambios no son validos.

    Ejemplo:
        config_corrida(pasos_totales=96)
    """
    base: dict[str, Any] = {"nombre": "corrida", "pasos_totales": 64, "checkpoint_cada": 32,
                            "pasos_calentamiento": 32, "beta_kl": 1.0, "pasos_decaimiento_kl": 1000}
    base.update(cambios)
    return config_ppo(**base)


class EntornoQueParaA(EntornoAleatorio):
    """Entorno que, tras unos pasos, crea el archivo de parada y anota el estado del PID."""

    def __init__(self, parada: Path, pid: Path, tras: int = 5) -> None:
        """
        Crea el entorno.

        Contexto:
            Pruebas de parada ordenada.

        Args:
            parada (Path): Archivo que se crea para pedir la parada.
            pid (Path): Archivo de PID cuyo contenido se anota durante el entrenamiento.
            tras (int): Paso en que se crea el archivo de parada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoQueParaA(Path("PARAR"), Path("pid"))
        """
        super().__init__()
        self.parada, self.pid, self.tras = parada, pid, tras
        self.pasos = 0
        self.pid_visto: str | None = None

    def step(self, action: int) -> Any:
        """
        Avanza un paso y pide parar cuando toca.

        Contexto:
            API de Gymnasium.

        Args:
            action (int): Ignorada.

        Returns:
            tuple: Resultado del paso del entorno aleatorio.

        Raises:
            Ninguna.

        Ejemplo:
            entorno.step(0)
        """
        self.pasos += 1
        if self.pasos == 1 and self.pid.exists():
            self.pid_visto = self.pid.read_text(encoding="utf-8")
        if self.pasos == self.tras:
            self.parada.write_text("", encoding="utf-8")
        return super().step(action)


class EntornoQueFalla(EntornoAleatorio):
    """Entorno que lanza una excepcion a partir de cierto paso."""

    def __init__(self, tras: int = 40) -> None:
        """
        Crea el entorno.

        Contexto:
            Pruebas de fallo durante el entrenamiento.

        Args:
            tras (int): Paso en que empieza a fallar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoQueFalla(10)
        """
        super().__init__()
        self.tras = tras
        self.pasos = 0

    def step(self, action: int) -> Any:
        """
        Avanza un paso o falla.

        Contexto:
            API de Gymnasium.

        Args:
            action (int): Ignorada.

        Returns:
            tuple: Resultado del paso del entorno aleatorio.

        Raises:
            RuntimeError: Desde el paso configurado.

        Ejemplo:
            entorno.step(0)
        """
        self.pasos += 1
        if self.pasos >= self.tras:
            raise RuntimeError("el juego se cerro")
        return super().step(action)


def test_rutas_de_una_corrida_cuelgan_de_su_carpeta() -> None:
    rutas = entj.rutas_ppo("mia")
    assert rutas.base == entj.CARPETA_RUNS / "mia"
    assert rutas.ultimo == rutas.base / "checkpoints" / "ultimo.zip"
    assert rutas.mejor == rutas.base / "mejor.zip"
    assert rutas.parada == rutas.base / "PARAR"


@pytest.mark.parametrize("catalogo", [cargar_catalogo(), catalogo_prueba()], ids=["real", "prueba"])
def test_con_cuatro_instancias_juice_tiene_una_y_los_jefes_de_amlo_tres(catalogo: Any) -> None:
    grupos = entj.repartir_entornos(catalogo, 4)
    assert grupos == [("juice",)] + [("amlo", "chamuco", "nahual")] * 3


@pytest.mark.parametrize("instancias", [4, 6, 8, 12])
def test_ningun_jefe_aporta_mas_de_una_vez_y_media_las_muestras_de_otro_porque_los_entornos_avanzan_a_la_vez(
        instancias: int) -> None:
    grupos = entj.repartir_entornos(cargar_catalogo(), instancias)
    assert len(grupos) == instancias
    muestras: dict[str, float] = {}
    for grupo in grupos:
        for jefe in grupo:
            muestras[jefe] = muestras.get(jefe, 0.0) + 1.0 / len(grupo)
    assert max(muestras.values()) / min(muestras.values()) <= 1.5 + 1e-9


def test_con_seis_instancias_juice_recibe_dos_y_el_mapa_de_amlo_cuatro() -> None:
    assert entj.repartir_entornos(cargar_catalogo(), 6) == (
        [("juice",)] * 2 + [("amlo", "chamuco", "nahual")] * 4)


def test_repartir_entornos_exige_al_menos_una_instancia_por_mapa() -> None:
    with pytest.raises(ValueError, match="menos instancias"):
        entj.repartir_entornos(catalogo_prueba(), 1)
    assert len(entj.repartir_entornos(catalogo_prueba(), 2)) == 2


def test_crear_entorno_jefe_ajusta_el_proceso_y_envuelve_el_entorno(
        monkeypatch: pytest.MonkeyPatch, aislar_sistema: dict[str, Any]) -> None:
    llamadas: list[tuple[Any, ...]] = []
    catalogo = catalogo_prueba()

    def falso(*args: Any, **kwargs: Any) -> EntornoAleatorio:
        """
        Sustituye a crear_entorno_jefes y anota sus argumentos.

        Contexto:
            Evita lanzar el juego.

        Args:
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            EntornoAleatorio: Entorno sin juego.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        llamadas.append((args, kwargs))
        return EntornoAleatorio()

    monkeypatch.setattr(entj, "crear_entorno_jefes", falso)
    monkeypatch.setattr(entj, "cargar_catalogo", lambda: catalogo)
    config = ConfigEntorno()
    env = entj.crear_entorno_jefe(2, ("amlo", "nahual"), config, 10)
    assert isinstance(env, Monitor) and isinstance(env.unwrapped, EntornoAleatorio)
    assert aislar_sistema["afinidad"] == [AFINIDAD_ENTRENAMIENTO]
    args, kwargs = llamadas[0]
    assert args == (catalogo, ("amlo", "nahual"), "mj2", config, 12, entj.SELECCION_ADAPTATIVA)
    assert kwargs == {"trazar": True, "recarga_cada": entj.RECARGA_CADA}
    entj.crear_entorno_jefe(0, ("juice",), config, 0, prefijo="zz", trazar=False)
    assert llamadas[1][0][2] == "zz0" and llamadas[1][1]["trazar"] is False


def test_crear_vec_env_jefes_exige_instancias() -> None:
    with pytest.raises(ValueError, match="al menos una instancia"):
        entj.crear_vec_env_jefes([], ConfigEntorno(), 0)


def test_crear_vec_env_jefes_con_fabricas_usa_un_vec_env_en_proceso() -> None:
    venv = entj.crear_vec_env_jefes([("amlo",), ("nahual",)], ConfigEntorno(), 0,
                                    fabricas=[EntornoAleatorio, EntornoAleatorio])
    assert isinstance(venv, DummyVecEnv) and venv.num_envs == 2


def test_crear_vec_env_jefes_real_crea_un_subproceso_por_grupo_con_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    capturado: dict[str, Any] = {}

    class SubprocFalso:
        """Anota las fabricas y el metodo de arranque que recibe SubprocVecEnv."""

        def __init__(self, fabricas: list[Any], start_method: str) -> None:
            """
            Guarda los argumentos.

            Contexto:
                Sustituye a SubprocVecEnv para no crear procesos.

            Args:
                fabricas (list[Any]): Funciones que crean cada entorno.
                start_method (str): Metodo de arranque.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (sustituto)
            """
            capturado["fabricas"], capturado["metodo"] = fabricas, start_method

    monkeypatch.setattr(entj, "SubprocVecEnv", SubprocFalso)
    config = ConfigEntorno()
    entj.crear_vec_env_jefes([("juice",), ("amlo", "nahual")], config, 5)
    assert capturado["metodo"] == "spawn"
    fabricas = capturado["fabricas"]
    assert [f.func for f in fabricas] == [entj.crear_entorno_jefe] * 2
    assert fabricas[1].args == (1, ("amlo", "nahual"), config, 5)
    assert isinstance(fabricas[0], functools.partial)


def test_parametros_de_carga_reflejan_la_configuracion_de_la_corrida(tmp_path: Path) -> None:
    config = ConfigPPOJefes(learning_rate=2e-4, clip_range=0.15, n_steps=64, batch_size=32)
    parametros = entj.parametros_de_carga(config, tmp_path)
    assert parametros["learning_rate"] == 2e-4 and parametros["clip_range"] == 0.15
    assert parametros["n_steps"] == 64 and parametros["batch_size"] == 32
    assert parametros["target_kl"] == config.target_kl and parametros["gae_lambda"] == config.gae_lambda
    assert parametros["tensorboard_log"] == str(tmp_path / "tb")


def test_cargar_modelo_ppo_con_archivo_inexistente_falla(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no existe el modelo"):
        entj.cargar_modelo_ppo(tmp_path / "no.zip", vec_prueba(), config_ppo(), tmp_path)


def test_cargar_modelo_ppo_sustituye_los_hiperparametros_guardados_por_los_de_la_corrida(tmp_path: Path) -> None:
    zip_modelo = guardar_modelo_pequeno(tmp_path / "m.zip")
    guardado = MaskablePPO.load(str(zip_modelo), device="cpu")
    assert guardado.learning_rate == 3e-4 and guardado.clip_range(1.0) == 0.2
    config = config_ppo(learning_rate=2e-4, clip_range=0.1, ent_coef=0.002)
    modelo = entj.cargar_modelo_ppo(zip_modelo, vec_prueba(), config, tmp_path / "corrida")
    assert isinstance(modelo, MaskablePPOKL)
    assert modelo.learning_rate == 2e-4 and modelo.clip_range(1.0) == 0.1 and modelo.ent_coef == 0.002
    assert modelo.n_steps == config.n_steps and modelo.target_kl == config.target_kl
    assert modelo.tensorboard_log == str(tmp_path / "corrida" / "tb")


def test_cargar_modelo_ppo_con_espacios_distintos_a_los_del_zip_falla(tmp_path: Path) -> None:
    zip_modelo = guardar_modelo_pequeno(tmp_path / "m.zip")
    otro = DummyVecEnv([lambda: EntornoEspacios(2, 3)])
    with pytest.raises(ValueError, match="do not match"):
        entj.cargar_modelo_ppo(zip_modelo, otro, config_ppo(), tmp_path)


def test_referencia_de_imitacion_lee_las_tasas_como_numeros(tmp_path: Path) -> None:
    ruta = tmp_path / "referencia.json"
    ruta.write_text(json.dumps({"tasas": {"amlo": 1, "nahual": "0.5"}}), encoding="utf-8")
    assert entj.referencia_de_imitacion(ruta) == {"amlo": 1.0, "nahual": 0.5}


@pytest.mark.parametrize("contenido", [None, "{}", '{"tasas": {}}', "no es json"])
def test_referencia_de_imitacion_sin_tasas_falla(tmp_path: Path, contenido: str | None) -> None:
    ruta = tmp_path / "referencia.json"
    if contenido is not None:
        ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tasas de victoria"):
        entj.referencia_de_imitacion(ruta)


def test_corrida_nueva_parte_de_la_imitacion_y_deja_config_checkpoints_y_estado(
        tmp_path: Path, aislar_sistema: dict[str, Any]) -> None:
    imitacion = preparar_imitacion(tmp_path)
    config = config_corrida()
    venv = vec_prueba()
    cierres: list[int] = []
    venv.close = lambda: cierres.append(1)  # type: ignore[method-assign]
    ultimo = entj.entrenar_jefes(config, venv, imitacion, catalogo_prueba(), LanzadorFalso())
    carpeta = entj.CARPETA_RUNS / "corrida"
    assert ultimo == carpeta / "checkpoints" / "ultimo.zip" and ultimo.exists()
    assert json.loads((carpeta / "config.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(asdict(config)))
    assert (carpeta / "mejor.zip").read_bytes() == imitacion.read_bytes()
    assert (carpeta / "checkpoints" / "estado_ppo.json").exists()
    assert list((carpeta / "checkpoints").glob("modelo_*.zip"))
    assert MaskablePPO.load(str(ultimo), device="cpu").num_timesteps == 64
    assert not entj.ARCHIVO_PID.exists() and not (carpeta / "PARAR").exists()
    assert cierres == [] and aislar_sistema["apagados"] == []  # el entorno inyectado no se cierra
    assert aislar_sistema["afinidad"] == [AFINIDAD_ENTRENAMIENTO]


def test_el_pid_de_la_corrida_existe_mientras_entrena_y_se_elimina_al_terminar(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    entorno = EntornoQueParaA(tmp_path / "no_se_usa", entj.ARCHIVO_PID, tras=10**9)
    venv = DummyVecEnv([lambda: entorno])
    entj.entrenar_jefes(config_corrida(pasos_totales=32), venv, imitacion, catalogo_prueba(), LanzadorFalso())
    assert entorno.pid_visto == str(os.getpid())
    assert not entj.ARCHIVO_PID.exists()


def test_reanudacion_continua_desde_ultimo_zip_y_no_pisa_el_mejor(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    config = config_corrida(pasos_totales=64)
    entj.entrenar_jefes(config, vec_prueba(), imitacion, catalogo_prueba(), LanzadorFalso())
    carpeta = entj.CARPETA_RUNS / "corrida"
    (carpeta / "mejor.zip").write_bytes(b"promovido")
    ultimo = entj.entrenar_jefes(config_corrida(pasos_totales=32), vec_prueba(), imitacion, catalogo_prueba(),
                                 LanzadorFalso())
    assert MaskablePPO.load(str(ultimo), device="cpu").num_timesteps == 96
    assert (carpeta / "mejor.zip").read_bytes() == b"promovido"


def test_la_parada_ordenada_guarda_el_modelo_y_limpia_parar_y_pid(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    parada = entj.rutas_ppo("corrida").parada
    entorno = EntornoQueParaA(parada, entj.ARCHIVO_PID, tras=5)
    venv = DummyVecEnv([lambda: entorno])
    ultimo = entj.entrenar_jefes(config_corrida(pasos_totales=5000), venv, imitacion, catalogo_prueba(),
                                 LanzadorFalso())
    pasos = MaskablePPO.load(str(ultimo), device="cpu").num_timesteps
    assert 0 < pasos < 5000
    assert not parada.exists() and not entj.ARCHIVO_PID.exists()


def test_un_archivo_parar_viejo_no_impide_empezar(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    parada = entj.rutas_ppo("corrida").parada
    parada.parent.mkdir(parents=True)
    parada.write_text("", encoding="utf-8")
    ultimo = entj.entrenar_jefes(config_corrida(pasos_totales=32), vec_prueba(), imitacion, catalogo_prueba(),
                                 LanzadorFalso())
    assert MaskablePPO.load(str(ultimo), device="cpu").num_timesteps == 32


def test_sin_referencia_no_se_arranca_ni_se_deja_pid(tmp_path: Path) -> None:
    modelo = guardar_modelo_pequeno(tmp_path / "solo" / "mejor.zip")
    with pytest.raises(FileNotFoundError, match="tasas de victoria"):
        entj.entrenar_jefes(config_corrida(), vec_prueba(), modelo, catalogo_prueba(), LanzadorFalso())
    assert not entj.ARCHIVO_PID.exists()
    assert not (entj.CARPETA_RUNS / "corrida").exists()


def test_un_modelo_incompatible_falla_pero_deja_limpio_el_pid(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    incompatible = DummyVecEnv([lambda: EntornoEspacios(2, 3)])
    with pytest.raises(ValueError, match="do not match"):
        entj.entrenar_jefes(config_corrida(), incompatible, imitacion, catalogo_prueba(), LanzadorFalso())
    assert not entj.ARCHIVO_PID.exists() and not entj.rutas_ppo("corrida").parada.exists()


def test_un_fallo_durante_el_entrenamiento_guarda_el_ultimo_modelo_y_propaga_el_error(tmp_path: Path) -> None:
    imitacion = preparar_imitacion(tmp_path)
    venv = DummyVecEnv([lambda: EntornoQueFalla(40)])
    with pytest.raises(RuntimeError, match="se cerro"):
        entj.entrenar_jefes(config_corrida(pasos_totales=5000), venv, imitacion, catalogo_prueba(), LanzadorFalso())
    carpeta = entj.CARPETA_RUNS / "corrida"
    assert (carpeta / "checkpoints" / "ultimo.zip").exists()
    assert list((carpeta / "checkpoints").glob("modelo_*.zip"))
    assert not entj.ARCHIVO_PID.exists()


def test_sin_entorno_ni_catalogo_ni_imitacion_usa_los_valores_por_defecto_y_cierra_lo_que_creo(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aislar_sistema: dict[str, Any]) -> None:
    carpeta_imitacion = entj.CARPETA_IMITACION
    carpeta_imitacion.mkdir(parents=True)
    (carpeta_imitacion / "referencia.json").write_text(json.dumps({"tasas": TASAS_IMITACION}), encoding="utf-8")
    guardar_modelo_pequeno(carpeta_imitacion / "mejor.zip", semilla=4)
    venv = vec_prueba()
    cierres: list[int] = []
    cerrar = venv.close
    venv.close = lambda: (cierres.append(1), cerrar())[1]  # type: ignore[method-assign]
    pedidos: list[Any] = []

    def crear(grupos: Any, config: Any, semilla: int, trazar: bool = True) -> Any:
        """
        Sustituye a crear_vec_env_jefes y anota la peticion.

        Contexto:
            Evita lanzar instancias del juego.

        Args:
            grupos (Any): Jefes de cada instancia.
            config (Any): Configuracion del entorno.
            semilla (int): Semilla base.
            trazar (bool): Si los entornos registran los golpes recibidos.

        Returns:
            Any: Entorno vectorizado de prueba.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        pedidos.append((grupos, config, semilla, trazar))
        return venv

    monkeypatch.setattr(entj, "crear_vec_env_jefes", crear)
    monkeypatch.setattr(entj, "cargar_catalogo", catalogo_prueba)
    entj.entrenar_jefes(config_corrida(pasos_totales=32, semilla=9, instancias=4), lanzador=LanzadorFalso())
    assert pedidos == [(entj.repartir_entornos(catalogo_prueba(), 4), ConfigEntorno(), 9, True)]
    assert cierres == [1] and aislar_sistema["apagados"] == [os.getpid()]


def test_si_cerrar_el_entorno_falla_igual_se_apagan_los_descendientes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aislar_sistema: dict[str, Any]) -> None:
    imitacion = preparar_imitacion(tmp_path)
    venv = vec_prueba()

    def falla() -> None:
        """
        Simula un cierre que lanza.

        Contexto:
            Un subproceso del juego que no responde al cierre.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Siempre.

        Ejemplo:
            (sustituto)
        """
        raise OSError("no cierra")

    venv.close = falla  # type: ignore[method-assign]
    monkeypatch.setattr(entj, "crear_vec_env_jefes", lambda *a, **k: venv)
    with pytest.raises(OSError, match="no cierra"):
        entj.entrenar_jefes(config_corrida(pasos_totales=32), None, imitacion, catalogo_prueba(), LanzadorFalso())
    assert aislar_sistema["apagados"] == [os.getpid()]
