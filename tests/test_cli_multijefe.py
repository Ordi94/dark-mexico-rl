"""Comandos de la linea de comandos del PPO multi-jefe: imitacion, evaluacion de modelos, entrenamiento y vista."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from dmrl.aplicacion import entrenamiento as ent
from dmrl.aplicacion import entrenamiento_jefes as entj
from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion import evaluacion_jefes as evj
from dmrl.aplicacion import evaluacion_modelo_jefes as evm
from dmrl.aplicacion import imitacion_jefes as imj
from dmrl.aplicacion import panel_vivo as pv
from dmrl.aplicacion import recoleccion_jefes as recj
from dmrl.aplicacion import visualizacion as vis
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import ConfigAprendizaje
from dmrl.entrada import cli
from dmrl.infraestructura.demostraciones import escribir_json
from tests.support_jefes import catalogo_prueba
from tests.support_ppo import guardar_modelo_pequeno


@dataclass
class Registro:
    """Llamadas que los sustitutos de la capa de aplicacion reciben de la linea de comandos."""

    llamadas: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def anotar(self, nombre: str, *args: Any, **kwargs: Any) -> None:
        """
        Guarda una llamada.

        Contexto:
            Cada sustituto anota lo que recibe para que la prueba verifique los argumentos.

        Args:
            nombre (str): Nombre de la funcion sustituida.
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            registro.anotar("medir", 1, a=2)
        """
        self.llamadas.append((nombre, args, kwargs))

    def de(self, nombre: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        """
        Llamadas hechas a una funcion.

        Contexto:
            Verificacion de argumentos.

        Args:
            nombre (str): Nombre de la funcion sustituida.

        Returns:
            list[tuple[tuple[Any, ...], dict[str, Any]]]: Argumentos de cada llamada, en orden.

        Raises:
            Ninguna.

        Ejemplo:
            registro.de("medir")
        """
        return [(a, k) for n, a, k in self.llamadas if n == nombre]


@pytest.fixture(autouse=True)
def aislar_sistema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Redirige runs/ y el PID a la carpeta temporal y usa el catalogo de prueba.

    Contexto:
        Todas las pruebas de este archivo: el runs/ real y su entrenamiento.pid pertenecen a una
        corrida en curso y no deben tocarse.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        tmp_path (Path): Carpeta temporal de la prueba.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path / "runs")
    monkeypatch.setattr(ent, "ARCHIVO_PID", tmp_path / "runs" / "entrenamiento.pid")
    monkeypatch.setattr(entj, "CARPETA_RUNS", tmp_path / "runs")
    monkeypatch.setattr(cli, "ajustar_proceso_actual", lambda *a, **k: None)
    monkeypatch.setattr(cli, "cargar_catalogo", catalogo_prueba)


def test_el_parser_de_imitar_jefes_tiene_valores_por_defecto_y_valida_las_opciones() -> None:
    p = cli.construir_parser()
    a = p.parse_args(["imitar-jefes"])
    assert (a.nombre, a.accion, a.arquitectura, a.dispositivo, a.semilla, a.evaluacion) == (
        "multijefe_imitacion", "resumen", "2x256", "auto", 0, None)
    b = p.parse_args(["imitar-jefes", "--accion", "publicar", "--dispositivo", "cpu", "--evaluacion", "final"])
    assert (b.accion, b.dispositivo, b.evaluacion) == ("publicar", "cpu", "final")
    with pytest.raises(SystemExit):
        p.parse_args(["imitar-jefes", "--accion", "inexistente"])
    with pytest.raises(SystemExit):
        p.parse_args(["imitar-jefes", "--arquitectura", "9x9"])


def test_el_parser_de_evaluar_modelo_jefes_evalua_todos_los_jefes_por_defecto() -> None:
    p = cli.construir_parser()
    a = p.parse_args(["evaluar-modelo-jefes", "--bot"])
    assert a.jefes == list(evm.JEFES_EVALUADOS) and a.bot and a.modelo is None
    assert (a.episodios, a.instancias, a.prefijo, a.epsilon_ruido, a.salida) == (20, 4, "ev", 0.0, None)
    b = p.parse_args(["evaluar-modelo-jefes", "--modelo", "m.zip", "--jefes", "amlo", "--salida", "s.json"])
    assert b.modelo == Path("m.zip") and b.jefes == ["amlo"] and b.salida == Path("s.json")


def test_el_parser_de_entrenar_jefes_y_ver_jefes_conoce_sus_opciones() -> None:
    p = cli.construir_parser()
    e = p.parse_args(["entrenar-jefes"])
    assert (e.nombre, e.instancias, e.pasos, e.learning_rate, e.minutos_evaluacion) == (
        "multijefe_ppo", 6, 200_000_000, 1e-4, 45.0)
    assert (e.episodios_evaluacion, e.beta_kl, e.pasos_calentamiento, e.imitacion, e.semilla) == (
        20, 0.3, 200_000, None, 0)
    v = p.parse_args(["ver-jefes", "--jefe", "nahual", "--max-segundos", "5"])
    assert (v.jefe, v.nombre, v.episodios, v.max_segundos, v.modelo) == ("nahual", "multijefe_ppo", 1, 5.0, None)
    with pytest.raises(SystemExit):
        p.parse_args(["ver-jefes"])
    with pytest.raises(SystemExit):
        p.parse_args(["ver-jefes", "--jefe", "no_existe"])


def test_recolectar_jefes_acepta_un_tope_de_minutos() -> None:
    p = cli.construir_parser()
    assert p.parse_args(["recolectar-jefes", "--jefes", "amlo"]).max_minutos is None
    assert p.parse_args(["recolectar-jefes", "--jefes", "amlo", "--max-minutos", "2.5"]).max_minutos == 2.5


def test_imitar_jefes_resumen_imprime_los_datos_por_jefe_de_la_carpeta_de_la_corrida(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()

    def datos(carpeta: Path) -> dict[str, dict[str, float]]:
        """
        Sustituye a datos_por_jefe_en_disco.

        Contexto:
            Evita leer demostraciones reales.

        Args:
            carpeta (Path): Carpeta de datos que recibe la linea de comandos.

        Returns:
            dict[str, dict[str, float]]: Resumen fijo.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("datos", carpeta)
        return {"amlo": {"episodios": 4.0}}

    monkeypatch.setattr(imj, "datos_por_jefe_en_disco", datos)
    assert cli.main(["imitar-jefes", "--nombre", "mi_imitacion"]) == 0
    assert json.loads(capsys.readouterr().out) == {"amlo": {"episodios": 4.0}}
    assert registro.de("datos")[0][0] == (ent.CARPETA_RUNS / "mi_imitacion" / "datos",)


def test_imitar_jefes_publicar_sin_evaluacion_falla_con_un_mensaje_claro() -> None:
    with pytest.raises(ValueError, match="--evaluacion"):
        cli.main(["imitar-jefes", "--accion", "publicar"])


def test_imitar_jefes_publicar_pasa_la_evaluacion_guardada_e_imprime_la_referencia(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()
    rutas = imj.RutasImitacionJefes(ent.CARPETA_RUNS / "multijefe_imitacion")
    escribir_json(rutas.evaluacion("final"), {"tasas": {"amlo": 0.9}})

    def publicar(rutas_: Any, arquitectura: str, evaluacion: dict[str, Any]) -> dict[str, float]:
        """
        Sustituye a publicar_modelo.

        Contexto:
            Evita copiar modelos reales.

        Args:
            rutas_ (Any): Rutas de la imitacion.
            arquitectura (str): Arquitectura publicada.
            evaluacion (dict[str, Any]): Contenido de la evaluacion.

        Returns:
            dict[str, float]: Referencia fija.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("publicar", rutas_, arquitectura, evaluacion)
        return {"amlo": 0.9}

    monkeypatch.setattr(imj, "publicar_modelo", publicar)
    assert cli.main(["imitar-jefes", "--accion", "publicar", "--arquitectura", "2x512",
                     "--evaluacion", "final"]) == 0
    (args, _), = registro.de("publicar")
    assert args == (rutas, "2x512", {"tasas": {"amlo": 0.9}})
    assert json.loads(capsys.readouterr().out) == {"mejor": str(rutas.mejor), "referencia": {"amlo": 0.9}}


def preparar_ajuste(monkeypatch: pytest.MonkeyPatch, registro: Registro,
                    tiempos: dict[str, float] | None = None) -> None:
    """
    Sustituye las funciones de imitacion que lanzarian calculo pesado.

    Contexto:
        Pruebas de imitar-jefes con las acciones dispositivos y entrenar.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        registro (Registro): Donde se anotan las llamadas.
        tiempos (dict, opcional): Segundos por epoca que devuelve la medicion.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        preparar_ajuste(monkeypatch, registro)
    """
    medicion = tiempos if tiempos is not None else {"cpu": 2.0}

    def cargar(carpeta: Path) -> Any:
        """
        Sustituye a cargar_conjunto_jefes.

        Contexto:
            Evita leer demostraciones.

        Args:
            carpeta (Path): Carpeta de datos.

        Returns:
            Any: Conjunto con unas demostraciones marcadas.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("cargar", carpeta)
        return SimpleNamespace(demostraciones="demos")

    def medir(*args: Any) -> dict[str, float]:
        """
        Sustituye a medir_dispositivos.

        Contexto:
            Evita entrenar para medir.

        Args:
            *args (Any): Catalogo, arquitectura, demostraciones y aprendizaje.

        Returns:
            dict[str, float]: Segundos por epoca fijos.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("medir", *args)
        return medicion

    def entrenar(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Sustituye a entrenar_arquitectura y emite un mensaje de progreso.

        Contexto:
            Evita entrenar de verdad.

        Args:
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos con nombre; incluye registrar.

        Returns:
            dict[str, Any]: Informe fijo.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("entrenar", *args)
        kwargs["registrar"]("epoca 1")
        return {"validacion": {"precision": 0.8}, "segundos": 3.5}

    monkeypatch.setattr(imj, "cargar_conjunto_jefes", cargar)
    monkeypatch.setattr(imj, "medir_dispositivos", medir)
    monkeypatch.setattr(imj, "entrenar_arquitectura", entrenar)


def test_imitar_jefes_dispositivos_mide_y_guarda_los_tiempos(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()
    preparar_ajuste(monkeypatch, registro, {"cpu": 2.0, "cuda": 0.5})
    assert cli.main(["imitar-jefes", "--accion", "dispositivos", "--arquitectura", "2x512"]) == 0
    (args, _), = registro.de("medir")
    assert args[1:] == ("2x512", "demos", ConfigAprendizaje())
    assert json.loads(capsys.readouterr().out) == {"segundos_por_epoca": {"cpu": 2.0, "cuda": 0.5}}
    guardado = json.loads((ent.CARPETA_RUNS / "multijefe_imitacion" / "dispositivos.json").read_text(encoding="utf-8"))
    assert guardado == {"arquitectura": "2x512", "segundos_por_epoca": {"cpu": 2.0, "cuda": 0.5}}
    assert registro.de("entrenar") == []


def test_imitar_jefes_entrenar_con_dispositivo_explicito_no_mide(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()
    preparar_ajuste(monkeypatch, registro)
    assert cli.main(["imitar-jefes", "--accion", "entrenar", "--dispositivo", "cpu", "--semilla", "7"]) == 0
    assert registro.de("medir") == []
    (args, _), = registro.de("entrenar")
    catalogo, conjunto, arquitectura, aprendizaje, semilla, dispositivo, rutas = args
    assert (arquitectura, aprendizaje, semilla, dispositivo) == ("2x256", ConfigAprendizaje(), 7, torch.device("cpu"))
    assert conjunto.demostraciones == "demos" and rutas.base == ent.CARPETA_RUNS / "multijefe_imitacion"
    assert catalogo.perfil("amlo").id == "amlo"
    salida = capsys.readouterr().out.splitlines()
    assert salida[0] == "epoca 1"
    assert json.loads(salida[1]) == {"arquitectura": "2x256", "validacion": {"precision": 0.8}, "segundos": 3.5}


def test_imitar_jefes_entrenar_en_auto_elige_el_dispositivo_mas_rapido_medido(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registro = Registro()
    preparar_ajuste(monkeypatch, registro, {"cpu": 1.0})
    assert cli.main(["imitar-jefes", "--accion", "entrenar"]) == 0
    assert len(registro.de("medir")) == 1
    (args, _), = registro.de("entrenar")
    assert args[5] == torch.device("cpu")


def test_evaluar_modelo_jefes_exige_exactamente_un_jugador() -> None:
    with pytest.raises(ValueError, match="--modelo o --bot"):
        cli.main(["evaluar-modelo-jefes"])
    with pytest.raises(ValueError, match="--modelo o --bot"):
        cli.main(["evaluar-modelo-jefes", "--bot", "--modelo", "m.zip"])


def test_evaluar_modelo_jefes_rechaza_un_jefe_que_no_esta_en_el_catalogo() -> None:
    with pytest.raises(KeyError):
        cli.main(["evaluar-modelo-jefes", "--bot", "--jefes", "amlo", "no_existe"])


def sustituir_evaluacion(monkeypatch: pytest.MonkeyPatch, registro: Registro,
                         tasas: dict[str, float]) -> None:
    """
    Sustituye la evaluacion en juego por episodios fijos.

    Contexto:
        Pruebas de evaluar-modelo-jefes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        registro (Registro): Donde se anota la llamada.
        tasas (dict[str, float]): Tasas por jefe que resume la evaluacion.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        sustituir_evaluacion(monkeypatch, registro, {"amlo": 1.0})
    """
    def evaluar(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Sustituye a evaluar_en_jefes.

        Contexto:
            Evita lanzar el juego.

        Args:
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            list[dict[str, Any]]: Un episodio fijo.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("evaluar", *args, **kwargs)
        return [{"jefe": "amlo", "victoria": True}]

    monkeypatch.setattr(evm, "evaluar_en_jefes", evaluar)
    monkeypatch.setattr(evm, "resumir_por_jefe", lambda episodios: {j: {"tasa_victoria": t} for j, t in tasas.items()})
    monkeypatch.setattr(evm, "tasas_por_jefe", lambda resumen: {j: r["tasa_victoria"] for j, r in resumen.items()})


def test_evaluar_modelo_jefes_con_modelo_pasa_los_argumentos_e_imprime_las_tasas(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()
    sustituir_evaluacion(monkeypatch, registro, {"amlo": 0.5, "nahual": 1.0})
    salida = tmp_path / "carpeta" / "eval.json"
    modelo = tmp_path / "m.zip"
    assert cli.main(["evaluar-modelo-jefes", "--modelo", str(modelo), "--jefes", "amlo", "nahual",
                     "--episodios", "3", "--instancias", "2", "--semilla", "9", "--prefijo", "pp",
                     "--epsilon-ruido", "0.05", "--salida", str(salida)]) == 0
    (args, kwargs), = registro.de("evaluar")
    assert args == (str(modelo), ["amlo", "nahual"], 3, 2, 9)
    assert kwargs["prefijo_id"] == "pp" and kwargs["epsilon_ruido"] == 0.05
    assert kwargs["catalogo"].perfil("amlo").id == "amlo"
    assert json.loads(capsys.readouterr().out) == {"tasas": {"amlo": 0.5, "nahual": 1.0}, "tasa_media": 0.75}
    guardado = json.loads(salida.read_text(encoding="utf-8"))
    assert guardado["modelo"] == str(modelo) and guardado["tasa_media"] == 0.75
    assert guardado["episodios"] == [{"jefe": "amlo", "victoria": True}]
    assert set(guardado) == {"modelo", "tasas", "tasa_media", "por_jefe", "episodios"}


def test_evaluar_modelo_jefes_con_bot_no_pasa_modelo_y_evalua_todos_los_jefes_por_defecto(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()
    sustituir_evaluacion(monkeypatch, registro, {})
    assert cli.main(["evaluar-modelo-jefes", "--bot"]) == 0
    (args, _), = registro.de("evaluar")
    assert args[0] is None and args[1] == list(evm.JEFES_EVALUADOS)
    assert json.loads(capsys.readouterr().out) == {"tasas": {}, "tasa_media": 0.0}


def test_entrenar_jefes_construye_la_configuracion_con_las_opciones_y_la_imitacion(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registro = Registro()

    def entrenar(config: Any, **kwargs: Any) -> Path:
        """
        Sustituye a entrenar_jefes.

        Contexto:
            Evita entrenar de verdad.

        Args:
            config (Any): Configuracion de la corrida.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            Path: Ruta fija del ultimo modelo.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("entrenar", config, **kwargs)
        return Path("runs/x/checkpoints/ultimo.zip")

    monkeypatch.setattr(entj, "entrenar_jefes", entrenar)
    imitacion = tmp_path / "mejor.zip"
    assert cli.main(["entrenar-jefes", "--nombre", "n1", "--instancias", "3", "--pasos", "1000",
                     "--learning-rate", "5e-5", "--minutos-evaluacion", "10", "--episodios-evaluacion", "6",
                     "--beta-kl", "0.2", "--pasos-calentamiento", "500", "--semilla", "4",
                     "--imitacion", str(imitacion)]) == 0
    (args, kwargs), = registro.de("entrenar")
    config = args[0]
    assert (config.nombre, config.instancias, config.pasos_totales, config.learning_rate) == ("n1", 3, 1000, 5e-5)
    assert (config.minutos_evaluacion, config.episodios_evaluacion, config.beta_kl) == (10.0, 6, 0.2)
    assert (config.pasos_calentamiento, config.semilla) == (500, 4)
    assert kwargs == {"ruta_imitacion": imitacion}
    assert str(Path("runs/x/checkpoints/ultimo.zip")) in capsys.readouterr().out


def test_entrenar_jefes_sin_imitacion_deja_que_el_entrenamiento_use_la_de_defecto(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registro = Registro()

    def entrenar(config: Any, **kwargs: Any) -> Path:
        """
        Sustituye a entrenar_jefes y anota la llamada.

        Contexto:
            Evita entrenar de verdad.

        Args:
            config (Any): Configuracion de la corrida.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            Path: Ruta fija.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("entrenar", config, **kwargs)
        return Path("u")

    monkeypatch.setattr(entj, "entrenar_jefes", entrenar)
    assert cli.main(["entrenar-jefes"]) == 0
    assert registro.de("entrenar")[0][1] == {"ruta_imitacion": None}


class EntornoVentana:
    """Entorno de ventana falso que anota su cierre."""

    def __init__(self) -> None:
        """
        Crea el entorno abierto.

        Contexto:
            Pruebas de ver-jefes.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoVentana()
        """
        self.cerrado = False

    def close(self) -> None:
        """
        Marca el entorno como cerrado.

        Contexto:
            ver-jefes lo cierra siempre.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            entorno.close()
        """
        self.cerrado = True


def preparar_vista(monkeypatch: pytest.MonkeyPatch, registro: Registro, entorno: EntornoVentana,
                   falla: bool = False) -> None:
    """
    Sustituye el entorno de ventana y la reproduccion de peleas.

    Contexto:
        Pruebas de ver-jefes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        registro (Registro): Donde se anotan las llamadas.
        entorno (EntornoVentana): Entorno que devuelve la creacion.
        falla (bool): Si es True la reproduccion lanza RuntimeError.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        preparar_vista(monkeypatch, registro, EntornoVentana())
    """
    def crear(*args: Any, **kwargs: Any) -> EntornoVentana:
        """
        Sustituye a crear_entorno_jefes.

        Contexto:
            Evita lanzar el juego con ventana.

        Args:
            *args (Any): Argumentos posicionales.
            **kwargs (Any): Argumentos con nombre.

        Returns:
            EntornoVentana: Entorno falso.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("crear", *args, **kwargs)
        return entorno

    def ver(*args: Any) -> list[dict[str, Any]]:
        """
        Sustituye a ver_pelea.

        Contexto:
            Evita jugar.

        Args:
            *args (Any): Entorno, politica, episodios, tope de segundos y opciones.

        Returns:
            list[dict[str, Any]]: Un episodio fijo.

        Raises:
            RuntimeError: Si la prueba lo pide.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("ver", *args)
        if falla:
            raise RuntimeError("la ventana se cerro")
        return [{"victoria": True}]

    monkeypatch.setattr(evj, "crear_entorno_jefes", crear)
    monkeypatch.setattr(vis, "ver_pelea", ver)
    monkeypatch.setattr(evl, "resumir", lambda resumenes: {"episodios": len(resumenes)})


def test_ver_jefes_sin_modelo_falla_antes_de_abrir_la_ventana() -> None:
    with pytest.raises(FileNotFoundError, match="no existe el modelo"):
        cli.main(["ver-jefes", "--jefe", "amlo"])


def test_ver_jefes_con_modelo_inexistente_no_crea_el_entorno(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registro = Registro()
    preparar_vista(monkeypatch, registro, EntornoVentana())
    with pytest.raises(FileNotFoundError, match="no_esta.zip"):
        cli.main(["ver-jefes", "--jefe", "amlo", "--modelo", str(tmp_path / "no_esta.zip")])
    assert registro.llamadas == []


def test_ver_jefes_juega_con_el_mejor_modelo_de_la_corrida_y_cierra_la_ventana(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro, entorno = Registro(), EntornoVentana()
    preparar_vista(monkeypatch, registro, entorno)
    guardar_modelo_pequeno(entj.rutas_ppo("multijefe_ppo").mejor)
    assert cli.main(["ver-jefes", "--jefe", "amlo", "--episodios", "2", "--max-segundos", "30"]) == 0
    (args, kwargs), = registro.de("crear")
    assert args[1:4] == (("amlo",), "ver", ConfigEntorno())
    assert kwargs == {"trazar": False, "params": vis.PARAMETROS_VENTANA}
    (argumentos, _), = registro.de("ver")
    env, politica, episodios, max_segundos, opciones = argumentos
    assert env is entorno and isinstance(politica, evl.PoliticaModelo)
    assert (episodios, max_segundos) == (2, 30.0)
    assert opciones == {"jefe": "amlo", "start_phase": catalogo_prueba().perfil("amlo").fases_inicio[0],
                        "boss_hp_mult": 1.0}
    assert entorno.cerrado
    assert json.loads(capsys.readouterr().out) == {"episodios": 1}


def test_ver_jefes_usa_el_modelo_pedido_y_cierra_la_ventana_si_falla_la_pelea(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registro, entorno = Registro(), EntornoVentana()
    preparar_vista(monkeypatch, registro, entorno, falla=True)
    modelo = guardar_modelo_pequeno(tmp_path / "otro.zip")
    with pytest.raises(RuntimeError, match="ventana"):
        cli.main(["ver-jefes", "--jefe", "nahual", "--modelo", str(modelo)])
    assert entorno.cerrado


def preparar_recoleccion(monkeypatch: pytest.MonkeyPatch, observado: dict[str, Any], faltan: int = 0,
                         falla: bool = False) -> None:
    """
    Sustituye la recoleccion y anota el estado del PID y de la parada mientras corre.

    Contexto:
        Pruebas del manejo de PID y PARAR de recolectar-jefes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        observado (dict[str, Any]): Donde se anota lo visto durante la recoleccion.
        faltan (int): Episodios que dice que faltan.
        falla (bool): Si es True la recoleccion lanza RuntimeError.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        preparar_recoleccion(monkeypatch, {})
    """
    def recolectar(carpeta: Path, ronda: int, jefes: list[str], episodios: int, instancias: int, semilla: int,
                   **kwargs: Any) -> int:
        """
        Sustituye a recolectar_jefes.

        Contexto:
            Evita lanzar el juego.

        Args:
            carpeta (Path): Carpeta de datos.
            ronda (int): Ronda.
            jefes (list[str]): Jefes.
            episodios (int): Peleas por jefe.
            instancias (int): Instancias.
            semilla (int): Semilla.
            **kwargs (Any): archivo_parada y limite_segundos.

        Returns:
            int: Episodios que faltan.

        Raises:
            RuntimeError: Si la prueba lo pide.

        Ejemplo:
            (sustituto)
        """
        observado.update(kwargs, carpeta=carpeta, ronda=ronda, jefes=jefes,
                         pid=ent.ARCHIVO_PID.read_text(encoding="utf-8") if ent.ARCHIVO_PID.exists() else None,
                         parada_vieja=kwargs["archivo_parada"].exists())
        if falla:
            raise RuntimeError("el juego se cayo")
        return faltan

    monkeypatch.setattr(recj, "recolectar_jefes", recolectar)
    monkeypatch.setattr(recj, "resumen_de_ronda", lambda carpeta, ronda: {})


def test_recolectar_jefes_registra_el_pid_borra_una_parada_vieja_y_limpia_al_terminar(
        monkeypatch: pytest.MonkeyPatch) -> None:
    observado: dict[str, Any] = {}
    preparar_recoleccion(monkeypatch, observado)
    parada = ent.CARPETA_RUNS / "mi_ronda" / "PARAR"
    parada.parent.mkdir(parents=True)
    parada.write_text("", encoding="utf-8")
    assert cli.main(["recolectar-jefes", "--jefes", "amlo", "--nombre", "mi_ronda", "--ronda", "2"]) == 0
    assert observado["pid"] == str(os.getpid()) and observado["parada_vieja"] is False
    assert observado["archivo_parada"] == parada and observado["limite_segundos"] is None
    assert observado["carpeta"] == ent.CARPETA_RUNS / "mi_ronda" / "datos" and observado["ronda"] == 2
    assert not ent.ARCHIVO_PID.exists() and not parada.exists()


def test_recolectar_jefes_convierte_el_tope_de_minutos_en_segundos(monkeypatch: pytest.MonkeyPatch) -> None:
    observado: dict[str, Any] = {}
    preparar_recoleccion(monkeypatch, observado)
    assert cli.main(["recolectar-jefes", "--jefes", "amlo", "--max-minutos", "1.5"]) == 0
    assert observado["limite_segundos"] == 90.0


def test_recolectar_jefes_si_faltan_episodios_devuelve_2_y_aun_asi_limpia(
        monkeypatch: pytest.MonkeyPatch) -> None:
    observado: dict[str, Any] = {}
    preparar_recoleccion(monkeypatch, observado, faltan=3)
    assert cli.main(["recolectar-jefes", "--jefes", "amlo"]) == 2
    assert not ent.ARCHIVO_PID.exists()


def test_recolectar_jefes_limpia_el_pid_y_la_parada_aunque_la_recoleccion_falle(
        monkeypatch: pytest.MonkeyPatch) -> None:
    observado: dict[str, Any] = {}
    preparar_recoleccion(monkeypatch, observado, falla=True)
    with pytest.raises(RuntimeError, match="se cayo"):
        cli.main(["recolectar-jefes", "--jefes", "amlo"])
    assert not ent.ARCHIVO_PID.exists()
    assert not (ent.CARPETA_RUNS / "imitacion_v2" / "PARAR").exists()


class SesionPanelFalsa:
    """Sesion del panel que anota si se cerro y entrega una politica reconocible."""

    def __init__(self) -> None:
        """
        Crea la sesion abierta.

        Contexto:
            Pruebas de ver-jefes con --panel.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            SesionPanelFalsa()
        """
        self.politica = object()
        self.url = "http://127.0.0.1:1/"
        self.cerrada = False
        self.terminados: list[Any] = []

    def al_terminar(self, resumen: Any) -> None:
        """
        Anota el resumen recibido.

        Contexto:
            Callback de ver_pelea.

        Args:
            resumen (Any): Resumen del episodio.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            sesion.al_terminar({})
        """
        self.terminados.append(resumen)

    def cerrar(self) -> None:
        """
        Marca la sesion como cerrada.

        Contexto:
            ver-jefes la cierra siempre.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            sesion.cerrar()
        """
        self.cerrada = True


def preparar_panel(monkeypatch: pytest.MonkeyPatch, registro: Registro, entorno: EntornoVentana,
                   falla: bool = False) -> SesionPanelFalsa:
    """
    Sustituye el panel, el entorno de ventana y la reproduccion de peleas.

    Contexto:
        Pruebas de ver-jefes con --panel.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        registro (Registro): Donde se anotan las llamadas.
        entorno (EntornoVentana): Entorno que devuelve la creacion.
        falla (bool): Si es True la reproduccion lanza RuntimeError.

    Returns:
        SesionPanelFalsa: La sesion que devolvera iniciar_sesion_panel.

    Raises:
        Ninguna.

    Ejemplo:
        sesion = preparar_panel(monkeypatch, registro, EntornoVentana())
    """
    sesion = SesionPanelFalsa()

    def iniciar(*args: Any) -> SesionPanelFalsa:
        """
        Sustituye a iniciar_sesion_panel.

        Contexto:
            Evita abrir un servidor y un navegador.

        Args:
            *args (Any): Modelo, politica, catalogo, jefe, nombre, puerto y navegador.

        Returns:
            SesionPanelFalsa: La sesion falsa.

        Raises:
            Ninguna.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("panel", *args)
        return sesion

    def ver(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Sustituye a ver_pelea.

        Contexto:
            Evita jugar.

        Args:
            *args (Any): Entorno, politica, episodios, tope de segundos y opciones.
            **kwargs (Any): Callback al_terminar.

        Returns:
            list[dict[str, Any]]: Un episodio fijo.

        Raises:
            RuntimeError: Si la prueba lo pide.

        Ejemplo:
            (sustituto)
        """
        registro.anotar("ver", *args, **kwargs)
        if falla:
            raise RuntimeError("la ventana se cerro")
        kwargs["al_terminar"]({"victoria": True})
        return [{"victoria": True}]

    monkeypatch.setattr(pv, "iniciar_sesion_panel", iniciar)
    monkeypatch.setattr(evj, "crear_entorno_jefes", lambda *a, **k: registro.anotar("crear") or entorno)
    monkeypatch.setattr(vis, "ver_pelea", ver)
    monkeypatch.setattr(evl, "resumir", lambda resumenes: {"episodios": len(resumenes)})
    return sesion


def test_el_parser_de_ver_jefes_conoce_las_opciones_del_panel() -> None:
    p = cli.construir_parser()
    assert p.parse_args(["ver-jefes", "--jefe", "amlo"]).panel is False
    v = p.parse_args(["ver-jefes", "--jefe", "amlo", "--panel", "--panel-puerto", "8800", "--sin-navegador"])
    assert (v.panel, v.panel_puerto, v.sin_navegador) == (True, 8800, True)
    assert p.parse_args(["ver-jefes", "--jefe", "amlo"]).panel_puerto == 8765


def test_ver_jefes_con_panel_abre_el_panel_antes_del_juego_y_lo_cierra(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registro, entorno = Registro(), EntornoVentana()
    sesion = preparar_panel(monkeypatch, registro, entorno)
    guardar_modelo_pequeno(entj.rutas_ppo("multijefe_ppo").mejor)
    assert cli.main(["ver-jefes", "--jefe", "amlo", "--panel", "--panel-puerto", "8801"]) == 0
    assert [nombre for nombre, _, _ in registro.llamadas] == ["panel", "crear", "ver"]
    (args, _), = registro.de("panel")
    assert args[3:7] == ("amlo", "multijefe_ppo", 8801, True)
    (argumentos, extras), = registro.de("ver")
    assert argumentos[1] is sesion.politica and extras == {"al_terminar": sesion.al_terminar}
    assert sesion.terminados == [{"victoria": True}] and sesion.cerrada and entorno.cerrado
    assert "panel en http://127.0.0.1:1/" in capsys.readouterr().out


def test_ver_jefes_con_panel_sin_navegador_y_modelo_pedido(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registro = Registro()
    preparar_panel(monkeypatch, registro, EntornoVentana())
    modelo = guardar_modelo_pequeno(tmp_path / "otro" / "m.zip")
    cli.main(["ver-jefes", "--jefe", "nahual", "--modelo", str(modelo), "--panel", "--sin-navegador"])
    (args, _), = registro.de("panel")
    assert args[4:7] == ("otro", 8765, False)


def test_ver_jefes_con_panel_cierra_panel_y_ventana_si_la_pelea_falla(monkeypatch: pytest.MonkeyPatch) -> None:
    registro, entorno = Registro(), EntornoVentana()
    sesion = preparar_panel(monkeypatch, registro, entorno, falla=True)
    guardar_modelo_pequeno(entj.rutas_ppo("multijefe_ppo").mejor)
    with pytest.raises(RuntimeError, match="ventana"):
        cli.main(["ver-jefes", "--jefe", "amlo", "--panel"])
    assert sesion.cerrada and entorno.cerrado


def test_ver_jefes_con_el_puerto_del_panel_ocupado_no_lanza_el_juego(monkeypatch: pytest.MonkeyPatch) -> None:
    registro = Registro()
    preparar_panel(monkeypatch, registro, EntornoVentana())

    def ocupado(*_args: Any) -> None:
        """
        Sustituye a iniciar_sesion_panel con un puerto ocupado.

        Contexto:
            Prueba de que el juego no se lanza si el panel no abre.

        Args:
            *_args (Any): Argumentos ignorados.

        Returns:
            None: Nunca retorna.

        Raises:
            OSError: Siempre.

        Ejemplo:
            (sustituto)
        """
        raise OSError("puerto ocupado")

    monkeypatch.setattr(pv, "iniciar_sesion_panel", ocupado)
    guardar_modelo_pequeno(entj.rutas_ppo("multijefe_ppo").mejor)
    with pytest.raises(OSError, match="ocupado"):
        cli.main(["ver-jefes", "--jefe", "amlo", "--panel"])
    assert registro.llamadas == []
