"""Callbacks del entrenamiento sobre un modelo falso."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dmrl.aplicacion.callbacks import (
    CallbackCheckpoint,
    CallbackCurriculo,
    CallbackMejorModelo,
    CallbackMetricas,
    CallbackParada,
    cargar_estado_curriculo,
    episodios_terminados,
    guardar_estado_curriculo,
)
from dmrl.dominio.configuracion import ConfigCurriculo
from dmrl.dominio.curriculo import Curriculo

CONFIG = ConfigCurriculo(niveles=(0.3, 1.0), tasa_victoria_subida=0.5, ventana=4, minimo_episodios=4)


class EntornoFalso:
    """Registra las llamadas env_method que recibe del callback de curriculo."""

    def __init__(self) -> None:
        """
        Crea el registro vacio.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoFalso()
        """
        self.llamadas: list[tuple[str, Any]] = []

    def env_method(self, nombre: str, valor: Any) -> None:
        """
        Anota la llamada.

        Contexto:
            Sustituye a VecEnv.env_method.

        Args:
            nombre (str): Metodo pedido a los entornos.
            valor (Any): Argumento.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            env.env_method("fijar_multiplicador", 0.3)
        """
        self.llamadas.append((nombre, valor))


class LoggerFalso:
    """Guarda los valores registrados."""

    def __init__(self) -> None:
        """
        Crea el diccionario de valores.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            LoggerFalso()
        """
        self.valores: dict[str, float] = {}

    def record(self, clave: str, valor: float) -> None:
        """
        Registra un valor.

        Contexto:
            Sustituye a Logger.record.

        Args:
            clave (str): Nombre de la metrica.
            valor (float): Valor.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            logger.record("dm/x", 1.0)
        """
        self.valores[clave] = valor


class ModeloFalso:
    """Modelo minimo: pasos, logger, entorno y guardado."""

    def __init__(self) -> None:
        """
        Crea el modelo con contadores en cero.

        Contexto:
            Cada prueba construye el suyo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ModeloFalso()
        """
        self.num_timesteps = 0
        self.logger = LoggerFalso()
        self.entorno = EntornoFalso()
        self.guardados: list[Path] = []

    def get_env(self) -> EntornoFalso:
        """
        Devuelve el entorno falso.

        Contexto:
            Sustituye a BaseAlgorithm.get_env.

        Args:
            Ninguno.

        Returns:
            EntornoFalso: Entorno que registra llamadas.

        Raises:
            Ninguna.

        Ejemplo:
            modelo.get_env()
        """
        return self.entorno

    def save(self, ruta: Path | str) -> None:
        """
        Escribe un archivo para poder comprobar que se guardo.

        Contexto:
            Sustituye a BaseAlgorithm.save.

        Args:
            ruta (Path | str): Destino.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            modelo.save("m.zip")
        """
        Path(ruta).write_bytes(b"modelo")
        self.guardados.append(Path(ruta))


def resumen(victoria: bool = False, fase_inicio: int = 1, resultado: str = "muerte", **extra: Any) -> dict[str, Any]:
    """
    Resumen de episodio como el que publica el entorno.

    Contexto:
        Entrada de los callbacks.

    Args:
        victoria (bool): Si gano.
        fase_inicio (int): Fase de inicio.
        resultado (str): Resultado del episodio.
        **extra (Any): Campos a sobrescribir.

    Returns:
        dict[str, Any]: Resumen completo.

    Raises:
        Ninguna.

    Ejemplo:
        resumen(victoria=True)
    """
    base = {
        "resultado": "victoria" if victoria else resultado, "victoria": victoria, "fase_inicio": fase_inicio,
        "danio_hecho": 100.0, "danio_recibido": 50.0, "parries": 2, "caguamas_usadas": 1,
        "caguamas_interrumpidas": 0, "fase_max": 1, "duracion_s": 60.0, "recompensa": 3.0,
        "progreso_jefe": 0.2, "curado": 0.0, "multiplicador": 0.3,
    }
    base.update(extra)
    return base


def paso(callback: Any, *resumenes: dict[str, Any]) -> bool:
    """
    Simula un paso de los entornos con episodios terminados.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        callback (Any): Callback bajo prueba (ya inicializado).
        *resumenes (dict[str, Any]): Resumenes de los episodios que terminan.

    Returns:
        bool: Valor devuelto por on_step.

    Raises:
        Ninguna.

    Ejemplo:
        paso(callback, resumen(True))
    """
    callback.update_locals({"infos": [{"episodio": r} for r in resumenes] + [{}]})
    return bool(callback.on_step())


def iniciar(callback: Any, modelo: ModeloFalso | None = None) -> ModeloFalso:
    """
    Inicializa un callback con un modelo falso.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        callback (Any): Callback a inicializar.
        modelo (ModeloFalso, opcional): Modelo a usar.

    Returns:
        ModeloFalso: Modelo conectado al callback.

    Raises:
        Ninguna.

    Ejemplo:
        modelo = iniciar(CallbackMetricas())
    """
    modelo = modelo or ModeloFalso()
    callback.init_callback(modelo)
    callback.on_training_start({}, {})
    return modelo


def test_episodios_terminados_ignora_las_infos_sin_resumen() -> None:
    """Solo se devuelven los resumenes presentes."""
    assert list(episodios_terminados({"infos": [{}, {"episodio": {"a": 1}}]})) == [{"a": 1}]
    assert list(episodios_terminados({})) == []


def test_metricas_publican_el_promedio_de_la_ventana_con_prefijo_dm() -> None:
    """Al cerrar el rollout se registran tasa de victoria, dano, parries, caguamas y fase."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackMetricas(ventana=10, curriculo=curriculo)
    modelo = iniciar(callback)
    modelo.num_timesteps = 100
    assert paso(callback, resumen(True, fase_max=2), resumen(False), resumen(False, fase_inicio=2))
    callback.on_rollout_end()
    v = modelo.logger.valores
    assert v["dm/tasa_victoria"] == pytest.approx(0.5)  # solo episodios que empiezan en fase 1
    assert v["dm/tasa_victoria_global"] == pytest.approx(1 / 3)
    assert v["dm/danio_al_jefe"] == pytest.approx(100.0)
    assert v["dm/parries"] == pytest.approx(2.0) and v["dm/caguamas_usadas"] == pytest.approx(1.0)
    assert v["dm/fase_max"] == pytest.approx(4 / 3)
    assert v["dm/nivel_curriculo"] == 0 and v["dm/multiplicador_vida_jefe"] == 0.3
    assert v["dm/episodios_total"] == 3 and v["dm/pasos_por_segundo"] > 0
    assert {"dm/progreso_jefe", "dm/danio_recibido", "dm/caguamas_interrumpidas", "dm/duracion_s",
            "dm/recompensa_episodio", "dm/tasa_victoria_nivel"} <= set(v)


def test_metricas_imprimen_una_linea_de_progreso_por_rollout(capsys: pytest.CaptureFixture[str]) -> None:
    """La consola recibe pasos, velocidad, tasa de victoria, dano y recompensa; imprimir=False la silencia."""
    callback = CallbackMetricas(ventana=10, curriculo=Curriculo(CONFIG))
    iniciar(callback)
    paso(callback, resumen(True))
    callback.on_rollout_end()
    linea = capsys.readouterr().out
    assert "victoria=1.00" in linea and "nivel=0" in linea and "danio_jefe=100" in linea
    silencioso = CallbackMetricas(ventana=10, imprimir=False)
    iniciar(silencioso)
    paso(silencioso, resumen(True))
    silencioso.on_rollout_end()
    assert capsys.readouterr().out == ""


def test_metricas_sin_episodios_solo_publican_velocidad() -> None:
    """Sin episodios todavia no hay promedios que registrar."""
    callback = CallbackMetricas()
    modelo = iniciar(callback)
    callback.on_rollout_end()
    assert "dm/tasa_victoria" not in modelo.logger.valores
    assert "dm/nivel_curriculo" not in modelo.logger.valores
    assert "dm/pasos_por_segundo" in modelo.logger.valores


def test_metricas_con_ventana_sin_episodios_de_fase_1_dan_tasa_cero() -> None:
    """Si solo hay episodios que empiezan en fase 2 la tasa de fase 1 es 0."""
    callback = CallbackMetricas()
    modelo = iniciar(callback)
    paso(callback, resumen(True, fase_inicio=2))
    callback.on_rollout_end()
    assert modelo.logger.valores["dm/tasa_victoria"] == 0.0


def test_metricas_no_cuentan_los_episodios_con_error() -> None:
    """Una caida del juego se cuenta aparte y no entra en los promedios."""
    callback = CallbackMetricas()
    modelo = iniciar(callback)
    paso(callback, resumen(resultado="error"), resumen(True))
    callback.on_rollout_end()
    assert modelo.logger.valores["dm/errores_instancia_total"] == 1
    assert modelo.logger.valores["dm/episodios_total"] == 1


def test_curriculo_publica_el_nivel_al_empezar_y_al_subir() -> None:
    """Al empezar y en cada subida se envian multiplicador y probabilidad de fase 2 a los entornos."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackCurriculo(curriculo)
    modelo = iniciar(callback)
    assert modelo.entorno.llamadas == [("fijar_multiplicador", 0.3), ("fijar_prob_fase2", 0.2)]
    paso(callback, resumen(True), resumen(True), resumen(True), resumen(True))
    assert curriculo.nivel == 1
    assert modelo.entorno.llamadas[-2:] == [("fijar_multiplicador", 1.0), ("fijar_prob_fase2", 0.2)]


def test_curriculo_ignora_los_episodios_con_error() -> None:
    """Un error de instancia no cuenta como derrota."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackCurriculo(curriculo)
    iniciar(callback)
    paso(callback, resumen(resultado="error"))
    assert curriculo.episodios_en_nivel == 0


def test_estado_del_curriculo_se_guarda_y_restaura(tmp_path: Path) -> None:
    """El nivel y la ventana viajan en curriculo.json; un archivo danado no rompe la reanudacion."""
    c = Curriculo(CONFIG)
    for _ in range(4):
        c.registrar_episodio(True, 1)
    ruta = tmp_path / "curriculo.json"
    guardar_estado_curriculo(c, ruta)
    d = Curriculo(CONFIG)
    assert cargar_estado_curriculo(d, ruta) and d.nivel == 1
    assert not cargar_estado_curriculo(d, tmp_path / "no_existe.json")
    ruta.write_text("no es json", encoding="utf-8")
    assert not cargar_estado_curriculo(d, ruta)
    ruta.write_text('{"nivel": 9}', encoding="utf-8")
    assert not cargar_estado_curriculo(d, ruta)


def test_checkpoint_guarda_ultimo_y_conserva_solo_los_ultimos(tmp_path: Path) -> None:
    """Cada multiplo de `cada` se guarda ultimo.zip, el curriculo y un checkpoint numerado."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackCheckpoint(tmp_path / "ck", curriculo, cada=100, conservar=2)
    modelo = iniciar(callback)
    for pasos in (50, 100, 150, 200, 250, 300, 400):
        modelo.num_timesteps = pasos
        callback.on_step()
    nombres = sorted(p.name for p in (tmp_path / "ck").iterdir())
    assert "ultimo.zip" in nombres and "curriculo.json" in nombres
    numerados = [n for n in nombres if n.startswith("modelo_")]
    assert len(numerados) == 2


def test_checkpoint_no_guarda_antes_de_cumplir_el_intervalo(tmp_path: Path) -> None:
    """Con menos pasos que `cada` no se escribe nada."""
    callback = CallbackCheckpoint(tmp_path / "ck", Curriculo(CONFIG), cada=100)
    modelo = iniciar(callback)
    modelo.num_timesteps = 99
    callback.on_step()
    assert list((tmp_path / "ck").iterdir()) == []


def test_mejor_modelo_se_guarda_al_mejorar_y_no_al_empeorar(tmp_path: Path) -> None:
    """Con 4 episodios minimos, mejor.zip se escribe al mejorar la tasa de victoria."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackMejorModelo(tmp_path / "mejor.zip", curriculo, ventana=4, minimo_episodios=4)
    iniciar(callback)
    paso(callback, resumen(False), resumen(False))
    callback.on_rollout_end()
    assert not (tmp_path / "mejor.zip").exists()
    paso(callback, resumen(True), resumen(False))
    callback.on_rollout_end()
    primero = callback.mejor
    assert (tmp_path / "mejor.zip").exists() and primero is not None
    paso(callback, resumen(False), resumen(False), resumen(False), resumen(False))
    callback.on_rollout_end()
    assert callback.mejor == primero


def test_mejor_modelo_reinicia_la_ventana_al_subir_de_nivel_y_prefiere_el_nivel_mayor(tmp_path: Path) -> None:
    """Un nivel mas alto siempre supera a cualquier puntaje del nivel anterior."""
    curriculo = Curriculo(CONFIG)
    callback = CallbackMejorModelo(tmp_path / "mejor.zip", curriculo, ventana=4, minimo_episodios=2)
    iniciar(callback)
    paso(callback, resumen(True), resumen(True))
    callback.on_rollout_end()
    nivel0 = callback.mejor
    curriculo.nivel = 1
    paso(callback, resumen(False, progreso_jefe=0.0), resumen(False, progreso_jefe=0.0), resumen(True, fase_inicio=2))
    callback.on_rollout_end()
    assert callback.mejor is not None and nivel0 is not None and callback.mejor > nivel0
    assert callback.mejor[0] == 1


def test_parada_detiene_el_entrenamiento_cuando_aparece_el_archivo(tmp_path: Path) -> None:
    """Se comprueba el disco cada `cada` pasos y False pide parar."""
    archivo = tmp_path / "PARAR"
    callback = CallbackParada(archivo, cada=3)
    iniciar(callback)
    resultados = [callback.on_step() for _ in range(3)]
    assert resultados == [True, True, True]
    archivo.write_text("", encoding="utf-8")
    assert [callback.on_step() for _ in range(3)] == [True, True, False]
