"""Ciclo de imitacion de punta a punta con recoleccion y evaluacion simuladas (el ajuste es real)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest
import torch

from dmrl.aplicacion import imitacion_ciclo as ciclo
from dmrl.dominio.imitacion import ConfigAprendizaje, ConfigImitacion, PlanEpisodio, resumen_de_juego
from dmrl.entrada import cli
from dmrl.infraestructura.demostraciones import escribir_json, guardar_lote, leer_json, metadatos_de_ronda
from tests.support_imitacion import demostraciones_sinteticas

CPU = torch.device("cpu")


def configuracion(**cambios: Any) -> ConfigImitacion:
    """
    Configuracion pequena para las pruebas del ciclo.

    Contexto:
        Atajo de las pruebas de este archivo.

    Args:
        **cambios (Any): Campos a sobrescribir.

    Returns:
        ConfigImitacion: Ciclo de 2 rondas con pocas peleas.

    Raises:
        Ninguna.

    Ejemplo:
        configuracion(rondas_dagger=0)
    """
    base: dict[str, Any] = dict(
        nombre="prueba", instancias=2, episodios_ronda0=12, episodios_dagger=8, rondas_dagger=1,
        episodios_evaluacion=4, episodios_fase2=2, episodios_ruido=2, episodios_confirmacion=3,
        aprendizaje=ConfigAprendizaje(epocas_maximas=2, paciencia=2, epocas_valor=1, tamano_lote=64),
    )
    base.update(cambios)
    return ConfigImitacion(**base)


class Simulacion:
    """
    Recoleccion y evaluacion simuladas que registran sus llamadas.

    Contexto:
        Sustituyen a recolectar_ronda y evaluar_en_paralelo para ejercitar el ciclo sin juego.
    """

    def __init__(self, faltan: int = 0) -> None:
        """
        Crea la simulacion.

        Contexto:
            Cada prueba construye la suya.

        Args:
            faltan (int): Episodios que la recoleccion dice no haber podido guardar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            Simulacion(faltan=1)
        """
        self.faltan = faltan
        self.recolecciones: list[tuple[int, Path | None]] = []
        self.evaluaciones: list[tuple[str, int, str, float]] = []
        self.crear_parada_al_evaluar: Path | None = None

    def recolectar(self, carpeta: Path, ronda: int, planes: list[PlanEpisodio], instancias: int,
                   previo: Path | None, entorno: Any, parada: Path) -> int:
        """
        Escribe un lote sintetico con un episodio por plan.

        Contexto:
            Sustituye a recolectar_ronda.

        Args:
            carpeta (Path): Carpeta raiz de los datos.
            ronda (int): Numero de ronda.
            planes (list[PlanEpisodio]): Plan de la ronda.
            instancias (int): Instancias (no usado).
            previo (Path, opcional): Modelo que ejecuta.
            entorno (Any): Configuracion del entorno (no usada).
            parada (Path): Archivo de parada (no usado).

        Returns:
            int: Episodios que se simulan como no guardados.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.recolecciones.append((ronda, previo))
        if self.faltan:
            return self.faltan
        if not metadatos_de_ronda(carpeta, ronda):
            datos = demostraciones_sinteticas(episodios=len(planes), pasos=20, semilla=ronda, ronda=ronda)
            meta = [{"indice": p.indice, "pasos": 20, "victoria": p.indice % 3 == 0, "fase_inicio": p.fase_inicio,
                     "epsilon": p.epsilon} for p in planes]
            guardar_lote(carpeta / f"ronda_{ronda:02d}" / "w0_00000.npz", datos, meta)
        return 0

    def evaluar(self, politica: str, episodios: int, instancias: int, ruta: Path | None, semilla: int,
                opciones: dict[str, Any] | None = None, epsilon_ruido: float = 0.0) -> list[dict[str, Any]]:
        """
        Devuelve peleas simuladas: el modelo de la ronda 1 gana mas que el de la 0.

        Contexto:
            Sustituye a evaluar_en_paralelo.

        Args:
            politica (str): "bot" o "modelo".
            episodios (int): Peleas.
            instancias (int): Instancias (no usado).
            ruta (Path, opcional): Modelo evaluado.
            semilla (int): Semilla base.
            opciones (dict, opcional): Opciones de reset.
            epsilon_ruido (float): Ruido de ejecucion.

        Returns:
            list[dict[str, Any]]: Resumenes de episodios.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.evaluaciones.append((politica, episodios, ruta.name if ruta else "", epsilon_ruido))
        if self.crear_parada_al_evaluar is not None:
            self.crear_parada_al_evaluar.parent.mkdir(parents=True, exist_ok=True)
            self.crear_parada_al_evaluar.write_text("", encoding="utf-8")
        ganadas = episodios if (ruta and ruta.name == "ronda_01.zip") else episodios // 2
        return [{"victoria": i < ganadas, "resultado": "victoria" if i < ganadas else "muerte",
                 "muerte": i >= ganadas, "fase_max": 2, "danio_hecho": 1000.0 + i, "parries": 3.0}
                for i in range(episodios)]


@pytest.fixture
def raiz(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Redirige runs/ a una carpeta temporal.

    Contexto:
        Las pruebas del ciclo escriben datos, modelos y evaluaciones.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.
        tmp_path (Path): Carpeta temporal de la prueba.

    Returns:
        Path: Carpeta que hace de runs/.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    monkeypatch.setattr(ciclo, "CARPETA_RUNS", tmp_path)
    return tmp_path


def test_las_rutas_cuelgan_de_la_carpeta_de_la_corrida(raiz: Path) -> None:
    """Todas las rutas derivan de runs/<nombre>."""
    r = ciclo.rutas_de("x")
    assert r.base == raiz / "x" and r.datos == raiz / "x" / "datos" and r.parada.name == "PARAR"
    assert r.mejor.name == "mejor.zip" and r.modelo(3).name == "ronda_03.zip"
    assert r.informe_ajuste(1).name == "ronda_01.json" and r.evaluacion("bot").parent.name == "evaluaciones"


def test_registrar_en_consola_antepone_la_hora(capsys: pytest.CaptureFixture[str]) -> None:
    """Formato [HH:MM:SS] mensaje."""
    ciclo.registrar_en_consola("hola")
    salida = capsys.readouterr().out
    assert salida.startswith("[") and salida.endswith("] hola\n") and salida[3] == ":"


def test_planes_de_ronda_usan_la_cantidad_de_su_ronda() -> None:
    """La ronda 0 usa la cantidad inicial; las de DAgger, la suya, con semillas distintas."""
    config = configuracion()
    assert len(ciclo.planes_de_ronda(config, 0)) == 12 and len(ciclo.planes_de_ronda(config, 1)) == 8
    assert ciclo.planes_de_ronda(config, 1) != ciclo.planes_de_ronda(config, 2)


def test_el_dispositivo_de_ajuste_es_cuda_si_existe(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU si hay, CPU si no."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ciclo.elegir_dispositivo_ajuste().type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ciclo.elegir_dispositivo_ajuste().type == "cpu"


def test_entrenar_ronda_agrega_rondas_guarda_modelo_e_informe(raiz: Path) -> None:
    """El informe trae curvas, brecha, precision por accion y la verificacion de recarga."""
    config, sim = configuracion(), Simulacion()
    rutas = ciclo.rutas_de(config.nombre)
    for ronda in (0, 1):
        sim.recolectar(rutas.datos, ronda, ciclo.planes_de_ronda(config, ronda), 2, None, None, rutas.parada)
    mensajes: list[str] = []
    informe = ciclo.entrenar_ronda(config, rutas, 1, CPU, mensajes.append)
    assert rutas.modelo(1).exists() and leer_json(rutas.informe_ajuste(1))["ronda"] == 1
    assert informe["pasos_totales"] == 20 * (12 + 8) and informe["coincidencia_tras_recargar"] == 1.0
    assert informe["acciones_por_ronda"] == {"0": 1, "1": 1} or set(informe["acciones_por_ronda"]) == {"0", "1"}
    assert len(informe["por_accion"]) == 22 and informe["por_accion"][19]["nombre"] == "parry"
    assert "brecha_perdida" in informe and "ecm_validacion" in informe["valor"]
    assert any("epoca" in m for m in mensajes)


def test_evaluar_condiciones_juega_las_tres_condiciones_o_solo_la_principal(raiz: Path) -> None:
    """Principal, fase 2 y ruido; con completa=False solo la principal."""
    config, sim = configuracion(), Simulacion()
    mensajes: list[str] = []
    completa = ciclo.evaluar_condiciones(config, "bot", None, 5, sim.evaluar, mensajes.append)
    assert set(completa) == {"principal", "fase2", "ruido"} and len(mensajes) == 3
    assert [e[3] for e in sim.evaluaciones] == [0.0, 0.0, config.epsilon_evaluacion]
    assert set(ciclo.evaluar_condiciones(config, "bot", None, 5, sim.evaluar, mensajes.append, False)) == {"principal"}


def test_el_ciclo_completo_elige_la_mejor_ronda_confirma_y_es_reanudable(raiz: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Dos rondas, linea base, mejor.zip de la ronda con mas victorias y confirmacion; repetirlo no reentrena."""
    config, sim = configuracion(), Simulacion()
    registro: list[str] = []
    estado = ciclo.ejecutar_ciclo(config, sim.recolectar, sim.evaluar, registro.append, CPU)
    rutas = ciclo.rutas_de(config.nombre)
    assert estado == {"estado": "completo", "ronda": 1}
    assert [r for r, _ in sim.recolecciones] == [0, 1] and sim.recolecciones[1][1] == rutas.modelo(0)
    assert rutas.mejor.read_bytes() == rutas.modelo(1).read_bytes()
    assert leer_json(rutas.base / "mejor.json")["ronda"] == 1
    assert leer_json(rutas.evaluacion("confirmacion_ronda_01"))["resumen"]["peleas"] == 3.0
    assert rutas.evaluacion("confirmacion_ronda_00").exists() and rutas.evaluacion("bot_confirmacion").exists()
    assert leer_json(rutas.base / "mejor.json")["criterio"] == "agrupado"
    assert leer_json(rutas.evaluacion("bot"))["principal"]["resumen"]["peleas"] == 4.0
    llamadas = len(sim.evaluaciones)
    entrenadas: list[int] = []
    monkeypatch.setattr(ciclo, "entrenar_ronda", lambda *a, **k: entrenadas.append(1))
    assert ciclo.ejecutar_ciclo(config, sim.recolectar, sim.evaluar, registro.append, CPU)["estado"] == "completo"
    assert entrenadas == [] and len(sim.evaluaciones) == llamadas


def test_el_ciclo_se_interrumpe_si_faltan_episodios_o_hay_parada(raiz: Path) -> None:
    """Sin recoleccion completa no se entrena; con la parada pedida se sale ordenadamente en cada punto."""
    config = configuracion()
    rutas = ciclo.rutas_de(config.nombre)
    registro: list[str] = []
    incompleta = Simulacion(faltan=2)
    assert ciclo.ejecutar_ciclo(config, incompleta.recolectar, incompleta.evaluar, registro.append, CPU) == {
        "estado": "interrumpido", "ronda": 0}
    assert not rutas.modelo(0).exists() and "quedan 2" in registro[-1]
    rutas.base.mkdir(parents=True, exist_ok=True)
    rutas.parada.write_text("", encoding="utf-8")
    assert ciclo.ejecutar_ciclo(config, incompleta.recolectar, incompleta.evaluar, registro.append, CPU) == {
        "estado": "interrumpido", "ronda": 0}
    rutas.parada.unlink()
    sim = Simulacion()
    sim.crear_parada_al_evaluar = rutas.parada
    final = ciclo.ejecutar_ciclo(configuracion(rondas_dagger=0), sim.recolectar, sim.evaluar, registro.append, CPU)
    assert final == {"estado": "interrumpido", "ronda": 1} and not rutas.evaluacion("bot").exists()


def test_seleccionar_y_confirmar_exigen_alguna_evaluacion(raiz: Path) -> None:
    """Sin evaluaciones de ninguna ronda no hay modelo que elegir ni candidata que confirmar."""
    config = configuracion()
    with pytest.raises(ValueError):
        ciclo.seleccionar_mejor(config, ciclo.rutas_de(config.nombre), print)
    with pytest.raises(ValueError):
        ciclo.confirmar_candidatas(config, ciclo.rutas_de(config.nombre), Simulacion().evaluar, print)


def escribir_evaluacion(rutas: ciclo.RutasImitacion, nombre: str, ganadas: int, total: int,
                        principal: bool = True) -> None:
    """
    Escribe una evaluacion sintetica con `ganadas` victorias de `total` peleas.

    Contexto:
        Pruebas de la seleccion en dos etapas.

    Args:
        rutas (RutasImitacion): Rutas de la corrida.
        nombre (str): Nombre de la evaluacion ("ronda_00", "confirmacion_ronda_00", ...).
        ganadas (int): Victorias.
        total (int): Peleas.
        principal (bool): Si es True el contenido va bajo la clave "principal".

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        escribir_evaluacion(rutas, "ronda_00", 40, 50)
    """
    episodios = [{"victoria": i < ganadas, "resultado": "victoria" if i < ganadas else "muerte", "fase_max": 2,
                  "danio_hecho": 2000.0, "parries": 1.0, "muerte": i >= ganadas} for i in range(total)]
    contenido: dict[str, Any] = {"resumen": resumen_de_juego(episodios), "episodios": episodios}
    escribir_json(rutas.evaluacion(nombre), {"principal": contenido} if principal else contenido)


def test_la_eleccion_final_usa_la_muestra_junta_y_no_solo_la_de_seleccion(raiz: Path) -> None:
    """La ronda que gano la seleccion por suerte pierde ante la que rinde parejo en las dos muestras."""
    config = configuracion(rondas_dagger=1)
    rutas = ciclo.rutas_de(config.nombre)
    for ronda in (0, 1):
        rutas.modelo(ronda).parent.mkdir(parents=True, exist_ok=True)
        rutas.modelo(ronda).write_bytes(f"modelo{ronda}".encode())
    escribir_evaluacion(rutas, "ronda_00", 45, 50)  # 90% en seleccion, 50% al confirmar
    escribir_evaluacion(rutas, "ronda_01", 40, 50)  # 80% y 80%
    escribir_evaluacion(rutas, "confirmacion_ronda_00", 50, 100, principal=False)
    escribir_evaluacion(rutas, "confirmacion_ronda_01", 80, 100, principal=False)
    assert ciclo.seleccionar_mejor(config, rutas, print) == 1
    assert rutas.mejor.read_bytes() == b"modelo1"
    informe = leer_json(rutas.base / "mejor.json")
    assert informe["principales"]["0"]["tasa_victoria"] > informe["principales"]["1"]["tasa_victoria"]
    assert informe["agrupados"]["1"]["peleas"] == 150.0


def test_sin_confirmaciones_la_eleccion_cae_a_la_evaluacion_principal(raiz: Path) -> None:
    """Con el ciclo cortado antes de confirmar, decide la evaluacion principal."""
    config = configuracion(rondas_dagger=1)
    rutas = ciclo.rutas_de(config.nombre)
    for ronda in (0, 1):
        rutas.modelo(ronda).parent.mkdir(parents=True, exist_ok=True)
        rutas.modelo(ronda).write_bytes(b"x")
    escribir_evaluacion(rutas, "ronda_00", 30, 50)
    escribir_evaluacion(rutas, "ronda_01", 40, 50)
    assert ciclo.seleccionar_mejor(config, rutas, print) == 1
    assert leer_json(rutas.base / "mejor.json")["criterio"] == "principal"


def test_solo_se_confirman_las_candidatas_pedidas_y_no_se_repiten(raiz: Path) -> None:
    """Con una candidata se juega una sola confirmacion; una segunda llamada no vuelve a jugar."""
    config = configuracion(rondas_dagger=2, candidatas_confirmacion=1)
    rutas = ciclo.rutas_de(config.nombre)
    for ronda, ganadas in ((0, 20), (1, 40), (2, 30)):
        escribir_evaluacion(rutas, f"ronda_{ronda:02d}", ganadas, 50)
    sim = Simulacion()
    assert ciclo.confirmar_candidatas(config, rutas, sim.evaluar, print) == [1]
    assert ciclo.confirmar_candidatas(config, rutas, sim.evaluar, print) == [1]
    assert len(sim.evaluaciones) == 1 and sim.evaluaciones[0][2] == "ronda_01.zip"


def test_episodios_por_ronda_resume_la_recoleccion(raiz: Path) -> None:
    """Episodios, pasos, victorias, fraccion en fase 2 y con ruido; las rondas vacias se omiten."""
    config, sim = configuracion(), Simulacion()
    rutas = ciclo.rutas_de(config.nombre)
    sim.recolectar(rutas.datos, 0, ciclo.planes_de_ronda(config, 0), 2, None, None, rutas.parada)
    resumen = ciclo.episodios_por_ronda(rutas, [0, 1])
    assert set(resumen) == {0} and resumen[0]["episodios"] == 12.0 and resumen[0]["pasos"] == 240.0
    assert resumen[0]["fase2"] == pytest.approx(round(12 * 0.2) / 12) and resumen[0]["con_ruido"] == 0.5


def test_comando_imitar_ejecuta_el_ciclo_y_limpia_su_pid(raiz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El PID existe mientras corre y desaparece al terminar; el codigo de salida refleja la interrupcion."""
    pid = raiz / "entrenamiento.pid"
    monkeypatch.setattr(cli.ent, "ARCHIVO_PID", pid)
    monkeypatch.setattr(cli, "ajustar_proceso_actual", lambda *a, **k: None)
    vistos: list[tuple[bool, ConfigImitacion]] = []

    def ciclo_falso(config: ConfigImitacion) -> dict[str, Any]:
        """
        Sustituye a ejecutar_ciclo registrando si el PID existia.

        Contexto:
            Prueba de comando_imitar.

        Args:
            config (ConfigImitacion): Configuracion recibida.

        Returns:
            dict[str, Any]: Estado completo o interrumpido segun la corrida.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        vistos.append((pid.exists(), config))
        return {"estado": "completo" if len(vistos) == 1 else "interrumpido", "ronda": 0}

    monkeypatch.setattr(cli.ciclo, "ejecutar_ciclo", ciclo_falso)
    args = ["imitar", "--nombre", "z", "--instancias", "2", "--episodios-inicial", "5", "--episodios-dagger", "3",
            "--rondas", "1", "--semilla", "9"]
    assert cli.main(args) == 0 and not pid.exists()
    assert vistos[0][0] is True and vistos[0][1].episodios_ronda0 == 5 and vistos[0][1].rondas_dagger == 1
    assert vistos[0][1].semilla == 9 and vistos[0][1].nombre == "z"
    assert cli.main(args) == 2


def test_comando_imitar_entrenar_reajusta_una_ronda(raiz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--accion entrenar solo reajusta la ronda pedida con los datos existentes."""
    monkeypatch.setattr(cli.ent, "ARCHIVO_PID", raiz / "entrenamiento.pid")
    monkeypatch.setattr(cli, "ajustar_proceso_actual", lambda *a, **k: None)
    pedidas: list[int] = []
    monkeypatch.setattr(cli.ciclo, "entrenar_ronda", lambda config, rutas, ronda, dispositivo: pedidas.append(ronda))
    assert cli.main(["imitar", "--accion", "entrenar", "--ronda", "2"]) == 0 and pedidas == [2]


def test_el_parser_conoce_imitar_con_sus_valores_por_defecto() -> None:
    """Defaults documentados del comando imitar."""
    a: argparse.Namespace = cli.construir_parser().parse_args(["imitar"])
    assert (a.nombre, a.instancias, a.episodios_inicial, a.episodios_dagger, a.rondas, a.accion) == (
        "imitacion", 4, 600, 300, 3, "todo")
    assert (a.episodios_evaluacion, a.episodios_secundarios, a.episodios_confirmacion, a.candidatas) == (50, 20, 100, 2)
