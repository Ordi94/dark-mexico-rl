"""Infraestructura: lanzamiento de instancias, procesos del sistema y linea de comandos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from dmrl.aplicacion import entrenamiento as ent
from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion import visualizacion as vis
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.entrada import cli
from dmrl.infraestructura import juego, procesos
from tests.support_dmrl import JuegoSintetico


class ClienteFalso:
    """BridgeClient falso que registra su ciclo de vida."""

    instancias: list[ClienteFalso] = []  # noqa: RUF012

    def __init__(self, id_instancia: str, reply_timeout: float) -> None:
        """
        Registra la creacion.

        Contexto:
            Sustituye a BridgeClient en las pruebas.

        Args:
            id_instancia (str): Id de la instancia.
            reply_timeout (float): Plazo de respuesta.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.id, self.plazo = id_instancia, reply_timeout
        self.lanzado: Any = None
        self.consola: list[str] | None = None
        self.cerrado = False
        self.falla_al_lanzar = False
        self.instancias.append(self)

    def launch(self, config: Any, connect_timeout: float) -> dict[str, Any]:
        """
        Registra el lanzamiento (o falla si se pidio).

        Contexto:
            Sustituye a BridgeClient.launch.

        Args:
            config (Any): LaunchConfig recibido.
            connect_timeout (float): Plazo de conexion.

        Returns:
            dict[str, Any]: Respuesta de saludo falsa.

        Raises:
            TimeoutError: Si falla_al_lanzar esta activo.

        Ejemplo:
            (uso interno)
        """
        if ClienteFalso.fallar:
            raise TimeoutError("sin conexion")
        self.lanzado = (config, connect_timeout)
        return {}

    fallar = False

    def config(self, console: list[str]) -> dict[str, Any]:
        """
        Registra los comandos de consola.

        Contexto:
            Sustituye a BridgeClient.config.

        Args:
            console (list[str]): Comandos.

        Returns:
            dict[str, Any]: Respuesta vacia.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.consola = console
        return {}

    def close(self, quit_game: bool = True) -> None:
        """
        Registra el cierre.

        Contexto:
            Sustituye a BridgeClient.close.

        Args:
            quit_game (bool): Ignorado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.cerrado = True


@pytest.fixture
def cliente_falso(monkeypatch: pytest.MonkeyPatch) -> type[ClienteFalso]:
    """
    Sustituye BridgeClient por el cliente falso.

    Contexto:
        Pruebas de crear_fabrica_puerto.

    Args:
        monkeypatch (pytest.MonkeyPatch): Parcheador de pytest.

    Returns:
        type[ClienteFalso]: Clase falsa con su registro limpio.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    ClienteFalso.instancias = []
    ClienteFalso.fallar = False
    monkeypatch.setattr(juego, "BridgeClient", ClienteFalso)
    return ClienteFalso


def test_la_fabrica_lanza_el_perfil_de_entrenamiento_con_afinidad_4_a_27(cliente_falso: type[ClienteFalso]) -> None:
    """La instancia usa su id, el perfil training, 30 Hz y deja libres las CPUs 0-3."""
    puerto = juego.crear_fabrica_puerto(juego.ParametrosInstancia(id_instancia="z9"))()
    c = cliente_falso.instancias[0]
    config, plazo = c.lanzado
    assert puerto is c and c.id == "z9" and config.profile == "training" and config.sim_hz == 30.0
    assert config.affinity == list(range(4, 28)) and config.show_window is False and plazo == 180.0
    assert c.consola is None


def test_la_fabrica_aplica_los_comandos_de_consola_pedidos(cliente_falso: type[ClienteFalso]) -> None:
    """Con consola, se envian tras conectar; sin afinidad no se limita ninguna CPU."""
    params = juego.ParametrosInstancia(perfil="window", afinidad=(), ventana_visible=True, consola=("t.MaxFPS 30",))
    juego.crear_fabrica_puerto(params)()
    c = cliente_falso.instancias[0]
    assert c.consola == ["t.MaxFPS 30"] and c.lanzado[0].affinity is None and c.lanzado[0].show_window


def test_la_fabrica_cierra_la_instancia_si_el_lanzamiento_falla(cliente_falso: type[ClienteFalso]) -> None:
    """Si el juego no se conecta no queda una instancia a medias."""
    cliente_falso.fallar = True
    with pytest.raises(TimeoutError):
        juego.crear_fabrica_puerto(juego.ParametrosInstancia())()
    assert cliente_falso.instancias[0].cerrado


def test_cerrar_puerto_tolera_none_y_errores() -> None:
    """Cerrar None no hace nada y un error al cerrar se ignora."""
    juego.cerrar_puerto(None)

    class Roto:
        """Puerto cuyo cierre falla."""

        def close(self, quit_game: bool = True) -> None:
            """
            Falla siempre.

            Contexto:
                Escenario de la prueba.

            Args:
                quit_game (bool): Ignorado.

            Returns:
                None: Nunca retorna.

            Raises:
                RuntimeError: Siempre.

            Ejemplo:
                (uso interno)
            """
            raise RuntimeError("ya muerto")

    juego.cerrar_puerto(Roto())  # type: ignore[arg-type]


def test_ajustar_proceso_actual_baja_la_prioridad_y_limita_las_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Se aplica Below Normal y la afinidad; los rechazos del sistema se toleran."""
    registro: dict[str, Any] = {}

    class ProcesoFalso:
        """Proceso que registra nice y cpu_affinity."""

        def nice(self, valor: int) -> None:
            """
            Registra la prioridad.

            Contexto:
                Sustituye a psutil.Process.nice.

            Args:
                valor (int): Clase de prioridad.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            registro["nice"] = valor

        def cpu_affinity(self, cpus: list[int]) -> None:
            """
            Registra la afinidad.

            Contexto:
                Sustituye a psutil.Process.cpu_affinity.

            Args:
                cpus (list[int]): CPUs permitidas.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            registro["cpus"] = cpus

    monkeypatch.setattr(procesos.psutil, "Process", lambda *a: ProcesoFalso())
    procesos.ajustar_proceso_actual(range(4, 8))
    assert registro == {"nice": psutil.BELOW_NORMAL_PRIORITY_CLASS, "cpus": [4, 5, 6, 7]}
    registro.clear()
    procesos.ajustar_proceso_actual(None, prioridad_baja=False)
    assert registro == {}

    class Rechaza(ProcesoFalso):
        """Proceso cuyo sistema rechaza los cambios."""

        def nice(self, valor: int) -> None:
            """
            Rechaza el cambio de prioridad.

            Contexto:
                Escenario de la prueba.

            Args:
                valor (int): Ignorado.

            Returns:
                None: Nunca retorna.

            Raises:
                psutil.AccessDenied: Siempre.

            Ejemplo:
                (uso interno)
            """
            raise psutil.AccessDenied()

        def cpu_affinity(self, cpus: list[int]) -> None:
            """
            Rechaza la afinidad.

            Contexto:
                Escenario de la prueba.

            Args:
                cpus (list[int]): Ignorado.

            Returns:
                None: Nunca retorna.

            Raises:
                ValueError: Siempre.

            Ejemplo:
                (uso interno)
            """
            raise ValueError("cpu invalida")

    monkeypatch.setattr(procesos.psutil, "Process", lambda *a: Rechaza())
    procesos.ajustar_proceso_actual([99])


def test_pid_se_escribe_se_lee_y_se_descarta_si_el_proceso_murio(tmp_path: Path) -> None:
    """leer_pid devuelve el PID vivo y None si falta el archivo, esta danado o el proceso no existe."""
    ruta = tmp_path / "carpeta" / "x.pid"
    procesos.escribir_pid(ruta)
    assert procesos.leer_pid(ruta) == psutil.Process().pid
    ruta.write_text("no es numero", encoding="utf-8")
    assert procesos.leer_pid(ruta) is None
    assert procesos.leer_pid(tmp_path / "no_existe.pid") is None
    ruta.write_text("4000000", encoding="utf-8")
    assert procesos.leer_pid(ruta) is None


def lanzar_dormilon(hijos: int = 0) -> subprocess.Popen[bytes]:
    """
    Lanza un proceso Python que duerme (y opcionalmente lanza hijos que tambien duermen).

    Contexto:
        Pruebas de apagado de procesos.

    Args:
        hijos (int): Cantidad de procesos hijos que lanza.

    Returns:
        subprocess.Popen[bytes]: Proceso raiz.

    Raises:
        OSError: Si no se puede lanzar.

    Ejemplo:
        proc = lanzar_dormilon(1)
    """
    codigo = (
        "import subprocess, sys, time\n"
        f"[subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']) for _ in range({hijos})]\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", codigo])
    deadline = time.monotonic() + 10
    while len(psutil.Process(proc.pid).children()) < hijos and time.monotonic() < deadline:
        time.sleep(0.05)
    return proc


def test_apagar_arbol_termina_al_proceso_y_a_sus_hijos() -> None:
    """Se matan raiz e hijos; un PID inexistente devuelve 0."""
    proc = lanzar_dormilon(hijos=2)
    vivos = [proc.pid, *(h.pid for h in psutil.Process(proc.pid).children(recursive=True))]
    # Windows puede interponer un lanzador por interprete: se exige al menos raiz + 2 hijos.
    assert procesos.apagar_arbol(proc.pid) >= 3
    assert proc.wait(timeout=10) is not None
    assert not any(psutil.pid_exists(pid) for pid in vivos)
    assert procesos.apagar_arbol(4000000) == 0


def test_apagar_descendientes_deja_vivo_al_proceso_raiz() -> None:
    """Solo los hijos mueren; la raiz sigue hasta que se la mata aparte."""
    proc = lanzar_dormilon(hijos=1)
    try:
        hijos = psutil.Process(proc.pid).children(recursive=True)
        assert procesos.apagar_descendientes(proc.pid) >= 1
        assert not any(h.is_running() for h in hijos)
        assert procesos.apagar_descendientes(4000000) == 0
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_ruta_de_modelo_prefiere_mejor_y_luego_el_ultimo_checkpoint(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    """mejor.zip si existe; si no, checkpoints/ultimo.zip; si no hay ninguno, error."""
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli.ruta_modelo_por_defecto("a")
    (tmp_path / "a" / "checkpoints").mkdir(parents=True)
    (tmp_path / "a" / "checkpoints" / "ultimo.zip").write_bytes(b"x")
    assert cli.ruta_modelo_por_defecto("a").name == "ultimo.zip"
    (tmp_path / "a" / "mejor.zip").write_bytes(b"x")
    assert cli.ruta_modelo_por_defecto("a").name == "mejor.zip"


def test_parser_de_la_linea_de_comandos_conoce_los_cinco_comandos() -> None:
    """Cada comando se parsea con sus valores por defecto documentados."""
    p = cli.construir_parser()
    e = p.parse_args(["entrenar"])
    assert (e.nombre, e.instancias, e.dispositivo, e.desde_cero, e.n_steps) == ("principal", 4, "auto", False, 1024)
    v = p.parse_args(["evaluar", "--politica", "bot"])
    assert (v.episodios, v.instancias, v.politica) == (50, 4, "bot")
    assert p.parse_args(["ver"]).politica == "modelo"
    assert p.parse_args(["verificar"]).comando == "verificar"
    assert p.parse_args(["especificacion"]).comando == "especificacion"
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_comando_entrenar_arma_la_configuracion_y_llama_al_entrenamiento(monkeypatch: pytest.MonkeyPatch,
                                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Las opciones de la linea de comandos llegan a ConfigEntrenamiento."""
    recibido: list[ent.ConfigEntrenamiento] = []
    monkeypatch.setattr(ent, "entrenar", lambda config: recibido.append(config) or Path("u.zip"))
    code = cli.main(["entrenar", "--nombre", "x", "--instancias", "2", "--pasos", "10", "--desde-cero",
                     "--dispositivo", "cpu", "--ent-coef", "0.02"])
    assert code == 0 and recibido[0].nombre == "x" and recibido[0].instancias == 2
    assert recibido[0].reanudar is False and recibido[0].ent_coef == 0.02 and recibido[0].pasos_totales == 10
    assert "u.zip" in capsys.readouterr().out


def test_comando_evaluar_imprime_el_resumen_y_escribe_el_detalle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """evaluar reparte las peleas, imprime las metricas y guarda el JSON si se pide."""
    llamadas: list[tuple[Any, ...]] = []
    resumen = {"victoria": True, "danio_hecho": 10.0, "fase_max": 1}
    monkeypatch.setattr(evl, "evaluar_en_paralelo", lambda *a: llamadas.append(a) or [resumen])
    salida = tmp_path / "sub" / "r.json"
    assert cli.main(["evaluar", "--politica", "azar", "--episodios", "3", "--instancias", "2",
                     "--salida", str(salida)]) == 0
    assert llamadas[0] == ("azar", 3, 2, None, 0)
    assert json.loads(salida.read_text())["resumen"]["tasa_victoria"] == 1.0
    assert "tasa_victoria" in capsys.readouterr().out
    cli.main(["evaluar", "--politica", "bot"])
    assert llamadas[1][3] is None


def test_comando_evaluar_pasa_el_ruido_solo_si_se_pide(monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """--epsilon-ruido llega a evaluar_en_paralelo como argumento nombrado; sin el, la llamada no cambia."""
    llamadas: list[dict[str, Any]] = []
    monkeypatch.setattr(evl, "evaluar_en_paralelo", lambda *a, **k: llamadas.append(k) or [])
    cli.main(["evaluar", "--politica", "bot", "--epsilon-ruido", "0.03"])
    cli.main(["evaluar", "--politica", "bot"])
    capsys.readouterr()
    assert llamadas == [{"epsilon_ruido": 0.03}, {}]


def test_comando_evaluar_con_modelo_usa_la_ruta_por_defecto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Con --politica modelo sin --modelo se busca mejor.zip de la corrida."""
    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "mejor.zip").write_bytes(b"x")
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path)
    llamadas: list[tuple[Any, ...]] = []
    monkeypatch.setattr(evl, "evaluar_en_paralelo", lambda *a: llamadas.append(a) or [])
    cli.main(["evaluar", "--politica", "modelo", "--nombre", "n", "--episodios", "1"])
    assert llamadas[0][3] == tmp_path / "n" / "mejor.zip"


def test_comando_ver_juega_con_ventana_y_cierra(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """ver crea el entorno con ventana, juega con la politica pedida y lo cierra al terminar."""
    juegos: list[JuegoSintetico] = []

    def entorno(tabla: TablaGolpes) -> Any:
        """
        Entorno sobre un juego sintetico en lugar de la ventana real.

        Contexto:
            Sustituye a crear_entorno_ventana.

        Args:
            tabla (TablaGolpes): Tabla de golpes.

        Returns:
            DarkMexicoEnv: Entorno de prueba.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        from dmrl.aplicacion.entorno import DarkMexicoEnv

        def lanzar() -> JuegoSintetico:
            """
            Lanza el juego sintetico registrado.

            Contexto:
                Fabrica de la prueba.

            Args:
                Ninguno.

            Returns:
                JuegoSintetico: Juego nuevo.

            Raises:
                Ninguna.

            Ejemplo:
                (uso interno)
            """
            juegos.append(JuegoSintetico(periodo=2))
            return juegos[-1]

        return DarkMexicoEnv(lanzar, tabla)

    monkeypatch.setattr(vis, "crear_entorno_ventana", entorno)
    assert cli.main(["ver", "--politica", "bot", "--episodios", "1"]) == 0
    assert juegos[0].cerrado and "episodios" in capsys.readouterr().out


def test_comando_ver_con_modelo_resuelve_la_ruta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Con --politica modelo la ruta del modelo se resuelve y se pasa a la politica."""
    monkeypatch.setattr(ent, "CARPETA_RUNS", tmp_path)
    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "mejor.zip").write_bytes(b"x")
    construidas: list[Any] = []

    def politica(nombre: str, tabla: TablaGolpes, ruta: str | None, semilla: int) -> evl.PoliticaAleatoria:
        """
        Sustituye a construir_politica registrando la ruta del modelo.

        Contexto:
            Prueba de comando_ver.

        Args:
            nombre (str): Nombre de la politica.
            tabla (TablaGolpes): Tabla de golpes.
            ruta (str | None): Ruta del modelo resuelta.
            semilla (int): Semilla.

        Returns:
            evl.PoliticaAleatoria: Politica de azar.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        construidas.append(ruta)
        return evl.PoliticaAleatoria(1)

    from dmrl.aplicacion.entorno import DarkMexicoEnv

    monkeypatch.setattr(evl, "construir_politica", politica)
    monkeypatch.setattr(vis, "crear_entorno_ventana",
                        lambda tabla: DarkMexicoEnv(lambda: JuegoSintetico(periodo=2), tabla))
    cli.main(["ver", "--nombre", "n", "--max-segundos", "0"])
    assert construidas == [str(tmp_path / "n" / "mejor.zip")]


def test_comando_verificar_ejecuta_check_env(monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """verificar corre check_env de SB3 sobre el entorno y lo cierra."""
    juegos: list[JuegoSintetico] = []

    def lanzar() -> JuegoSintetico:
        """
        Lanza el juego sintetico registrado.

        Contexto:
            Fabrica de la prueba.

        Args:
            Ninguno.

        Returns:
            JuegoSintetico: Juego nuevo.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        juegos.append(JuegoSintetico(periodo=50))
        return juegos[-1]

    monkeypatch.setattr(cli, "crear_fabrica_puerto", lambda params: lanzar)
    monkeypatch.setattr(cli, "cargar_tabla_golpes", lambda: TablaGolpes())
    assert cli.main(["verificar"]) == 0
    assert "check_env: OK" in capsys.readouterr().out and juegos[0].cerrado


def test_comando_especificacion_escribe_el_documento(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Se genera el Markdown de la observacion en la ruta configurada."""
    destino = tmp_path / "docs" / "ESPEC.md"
    monkeypatch.setattr(cli, "ARCHIVO_ESPECIFICACION", destino)
    assert cli.comando_especificacion(argparse.Namespace()) == 0
    assert "Especificacion de la observacion" in destino.read_text(encoding="utf-8")


def test_python_m_dmrl_ejecuta_la_linea_de_comandos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El punto de entrada del paquete llama a main con los argumentos del proceso."""
    import runpy

    destino = tmp_path / "ESPEC.md"
    monkeypatch.setattr(cli, "ARCHIVO_ESPECIFICACION", destino)
    monkeypatch.setattr(sys, "argv", ["dmrl", "especificacion"])
    with pytest.raises(SystemExit) as salida:
        runpy.run_module("dmrl", run_name="__main__")
    assert salida.value.code == 0 and destino.exists()
