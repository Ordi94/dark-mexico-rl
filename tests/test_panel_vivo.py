"""PoliticaPanel, sesion del panel y su integracion con ver_pelea y ver-jefes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from dmrl.aplicacion import panel_vivo as pv
from dmrl.aplicacion import visualizacion as vis
from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento
from dmrl.aplicacion.evaluacion import PoliticaModelo
from dmrl.aplicacion.imitacion_red import crear_modelo_base
from dmrl.aplicacion.inspector_red import InspectorRed
from dmrl.dominio.acciones import ACCION_PARRY, N_ACCIONES
from dmrl.dominio.observacion_v2 import campos_v2, tamano_simple_v2
from dmrl.infraestructura.cliente_panel import ClientePanel
from dmrl.infraestructura.servidor_panel import ServidorPanel
from tests.support_jefes import catalogo_prueba

CATALOGO = catalogo_prueba()
NOMBRES = [n for n, _ in campos_v2(CATALOGO)]
TAMANO = tamano_simple_v2(CATALOGO)


class EntornoFalso:
    """Entorno minimo: observacion cruda mutable y mascara con todo permitido."""

    def __init__(self) -> None:
        """
        Crea el entorno con una pelea recien empezada.

        Contexto:
            Pruebas de PoliticaPanel.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoFalso()
        """
        self.observacion_cruda: dict[str, Any] = self.cruda()

    @staticmethod
    def cruda(t: float = 10.0, hp: float = 100.0, caguamas: int = 3, parries: int = 0,
              clase: str = "BP_AMLOBOSS_C") -> dict[str, Any]:
        """
        Observacion cruda con jugador y jefe.

        Contexto:
            Pruebas de PoliticaPanel.

        Args:
            t (float): Tiempo de juego.
            hp (float): Vida del jugador.
            caguamas (int): Caguamas restantes.
            parries (int): Parries del paso.
            clase (str): Clase del actor jefe.

        Returns:
            dict[str, Any]: Observacion cruda.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoFalso.cruda(hp=50)
        """
        return {"t": t, "player": {"hp": hp, "hp_max": 100.0, "caguamas": caguamas},
                "boss": {"cls": clase, "hp": 900.0, "hp_max": 1000.0}, "ev": {"parries": parries}}

    def action_masks(self) -> NDArray[np.bool_]:
        """
        Mascara con una accion prohibida.

        Contexto:
            Interfaz que usa PoliticaPanel.

        Args:
            Ninguno.

        Returns:
            NDArray[np.bool_]: 22 valores.

        Raises:
            Ninguna.

        Ejemplo:
            env.action_masks()
        """
        valida = np.ones(N_ACCIONES, dtype=np.bool_)
        valida[21] = False
        return valida


class BaseFija:
    """Politica base que siempre elige la misma accion y cuenta los reinicios."""

    def __init__(self, accion: int = ACCION_PARRY) -> None:
        """
        Crea la politica.

        Contexto:
            Pruebas de PoliticaPanel.

        Args:
            accion (int): Accion que elige siempre.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            BaseFija(3)
        """
        self.accion = accion
        self.reinicios = 0

    def reiniciar(self) -> None:
        """
        Cuenta el reinicio.

        Contexto:
            Interfaz Politica.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            base.reiniciar()
        """
        self.reinicios += 1

    def decidir(self, env: Any, obs: NDArray[np.float32]) -> int:
        """
        Devuelve la accion fija.

        Contexto:
            Interfaz Politica.

        Args:
            env (Any): Entorno (no usado).
            obs (NDArray[np.float32]): Observacion (no usada).

        Returns:
            int: La accion fija.

        Raises:
            Ninguna.

        Ejemplo:
            base.decidir(env, obs)
        """
        return self.accion


@pytest.fixture(scope="module")
def modelo() -> MaskablePPO:
    """
    Modelo sin entrenar con la forma de la observacion v2 del catalogo de prueba.

    Contexto:
        Pruebas del panel.

    Args:
        Ninguno.

    Returns:
        MaskablePPO: Modelo en CPU con capas de 8 neuronas.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    return crear_modelo_base(ConfigEntrenamiento(nombre="x", capas=(8, 8)), TAMANO)  # type: ignore[no-any-return]


def observacion(semilla: int = 0) -> NDArray[np.float32]:
    """
    Observacion apilada aleatoria del tamano del modelo.

    Contexto:
        Pruebas del panel.

    Args:
        semilla (int): Semilla del generador.

    Returns:
        NDArray[np.float32]: Vector de 4 x tamano simple valores.

    Raises:
        Ninguna.

    Ejemplo:
        observacion(2)
    """
    return np.random.default_rng(semilla).random(4 * TAMANO).astype(np.float32)


def crear_politica(modelo: MaskablePPO, base: Any = None) -> tuple[pv.PoliticaPanel, list[dict[str, Any]]]:
    """
    PoliticaPanel que acumula en una lista lo que publica.

    Contexto:
        Pruebas del panel.

    Args:
        modelo (MaskablePPO): Modelo a inspeccionar.
        base (Any, opcional): Politica base; por defecto BaseFija.

    Returns:
        tuple[pv.PoliticaPanel, list[dict[str, Any]]]: Politica y lista de mensajes publicados.

    Raises:
        Ninguna.

    Ejemplo:
        politica, publicados = crear_politica(modelo)
    """
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(base or BaseFija(), InspectorRed(modelo.policy, TAMANO), NOMBRES, publicados.append,
                                CATALOGO, "amlo", "ppo", 15.0)
    return politica, publicados


def test_decide_lo_mismo_que_la_politica_base_y_publica_un_cuadro(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo, BaseFija(7))
    politica.reiniciar()
    assert politica.decidir(EntornoFalso(), observacion()) == 7  # type: ignore[arg-type]
    (cuadro,) = publicados
    assert cuadro["tipo"] == "cuadro" and cuadro["paso"] == 1 and cuadro["accionElegida"] == 7
    assert cuadro["encabezado"]["jefe"] == CATALOGO.perfil("amlo").nombre and cuadro["encabezado"]["tiempoPelea"] == 0.0
    assert cuadro["mascara"][21] is False and len(cuadro["probabilidades"]) == N_ACCIONES
    assert len(cuadro["activacionesPolitica"]) == 2 and len(cuadro["activacionesPolitica"][0]) == 8
    assert cuadro["eventos"] == [] and len(cuadro["atencion"]) <= 10


def test_reiniciar_reinicia_la_politica_base(modelo: MaskablePPO) -> None:
    base = BaseFija()
    politica, _ = crear_politica(modelo, base)
    politica.reiniciar()
    politica.reiniciar()
    assert base.reinicios == 2


def test_los_eventos_se_detectan_entre_decisiones_y_el_tiempo_corre(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo)
    env = EntornoFalso()
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    env.observacion_cruda = EntornoFalso.cruda(t=10.5, hp=88.0, caguamas=2, parries=1, clase="BP_Chamuco_C")
    politica.decidir(env, observacion(1))  # type: ignore[arg-type]
    segundo = publicados[1]
    tipos = {e["tipo"] for e in segundo["eventos"]}
    assert tipos == {"parry", "golpe", "caguama", "fase"} and segundo["paso"] == 2
    assert segundo["encabezado"]["tiempoPelea"] == 0.5
    assert segundo["encabezado"]["jefe"] == CATALOGO.perfil("chamuco").nombre
    assert segundo["encabezado"]["fase"] == 2 and segundo["milisegundosCalculo"] >= 0


def test_el_historial_de_quien_se_conecta_despues_trae_valores_y_eventos(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo)
    env = EntornoFalso()
    politica.reiniciar()
    assert [m["tipo"] for m in politica.mensajes_iniciales()] == ["configuracion", "historial"]
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    env.observacion_cruda = EntornoFalso.cruda(t=11.0, hp=90.0)
    politica.decidir(env, observacion(1))  # type: ignore[arg-type]
    configuracion, historial, ultimo = politica.mensajes_iniciales()
    assert configuracion["capasPolitica"] == [8, 8] and configuracion["activacion"] == "tanh"
    assert [v[0] for v in historial["valores"]] == [0.0, 1.0] and ultimo == publicados[-1]
    assert historial["eventos"] == [{"t": 1.0, "tipo": "golpe", "detalle": 10.0}]


def test_el_historial_solo_conserva_la_ventana_de_la_grafica(modelo: MaskablePPO) -> None:
    politica, _ = crear_politica(modelo)
    env = EntornoFalso()
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    env.observacion_cruda = EntornoFalso.cruda(t=11.0, hp=90.0)
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    env.observacion_cruda = EntornoFalso.cruda(t=40.0, hp=90.0)
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    assert politica.mensajes_iniciales()[1]["eventos"] == []


def test_un_episodio_nuevo_borra_el_historial_y_reinicia_el_contador(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo)
    env = EntornoFalso()
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    politica.reiniciar()
    env.observacion_cruda = EntornoFalso.cruda(t=100.0, hp=50.0)
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    assert publicados[-1]["paso"] == 1 and publicados[-1]["eventos"] == []
    assert len(politica.mensajes_iniciales()[1]["valores"]) == 1


def test_finalizar_publica_el_resultado_y_lo_deja_en_el_historial(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo)
    politica.finalizar({"victoria": True, "resultado": "victoria", "danio_hecho": 900.0, "parries": 5})
    politica.finalizar({"victoria": False, "resultado": "muerte"})
    gano, murio = publicados
    assert gano["tipo"] == "fin" and gano["hito"]["tipo"] == "victoria" and gano["parries"] == 5
    assert murio["hito"]["tipo"] == "muerte" and murio["danioRecibido"] == 0.0
    assert [e["tipo"] for e in politica.mensajes_iniciales()[1]["eventos"]] == ["victoria", "muerte"]


def test_un_fallo_del_panel_no_cambia_la_accion_ni_detiene_la_pelea(
        modelo: MaskablePPO, capsys: pytest.CaptureFixture[str]) -> None:
    politica, publicados = crear_politica(modelo, BaseFija(3))
    env = EntornoFalso()
    politica.reiniciar()
    assert politica.decidir(env, np.zeros(3, dtype=np.float32)) == 3  # type: ignore[arg-type]
    assert politica.decidir(env, np.zeros(3, dtype=np.float32)) == 3  # type: ignore[arg-type]
    assert publicados == [] and politica.errores == 2
    assert capsys.readouterr().err.count("se omite un cuadro") == 1


def test_si_publicar_falla_la_decision_se_devuelve_igual(modelo: MaskablePPO) -> None:
    def roto(_mensaje: dict[str, Any]) -> None:
        """
        Publicador que siempre falla.

        Contexto:
            Prueba de aislamiento de fallos.

        Args:
            _mensaje (dict[str, Any]): Mensaje ignorado.

        Returns:
            None: No retorna valor.

        Raises:
            RuntimeError: Siempre.

        Ejemplo:
            roto({})
        """
        raise RuntimeError("navegador roto")

    politica = pv.PoliticaPanel(BaseFija(4), InspectorRed(modelo.policy, TAMANO), NOMBRES, roto, CATALOGO,
                                "amlo", "ppo", 15.0)
    assert politica.decidir(EntornoFalso(), observacion()) == 4  # type: ignore[arg-type]
    politica.finalizar({"victoria": True})
    assert politica.errores == 2


def test_una_clase_de_jefe_desconocida_usa_el_jefe_de_la_sesion(modelo: MaskablePPO) -> None:
    politica, publicados = crear_politica(modelo)
    env = EntornoFalso()
    env.observacion_cruda = EntornoFalso.cruda(clase="BP_Misterio_C")
    politica._jefe = "sesion"
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    assert publicados[0]["encabezado"]["jefe"] == "sesion"


def test_la_sesion_abre_el_navegador_publica_y_se_cierra(modelo: MaskablePPO) -> None:
    abiertas: list[str] = []
    sesion = pv.iniciar_sesion_panel(modelo, PoliticaModelo(modelo), CATALOGO, "amlo", "prueba", puerto=0,
                                     abrir=lambda url: abiertas.append(url) or True)
    try:
        assert abiertas == [sesion.url] and sesion.url.startswith("http://127.0.0.1:")
        cliente = ClientePanel(sesion.servidor.puerto)
        assert cliente.recibir()["tipo"] == "configuracion" and cliente.recibir()["tipo"] == "historial"
        sesion.politica.reiniciar()
        env = EntornoFalso()
        accion = sesion.politica.decidir(env, observacion())  # type: ignore[arg-type]
        cuadro = cliente.recibir()
        assert cuadro is not None and cuadro["accionElegida"] == accion
        sesion.al_terminar({"victoria": True, "resultado": "victoria"})
        assert cliente.recibir()["tipo"] == "fin"  # type: ignore[index]
        cliente.cerrar()
    finally:
        sesion.cerrar()


def test_sin_navegador_no_se_abre_ninguna_pagina(modelo: MaskablePPO) -> None:
    abiertas: list[str] = []
    sesion = pv.iniciar_sesion_panel(modelo, PoliticaModelo(modelo), CATALOGO, "amlo", "prueba", puerto=0,
                                     abrir_navegador=False, abrir=lambda url: abiertas.append(url) or True)
    sesion.cerrar()
    assert abiertas == []


def test_un_modelo_que_no_es_v2_o_un_puerto_ocupado_fallan_antes_de_abrir_nada(
        modelo: MaskablePPO, tmp_path: Path) -> None:
    abiertas: list[str] = []
    pequeno = crear_modelo_base(ConfigEntrenamiento(nombre="x", capas=(8, 8)), 6)
    with pytest.raises(ValueError, match="no es multiplo"):
        pv.iniciar_sesion_panel(pequeno, PoliticaModelo(pequeno), CATALOGO, "amlo", "x", puerto=0,
                                abrir=lambda url: abiertas.append(url) or True)
    ocupante = ServidorPanel(0)
    puerto = ocupante.iniciar()
    try:
        with pytest.raises(OSError):
            pv.iniciar_sesion_panel(modelo, PoliticaModelo(modelo), CATALOGO, "amlo", "x", puerto=puerto,
                                    abrir=lambda url: abiertas.append(url) or True)
    finally:
        ocupante.detener()
    assert abiertas == []


class EntornoDePelea:
    """Entorno que termina la pelea tras dos pasos con un resumen de victoria."""

    def __init__(self) -> None:
        """
        Crea el entorno.

        Contexto:
            Pruebas de ver_pelea.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            EntornoDePelea()
        """
        self.observacion_cruda: dict[str, Any] = {"t": 0.0}
        self.pasos = 0

    def reset(self, options: dict[str, Any] | None = None) -> tuple[None, dict[str, Any]]:
        """
        Reinicia la cuenta de pasos.

        Contexto:
            Interfaz que usa ver_pelea.

        Args:
            options (dict, opcional): Opciones de reset (ignoradas).

        Returns:
            tuple[None, dict[str, Any]]: Sin observacion ni info.

        Raises:
            Ninguna.

        Ejemplo:
            env.reset()
        """
        self.pasos = 0
        return None, {}

    def step(self, accion: int) -> tuple[None, float, bool, bool, dict[str, Any]]:
        """
        Avanza un paso; el segundo termina la pelea.

        Contexto:
            Interfaz que usa ver_pelea.

        Args:
            accion (int): Accion (ignorada).

        Returns:
            tuple: (obs, recompensa, terminado, truncado, info).

        Raises:
            Ninguna.

        Ejemplo:
            env.step(0)
        """
        self.pasos += 1
        fin = self.pasos >= 2
        return None, 0.0, fin, False, ({"episodio": {"victoria": True, "resultado": "victoria"}} if fin else {})


def test_ver_pelea_avisa_al_terminar_cada_episodio() -> None:
    avisos: list[dict[str, Any]] = []

    class Politica:
        """Politica que no hace nada."""

        def reiniciar(self) -> None:
            """
            Sin estado.

            Contexto:
                Interfaz Politica.

            Args:
                Ninguno.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                Politica().reiniciar()
            """

        def decidir(self, env: Any, obs: Any) -> int:
            """
            Siempre "nada".

            Contexto:
                Interfaz Politica.

            Args:
                env (Any): Entorno.
                obs (Any): Observacion.

            Returns:
                int: Cero.

            Raises:
                Ninguna.

            Ejemplo:
                Politica().decidir(None, None)
            """
            return 0

    resumenes = vis.ver_pelea(EntornoDePelea(), Politica(), episodios=2, al_terminar=avisos.append,  # type: ignore[arg-type]
                              dormir=lambda _s: None)
    assert avisos == resumenes and len(avisos) == 2
