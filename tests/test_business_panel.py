"""Reglas de negocio del panel en vivo: fidelidad con la politica, lectura honesta y la pelea primero."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from torch import nn

from dmrl.aplicacion import panel_vivo as pv
from dmrl.aplicacion.entrenamiento import ConfigEntrenamiento
from dmrl.aplicacion.evaluacion import PoliticaModelo
from dmrl.aplicacion.imitacion_red import crear_modelo_base
from dmrl.aplicacion.inspector_red import InspectorRed
from dmrl.dominio import acciones as acc
from dmrl.dominio.observacion_v2 import campos_v2, tamano_simple_v2
from dmrl.dominio.panel import TOP_ATENCION
from dmrl.infraestructura.cliente_panel import ClientePanel
from dmrl.infraestructura.servidor_panel import HOST_LOCAL, ServidorPanel
from tests.support_jefes import catalogo_prueba
from tests.test_panel_vivo import CATALOGO, NOMBRES, TAMANO, BaseFija, EntornoFalso, observacion


@pytest.fixture(scope="module")
def modelo() -> MaskablePPO:
    """
    Modelo sin entrenar con la observacion v2 del catalogo de prueba.

    Contexto:
        Reglas de negocio del panel.

    Args:
        Ninguno.

    Returns:
        MaskablePPO: Modelo en CPU con capas de 16 neuronas.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    return crear_modelo_base(ConfigEntrenamiento(nombre="x", capas=(16, 16), semilla=3), TAMANO)  # type: ignore[no-any-return]


class EntornoReal(EntornoFalso):
    """Entorno cuya mascara sale de las reglas reales del juego (mascara_acciones)."""

    def action_masks(self) -> NDArray[np.bool_]:
        """
        Mascara del juego para el jugador de la observacion cruda.

        Contexto:
            Reglas de negocio del panel.

        Args:
            Ninguno.

        Returns:
            NDArray[np.bool_]: 22 valores segun acciones.mascara_acciones.

        Raises:
            Ninguna.

        Ejemplo:
            env.action_masks()
        """
        return np.asarray(acc.mascara_acciones(self.observacion_cruda["player"]), dtype=np.bool_)


def test_acciones_enmascaradas_se_marcan_no_disponibles(modelo: MaskablePPO) -> None:
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(BaseFija(acc.ACCION_ATAQUE), InspectorRed(modelo.policy, TAMANO), NOMBRES,
                                publicados.append, CATALOGO, "amlo", "ppo", 15.0)
    env = EntornoReal()
    env.observacion_cruda["player"].update({"stamina": 1.0, "can_attack": True, "can_roll": True, "can_heal": True,
                                            "drinking": False, "power_ready": False})
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    cuadro = publicados[0]
    # Vida al maximo: la caguama no cura; sin barra de poder el especial no se acepta.
    assert cuadro["mascara"][acc.ACCION_CAGUAMA] is False and cuadro["mascara"][acc.ACCION_ESPECIAL] is False
    assert cuadro["probabilidades"][acc.ACCION_CAGUAMA] == 0.0 and cuadro["probabilidades"][acc.ACCION_ESPECIAL] == 0.0
    assert cuadro["mascara"][acc.ACCION_ATAQUE] is True and cuadro["mascara"][acc.ACCION_NADA] is True
    assert sum(cuadro["probabilidades"]) == pytest.approx(1.0, abs=1e-3)


def test_acciones_validas_cambian_de_estado_cuando_el_juego_las_habilita(modelo: MaskablePPO) -> None:
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(BaseFija(), InspectorRed(modelo.policy, TAMANO), NOMBRES, publicados.append,
                                CATALOGO, "amlo", "ppo", 15.0)
    env = EntornoReal()
    env.observacion_cruda["player"].update({"hp": 40.0, "stamina": 1.0, "can_heal": True, "can_attack": True})
    politica.reiniciar()
    politica.decidir(env, observacion())  # type: ignore[arg-type]
    assert publicados[0]["mascara"][acc.ACCION_CAGUAMA] is True


def test_panel_no_bloquea_si_no_hay_clientes(modelo: MaskablePPO) -> None:
    servidor = ServidorPanel(0)
    politica = pv.PoliticaPanel(BaseFija(), InspectorRed(modelo.policy, TAMANO), NOMBRES, servidor.publicar,
                                CATALOGO, "amlo", "ppo", 15.0)
    env = EntornoFalso()
    politica.reiniciar()
    inicio = time.perf_counter()
    for i in range(200):
        politica.decidir(env, observacion(i))  # type: ignore[arg-type]
    promedio_ms = (time.perf_counter() - inicio) / 200 * 1000
    assert promedio_ms < 33.0, f"{promedio_ms:.1f} ms por decision: no cabe en el periodo de 15 Hz (66 ms)"
    assert servidor.cuadros_publicados == 0 and politica.errores == 0


def test_un_navegador_lento_no_frena_las_decisiones(modelo: MaskablePPO) -> None:
    sesion = pv.iniciar_sesion_panel(modelo, PoliticaModelo(modelo), CATALOGO, "amlo", "ppo", puerto=0,
                                     abrir_navegador=False)
    lento = ClientePanel(sesion.servidor.puerto)
    try:
        env = EntornoFalso()
        sesion.politica.reiniciar()
        inicio = time.perf_counter()
        for i in range(150):
            sesion.politica.decidir(env, observacion(i))  # type: ignore[arg-type]
        assert (time.perf_counter() - inicio) / 150 < 0.033
        assert sesion.politica.errores == 0
    finally:
        lento.cerrar()
        sesion.cerrar()


def test_el_panel_solo_escucha_en_el_bucle_local() -> None:
    with pytest.raises(ValueError):
        ServidorPanel(8765, host="0.0.0.0")
    servidor = ServidorPanel(0)
    servidor.iniciar()
    try:
        assert servidor._escucha is not None and servidor._escucha.getsockname()[0] == HOST_LOCAL == "127.0.0.1"
    finally:
        servidor.detener()


def politica_lineal(pesos: list[float], tamano_simple: int, copias: int) -> SimpleNamespace:
    """
    Politica de una capa lineal sin sesgo cuya accion 0 pesa cada caracteristica con `pesos`.

    Contexto:
        Reglas de atribucion: con un modelo lineal gradiente x entrada es exactamente w * x.

    Args:
        pesos (list[float]): Peso de cada caracteristica de la observacion simple (se repite por copia).
        tamano_simple (int): Caracteristicas por observacion simple.
        copias (int): Observaciones apiladas.

    Returns:
        SimpleNamespace: Politica con la interfaz que usa el inspector.

    Raises:
        Ninguna.

    Ejemplo:
        politica_lineal([1.0, 2.0], 2, 4)
    """
    entrada = tamano_simple * copias
    capa = nn.Linear(entrada, 1, bias=False)
    with torch.no_grad():
        capa.weight[:] = torch.tensor(pesos * copias).reshape(1, -1)

    class Extractor(nn.Module):
        """Extractor cuyas redes son identidades; el logit sale de action_net."""

        def __init__(self) -> None:
            """
            Crea las redes identidad.

            Contexto:
                Politica lineal de prueba.

            Args:
                Ninguno.

            Returns:
                None: No retorna valor.

            Raises:
                Ninguna.

            Ejemplo:
                Extractor()
            """
            super().__init__()
            self.policy_net = nn.Sequential(nn.Linear(entrada, entrada, bias=False))
            self.value_net = nn.Sequential(nn.Linear(entrada, entrada, bias=False))
            with torch.no_grad():
                self.policy_net[0].weight.copy_(torch.eye(entrada))
                self.value_net[0].weight.copy_(torch.eye(entrada))

        def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
            """
            Pasa la entrada por la red de politica identidad.

            Contexto:
                Politica lineal de prueba.

            Args:
                x (torch.Tensor): Entrada.

            Returns:
                torch.Tensor: La misma entrada.

            Raises:
                Ninguna.

            Ejemplo:
                extractor.forward_actor(x)
            """
            return self.policy_net(x)

        def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
            """
            Pasa la entrada por la red de valor identidad.

            Contexto:
                Politica lineal de prueba.

            Args:
                x (torch.Tensor): Entrada.

            Returns:
                torch.Tensor: La misma entrada.

            Raises:
                Ninguna.

            Ejemplo:
                extractor.forward_critic(x)
            """
            return self.value_net(x)

    accion = nn.Linear(entrada, 2, bias=False)
    with torch.no_grad():
        accion.weight[0] = capa.weight[0]
        accion.weight[1] = 0.0
    return SimpleNamespace(
        mlp_extractor=Extractor(), action_net=accion, value_net=nn.Linear(entrada, 1),
        observation_space=spaces.Box(-1, 1, (entrada,), np.float32), share_features_extractor=True,
        extract_features=lambda x: x)


def test_atribucion_agrega_las_cuatro_copias_apiladas() -> None:
    pesos = [2.0, -1.0, 0.5]
    inspector = InspectorRed(politica_lineal(pesos, 3, 4), 3)
    # La caracteristica 0 solo esta activa en la copia mas vieja; la 1 en las cuatro; la 2 en la mas nueva.
    obs = np.zeros((4, 3), dtype=np.float32)
    obs[0, 0] = 1.0
    obs[:, 1] = [0.5, 0.5, 0.5, 0.5]
    obs[3, 2] = 4.0
    resultado = inspector.inspeccionar(obs.reshape(-1), np.array([True, True]), 0)
    assert resultado.atribucion.tolist() == pytest.approx([2.0 * 1.0, -1.0 * 2.0, 0.5 * 4.0])


def test_la_atribucion_puede_ser_negativa_si_la_entrada_empuja_en_contra() -> None:
    inspector = InspectorRed(politica_lineal([1.0, -3.0], 2, 4), 2)
    obs = np.tile(np.array([0.1, 0.9], dtype=np.float32), 4)
    atribucion = inspector.inspeccionar(obs, np.array([True, True]), 0).atribucion
    assert atribucion[0] > 0 > atribucion[1]


def test_las_diez_entradas_mas_influyentes_vienen_ordenadas_y_legibles(modelo: MaskablePPO) -> None:
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(BaseFija(), InspectorRed(modelo.policy, TAMANO), NOMBRES, publicados.append,
                                CATALOGO, "amlo", "ppo", 15.0)
    politica.reiniciar()
    politica.decidir(EntornoFalso(), observacion(7))  # type: ignore[arg-type]
    atencion = publicados[0]["atencion"]
    assert len(atencion) == TOP_ATENCION
    magnitudes = [abs(a["atribucion"]) for a in atencion]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(a["etiqueta"] and a["valorTexto"] for a in atencion)
    assert {a["nombre"] for a in atencion} <= set(NOMBRES)


def test_el_color_de_las_neuronas_no_se_renormaliza_entre_cuadros(modelo: MaskablePPO) -> None:
    publicados: list[dict[str, Any]] = []
    inspector = InspectorRed(modelo.policy, TAMANO)
    politica = pv.PoliticaPanel(BaseFija(), inspector, NOMBRES, publicados.append, CATALOGO, "amlo", "ppo", 15.0)
    politica.reiniciar()
    politica.decidir(EntornoFalso(), np.zeros(4 * TAMANO, dtype=np.float32))  # type: ignore[arg-type]
    politica.decidir(EntornoFalso(), observacion(5) * 50)  # type: ignore[arg-type]
    configuracion = politica.mensajes_iniciales()[0]
    assert configuracion["escalaColor"] == inspector.escala_color() == 1.0
    reposo, fuerte = publicados
    # Tanh acotada: los valores publicados son los reales (no divididos por el maximo del cuadro).
    pico_reposo = max(abs(v) for v in reposo["activacionesPolitica"][0])
    assert pico_reposo < max(abs(v) for v in fuerte["activacionesPolitica"][0])
    assert max(abs(v) for capa in fuerte["activacionesPolitica"] for v in capa) <= 1.0


def test_el_panel_no_cambia_la_accion_ni_los_pesos_del_modelo(modelo: MaskablePPO) -> None:
    antes = [p.detach().clone() for p in modelo.policy.parameters()]
    base = PoliticaModelo(modelo)
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(base, InspectorRed(modelo.policy, TAMANO), NOMBRES, publicados.append, CATALOGO,
                                "amlo", "ppo", 15.0)
    env = EntornoFalso()
    politica.reiniciar()
    for i in range(30):
        obs = observacion(i)
        esperada = base.decidir(env, obs)  # type: ignore[arg-type]
        assert politica.decidir(env, obs) == esperada  # type: ignore[arg-type]
        assert publicados[-1]["accionElegida"] == esperada
        # Las probabilidades se publican con 4 decimales: una politica casi uniforme puede empatar al redondear.
        assert publicados[-1]["probabilidades"][esperada] >= max(publicados[-1]["probabilidades"]) - 1e-4
    assert all(torch.equal(a, p.detach()) for a, p in zip(antes, modelo.policy.parameters(), strict=True))
    assert all(p.grad is None for p in modelo.policy.parameters())


def test_cada_cuadro_es_autocontenido_y_se_puede_volcar_a_jsonl(modelo: MaskablePPO) -> None:
    publicados: list[dict[str, Any]] = []
    politica = pv.PoliticaPanel(BaseFija(), InspectorRed(modelo.policy, TAMANO), NOMBRES, publicados.append,
                                CATALOGO, "amlo", "ppo", 15.0)
    politica.reiniciar()
    for i in range(3):
        politica.decidir(EntornoFalso(), observacion(i))  # type: ignore[arg-type]
    lineas = [json.dumps(c, allow_nan=False) for c in publicados]
    assert all("\n" not in linea for linea in lineas)
    assert [json.loads(linea) for linea in lineas] == publicados
    claves = {"tipo", "paso", "encabezado", "activacionesPolitica", "activacionesValor", "probabilidades", "mascara",
              "accionElegida", "valorEstimado", "eventos", "atencion", "milisegundosCalculo"}
    assert all(set(c) == claves for c in publicados)


def test_el_catalogo_de_prueba_y_el_modelo_comparten_la_misma_observacion() -> None:
    assert len(campos_v2(catalogo_prueba())) == tamano_simple_v2(catalogo_prueba()) == TAMANO
