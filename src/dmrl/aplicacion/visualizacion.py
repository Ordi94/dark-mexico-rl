"""
Ver pelear a una politica en una ventana del juego a velocidad normal.

Usa una sola instancia con ventana, sin -nullrhi, y el mismo paso fijo de 1/30 s del entrenamiento
(decisiones cada 2 frames, 15 Hz), de modo que la dinamica es la que aprendio el modelo. El motor
dibuja a ~56 fps (limite de la pantalla; t.MaxFPS no lo reduce con paso fijo), es decir ~1.9x, asi
que el ritmo de 1x lo marca Python: tras cada decision espera hasta que el reloj real alcance el
tiempo de juego transcurrido.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.aplicacion.evaluacion import OPCIONES_PELEA_COMPLETA, Politica
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.infraestructura.juego import ParametrosInstancia, crear_fabrica_puerto

PARAMETROS_VENTANA = ParametrosInstancia(
    id_instancia="ver", perfil="window", sim_hz=30.0, afinidad=(), ventana_visible=True,
)


def crear_entorno_ventana(tabla: TablaGolpes, config: ConfigEntorno | None = None,
                          params: ParametrosInstancia = PARAMETROS_VENTANA) -> DarkMexicoEnv:
    """
    Crea un entorno cuya instancia del juego tiene ventana y corre a velocidad normal.

    Contexto:
        ver_pelea.ps1.

    Args:
        tabla (TablaGolpes): Golpes medidos del jefe.
        config (ConfigEntorno, opcional): Configuracion del entorno.
        params (ParametrosInstancia): Parametros de lanzamiento de la ventana.

    Returns:
        DarkMexicoEnv: Entorno con el juego ya abierto.

    Raises:
        TimeoutError: Si el juego no se conecta.

    Ejemplo:
        env = crear_entorno_ventana(tabla)
    """
    return DarkMexicoEnv(crear_fabrica_puerto(params), tabla, config)


def ver_pelea(env: DarkMexicoEnv, politica: Politica, episodios: int = 1, max_segundos: float | None = None,
              opciones: dict[str, Any] | None = None, reloj: Callable[[], float] = time.monotonic,
              dormir: Callable[[float], None] = time.sleep,
              al_terminar: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """
    Juega episodios con la politica a velocidad normal hasta agotarlos o hasta un limite de tiempo real.

    Contexto:
        Comando ver. El limite de tiempo permite comprobar que la ventana abre y cierra sin
        esperar una pelea completa.

    Args:
        env (DarkMexicoEnv): Entorno con ventana.
        politica (Politica): Politica a observar.
        episodios (int): Peleas a jugar.
        max_segundos (float, opcional): Tiempo real maximo total; None = sin limite.
        opciones (dict, opcional): Opciones de reset (por defecto, pelea completa al 100%).
        reloj (Callable): Reloj monotono en segundos (inyectable para pruebas).
        dormir (Callable): Espera en segundos (inyectable para pruebas).
        al_terminar (Callable, opcional): Se llama con el resumen de cada episodio terminado (el panel
            en vivo lo usa para marcar la victoria o la muerte).

    Returns:
        list[dict[str, Any]]: Resumenes de los episodios terminados antes del limite.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ver_pelea(env, politica, episodios=1, max_segundos=20)
    """
    opciones = OPCIONES_PELEA_COMPLETA if opciones is None else opciones
    inicio = reloj()
    resumenes: list[dict[str, Any]] = []
    for _ in range(episodios):
        obs, _info = env.reset(options=opciones)
        politica.reiniciar()
        ancla_reloj, ancla_juego = reloj(), float(env.observacion_cruda.get("t", 0.0))
        while True:
            if max_segundos is not None and reloj() - inicio >= max_segundos:
                return resumenes
            obs, _recompensa, terminado, truncado, info = env.step(politica.decidir(env, obs))
            if terminado or truncado:
                resumenes.append(info["episodio"])
                if al_terminar is not None:
                    al_terminar(info["episodio"])
                break
            transcurrido_juego = float(env.observacion_cruda.get("t", ancla_juego)) - ancla_juego
            espera = transcurrido_juego - (reloj() - ancla_reloj)
            if espera > 0:
                dormir(espera)
    return resumenes
