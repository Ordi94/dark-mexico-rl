"""
Panel en vivo: muestra a la red "pensando" mientras pelea en la ventana del juego.

PoliticaPanel envuelve a la politica que juega: la accion la sigue eligiendo la politica original,
sin cambios; despues el inspector calcula (en el mismo proceso, hilo de la pelea) activaciones,
probabilidades, valor y atribucion, y el cuadro resultante se entrega a un publicador. El publicador
es una funcion cualquiera mensaje -> None: hoy es ServidorPanel.publicar (navegador por WebSocket) y
manana puede ser tambien una escritura a archivo JSONL sin tocar el calculo.

Todo fallo del panel se contiene: la pelea nunca depende de que el panel funcione.
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmrl.aplicacion.entorno import DarkMexicoEnv
from dmrl.aplicacion.evaluacion import Politica
from dmrl.aplicacion.inspector_red import InspectorRed
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.observacion_v2 import campos_v2
from dmrl.dominio.panel import (
    EVENTO_MUERTE,
    EVENTO_VICTORIA,
    SEGUNDOS_HISTORIAL,
    DatosCuadro,
    DetectorEventos,
    construir_atencion,
    construir_configuracion,
    construir_cuadro,
    encabezado_de,
)
from dmrl.dominio.reglas_jefes import resolver_fase_de
from dmrl.infraestructura.servidor_panel import PUERTO_DEFECTO, ServidorPanel

Publicador = Callable[[dict[str, Any]], None]
MAXIMO_PUNTOS_HISTORIAL = 600


def avisar(mensaje: str) -> None:
    """
    Escribe un aviso del panel en la salida de error.

    Contexto:
        Fallos no fatales del panel (el combate sigue).

    Args:
        mensaje (str): Texto del aviso.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        avisar("panel: cuadro descartado")
    """
    print(mensaje, file=sys.stderr, flush=True)


class PoliticaPanel:
    """
    Politica que decide como la original y publica en cada decision lo que la red calcula.

    Contexto:
        Se pasa a ver_pelea en lugar de PoliticaModelo cuando se pide el panel.
    """

    def __init__(self, base: Politica, inspector: InspectorRed, nombres: Sequence[str], publicar: Publicador,
                 catalogo: CatalogoJefes, jefe: str, modelo: str, hz: float,
                 reloj: Callable[[], float] = time.perf_counter) -> None:
        """
        Crea la politica con el panel.

        Contexto:
            iniciar_sesion_panel.

        Args:
            base (Politica): Politica que elige la accion (PoliticaModelo).
            inspector (InspectorRed): Observador de la red de esa politica.
            nombres (Sequence[str]): Nombre de cada caracteristica de la observacion simple (campos_v2).
            publicar (Callable): Recibe cada mensaje JSON; no debe bloquear.
            catalogo (CatalogoJefes): Para el nombre y la fase del jefe activo.
            jefe (str): Jefe de episodio de la sesion.
            modelo (str): Nombre del modelo mostrado en el panel.
            hz (float): Frecuencia de decision esperada.
            reloj (Callable): Reloj en segundos (inyectable para pruebas).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            PoliticaPanel(base, inspector, nombres, servidor.publicar, catalogo, "amlo", "ppo", 15.0)
        """
        self._base = base
        self._inspector = inspector
        self._nombres = tuple(nombres)
        self._publicar = publicar
        self._catalogo = catalogo
        self._jefe = jefe
        self._reloj = reloj
        self._fase_de = resolver_fase_de(catalogo)
        self._detector = DetectorEventos(self._fase_de)
        estructura = inspector.estructura()
        self._configuracion = construir_configuracion(
            estructura.capas_politica, estructura.capas_valor, estructura.activacion, inspector.escala_color(),
            jefe, modelo, hz)
        self._candado = threading.Lock()
        self._valores: deque[tuple[float, float]] = deque(maxlen=MAXIMO_PUNTOS_HISTORIAL)
        self._eventos: list[dict[str, Any]] = []
        self._ultimo_cuadro: dict[str, Any] | None = None
        self._episodio_nuevo = True
        self._paso = 0
        self._t0 = 0.0
        self._segundos = 0.0
        self.errores = 0

    def reiniciar(self) -> None:
        """
        Prepara la politica y el panel para un episodio nuevo.

        Contexto:
            ver_pelea, tras el reset del entorno. La referencia de eventos se toma en la primera
            decision, cuando ya hay observacion cruda.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            politica.reiniciar()
        """
        self._base.reiniciar()
        self._episodio_nuevo = True

    def decidir(self, env: DarkMexicoEnv, obs: NDArray[np.float32]) -> int:
        """
        Accion de la politica original; despues publica el cuadro de esta decision.

        Contexto:
            Bucle de ver_pelea, una vez por decision. Un fallo del panel se cuenta y se avisa una vez,
            pero nunca altera la accion ni detiene la pelea.

        Args:
            env (DarkMexicoEnv): Entorno (observacion cruda y mascara).
            obs (NDArray[np.float32]): Observacion apilada.

        Returns:
            int: La accion que eligio la politica original.

        Raises:
            Ninguna: Las excepciones del panel se contienen; las de la politica base se propagan.

        Ejemplo:
            accion = politica.decidir(env, obs)
        """
        accion = self._base.decidir(env, obs)
        try:
            self._publicar_decision(env, obs, accion)
        except Exception as exc:
            self.errores += 1
            if self.errores == 1:
                avisar(f"panel: se omite un cuadro por un error ({exc!r}); la pelea continua")
        return accion

    def _publicar_decision(self, env: DarkMexicoEnv, obs: NDArray[np.float32], accion: int) -> None:
        """
        Calcula y publica el cuadro de una decision.

        Contexto:
            decidir.

        Args:
            env (DarkMexicoEnv): Entorno (observacion cruda y mascara).
            obs (NDArray[np.float32]): Observacion apilada.
            accion (int): Accion ya elegida.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si la observacion no corresponde a la red inspeccionada.

        Ejemplo:
            (uso interno)
        """
        inicio = self._reloj()
        cruda = env.observacion_cruda
        if self._episodio_nuevo:
            self._detector.reiniciar(cruda)
            self._t0 = float(cruda.get("t", 0.0))
            self._paso = 0
            self._episodio_nuevo = False
            eventos: list[dict[str, Any]] = []
            with self._candado:
                self._valores.clear()
                self._eventos.clear()
        else:
            eventos = self._detector.observar(cruda)
        self._paso += 1
        mascara = env.action_masks()
        inspeccion = self._inspector.inspeccionar(obs, mascara, accion)
        self._segundos = max(float(cruda.get("t", self._t0)) - self._t0, 0.0)
        fase = self._fase_de(cruda, 1)
        atencion = construir_atencion(inspeccion.atribucion, np.asarray(obs)[-len(self._nombres):], self._nombres)
        cuadro = construir_cuadro(DatosCuadro(
            paso=self._paso, encabezado=encabezado_de(cruda, self._nombre_de_clase, fase, self._segundos, self._jefe),
            activaciones_politica=inspeccion.activaciones_politica,
            activaciones_valor=inspeccion.activaciones_valor, probabilidades=inspeccion.probabilidades,
            mascara=mascara.tolist(), accion=accion, valor=inspeccion.valor, eventos=eventos, atencion=atencion,
            milisegundos=(self._reloj() - inicio) * 1000.0))
        self._registrar(cuadro["valorEstimado"], eventos)
        with self._candado:
            self._ultimo_cuadro = cuadro
        self._publicar(cuadro)

    def _nombre_de_clase(self, clase: str | None) -> str | None:
        """
        Nombre para mostrar del jefe activo a partir de su clase Blueprint.

        Contexto:
            encabezado_de. El actor activo cambia durante la pelea (AMLO pasa a Chamuco).

        Args:
            clase (str | None): Clase del actor jefe en la observacion cruda.

        Returns:
            str | None: Nombre del perfil, o None si la clase no se reconoce.

        Raises:
            Ninguna.

        Ejemplo:
            self._nombre_de_clase("BP_Chamuco_C")  # "Chamuco"
        """
        perfil = self._catalogo.por_clase(clase)
        return perfil.nombre if perfil is not None else None

    def _registrar(self, valor: float, eventos: Sequence[dict[str, Any]]) -> None:
        """
        Agrega la decision al historial que reciben los navegadores que se conectan despues.

        Contexto:
            _publicar_decision. Solo se conserva la ventana de la grafica.

        Args:
            valor (float): V(s) de esta decision.
            eventos (Sequence[dict[str, Any]]): Hitos detectados.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        with self._candado:
            self._valores.append((round(self._segundos, 2), valor))
            self._eventos.extend({"t": round(self._segundos, 2), **e} for e in eventos)
            corte = self._segundos - SEGUNDOS_HISTORIAL
            self._eventos[:] = [e for e in self._eventos if e["t"] >= corte]

    def finalizar(self, resumen: dict[str, Any]) -> None:
        """
        Publica el final de la pelea (victoria o muerte) como hito del historial.

        Contexto:
            ver_pelea, al terminar cada episodio.

        Args:
            resumen (dict[str, Any]): Resumen del episodio (info["episodio"] del entorno).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Un fallo del panel se contiene.

        Ejemplo:
            politica.finalizar({"victoria": True, "parries": 12})
        """
        try:
            tipo = EVENTO_VICTORIA if resumen.get("victoria") else EVENTO_MUERTE
            hito = {"tipo": tipo, "detalle": resumen.get("resultado", tipo)}
            self._registrar_hito(hito)
            self._publicar({
                "tipo": "fin", "resultado": resumen.get("resultado"), "hito": hito,
                "tiempoPelea": round(self._segundos, 2), "danioHecho": resumen.get("danio_hecho", 0.0),
                "danioRecibido": resumen.get("danio_recibido", 0.0), "parries": resumen.get("parries", 0),
                "caguamasUsadas": resumen.get("caguamas_usadas", 0)})
        except Exception as exc:
            self.errores += 1
            avisar(f"panel: no se pudo publicar el final de la pelea ({exc!r})")

    def _registrar_hito(self, hito: dict[str, Any]) -> None:
        """
        Agrega un hito al historial en el instante actual de la pelea.

        Contexto:
            finalizar.

        Args:
            hito (dict[str, Any]): Evento con tipo y detalle.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        with self._candado:
            self._eventos.append({"t": round(self._segundos, 2), **hito})

    def mensajes_iniciales(self) -> list[dict[str, Any]]:
        """
        Mensajes que recibe un navegador al conectarse: configuracion, historial y ultimo cuadro.

        Contexto:
            ServidorPanel, en el hilo de la conexion; por eso toma una copia bajo candado.

        Args:
            Ninguno.

        Returns:
            list[dict[str, Any]]: Configuracion, historial de V(s) y eventos y, si ya hubo, el ultimo cuadro.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.fijar_proveedor(politica.mensajes_iniciales)
        """
        with self._candado:
            historial = {"tipo": "historial", "valores": [[t, v] for t, v in self._valores],
                         "eventos": [dict(e) for e in self._eventos]}
            ultimo = self._ultimo_cuadro
        return [self._configuracion, historial, *([ultimo] if ultimo is not None else [])]


@dataclass
class SesionPanel:
    """
    Panel iniciado: servidor, politica envuelta y limpieza.

    Contexto:
        Lo devuelve iniciar_sesion_panel y lo consume comando_ver_jefes.

    Atributos:
        servidor: servidor HTTP/WebSocket en marcha.
        politica: politica que debe jugar la pelea.
        url: direccion de la pagina del panel.
    """

    servidor: ServidorPanel
    politica: PoliticaPanel
    url: str

    def al_terminar(self, resumen: dict[str, Any]) -> None:
        """
        Notifica el final de un episodio al panel.

        Contexto:
            Callback al_terminar de ver_pelea.

        Args:
            resumen (dict[str, Any]): Resumen del episodio.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            ver_pelea(env, sesion.politica, al_terminar=sesion.al_terminar)
        """
        self.politica.finalizar(resumen)

    def cerrar(self) -> None:
        """
        Detiene el servidor del panel.

        Contexto:
            Fin de ver-jefes, tambien si la pelea fallo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            sesion.cerrar()
        """
        self.servidor.detener()


def iniciar_sesion_panel(modelo: Any, base: Politica, catalogo: CatalogoJefes, jefe: str, nombre_modelo: str,
                         puerto: int = PUERTO_DEFECTO, abrir_navegador: bool = True,
                         abrir: Callable[[str], bool] = webbrowser.open,
                         config: ConfigEntorno | None = None) -> SesionPanel:
    """
    Arranca el servidor del panel y envuelve la politica con el inspector.

    Contexto:
        comando_ver_jefes con --panel, ANTES de lanzar el juego: si el puerto esta ocupado o el
        modelo no usa la observacion v2 se falla sin abrir la ventana del juego.

    Args:
        modelo (Any): MaskablePPO cargado (se lee model.policy, no se modifica).
        base (Politica): Politica que elige las acciones.
        catalogo (CatalogoJefes): Jefes y vocabulario de la observacion v2.
        jefe (str): Jefe de episodio de la sesion.
        nombre_modelo (str): Nombre mostrado en el panel.
        puerto (int): Puerto local del panel.
        abrir_navegador (bool): Si es True abre la pagina en el navegador predeterminado.
        abrir (Callable): Funcion que abre una URL (inyectable para pruebas).
        config (ConfigEntorno, opcional): Para la frecuencia de decision mostrada.

    Returns:
        SesionPanel: Servidor en marcha y politica lista.

    Raises:
        ValueError: Si la red no usa la observacion v2 del catalogo.
        OSError: Si el puerto esta ocupado.

    Ejemplo:
        sesion = iniciar_sesion_panel(modelo, PoliticaModelo(modelo), catalogo, "amlo", "multijefe_ppo")
    """
    nombres = [nombre for nombre, _ in campos_v2(catalogo)]
    inspector = InspectorRed(modelo.policy, len(nombres))
    entorno = config or ConfigEntorno()
    servidor = ServidorPanel(puerto)
    politica = PoliticaPanel(base, inspector, nombres, servidor.publicar, catalogo, jefe, nombre_modelo,
                             entorno.sim_hz / entorno.frames_por_paso)
    servidor.fijar_proveedor(politica.mensajes_iniciales)
    servidor.iniciar()
    if abrir_navegador:
        abrir(servidor.url)
    return SesionPanel(servidor, politica, servidor.url)
