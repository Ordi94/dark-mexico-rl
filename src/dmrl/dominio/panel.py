"""
Reglas puras del panel en vivo: que se muestra de cada decision de la red y como se lee.

Este modulo no conoce torch, sockets ni el juego. Recibe arreglos de numpy y observaciones crudas y
devuelve estructuras JSON (camelCase) listas para enviar:
    * probabilidades de las 22 acciones con su mascara (distribucion que realmente muestrea la politica);
    * atribucion gradiente x entrada agregada sobre las copias apiladas de la observacion;
    * valores legibles de las caracteristicas (segundos, distancias, si/no);
    * deteccion de hitos del combate (parry, golpe recibido, caguama, cambio de fase).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmrl.dominio import acciones as acc

Vector = NDArray[np.float32]

TOP_ATENCION = 10
DECIMALES_ACTIVACION = 2
DECIMALES_PROBABILIDAD = 4
SEGUNDOS_HISTORIAL = 20.0
EVENTO_PARRY = "parry"
EVENTO_GOLPE = "golpe"
EVENTO_CAGUAMA = "caguama"
EVENTO_FASE = "fase"
EVENTO_VICTORIA = "victoria"
EVENTO_MUERTE = "muerte"

ETIQUETAS_DIRECCION: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")
ETIQUETAS_ACCIONES: tuple[str, ...] = (
    ("Nada",)
    + tuple(f"Moverse {d}" for d in ETIQUETAS_DIRECCION)
    + tuple(f"Rodar {d}" for d in ETIQUETAS_DIRECCION)
    + ("Ataque", "Especial", "Parry", "Bloquear", "Caguama")
)

# Escalas de lectura de los campos normalizados de la observacion v2 (ver ESPECIFICACION_OBSERVACION_V2).
VELOCIDAD_MAX = 800.0
RESTANTE_MAX = 4.0
RATE_MAX = 2.0
GOLPE_HORIZONTE = 1.5
REMOTA_HORIZONTE = 2.5
DANIO_MAX = 50.0
GOLPES_MAX = 5.0
ORIGEN_MAX = 1000.0
DISTANCIA_MAX = 4000.0
CAGUAMAS_MAX = 3.0
UU_POR_METRO = 100.0

# nombre de caracteristica -> (etiqueta, tipo de lectura, escala). Tipos: pct, bool, seg, dist, num, uu,
# vel, rate, signo, cuenta.
_FORMATOS: dict[str, tuple[str, str, float]] = {
    "jugador_vida": ("Vida del jugador", "pct", 1.0),
    "jugador_resistencia": ("Resistencia del jugador", "pct", 1.0),
    "jugador_caguamas": ("Caguamas restantes", "cuenta", CAGUAMAS_MAX),
    "jugador_poder": ("Barra de poder", "pct", 1.0),
    "jugador_bloquea": ("Jugador bloqueando", "bool", 1.0),
    "jugador_bebiendo": ("Jugador bebiendo", "bool", 1.0),
    "jugador_progreso_bebida": ("Progreso de la bebida", "pct", 1.0),
    "jugador_reaccion_golpe": ("Jugador en reaccion a golpe", "bool", 1.0),
    "jugador_rueda": ("Jugador rodando", "bool", 1.0),
    "jugador_ventana_parry": ("Ventana de parry abierta", "bool", 1.0),
    "jugador_ataca": ("Jugador atacando", "bool", 1.0),
    "jugador_puede_atacar": ("Puede atacar", "bool", 1.0),
    "jugador_puede_rodar": ("Puede rodar", "bool", 1.0),
    "jugador_puede_curar": ("Puede curarse", "bool", 1.0),
    "jugador_enfocado": ("Enfocado al jefe", "bool", 1.0),
    "jugador_velocidad": ("Velocidad del jugador", "vel", VELOCIDAD_MAX),
    "jugador_montage_tiempo": ("Avance de la animacion del jugador", "pct", 1.0),
    "jefe_vida": ("Vida del jefe", "pct", 1.0),
    "jefe_fase_avanzada": ("Jefe en segunda fase", "bool", 1.0),
    "jefe_aturdido": ("Jefe aturdido", "bool", 1.0),
    "jefe_ataca": ("Jefe atacando", "bool", 1.0),
    "jefe_teletransportando": ("Jefe teletransportandose", "bool", 1.0),
    "jefe_velocidad": ("Velocidad del jefe", "vel", VELOCIDAD_MAX),
    "jefe_montage_tiempo": ("Avance de la animacion del jefe", "pct", 1.0),
    "jefe_montage_restante": ("Tiempo restante del ataque del jefe", "seg", RESTANTE_MAX),
    "jefe_montage_rate": ("Velocidad de la animacion del jefe", "rate", RATE_MAX),
    "amenaza_hay": ("Hay un golpe pendiente", "bool", 1.0),
    "amenaza_golpe_tiempo": ("Tiempo al proximo golpe", "seg", GOLPE_HORIZONTE),
    "amenaza_golpe_parable": ("El proximo golpe se puede parar", "bool", 1.0),
    "amenaza_golpe_danio": ("Dano del proximo golpe", "num", DANIO_MAX),
    "amenaza_golpes_pendientes": ("Golpes que quedan en el ataque", "cuenta", GOLPES_MAX),
    "amenaza_en_alcance": ("Jugador al alcance del golpe", "bool", 1.0),
    "amenaza_parry_tiempo": ("Tiempo al proximo golpe parable", "seg", GOLPE_HORIZONTE),
    "amenaza_esquivar_tiempo": ("Tiempo al proximo golpe esquivable", "seg", GOLPE_HORIZONTE),
    "remota_activa": ("Ataque a distancia pendiente", "bool", 1.0),
    "remota_tiempo": ("Tiempo al ataque a distancia", "seg", REMOTA_HORIZONTE),
    "remota_esquivable": ("Ataque a distancia esquivable", "bool", 1.0),
    "remota_parable": ("Ataque a distancia parable", "bool", 1.0),
    "remota_danio": ("Dano del ataque a distancia", "num", DANIO_MAX),
    "remota_origen_distancia": ("Distancia al origen del ataque", "uu", ORIGEN_MAX),
    "remota_origen_cos": ("Origen del ataque a distancia (frente/atras)", "signo", 1.0),
    "remota_origen_sen": ("Origen del ataque a distancia (lado)", "signo", 1.0),
    "remota_sobre_jugador": ("Ataque cae sobre el jugador", "bool", 1.0),
    "rel_distancia": ("Distancia al jefe", "uu", DISTANCIA_MAX),
    "rel_jefe_cos": ("Jefe de frente/atras", "signo", 1.0),
    "rel_jefe_sen": ("Jefe a un lado", "signo", 1.0),
    "rel_mirada_cos": ("El jefe lo mira (frente)", "signo", 1.0),
    "rel_mirada_sen": ("El jefe lo mira (lado)", "signo", 1.0),
    "tiempo_episodio": ("Tiempo de pelea (fraccion del limite)", "pct", 1.0),
}
_PREFIJOS_UNO_HOT: tuple[tuple[str, str], ...] = (
    ("jugador_montage_", "Animacion del jugador"),
    ("jefe_montage_", "Animacion del jefe"),
    ("jefe_id_", "Jefe activo"),
    ("accion_previa_", "Accion previa"),
)
_ACCION_PREVIA = {f"accion_previa_{n}": e for n, e in zip(acc.NOMBRES_ACCIONES, ETIQUETAS_ACCIONES, strict=True)}


def distribucion_enmascarada(logits: Sequence[float] | NDArray[Any], mascara: Sequence[bool] | NDArray[Any]
                             ) -> NDArray[np.float64]:
    """
    Probabilidades de la politica con las acciones no validas en cero.

    Contexto:
        Panel en vivo, barra de intencion: es la misma distribucion que muestrea MaskablePPO (softmax
        sobre los logits con la mascara aplicada).

    Args:
        logits (Sequence[float] | NDArray): Logits sin enmascarar, uno por accion.
        mascara (Sequence[bool] | NDArray): True para las acciones que el juego acepta.

    Returns:
        NDArray[np.float64]: Probabilidades que suman 1; las acciones enmascaradas valen 0.

    Raises:
        ValueError: Si las longitudes difieren o no hay ninguna accion valida.

    Ejemplo:
        distribucion_enmascarada([1.0, 2.0, 3.0], [True, True, False])  # [0.269, 0.731, 0.0]
    """
    valores = np.asarray(logits, dtype=np.float64)
    validas = np.asarray(mascara, dtype=bool)
    if valores.shape != validas.shape or valores.ndim != 1:
        raise ValueError(f"logits {valores.shape} y mascara {validas.shape} deben ser vectores del mismo largo")
    if not validas.any():
        raise ValueError("la mascara no deja ninguna accion valida")
    visibles = np.where(validas, valores, -np.inf)
    exponentes = np.exp(visibles - visibles.max())
    return np.asarray(exponentes / exponentes.sum(), dtype=np.float64)


def agregar_copias(valores: Sequence[float] | NDArray[Any], tamano_simple: int) -> NDArray[np.float64]:
    """
    Suma por caracteristica las copias apiladas de un vector de la observacion.

    Contexto:
        Atribucion: la politica ve las ultimas N observaciones concatenadas y la importancia de una
        caracteristica es la suma de lo que aporta en cada una de sus copias.

    Args:
        valores (Sequence[float] | NDArray): Vector de N * tamano_simple valores (la mas vieja primero).
        tamano_simple (int): Longitud de una observacion simple.

    Returns:
        NDArray[np.float64]: Vector de tamano_simple valores.

    Raises:
        ValueError: Si el largo no es un multiplo exacto de tamano_simple.

    Ejemplo:
        agregar_copias([1, 2, 3, 4], 2)  # [4.0, 6.0]
    """
    vector = np.asarray(valores, dtype=np.float64).reshape(-1)
    if tamano_simple <= 0 or vector.size == 0 or vector.size % tamano_simple != 0:
        raise ValueError(f"{vector.size} valores no se reparten en copias de {tamano_simple}")
    return np.asarray(vector.reshape(-1, tamano_simple).sum(axis=0), dtype=np.float64)


def atribucion_por_caracteristica(gradiente: Sequence[float] | NDArray[Any],
                                  entrada: Sequence[float] | NDArray[Any],
                                  tamano_simple: int) -> NDArray[np.float64]:
    """
    Gradiente x entrada del logit elegido, agregado sobre las copias apiladas.

    Contexto:
        Panel en vivo, seccion Atencion. Un valor positivo empuja hacia la accion elegida y uno
        negativo en contra; los positivos y negativos de copias distintas de la misma caracteristica
        se compensan, que es lo que cuenta para el logit.

    Args:
        gradiente (Sequence[float] | NDArray): d logit / d observacion, de N * tamano_simple valores.
        entrada (Sequence[float] | NDArray): Observacion apilada que produjo el logit.
        tamano_simple (int): Longitud de una observacion simple.

    Returns:
        NDArray[np.float64]: Atribucion con signo por caracteristica (tamano_simple valores).

    Raises:
        ValueError: Si los vectores difieren de largo o no son multiplos de tamano_simple.

    Ejemplo:
        atribucion_por_caracteristica([1, 1, 2, 2], [1, 0, 1, 1], 2)  # [3.0, 2.0]
    """
    g = np.asarray(gradiente, dtype=np.float64).reshape(-1)
    x = np.asarray(entrada, dtype=np.float64).reshape(-1)
    if g.size != x.size:
        raise ValueError(f"gradiente ({g.size}) y entrada ({x.size}) deben tener el mismo largo")
    return agregar_copias(g * x, tamano_simple)


def indices_mas_influyentes(atribucion: Sequence[float] | NDArray[Any], cantidad: int = TOP_ATENCION) -> list[int]:
    """
    Indices de las caracteristicas con mayor atribucion absoluta.

    Contexto:
        Panel en vivo, seccion Atencion. Las caracteristicas con atribucion exactamente cero (por
        ejemplo los one-hot apagados) no se listan: no pesaron en la decision.

    Args:
        atribucion (Sequence[float] | NDArray): Atribucion con signo por caracteristica.
        cantidad (int): Maximo de indices a devolver.

    Returns:
        list[int]: Indices ordenados de mayor a menor magnitud (los empates, por indice).

    Raises:
        ValueError: Si cantidad no es positiva.

    Ejemplo:
        indices_mas_influyentes([0.1, -0.5, 0.0, 0.3], 2)  # [1, 3]
    """
    if cantidad <= 0:
        raise ValueError("cantidad debe ser positiva")
    magnitud = np.abs(np.asarray(atribucion, dtype=np.float64))
    orden = np.argsort(-magnitud, kind="stable")
    return [int(i) for i in orden[:cantidad] if magnitud[i] > 0.0]


def _decimales(valor: float, cifras: int = 1) -> str:
    """
    Formatea un numero con pocas cifras sin ceros inutiles al final.

    Contexto:
        formatear_caracteristica.

    Args:
        valor (float): Numero a escribir.
        cifras (int): Cifras decimales maximas.

    Returns:
        str: Texto como "2.5" o "3".

    Raises:
        Ninguna.

    Ejemplo:
        _decimales(3.0)  # "3"
    """
    texto = f"{valor:.{cifras}f}"
    return texto.rstrip("0").rstrip(".") if "." in texto else texto


def formatear_caracteristica(nombre: str, valor: float) -> tuple[str, str]:
    """
    Etiqueta en espanol y valor legible de una caracteristica de la observacion v2.

    Contexto:
        Panel en vivo, seccion Atencion: el valor normalizado de la red se convierte a las unidades
        del juego (segundos, distancia, si/no) deshaciendo la escala de la especificacion v2.

    Args:
        nombre (str): Nombre de la caracteristica (campos_v2).
        valor (float): Valor normalizado de la copia mas reciente.

    Returns:
        tuple[str, str]: (etiqueta, valor legible). Un nombre sin regla conserva su nombre y se
            muestra con dos decimales.

    Raises:
        Ninguna.

    Ejemplo:
        formatear_caracteristica("amenaza_golpe_tiempo", 0.4)  # ("Tiempo al proximo golpe", "0.6 s")
    """
    if nombre in _ACCION_PREVIA:
        return f"Accion previa: {_ACCION_PREVIA[nombre]}", "si" if valor >= 0.5 else "no"
    for prefijo, titulo in _PREFIJOS_UNO_HOT:
        if nombre.startswith(prefijo) and nombre not in _FORMATOS:
            return f"{titulo}: {nombre[len(prefijo):]}", "si" if valor >= 0.5 else "no"
    if nombre not in _FORMATOS:
        return nombre, f"{valor:.2f}"
    etiqueta, tipo, escala = _FORMATOS[nombre]
    real = valor * escala
    if tipo == "bool":
        texto = "si" if valor >= 0.5 else "no"
    elif tipo == "pct":
        texto = f"{valor * 100:.0f}%"
    elif tipo == "seg":
        texto = "sin golpe" if valor >= 1.0 and nombre.startswith("amenaza_") else f"{_decimales(real)} s"
    elif tipo == "uu":
        texto = f"{real:.0f} uu ({_decimales(real / UU_POR_METRO)} m)"
    elif tipo == "vel":
        texto = f"{real:.0f} uu/s"
    elif tipo == "rate":
        texto = f"x{_decimales(real, 2)}"
    elif tipo == "signo":
        texto = f"{valor:+.2f}"
    else:
        # num y cuenta: magnitud en las unidades del juego (dano, caguamas, golpes).
        texto = _decimales(real)
    return etiqueta, texto


def construir_atencion(atribucion: NDArray[np.float64], ultima_copia: Sequence[float] | NDArray[Any],
                       nombres: Sequence[str], cantidad: int = TOP_ATENCION) -> list[dict[str, Any]]:
    """
    Las entradas que mas pesaron en la decision, con nombre y valor legibles.

    Contexto:
        Panel en vivo, seccion Atencion.

    Args:
        atribucion (NDArray[np.float64]): Atribucion con signo por caracteristica.
        ultima_copia (Sequence[float] | NDArray): Valores normalizados de la observacion mas reciente.
        nombres (Sequence[str]): Nombre de cada caracteristica (campos_v2).
        cantidad (int): Cuantas entradas listar.

    Returns:
        list[dict[str, Any]]: Una entrada por caracteristica con indice, nombre, etiqueta, valor,
            valorTexto y atribucion, de mayor a menor magnitud.

    Raises:
        ValueError: Si los largos no coinciden con la cantidad de nombres.

    Ejemplo:
        construir_atencion(atribucion, ultima, nombres)[0]["etiqueta"]
    """
    ultima = np.asarray(ultima_copia, dtype=np.float64).reshape(-1)
    if not len(nombres) == atribucion.size == ultima.size:
        raise ValueError(f"{len(nombres)} nombres, {atribucion.size} atribuciones y {ultima.size} valores no coinciden")
    filas = []
    for indice in indices_mas_influyentes(atribucion, cantidad):
        etiqueta, texto = formatear_caracteristica(nombres[indice], float(ultima[indice]))
        filas.append({"indice": indice, "nombre": nombres[indice], "etiqueta": etiqueta,
                      "valor": round(float(ultima[indice]), 4), "valorTexto": texto,
                      "atribucion": round(float(atribucion[indice]), 5)})
    return filas


def encabezado_de(cruda: dict[str, Any], nombre_de_clase: Callable[[str | None], str | None],
                  fase: int, segundos: float, jefe: str) -> dict[str, Any]:
    """
    Datos del encabezado del panel a partir de la observacion cruda.

    Contexto:
        Panel en vivo, barra superior. Cada campo ausente en la observacion (una carga, el jefe
        aun no presente) se publica como None o cero en lugar de romper el cuadro.

    Args:
        cruda (dict[str, Any]): Observacion cruda del puente.
        nombre_de_clase (Callable): Clase Blueprint del jefe -> nombre para mostrar (o None).
        fase (int): Fase de la pelea (1 o 2).
        segundos (float): Segundos de pelea transcurridos.
        jefe (str): Jefe de episodio elegido, usado si el actor activo no se reconoce.

    Returns:
        dict[str, Any]: jefe, fase, vidaJugador, vidaJugadorMax, vidaJefe, vidaJefeMax, caguamas y
            tiempoPelea.

    Raises:
        Ninguna.

    Ejemplo:
        encabezado_de(cruda, lambda c: "AMLO", 1, 12.5, "amlo")["jefe"]  # "AMLO"
    """
    jugador = cruda.get("player") or {}
    actor = cruda.get("boss") or {}
    nombre = nombre_de_clase(actor.get("cls")) or jefe
    return {
        "jefe": nombre,
        "fase": int(fase),
        "vidaJugador": round(max(float(jugador.get("hp", 0.0)), 0.0), 1),
        "vidaJugadorMax": round(float(jugador.get("hp_max", 0.0)), 1),
        "vidaJefe": round(max(float(actor.get("hp", 0.0)), 0.0), 1),
        "vidaJefeMax": round(float(actor.get("hp_max", 0.0)), 1),
        "caguamas": int(jugador.get("caguamas", 0)),
        "tiempoPelea": round(float(segundos), 2),
    }


class DetectorEventos:
    """
    Detecta hitos del combate comparando cada observacion cruda con la anterior.

    Contexto:
        PoliticaPanel lo consulta en cada decision. Usa los mismos hechos medidos que la recompensa:
        el contador de parries del paso, la caida real de la vida, el gasto de una caguama y el
        cambio de fase.
    """

    def __init__(self, fase_de: Callable[[dict[str, Any], int], int]) -> None:
        """
        Crea el detector sin referencia previa.

        Contexto:
            Arranque de la sesion del panel.

        Args:
            fase_de (Callable): Regla (observacion, fase por defecto) -> fase (1 o 2).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            detector = DetectorEventos(resolver_fase_de(catalogo))
        """
        self._fase_de = fase_de
        self._vida: float | None = None
        self._caguamas: int | None = None
        self._fase: int | None = None

    def reiniciar(self, cruda: dict[str, Any]) -> None:
        """
        Toma la observacion inicial del episodio como referencia, sin emitir eventos.

        Contexto:
            PoliticaPanel.reiniciar, justo despues del reset del entorno.

        Args:
            cruda (dict[str, Any]): Observacion cruda inicial.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            detector.reiniciar(env.observacion_cruda)
        """
        jugador = cruda.get("player") or {}
        self._vida = float(jugador.get("hp", 0.0))
        self._caguamas = int(jugador.get("caguamas", 0))
        self._fase = self._fase_de(cruda, 1)

    def observar(self, cruda: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Eventos ocurridos desde la observacion anterior.

        Contexto:
            PoliticaPanel.decidir, una vez por decision.

        Args:
            cruda (dict[str, Any]): Observacion cruda actual.

        Returns:
            list[dict[str, Any]]: Eventos con "tipo" y "detalle" (parries del paso, vida perdida,
                caguamas gastadas o nueva fase). Vacia en la primera observacion si no hubo reinicio.

        Raises:
            Ninguna.

        Ejemplo:
            detector.observar(cruda)  # [{"tipo": "golpe", "detalle": 12.0}]
        """
        jugador = cruda.get("player") or {}
        vida = float(jugador.get("hp", 0.0))
        caguamas = int(jugador.get("caguamas", 0))
        fase = self._fase_de(cruda, self._fase or 1)
        eventos: list[dict[str, Any]] = []
        parries = int((cruda.get("ev") or {}).get("parries", 0))
        if parries > 0:
            eventos.append({"tipo": EVENTO_PARRY, "detalle": parries})
        if self._vida is not None and vida < self._vida:
            eventos.append({"tipo": EVENTO_GOLPE, "detalle": round(self._vida - max(vida, 0.0), 1)})
        if self._caguamas is not None and caguamas < self._caguamas:
            eventos.append({"tipo": EVENTO_CAGUAMA, "detalle": self._caguamas - caguamas})
        if self._fase is not None and fase > self._fase:
            eventos.append({"tipo": EVENTO_FASE, "detalle": fase})
        self._vida, self._caguamas, self._fase = vida, caguamas, fase
        return eventos


@dataclass(frozen=True)
class DatosCuadro:
    """
    Todo lo que describe una decision de la red, antes de convertirlo a JSON.

    Contexto:
        PoliticaPanel lo arma con el resultado del inspector y la observacion cruda.

    Atributos:
        paso: numero de decision dentro de la pelea.
        encabezado: salida de encabezado_de.
        activaciones_politica: salida de cada capa oculta de la red de politica.
        activaciones_valor: salida de cada capa oculta de la red de valor.
        probabilidades: probabilidad de cada accion.
        mascara: acciones validas.
        accion: accion elegida.
        valor: estimacion V(s).
        eventos: hitos detectados desde la decision anterior.
        atencion: salida de construir_atencion.
        milisegundos: tiempo de calculo del panel en esta decision.
    """

    paso: int
    encabezado: dict[str, Any]
    activaciones_politica: Sequence[NDArray[Any]]
    activaciones_valor: Sequence[NDArray[Any]]
    probabilidades: NDArray[Any]
    mascara: Sequence[bool]
    accion: int
    valor: float
    eventos: Sequence[dict[str, Any]]
    atencion: Sequence[dict[str, Any]]
    milisegundos: float


def _redondeada(capa: NDArray[Any]) -> list[float]:
    """
    Activaciones de una capa como lista JSON con pocos decimales.

    Contexto:
        construir_cuadro. Dos decimales bastan para el color y reducen el trafico.

    Args:
        capa (NDArray): Activaciones de la capa.

    Returns:
        list[float]: Valores finitos redondeados.

    Raises:
        Ninguna.

    Ejemplo:
        _redondeada(np.array([0.123456]))  # [0.12]
    """
    return list(np.round(np.nan_to_num(np.asarray(capa, dtype=np.float64)), DECIMALES_ACTIVACION).tolist())


def construir_cuadro(datos: DatosCuadro) -> dict[str, Any]:
    """
    Cuadro JSON (camelCase) de una decision.

    Contexto:
        PoliticaPanel lo publica en cada decision; ServidorPanel lo envia a los navegadores.
        Cada cuadro es autocontenido, de modo que un archivo con un cuadro por linea (JSONL) basta
        para reproducirlo despues.

    Args:
        datos (DatosCuadro): Resultado de la decision.

    Returns:
        dict[str, Any]: Mensaje con tipo "cuadro".

    Raises:
        Ninguna.

    Ejemplo:
        construir_cuadro(datos)["accionElegida"]
    """
    return {
        "tipo": "cuadro",
        "paso": datos.paso,
        "encabezado": datos.encabezado,
        "activacionesPolitica": [_redondeada(c) for c in datos.activaciones_politica],
        "activacionesValor": [_redondeada(c) for c in datos.activaciones_valor],
        "probabilidades": [round(float(p), DECIMALES_PROBABILIDAD) for p in datos.probabilidades],
        "mascara": [bool(m) for m in datos.mascara],
        "accionElegida": int(datos.accion),
        "valorEstimado": round(float(np.nan_to_num(datos.valor)), 4),
        "eventos": [dict(e) for e in datos.eventos],
        "atencion": [dict(a) for a in datos.atencion],
        "milisegundosCalculo": round(float(datos.milisegundos), 2),
    }


def construir_configuracion(capas_politica: Sequence[int], capas_valor: Sequence[int], activacion: str,
                            escala: float, jefe: str, modelo: str, hz: float) -> dict[str, Any]:
    """
    Mensaje inicial con lo que no cambia entre cuadros.

    Contexto:
        ServidorPanel lo envia al conectarse cada navegador, antes del historial y los cuadros.

    Args:
        capas_politica (Sequence[int]): Neuronas de cada capa oculta de la politica.
        capas_valor (Sequence[int]): Neuronas de cada capa oculta del valor.
        activacion (str): Funcion de activacion de las capas ocultas (por ejemplo "tanh").
        escala (float): Valor absoluto que corresponde al color extremo, fijo en el tiempo.
        jefe (str): Jefe de episodio de la sesion.
        modelo (str): Nombre del modelo mostrado.
        hz (float): Frecuencia de decision esperada.

    Returns:
        dict[str, Any]: Mensaje con tipo "configuracion".

    Raises:
        Ninguna.

    Ejemplo:
        construir_configuracion([256, 256], [256, 256], "tanh", 1.0, "amlo", "multijefe_ppo", 15.0)
    """
    return {
        "tipo": "configuracion",
        "capasPolitica": [int(n) for n in capas_politica],
        "capasValor": [int(n) for n in capas_valor],
        "activacion": activacion,
        "escalaColor": float(escala),
        "acciones": list(ETIQUETAS_ACCIONES),
        "jefe": jefe,
        "modelo": modelo,
        "frecuenciaHz": float(hz),
        "segundosHistorial": SEGUNDOS_HISTORIAL,
    }
