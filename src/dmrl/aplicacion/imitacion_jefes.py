"""
Imitacion multi-jefe: una sola politica MaskablePPO (observacion v2) ajustada a las demostraciones del
bot contra todos los jefes.

Reutiliza el ajuste supervisado de la imitacion de un jefe (imitacion_red): perdida enmascarada, balance
moderado de clases, weight decay, parada temprana por perdida de validacion y ajuste aparte de la
cabeza de valor. Lo propio del caso multi-jefe es la particion estratificada por jefe, las metricas por
jefe, la comparacion de arquitecturas y la medicion del dispositivo (CPU contra GPU) para elegir el mas
rapido.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dmrl.aplicacion.entrenamiento import CARPETA_RUNS, ConfigEntrenamiento
from dmrl.aplicacion.imitacion_red import (
    ResultadoAjuste,
    coincidencia_tras_cargar,
    crear_modelo_base,
    entrenar_politica,
    entrenar_valor,
    guardar_modelo,
)
from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.imitacion import ConfigAprendizaje, Demostraciones, concatenar, retornos_descontados
from dmrl.dominio.imitacion_jefes import (
    arquitectura_por_nombre,
    informe_por_jefe,
    jefe_por_paso,
    metricas_por_jefe,
    particionar_estratificado,
)
from dmrl.dominio.jefes import CatalogoJefes
from dmrl.dominio.observacion_v2 import tamano_simple_v2
from dmrl.infraestructura.demostraciones import cargar_lote, escribir_json, lotes_de_ronda, metadatos_de_ronda

CARPETA_IMITACION_JEFES = CARPETA_RUNS / "multijefe_imitacion"
FRACCION_VALIDACION = 0.15
MUESTRA_RECARGA = 300
RONDAS_MAXIMAS = 50

Registrar = Callable[[str], None]


@dataclass(frozen=True)
class RutasImitacionJefes:
    """
    Ubicacion de los archivos de la imitacion multi-jefe.

    Contexto:
        Recoleccion, ajuste, evaluacion y publicacion del modelo; centralizar las rutas evita que
        cada etapa discrepe de las demas.

    Atributos:
        base: carpeta runs/<nombre>.
    """

    base: Path = CARPETA_IMITACION_JEFES

    @property
    def datos(self) -> Path:
        """
        Carpeta de las demostraciones.

        Contexto:
            Recoleccion y carga.

        Args:
            Ninguno.

        Returns:
            Path: base/datos.

        Raises:
            Ninguna.

        Ejemplo:
            RutasImitacionJefes().datos
        """
        return self.base / "datos"

    @property
    def mejor(self) -> Path:
        """
        Modelo elegido, cargable por MaskablePPO.

        Contexto:
            Evaluacion, ver_pelea y arranque del PPO.

        Args:
            Ninguno.

        Returns:
            Path: base/mejor.zip.

        Raises:
            Ninguna.

        Ejemplo:
            RutasImitacionJefes().mejor
        """
        return self.base / "mejor.zip"

    def modelo(self, arquitectura: str) -> Path:
        """
        Modelo ajustado de una arquitectura.

        Contexto:
            Ajuste y comparacion en juego.

        Args:
            arquitectura (str): Nombre de la arquitectura candidata.

        Returns:
            Path: base/modelos/<arquitectura>.zip.

        Raises:
            Ninguna.

        Ejemplo:
            RutasImitacionJefes().modelo("2x256")
        """
        return self.base / "modelos" / f"{arquitectura}.zip"

    def informe(self, arquitectura: str) -> Path:
        """
        Informe del ajuste de una arquitectura (curvas, precision por accion y por jefe).

        Contexto:
            Reporte final.

        Args:
            arquitectura (str): Nombre de la arquitectura candidata.

        Returns:
            Path: base/modelos/<arquitectura>.json.

        Raises:
            Ninguna.

        Ejemplo:
            RutasImitacionJefes().informe("2x256")
        """
        return self.base / "modelos" / f"{arquitectura}.json"

    def evaluacion(self, nombre: str) -> Path:
        """
        Resultado de una evaluacion en juego.

        Contexto:
            Seleccion de la arquitectura y reporte.

        Args:
            nombre (str): Nombre de la evaluacion (arquitectura, "bot", "final", ...).

        Returns:
            Path: base/evaluaciones/<nombre>.json.

        Raises:
            Ninguna.

        Ejemplo:
            RutasImitacionJefes().evaluacion("bot")
        """
        return self.base / "evaluaciones" / f"{nombre}.json"


@dataclass
class ConjuntoJefes:
    """
    Todas las demostraciones recolectadas con el jefe de cada episodio.

    Contexto:
        Entrada del ajuste multi-jefe y de los informes de datos por jefe.

    Atributos:
        demostraciones: pasos de todas las rondas, en el orden de los lotes.
        jefe_de_episodio: jefe de cada identificador de episodio.
        metadatos: metadatos de todos los episodios (fase de inicio, ruido, resultado, pasos).
    """

    demostraciones: Demostraciones
    jefe_de_episodio: dict[int, str]
    metadatos: list[dict[str, Any]]


def cargar_conjunto_jefes(carpeta_datos: Path) -> ConjuntoJefes:
    """
    Carga las demostraciones de todas las rondas existentes y asigna un jefe a cada episodio.

    Contexto:
        Inicio del ajuste. La recoleccion deja una ronda por tanda (el mapa de Juice y el de AMLO se
        recolectan por separado); cualquier ronda presente cuenta.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.

    Returns:
        ConjuntoJefes: Demostraciones unidas, jefe por episodio y metadatos.

    Raises:
        ValueError: Si no hay ningun lote guardado.

    Ejemplo:
        conjunto = cargar_conjunto_jefes(Path("runs/multijefe_imitacion/datos"))
    """
    partes: list[Demostraciones] = []
    metadatos: list[dict[str, Any]] = []
    for ronda in range(RONDAS_MAXIMAS):
        for lote in lotes_de_ronda(carpeta_datos, ronda):
            datos, meta = cargar_lote(lote)
            partes.append(datos)
            metadatos += meta
    if not partes:
        raise ValueError(f"no hay demostraciones guardadas en {carpeta_datos}")
    jefe_de_episodio = {int(m["id"]): str(m["jefe"]) for m in metadatos}
    return ConjuntoJefes(concatenar(partes), jefe_de_episodio, metadatos)


def resumen_de_datos(metadatos: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Cantidad de datos recolectados por jefe.

    Contexto:
        Reporte de la recoleccion y del ajuste.

    Args:
        metadatos (list[dict[str, Any]]): Metadatos de los episodios.

    Returns:
        dict[str, dict[str, float]]: jefe -> peleas, pasos, victorias del bot, peleas con ruido,
            peleas que empiezan en la segunda fase y peleas que pasaron por la segunda fase.

    Raises:
        Ninguna.

    Ejemplo:
        resumen_de_datos(metadatos)["amlo"]["peleas"]
    """
    por_jefe: dict[str, list[dict[str, Any]]] = {}
    for m in metadatos:
        por_jefe.setdefault(str(m["jefe"]), []).append(m)
    return {
        jefe: {
            "peleas": float(len(lista)), "pasos": float(sum(int(e.get("pasos", 0)) for e in lista)),
            "victorias_bot": float(sum(1 for e in lista if e.get("victoria"))),
            "con_ruido": float(sum(1 for e in lista if float(e.get("epsilon", 0.0)) > 0.0)),
            "inicio_fase2": float(sum(1 for e in lista if int(e.get("fase_inicio", 1)) == 2)),
            "pasaron_fase2": float(sum(1 for e in lista if int(e.get("fase_max", 1)) >= 2)),
        }
        for jefe, lista in sorted(por_jefe.items())
    }


def _informe_ajuste(resultado: ResultadoAjuste) -> dict[str, Any]:
    """
    Serializa un ResultadoAjuste con nombres de accion legibles.

    Contexto:
        Informe JSON de cada arquitectura.

    Args:
        resultado (ResultadoAjuste): Resultado del ajuste.

    Returns:
        dict[str, Any]: Estructura de tipos JSON.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    datos = dataclasses.asdict(resultado)
    datos["por_accion"] = [
        {"accion": m.accion, "nombre": acc.NOMBRES_ACCIONES[m.accion], "soporte": m.soporte,
         "recall": m.recall, "precision": m.precision}
        for m in resultado.por_accion
    ]
    return datos


def config_modelo(catalogo: CatalogoJefes, arquitectura: str, semilla: int) -> tuple[ConfigEntrenamiento, int]:
    """
    Configuracion de entrenamiento y tamano de observacion de una arquitectura.

    Contexto:
        Ajuste y PPO: ambos deben crear el modelo con las mismas capas y la misma observacion.

    Args:
        catalogo (CatalogoJefes): Catalogo de jefes (da el tamano de la observacion v2).
        arquitectura (str): Nombre de la arquitectura candidata.
        semilla (int): Semilla de la inicializacion.

    Returns:
        tuple[ConfigEntrenamiento, int]: Configuracion con las capas elegidas y valores de una
            observacion sin apilar.

    Raises:
        ValueError: Si la arquitectura no es candidata.

    Ejemplo:
        config, tamano = config_modelo(catalogo, "2x256", 0)
    """
    config = ConfigEntrenamiento(nombre="multijefe_imitacion", capas=arquitectura_por_nombre(arquitectura),
                                 semilla=semilla, entorno=ConfigEntorno())
    return config, tamano_simple_v2(catalogo)


def medir_dispositivos(catalogo: CatalogoJefes, arquitectura: str, datos: Demostraciones,
                       config: ConfigAprendizaje, pasos_muestra: int = 200_000,
                       dispositivos: tuple[str, ...] = ("cpu", "cuda")) -> dict[str, float]:
    """
    Mide cuanto tarda una epoca de ajuste (extrapolada al conjunto completo) en cada dispositivo.

    Contexto:
        Antes del ajuste: la CPU y la GPU compiten con el juego por recursos distintos, asi que se
        elige el mas rapido con una medicion en vez de suponerlo. Se ajusta una epoca sobre una
        muestra con un modelo desechable; la parada temprana y la validacion no intervienen.

    Args:
        catalogo (CatalogoJefes): Catalogo de jefes.
        arquitectura (str): Arquitectura con la que medir.
        datos (Demostraciones): Conjunto del que se toma la muestra.
        config (ConfigAprendizaje): Hiperparametros (tamano de lote, tasa).
        pasos_muestra (int): Pasos de la muestra.
        dispositivos (tuple[str, ...]): Dispositivos a medir; los no disponibles se omiten.

    Returns:
        dict[str, float]: dispositivo -> segundos estimados por epoca sobre todo el conjunto.

    Raises:
        ValueError: Si el conjunto esta vacio.

    Ejemplo:
        medir_dispositivos(catalogo, "2x256", demostraciones, ConfigAprendizaje())
    """
    from dmrl.aplicacion.imitacion_red import logits_actor, perdida_enmascarada

    n = len(datos)
    if n == 0:
        raise ValueError("no hay datos con que medir")
    muestra = min(n, pasos_muestra)
    filas = np.random.default_rng(0).choice(n, size=muestra, replace=False)
    obs = torch.from_numpy(datos.obs[filas])
    mascaras = torch.from_numpy(datos.mascaras[filas])
    etiquetas = torch.from_numpy(datos.etiquetas[filas].astype(np.int64))
    tiempos: dict[str, float] = {}
    for nombre in dispositivos:
        if nombre == "cuda" and not torch.cuda.is_available():
            continue
        dispositivo = torch.device(nombre)
        configuracion, tamano = config_modelo(catalogo, arquitectura, 0)
        modelo = crear_modelo_base(configuracion, tamano)
        politica = modelo.policy.to(dispositivo)
        parametros = [*politica.mlp_extractor.policy_net.parameters(), *politica.action_net.parameters()]
        optimizador = torch.optim.AdamW(parametros, lr=config.tasa_aprendizaje)
        obs_d, mascaras_d, etiquetas_d = obs.to(dispositivo), mascaras.to(dispositivo), etiquetas.to(dispositivo)
        if dispositivo.type == "cuda":  # pragma: no cover - solo con GPU CUDA
            torch.cuda.synchronize()
        inicio = time.perf_counter()
        for desde in range(0, muestra, config.tamano_lote):
            hasta = desde + config.tamano_lote
            perdida = perdida_enmascarada(logits_actor(politica, obs_d[desde:hasta]), etiquetas_d[desde:hasta],
                                          mascaras_d[desde:hasta])
            optimizador.zero_grad(set_to_none=True)
            torch.autograd.backward(perdida)
            optimizador.step()
        if dispositivo.type == "cuda":  # pragma: no cover - solo con GPU CUDA
            torch.cuda.synchronize()
        tiempos[nombre] = (time.perf_counter() - inicio) * n / muestra
    return tiempos


def elegir_dispositivo_mas_rapido(tiempos: Mapping[str, float]) -> torch.device:
    """
    Dispositivo con menor tiempo por epoca.

    Contexto:
        Despues de medir_dispositivos.

    Args:
        tiempos (Mapping[str, float]): dispositivo -> segundos por epoca.

    Returns:
        torch.device: El mas rapido; la CPU si no hay mediciones.

    Raises:
        Ninguna.

    Ejemplo:
        elegir_dispositivo_mas_rapido({"cpu": 30.0, "cuda": 4.0})  # cuda
    """
    if not tiempos:
        return torch.device("cpu")
    return torch.device(min(tiempos, key=lambda d: tiempos[d]))


def entrenar_arquitectura(catalogo: CatalogoJefes, conjunto: ConjuntoJefes, arquitectura: str,
                          aprendizaje: ConfigAprendizaje, semilla: int, dispositivo: torch.device,
                          rutas: RutasImitacionJefes, registrar: Registrar = print) -> dict[str, Any]:
    """
    Ajusta politica y valor de una arquitectura con todas las demostraciones y la guarda.

    Contexto:
        Una vez por arquitectura candidata. La particion es estratificada por jefe y estable (un
        episodio no cambia de lado entre arquitecturas), asi que las curvas son comparables.

    Args:
        catalogo (CatalogoJefes): Catalogo de jefes.
        conjunto (ConjuntoJefes): Demostraciones con el jefe de cada episodio.
        arquitectura (str): Nombre de la arquitectura candidata.
        aprendizaje (ConfigAprendizaje): Hiperparametros del ajuste.
        semilla (int): Semilla del ajuste (el orden de los lotes y la inicializacion).
        dispositivo (torch.device): Dispositivo del ajuste.
        rutas (RutasImitacionJefes): Rutas de la corrida.
        registrar (Callable): Registro de progreso.

    Returns:
        dict[str, Any]: Informe del ajuste (tambien guardado en el JSON de la arquitectura).

    Raises:
        ValueError: Si no hay episodios suficientes para particionar.

    Ejemplo:
        informe = entrenar_arquitectura(catalogo, conjunto, "2x256", ConfigAprendizaje(), 0, cpu, rutas)
    """
    inicio = time.monotonic()
    datos = conjunto.demostraciones
    es_train, es_val = particionar_estratificado(datos.episodios, conjunto.jefe_de_episodio, FRACCION_VALIDACION,
                                                 semilla)
    entrenamiento, validacion = datos.seleccionar(es_train), datos.seleccionar(es_val)
    registrar(f"{arquitectura}: {len(datos)} pasos, {np.unique(datos.episodios).size} episodios "
              f"(entrenamiento {len(entrenamiento)}, validacion {len(validacion)}) en {dispositivo}")
    configuracion, tamano = config_modelo(catalogo, arquitectura, semilla)
    modelo = crear_modelo_base(configuracion, tamano)

    def mostrar(punto: Any) -> None:
        """
        Muestra una epoca del ajuste.

        Contexto:
            entrenar_arquitectura.

        Args:
            punto (Any): PuntoCurva de la epoca.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        registrar(f"  {arquitectura} epoca {punto.epoca}: train {punto.perdida_entrenamiento:.4f}/"
                  f"{punto.precision_entrenamiento:.3f}  val {punto.perdida_validacion:.4f}/"
                  f"{punto.precision_validacion:.3f}")

    ajuste = entrenar_politica(modelo, entrenamiento, validacion, aprendizaje, semilla, dispositivo, mostrar)
    retornos_train = retornos_descontados(entrenamiento.recompensas, entrenamiento.episodios, aprendizaje.gamma)
    retornos_val = retornos_descontados(validacion.recompensas, validacion.episodios, aprendizaje.gamma)
    valor = entrenar_valor(modelo, entrenamiento, validacion, retornos_train, retornos_val, aprendizaje, semilla,
                           dispositivo)
    guardar_modelo(modelo, rutas.modelo(arquitectura))
    predicciones = _predecir_validacion(modelo, validacion)
    por_jefe = metricas_por_jefe(
        predicciones, validacion.etiquetas.astype(np.int64),
        jefe_por_paso(validacion.episodios, conjunto.jefe_de_episodio), validacion.episodios)
    muestra = np.random.default_rng(semilla).choice(len(validacion), size=min(MUESTRA_RECARGA, len(validacion)),
                                                    replace=False)
    esperadas = predicciones[muestra]
    coincidencia = coincidencia_tras_cargar(rutas.modelo(arquitectura), validacion.obs[muestra],
                                            validacion.mascaras[muestra], esperadas)
    informe = _informe_ajuste(ajuste)
    informe.update({
        "arquitectura": arquitectura, "dispositivo": str(dispositivo), "valor": valor,
        "segundos": time.monotonic() - inicio, "coincidencia_tras_recargar": coincidencia,
        "por_jefe": informe_por_jefe(por_jefe),
    })
    escribir_json(rutas.informe(arquitectura), informe)
    registrar(f"{arquitectura}: precision de validacion {ajuste.validacion['precision']:.4f}; por jefe "
              + ", ".join(f"{m.jefe} {m.precision:.3f}" for m in por_jefe)
              + f"; recarga {coincidencia:.3f}")
    return informe


def _predecir_validacion(modelo: Any, validacion: Demostraciones, tamano_lote: int = 8192) -> np.ndarray[Any, Any]:
    """
    Accion determinista (la de mayor probabilidad entre las validas) en cada paso de validacion.

    Contexto:
        Metricas por jefe y verificacion de la recarga desde disco.

    Args:
        modelo (Any): MaskablePPO ya ajustado.
        validacion (Demostraciones): Pasos de validacion.
        tamano_lote (int): Filas por lote de inferencia.

    Returns:
        np.ndarray: Accion elegida en cada paso, de tipo int64.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    from dmrl.aplicacion.imitacion_red import logits_actor, predecir_enmascarado

    politica = modelo.policy.to("cpu")
    salida: list[np.ndarray[Any, Any]] = []
    with torch.no_grad():
        for desde in range(0, len(validacion), tamano_lote):
            hasta = desde + tamano_lote
            logits = logits_actor(politica, torch.from_numpy(validacion.obs[desde:hasta]))
            salida.append(predecir_enmascarado(logits, torch.from_numpy(validacion.mascaras[desde:hasta])).numpy())
    return np.concatenate(salida).astype(np.int64)


def datos_por_jefe_en_disco(carpeta_datos: Path) -> dict[str, dict[str, float]]:
    """
    Resumen de lo recolectado leyendo solo los metadatos de los lotes.

    Contexto:
        Reporte: no carga las observaciones, que ocupan varios GB.

    Args:
        carpeta_datos (Path): Carpeta raiz de los datos.

    Returns:
        dict[str, dict[str, float]]: Igual que resumen_de_datos.

    Raises:
        Ninguna.

    Ejemplo:
        datos_por_jefe_en_disco(Path("runs/multijefe_imitacion/datos"))
    """
    metadatos: list[dict[str, Any]] = []
    for ronda in range(RONDAS_MAXIMAS):
        metadatos += metadatos_de_ronda(carpeta_datos, ronda)
    return resumen_de_datos(metadatos)


def publicar_modelo(rutas: RutasImitacionJefes, arquitectura: str, evaluacion: Mapping[str, Any]) -> dict[str, float]:
    """
    Publica el modelo elegido como mejor.zip y fija la referencia de la imitacion.

    Contexto:
        Fin de la imitacion. referencia.json es el piso con que el PPO decide si un checkpoint cae
        demasiado en algun jefe; mejor.json documenta de donde sale cada cifra.

    Args:
        rutas (RutasImitacionJefes): Rutas de la corrida.
        arquitectura (str): Arquitectura elegida.
        evaluacion (Mapping[str, Any]): Resultado de evaluar-modelo-jefes de esa arquitectura (con
            "por_jefe" y, en cada jefe, "tasa" y "juego").

    Returns:
        dict[str, float]: Jefe -> tasa de victoria que queda como referencia.

    Raises:
        FileNotFoundError: Si la arquitectura no tiene modelo ajustado.
        ValueError: Si la evaluacion no trae tasas.

    Ejemplo:
        publicar_modelo(rutas, "2x256", leer_json(rutas.evaluacion("final")))
    """
    por_jefe = dict(evaluacion.get("por_jefe", {}))
    if not por_jefe:
        raise ValueError("la evaluacion no trae tasas de victoria por jefe")
    origen = rutas.modelo(arquitectura)
    if not origen.exists():
        raise FileNotFoundError(f"no existe el modelo ajustado {origen}")
    tasas = {str(jefe): float(datos["tasa"]) for jefe, datos in por_jefe.items()}
    peleas = {str(jefe): int(datos["juego"].get("peleas", 0)) for jefe, datos in por_jefe.items()}
    shutil.copyfile(origen, rutas.mejor)
    escribir_json(rutas.base / "referencia.json", {"arquitectura": arquitectura, "tasas": tasas, "peleas": peleas})
    escribir_json(rutas.base / "mejor.json", {"arquitectura": arquitectura, "tasas": tasas, "peleas": peleas,
                                              "por_jefe": {j: d["juego"] for j, d in por_jefe.items()}})
    return tasas
