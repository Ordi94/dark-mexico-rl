"""
Ciclo completo de imitacion: recolectar, ajustar, jugar, repetir con DAgger y elegir por desempeno.

Cada etapa deja su resultado en disco (datos, modelo, evaluacion) y se omite si ya existe, de modo
que el ciclo se reanuda tras cualquier interrupcion sin repetir trabajo. El modelo elegido es el que
mejor juega, no el de menor perdida: una perdida de validacion baja con tasa de victoria baja es
desfase de distribucion, y se corrige con mas rondas de DAgger, no con mas epocas.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dmrl.aplicacion import evaluacion as evl
from dmrl.aplicacion import recoleccion as rec
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
from dmrl.dominio.imitacion import (
    ConfigImitacion,
    PlanEpisodio,
    brecha_sobreajuste,
    clave_de_seleccion,
    elegir_mejor,
    particionar_episodios,
    planificar_episodios,
    resumen_de_juego,
    retornos_descontados,
)
from dmrl.infraestructura.demostraciones import cargar_rondas, escribir_json, leer_json, metadatos_de_ronda

OPCIONES_FASE1: dict[str, Any] = {"start_phase": 1, "boss_hp_mult": 1.0}
OPCIONES_FASE2: dict[str, Any] = {"start_phase": 2, "boss_hp_mult": 1.0}
MUESTRA_RECARGA = 300
SEMILLA_EVALUACION = 1000
SEMILLA_CONFIRMACION = 9000

Registrar = Callable[[str], None]
Recolectar = Callable[..., int]
Evaluar = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class RutasImitacion:
    """
    Ubicacion de los archivos de una corrida de imitacion.

    Contexto:
        Todo el ciclo; centralizar las rutas evita que recoleccion, ajuste y evaluacion discrepen.

    Atributos:
        base: carpeta runs/<nombre>.
    """

    base: Path

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
            rutas.datos
        """
        return self.base / "datos"

    @property
    def parada(self) -> Path:
        """
        Archivo cuya existencia pide terminar ordenadamente.

        Contexto:
            detener.ps1 y los bucles de recoleccion.

        Args:
            Ninguno.

        Returns:
            Path: base/PARAR.

        Raises:
            Ninguna.

        Ejemplo:
            rutas.parada.exists()
        """
        return self.base / "PARAR"

    @property
    def mejor(self) -> Path:
        """
        Modelo elegido, cargable por MaskablePPO.

        Contexto:
            evaluar.ps1, ver_pelea.ps1 y el arranque de PPO.

        Args:
            Ninguno.

        Returns:
            Path: base/mejor.zip.

        Raises:
            Ninguna.

        Ejemplo:
            rutas.mejor
        """
        return self.base / "mejor.zip"

    def modelo(self, ronda: int) -> Path:
        """
        Modelo de una ronda.

        Contexto:
            Ajuste, DAgger y seleccion.

        Args:
            ronda (int): Numero de ronda.

        Returns:
            Path: base/modelos/ronda_XX.zip.

        Raises:
            Ninguna.

        Ejemplo:
            rutas.modelo(1)
        """
        return self.base / "modelos" / f"ronda_{ronda:02d}.zip"

    def informe_ajuste(self, ronda: int) -> Path:
        """
        Informe del ajuste de una ronda (curvas, precision por accion).

        Contexto:
            Reporte final.

        Args:
            ronda (int): Numero de ronda.

        Returns:
            Path: base/modelos/ronda_XX.json.

        Raises:
            Ninguna.

        Ejemplo:
            rutas.informe_ajuste(0)
        """
        return self.base / "modelos" / f"ronda_{ronda:02d}.json"

    def evaluacion(self, nombre: str) -> Path:
        """
        Resultado de una evaluacion en juego.

        Contexto:
            Seleccion y reporte.

        Args:
            nombre (str): "ronda_XX", "bot", "bot_confirmacion" o "confirmacion_ronda_XX".

        Returns:
            Path: base/evaluaciones/<nombre>.json.

        Raises:
            Ninguna.

        Ejemplo:
            rutas.evaluacion("bot")
        """
        return self.base / "evaluaciones" / f"{nombre}.json"


def rutas_de(nombre: str) -> RutasImitacion:
    """
    Rutas de la corrida con ese nombre.

    Contexto:
        CLI y scripts.

    Args:
        nombre (str): Nombre de la corrida.

    Returns:
        RutasImitacion: Rutas bajo runs/<nombre>.

    Raises:
        Ninguna.

    Ejemplo:
        rutas_de("imitacion")
    """
    return RutasImitacion(CARPETA_RUNS / nombre)


def registrar_en_consola(mensaje: str) -> None:
    """
    Imprime una linea con la hora.

    Contexto:
        Registro por defecto del ciclo.

    Args:
        mensaje (str): Texto a mostrar.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        registrar_en_consola("ronda 0 lista")
    """
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def planes_de_ronda(config: ConfigImitacion, ronda: int) -> list[PlanEpisodio]:
    """
    Plan de episodios de una ronda.

    Contexto:
        Recoleccion; la semilla depende de la ronda para que rondas distintas no repitan sorteos.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        ronda (int): Numero de ronda (0 = bot, 1.. = DAgger).

    Returns:
        list[PlanEpisodio]: Plan completo de la ronda.

    Raises:
        ValueError: Si los parametros del plan son invalidos.

    Ejemplo:
        planes_de_ronda(ConfigImitacion(), 0)
    """
    cantidad = config.episodios_ronda0 if ronda == 0 else config.episodios_dagger
    return planificar_episodios(cantidad, config.prob_fase2, config.fraccion_ruido, config.epsilon_min,
                                config.epsilon_max, config.semilla + ronda)


def elegir_dispositivo_ajuste() -> torch.device:
    """
    Dispositivo del ajuste supervisado: la GPU si existe.

    Contexto:
        El ajuste no esta en el bucle del juego, asi que la GPU acorta minutos sin quitar pasos/s
        al entrenamiento ni a las instancias.

    Args:
        Ninguno.

    Returns:
        torch.device: cuda si esta disponible, si no cpu.

    Raises:
        Ninguna.

    Ejemplo:
        elegir_dispositivo_ajuste()
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _informe_ajuste(resultado: ResultadoAjuste) -> dict[str, Any]:
    """
    Serializa un ResultadoAjuste con nombres de accion legibles.

    Contexto:
        Informe JSON de cada ronda.

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


def entrenar_ronda(config: ConfigImitacion, rutas: RutasImitacion, ronda: int, dispositivo: torch.device,
                   registrar: Registrar = registrar_en_consola) -> dict[str, Any]:
    """
    Reentrena la red desde cero con las demostraciones de las rondas 0..ronda y la guarda.

    Contexto:
        Una vez por ronda. Se reinicia desde pesos nuevos en cada ronda para que ninguna ronda
        herede el sobreajuste de la anterior; el conjunto agregado de DAgger es el que da la
        cobertura de estados.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        rutas (RutasImitacion): Rutas de la corrida.
        ronda (int): Ronda a entrenar.
        dispositivo (torch.device): Dispositivo del ajuste.
        registrar (Callable): Registro de progreso.

    Returns:
        dict[str, Any]: Informe del ajuste (tambien guardado en el JSON de la ronda).

    Raises:
        ValueError: Si no hay demostraciones suficientes para particionar.

    Ejemplo:
        informe = entrenar_ronda(config, rutas, 0, torch.device("cpu"))
    """
    inicio = time.monotonic()
    datos = cargar_rondas(rutas.datos, list(range(ronda + 1)))
    es_train, es_val = particionar_episodios(datos.episodios, config.fraccion_validacion, config.semilla)
    entrenamiento, validacion = datos.seleccionar(es_train), datos.seleccionar(es_val)
    registrar(f"ronda {ronda}: {len(datos)} pasos, {np.unique(datos.episodios).size} episodios "
              f"(entrenamiento {len(entrenamiento)}, validacion {len(validacion)})")
    modelo = crear_modelo_base(ConfigEntrenamiento(nombre=config.nombre, semilla=config.semilla + ronda))

    def mostrar(punto: Any) -> None:
        """
        Muestra una epoca del ajuste.

        Contexto:
            entrenar_ronda.

        Args:
            punto (Any): PuntoCurva de la epoca.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        registrar(f"  epoca {punto.epoca}: train {punto.perdida_entrenamiento:.4f}/{punto.precision_entrenamiento:.3f}"
                  f"  val {punto.perdida_validacion:.4f}/{punto.precision_validacion:.3f}")

    ajuste = entrenar_politica(modelo, entrenamiento, validacion, config.aprendizaje, config.semilla + ronda,
                               dispositivo, mostrar)
    retornos_train = retornos_descontados(entrenamiento.recompensas, entrenamiento.episodios,
                                          config.aprendizaje.gamma)
    retornos_val = retornos_descontados(validacion.recompensas, validacion.episodios, config.aprendizaje.gamma)
    valor = entrenar_valor(modelo, entrenamiento, validacion, retornos_train, retornos_val, config.aprendizaje,
                           config.semilla + ronda, dispositivo)
    guardar_modelo(modelo, rutas.modelo(ronda))
    muestra = np.random.default_rng(config.semilla).choice(len(validacion), size=min(MUESTRA_RECARGA, len(validacion)),
                                                           replace=False)
    esperadas = _acciones_del_modelo(modelo, validacion.obs[muestra], validacion.mascaras[muestra])
    coincidencia = coincidencia_tras_cargar(rutas.modelo(ronda), validacion.obs[muestra],
                                            validacion.mascaras[muestra], esperadas)
    informe = _informe_ajuste(ajuste)
    informe.update({
        "ronda": ronda, "pasos_totales": len(datos), "valor": valor, "segundos": time.monotonic() - inicio,
        "brecha_perdida": brecha_sobreajuste(ajuste.entrenamiento["perdida"], ajuste.validacion["perdida"]),
        "coincidencia_tras_recargar": coincidencia,
        "acciones_por_ronda": {str(r): int(len(metadatos_de_ronda(rutas.datos, r))) for r in range(ronda + 1)},
    })
    escribir_json(rutas.informe_ajuste(ronda), informe)
    registrar(f"ronda {ronda}: val precision {ajuste.validacion['precision']:.3f}, brecha de perdida "
              f"{informe['brecha_perdida']:+.3f}, recarga {coincidencia:.3f}")
    return informe


def _acciones_del_modelo(modelo: Any, obs: np.ndarray[Any, Any],
                         mascaras: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Acciones deterministas del modelo en memoria sobre una muestra.

    Contexto:
        entrenar_ronda; referencia para verificar la recarga desde disco.

    Args:
        modelo (Any): MaskablePPO.
        obs (np.ndarray): Observaciones de la muestra.
        mascaras (np.ndarray): Mascaras de la muestra.

    Returns:
        np.ndarray: Accion elegida en cada fila.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    modelo.policy.to("cpu")
    return np.asarray([
        int(modelo.predict(obs[i], action_masks=mascaras[i], deterministic=True)[0]) for i in range(obs.shape[0])
    ], dtype=np.int64)


def evaluar_condiciones(config: ConfigImitacion, politica: str, ruta_modelo: Path | None, semilla: int,
                        evaluar: Evaluar, registrar: Registrar, completa: bool = True) -> dict[str, Any]:
    """
    Juega las evaluaciones en juego de una politica: principal, inicio en fase 2 y con ruido.

    Contexto:
        Fin de cada ronda y linea base del bot. La condicion principal (jefe al 100%, fase 1) es la
        que decide la seleccion; las otras dos miden si lo aprendido depende de la trayectoria
        exacta de inicio.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        politica (str): "bot" o "modelo".
        ruta_modelo (Path, opcional): Modelo para la politica "modelo".
        semilla (int): Semilla base de la evaluacion.
        evaluar (Callable): Funcion con la firma de evaluar_en_paralelo.
        registrar (Callable): Registro de progreso.
        completa (bool): Si es False solo se juega la condicion principal.

    Returns:
        dict[str, Any]: Por condicion, el resumen de juego y los episodios.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        resultado = evaluar_condiciones(config, "bot", None, 1000, evaluar_en_paralelo, print)
    """
    condiciones: list[tuple[str, int, dict[str, Any], float]] = [
        ("principal", config.episodios_evaluacion, OPCIONES_FASE1, 0.0),
    ]
    if completa:
        condiciones += [
            ("fase2", config.episodios_fase2, OPCIONES_FASE2, 0.0),
            ("ruido", config.episodios_ruido, OPCIONES_FASE1, config.epsilon_evaluacion),
        ]
    salida: dict[str, Any] = {}
    for nombre, cantidad, opciones, epsilon in condiciones:
        episodios = evaluar(politica, cantidad, config.instancias, ruta_modelo, semilla, opciones,
                            epsilon_ruido=epsilon)
        resumen = resumen_de_juego(episodios)
        salida[nombre] = {"resumen": resumen, "episodios": episodios}
        registrar(f"  {politica} {nombre}: {int(resumen.get('victorias', 0))}/{int(resumen.get('peleas', 0))} "
                  f"victorias, dano {resumen.get('danio_hecho', 0):.0f}, parries {resumen.get('parries', 0):.1f}")
    return salida


def ejecutar_ciclo(config: ConfigImitacion, recolectar: Recolectar = rec.recolectar_ronda,
                   evaluar: Evaluar = evl.evaluar_en_paralelo, registrar: Registrar = registrar_en_consola,
                   dispositivo: torch.device | None = None, entorno: ConfigEntorno | None = None) -> dict[str, Any]:
    """
    Ejecuta (o reanuda) el ciclo completo de imitacion.

    Contexto:
        Comando imitar. Orden: ronda 0 con el bot, ajuste, evaluacion; despues cada ronda de DAgger
        (la red anterior juega, el bot etiqueta, se reentrena con todo lo acumulado y se evalua);
        linea base del bot; eleccion por desempeno en juego y confirmacion con peleas nuevas.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        recolectar (Callable): Funcion con la firma de recolectar_ronda.
        evaluar (Callable): Funcion con la firma de evaluar_en_paralelo.
        registrar (Callable): Registro de progreso.
        dispositivo (torch.device, opcional): Dispositivo del ajuste; por defecto la GPU si existe.
        entorno (ConfigEntorno, opcional): Configuracion del entorno.

    Returns:
        dict[str, Any]: Estado final: "completo" o "interrumpido" y, si termino, la ronda elegida.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        ejecutar_ciclo(ConfigImitacion(episodios_ronda0=20))
    """
    rutas = rutas_de(config.nombre)
    rutas.base.mkdir(parents=True, exist_ok=True)
    dispositivo = dispositivo or elegir_dispositivo_ajuste()
    for ronda in range(config.rondas_dagger + 1):
        if rutas.parada.exists():
            return {"estado": "interrumpido", "ronda": ronda}
        previo = rutas.modelo(ronda - 1) if ronda > 0 else None
        faltan = recolectar(rutas.datos, ronda, planes_de_ronda(config, ronda), config.instancias, previo,
                            entorno, rutas.parada)
        if faltan:
            registrar(f"ronda {ronda}: quedan {faltan} episodios sin recolectar")
            return {"estado": "interrumpido", "ronda": ronda}
        registrar(f"ronda {ronda}: recoleccion completa ({len(metadatos_de_ronda(rutas.datos, ronda))} episodios)")
        if not rutas.modelo(ronda).exists():
            entrenar_ronda(config, rutas, ronda, dispositivo, registrar)
        if not rutas.evaluacion(f"ronda_{ronda:02d}").exists():
            resultado = evaluar_condiciones(config, "modelo", rutas.modelo(ronda), config.semilla + SEMILLA_EVALUACION
                                            + ronda, evaluar, registrar)
            escribir_json(rutas.evaluacion(f"ronda_{ronda:02d}"), resultado)
    if rutas.parada.exists():
        return {"estado": "interrumpido", "ronda": config.rondas_dagger + 1}
    if not rutas.evaluacion("bot").exists():
        escribir_json(rutas.evaluacion("bot"),
                      evaluar_condiciones(config, "bot", None, config.semilla + SEMILLA_EVALUACION, evaluar, registrar))
    if not rutas.evaluacion("bot_confirmacion").exists():
        escribir_json(rutas.evaluacion("bot_confirmacion"), _confirmar(config, "bot", None, evaluar, registrar, 0))
    confirmar_candidatas(config, rutas, evaluar, registrar)
    return {"estado": "completo", "ronda": seleccionar_mejor(config, rutas, registrar)}


def _confirmar(config: ConfigImitacion, politica: str, ruta_modelo: Path | None, evaluar: Evaluar,
               registrar: Registrar, ronda: int) -> dict[str, Any]:
    """
    Juega peleas nuevas (jefe al 100%, fase 1) que no intervinieron en ninguna decision previa.

    Contexto:
        confirmar_candidatas y la linea base del bot. La muestra de seleccion (50 peleas) esta
        sesgada a favor del ganador por haberlo elegido: la confirmacion es la que se informa.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        politica (str): "bot" o "modelo".
        ruta_modelo (Path, opcional): Modelo para la politica "modelo".
        evaluar (Callable): Funcion con la firma de evaluar_en_paralelo.
        registrar (Callable): Registro de progreso.
        ronda (int): Ronda del modelo (0 para el bot); desplaza la semilla.

    Returns:
        dict[str, Any]: Ronda, resumen de juego y episodios.

    Raises:
        RuntimeError: Si el juego no consigue reiniciar.

    Ejemplo:
        _confirmar(config, "bot", None, evaluar_en_paralelo, print, 0)
    """
    episodios = evaluar(politica, config.episodios_confirmacion, config.instancias, ruta_modelo,
                        config.semilla + SEMILLA_CONFIRMACION + ronda, dict(OPCIONES_FASE1))
    resumen = resumen_de_juego(episodios)
    registrar(f"  {politica} confirmacion (ronda {ronda}): {int(resumen.get('victorias', 0))}/"
              f"{int(resumen.get('peleas', 0))} victorias")
    return {"ronda": ronda, "resumen": resumen, "episodios": episodios}


def confirmar_candidatas(config: ConfigImitacion, rutas: RutasImitacion, evaluar: Evaluar,
                         registrar: Registrar) -> list[int]:
    """
    Confirma con peleas nuevas a las rondas que mejor jugaron en la evaluacion principal.

    Contexto:
        Fin del ciclo. Elegir al mejor de varios modelos con 50 peleas favorece al que tuvo suerte
        (el intervalo de una tasa con 50 peleas es de +-12 puntos); las candidatas se vuelven a
        jugar con peleas nuevas y la eleccion final se hace con ambas muestras juntas.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        rutas (RutasImitacion): Rutas de la corrida.
        evaluar (Callable): Funcion con la firma de evaluar_en_paralelo.
        registrar (Callable): Registro de progreso.

    Returns:
        list[int]: Rondas confirmadas, de mejor a peor en la evaluacion principal.

    Raises:
        ValueError: Si ninguna ronda tiene evaluacion principal.

    Ejemplo:
        confirmar_candidatas(config, rutas, evaluar_en_paralelo, print)
    """
    principales = _resumenes_principales(config, rutas)
    if not principales:
        raise ValueError("ninguna ronda tiene evaluacion principal")
    orden = sorted(principales, key=lambda r: (clave_de_seleccion(principales[r]), r), reverse=True)
    candidatas = orden[: max(1, config.candidatas_confirmacion)]
    for ronda in candidatas:
        archivo = rutas.evaluacion(f"confirmacion_ronda_{ronda:02d}")
        if not archivo.exists():
            escribir_json(archivo, _confirmar(config, "modelo", rutas.modelo(ronda), evaluar, registrar, ronda))
    return candidatas


def _resumenes_principales(config: ConfigImitacion, rutas: RutasImitacion) -> dict[int, dict[str, float]]:
    """
    Resumen de la evaluacion principal de cada ronda evaluada.

    Contexto:
        confirmar_candidatas y seleccionar_mejor.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        rutas (RutasImitacion): Rutas de la corrida.

    Returns:
        dict[int, dict[str, float]]: Ronda -> resumen; sin las rondas sin evaluar.

    Raises:
        Ninguna.

    Ejemplo:
        (uso interno)
    """
    resumenes: dict[int, dict[str, float]] = {}
    for ronda in range(config.rondas_dagger + 1):
        datos = leer_json(rutas.evaluacion(f"ronda_{ronda:02d}"))
        if "principal" in datos:
            resumenes[ronda] = datos["principal"]["resumen"]
    return resumenes


def seleccionar_mejor(config: ConfigImitacion, rutas: RutasImitacion, registrar: Registrar) -> int:
    """
    Elige la ronda que mejor juega y publica su modelo como mejor.zip.

    Contexto:
        Fin del ciclo. El criterio es el desempeno en juego, no la perdida de validacion. Entre
        las rondas confirmadas decide la muestra junta (seleccion mas confirmacion); si ninguna
        esta confirmada decide la evaluacion principal.

    Args:
        config (ConfigImitacion): Parametros del ciclo.
        rutas (RutasImitacion): Rutas de la corrida.
        registrar (Callable): Registro de progreso.

    Returns:
        int: Ronda elegida.

    Raises:
        ValueError: Si ninguna ronda tiene evaluacion.

    Ejemplo:
        seleccionar_mejor(config, rutas, print)
    """
    principales = _resumenes_principales(config, rutas)
    agrupados: dict[int, dict[str, float]] = {}
    for ronda in principales:
        confirmacion = leer_json(rutas.evaluacion(f"confirmacion_ronda_{ronda:02d}"))
        if "episodios" in confirmacion:
            episodios = [*leer_json(rutas.evaluacion(f"ronda_{ronda:02d}"))["principal"]["episodios"],
                         *confirmacion["episodios"]]
            agrupados[ronda] = resumen_de_juego(episodios)
    base_de_eleccion = agrupados or principales
    mejor = elegir_mejor(base_de_eleccion)
    shutil.copyfile(rutas.modelo(mejor), rutas.mejor)
    escribir_json(rutas.base / "mejor.json", {
        "ronda": mejor, "resumen": base_de_eleccion[mejor], "criterio": "agrupado" if agrupados else "principal",
        "principales": {str(r): s for r, s in principales.items()},
        "agrupados": {str(r): s for r, s in agrupados.items()},
    })
    registrar(f"modelo elegido: ronda {mejor} ({base_de_eleccion[mejor].get('tasa_victoria', 0):.0%} de victorias, "
              f"{int(base_de_eleccion[mejor].get('peleas', 0))} peleas)")
    return mejor


def episodios_por_ronda(rutas: RutasImitacion, rondas: Sequence[int]) -> dict[int, dict[str, float]]:
    """
    Estadisticas de recoleccion de cada ronda para el reporte.

    Contexto:
        Reporte final.

    Args:
        rutas (RutasImitacion): Rutas de la corrida.
        rondas (Sequence[int]): Rondas a resumir.

    Returns:
        dict[int, dict[str, float]]: Episodios, pasos, victorias del ejecutor y fraccion en fase 2.

    Raises:
        Ninguna.

    Ejemplo:
        episodios_por_ronda(rutas, [0, 1])
    """
    salida: dict[int, dict[str, float]] = {}
    for ronda in rondas:
        meta = metadatos_de_ronda(rutas.datos, ronda)
        if not meta:
            continue
        n = len(meta)
        salida[ronda] = {
            "episodios": float(n), "pasos": float(sum(int(m["pasos"]) for m in meta)),
            "victorias": float(sum(1 for m in meta if m.get("victoria"))),
            "fase2": sum(1 for m in meta if m["fase_inicio"] == 2) / n,
            "con_ruido": sum(1 for m in meta if m["epsilon"] > 0) / n,
        }
    return salida
