"""
Callbacks del PPO multi-jefe: metricas por jefe, checkpoints, y evaluacion automatica en juego con
compuerta de promocion y correccion por regresion.

La evaluacion corre en un proceso aparte (python -m dmrl evaluar-modelo-jefes) con sus propias
instancias del juego, para no detener el entrenamiento ni mezclar sus episodios con los de
aprendizaje. El callback solo guarda una copia del modelo, lanza el proceso, revisa si ya termino y
aplica las reglas de dominio.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import ConstantSchedule

from dmrl.aplicacion.callbacks import RESULTADO_ERROR, episodios_terminados
from dmrl.dominio.ppo_jefes import (
    ConfigPPOJefes,
    ControlRegresion,
    EstadoEvaluacion,
    evaluar_checkpoint,
    metricas_de_ventana,
    tasa_aprendizaje_corregida,
    tasa_media,
)
from dmrl.infraestructura.demostraciones import escribir_json, leer_json

VENTANA_POR_JEFE = 50
NOMBRE_ESTADO = "estado_ppo.json"
NOMBRE_HISTORIAL = "evaluaciones.jsonl"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NO_WINDOW = 0x08000000
PREFIJO_INSTANCIAS_EVALUACION = "evppo"


class ProcesoEvaluacion(Protocol):
    """Lo que el callback necesita de un proceso de evaluacion en marcha."""

    returncode: int | None

    def poll(self) -> int | None:
        """Codigo de salida si ya termino, None si sigue corriendo."""
        ...

    def terminate(self) -> None:
        """Pide al proceso que termine."""
        ...


Lanzador = Callable[[Path, Path, ConfigPPOJefes], ProcesoEvaluacion]


def lanzar_evaluacion(modelo: Path, salida: Path, config: ConfigPPOJefes) -> ProcesoEvaluacion:
    """
    Lanza la evaluacion de un modelo en un proceso nuevo, con prioridad baja.

    Contexto:
        CallbackEvaluacionPeriodica, cuando toca evaluar. El proceso escribe el resultado en `salida`
        y su registro en el mismo nombre con extension .log.

    Args:
        modelo (Path): Copia del modelo a evaluar.
        salida (Path): JSON donde el proceso deja el resultado.
        config (ConfigPPOJefes): Peleas por jefe de la evaluacion.

    Returns:
        subprocess.Popen[bytes]: Proceso lanzado (sin ventana).

    Raises:
        OSError: Si no se puede crear el proceso o su registro.

    Ejemplo:
        proceso = lanzar_evaluacion(Path("snap.zip"), Path("eval.json"), ConfigPPOJefes())
    """
    argumentos = [sys.executable, "-m", "dmrl", "evaluar-modelo-jefes", "--modelo", str(modelo),
                  "--episodios", str(config.episodios_evaluacion), "--instancias", "4",
                  "--semilla", str(config.semilla + 7000), "--prefijo", PREFIJO_INSTANCIAS_EVALUACION,
                  "--salida", str(salida)]
    salida.parent.mkdir(parents=True, exist_ok=True)
    registro = salida.with_suffix(".log").open("wb")
    try:
        proceso = subprocess.Popen(argumentos, stdout=registro, stderr=subprocess.STDOUT,
                                   creationflags=BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW)
    finally:
        registro.close()
    return proceso


def aplicar_tasa_aprendizaje(modelo: Any, tasa: float) -> None:
    """
    Cambia la tasa de aprendizaje de un modelo en marcha.

    Contexto:
        Correccion por regresion y reanudacion con la tasa ya corregida. SB3 recalcula la tasa del
        optimizador desde lr_schedule en cada actualizacion, asi que hay que cambiar ambos.

    Args:
        modelo (Any): MaskablePPO.
        tasa (float): Nueva tasa de aprendizaje.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        aplicar_tasa_aprendizaje(modelo, 5e-5)
    """
    modelo.learning_rate = tasa
    modelo.lr_schedule = ConstantSchedule(tasa)
    for grupo in modelo.policy.optimizer.param_groups:
        grupo["lr"] = tasa


def restaurar_pesos(modelo: Any, ruta: Path) -> None:
    """
    Devuelve la politica de un modelo en marcha a los pesos guardados en un zip.

    Contexto:
        Correccion por regresion: volver al ultimo mejor modelo. El estado del optimizador (los
        momentos de Adam) se descarta, porque describe la trayectoria que se abandona.

    Args:
        modelo (Any): MaskablePPO en entrenamiento.
        ruta (Path): Zip de un MaskablePPO con la misma arquitectura.

    Returns:
        None: No retorna valor.

    Raises:
        FileNotFoundError: Si el archivo no existe.

    Ejemplo:
        restaurar_pesos(modelo, Path("runs/multijefe_ppo/mejor.zip"))
    """
    from sb3_contrib import MaskablePPO

    if not ruta.exists():
        raise FileNotFoundError(f"no existe el modelo {ruta}")
    guardado = MaskablePPO.load(str(ruta), device="cpu")
    modelo.policy.load_state_dict({k: v.to(modelo.device) for k, v in guardado.policy.state_dict().items()})
    modelo.policy.optimizer.state.clear()


class CallbackMetricasJefes(BaseCallback):
    """
    Publica en TensorBoard las metricas de cada jefe (prefijo dm/<jefe>/) al final de cada rollout.

    Contexto:
        Se pasa a MaskablePPO.learn. Los episodios en error (caida del juego) no cuentan.
    """

    def __init__(self, fase_normal: Mapping[str, int], ventana: int = VENTANA_POR_JEFE, imprimir: bool = True) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            fase_normal (Mapping[str, int]): Jefe -> fase de inicio de una pelea completa.
            ventana (int): Episodios recientes por jefe sobre los que se promedia.
            imprimir (bool): Si es True escribe una linea de progreso por rollout.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            CallbackMetricasJefes({"amlo": 1, "chamuco": 2})
        """
        super().__init__()
        self.fase_normal = dict(fase_normal)
        self.imprimir = imprimir
        self.recientes: dict[str, deque[dict[str, Any]]] = {j: deque(maxlen=ventana) for j in fase_normal}
        self.episodios_total = 0
        self.errores_total = 0
        self._t0 = 0.0
        self._pasos0 = 0

    def _on_training_start(self) -> None:
        """
        Marca el inicio para medir pasos por segundo.

        Contexto:
            SB3 lo llama una vez antes del primer rollout.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        self._t0 = time.perf_counter()
        self._pasos0 = self.num_timesteps

    def _on_step(self) -> bool:
        """
        Acumula los episodios terminados en este paso, por jefe.

        Contexto:
            SB3 lo llama tras cada paso de los entornos.

        Args:
            Ninguno.

        Returns:
            bool: Siempre True.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        for resumen in episodios_terminados(self.locals):
            if resumen.get("resultado") == RESULTADO_ERROR:
                self.errores_total += 1
                continue
            self.episodios_total += 1
            jefe = str(resumen.get("jefe", "?"))
            self.recientes.setdefault(jefe, deque(maxlen=VENTANA_POR_JEFE)).append(resumen)
        return True

    def _on_rollout_end(self) -> None:
        """
        Registra las metricas de cada jefe y la velocidad de muestreo.

        Contexto:
            SB3 lo llama al terminar de recolectar cada rollout, antes de actualizar la red.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        transcurrido = max(time.perf_counter() - self._t0, 1e-9)
        pasos_s = (self.num_timesteps - self._pasos0) / transcurrido
        self.logger.record("dm/pasos_por_segundo", pasos_s)
        self.logger.record("dm/episodios_total", self.episodios_total)
        self.logger.record("dm/errores_instancia_total", self.errores_total)
        tasas: list[float] = []
        partes: list[str] = []
        for jefe, ventana in sorted(self.recientes.items()):
            metricas = metricas_de_ventana(list(ventana), self.fase_normal.get(jefe, 1))
            for nombre, valor in metricas.items():
                self.logger.record(f"dm/{jefe}/{nombre}", valor)
            if metricas:
                tasas.append(metricas["tasa_victoria"])
                partes.append(f"{jefe}={metricas['tasa_victoria']:.2f}")
        if tasas:
            self.logger.record("dm/tasa_victoria_media", sum(tasas) / len(tasas))
        if self.imprimir:
            print(f"pasos={self.num_timesteps} sps={pasos_s:.0f} episodios={self.episodios_total} "
                  f"victoria[{' '.join(partes)}]", flush=True)


class CallbackCheckpointJefes(BaseCallback):
    """
    Guarda ultimo.zip periodicamente y un checkpoint numerado cada varios guardados.

    Contexto:
        Permite reanudar con ultimo.zip. Los numerados dan puntos de retorno si una mala racha
        degrada la politica; se conservan solo los ultimos.
    """

    def __init__(self, directorio: Path, cada: int = 100_000, cada_numerado: int = 10, conservar: int = 5) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            directorio (Path): Carpeta de checkpoints (se crea si falta).
            cada (int): Pasos entre guardados de ultimo.zip.
            cada_numerado (int): Cada cuantos guardados se conserva ademas un modelo_<pasos>.zip.
            conservar (int): Cuantos checkpoints numerados se mantienen.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede crear la carpeta.

        Ejemplo:
            CallbackCheckpointJefes(Path("runs/multijefe_ppo/checkpoints"))
        """
        super().__init__()
        self.directorio = directorio
        self.cada = cada
        self.cada_numerado = cada_numerado
        self.conservar = conservar
        self._proximo = cada
        self._guardados = 0
        directorio.mkdir(parents=True, exist_ok=True)

    def guardar(self, numerado: bool = False) -> Path:
        """
        Guarda ultimo.zip y, si se pide, un checkpoint numerado.

        Contexto:
            _on_step y el cierre del entrenamiento. Se escribe a un temporal y se reemplaza, de modo
            que un corte durante el guardado nunca deja un ultimo.zip a medias.

        Args:
            numerado (bool): Si es True conserva ademas modelo_<pasos>.zip.

        Returns:
            Path: Ruta de ultimo.zip.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            callback.guardar()
        """
        ultimo = self.directorio / "ultimo.zip"
        temporal = self.directorio / "ultimo_tmp.zip"
        self.model.save(temporal)
        temporal.replace(ultimo)
        if numerado:
            shutil.copyfile(ultimo, self.directorio / f"modelo_{self.num_timesteps}.zip")
            antiguos = sorted(self.directorio.glob("modelo_*.zip"), key=lambda p: p.stat().st_mtime)
            for viejo in antiguos[: max(0, len(antiguos) - self.conservar)]:
                viejo.unlink()
        return ultimo

    def _on_step(self) -> bool:
        """
        Guarda cuando se cruza el siguiente multiplo de `cada` pasos.

        Contexto:
            SB3 lo llama tras cada paso de los entornos.

        Args:
            Ninguno.

        Returns:
            bool: Siempre True.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            (llamado por SB3)
        """
        if self.num_timesteps >= self._proximo:
            self._guardados += 1
            self.guardar(numerado=self._guardados % self.cada_numerado == 0)
            self._proximo = (self.num_timesteps // self.cada + 1) * self.cada
        return True


class CallbackEvaluacionPeriodica(BaseCallback):
    """
    Evalua en juego el modelo cada cierto tiempo y aplica la compuerta de promocion y la correccion.

    Contexto:
        Reglas (documentadas en dominio.ppo_jefes):
        * mejor.zip solo se reemplaza si la tasa de victoria media supera a la del mejor vigente y
          ningun jefe cae mas de caida_maxima_por_jefe respecto a la imitacion;
        * si la media queda claramente por debajo de la imitacion en evaluaciones_para_revertir
          evaluaciones seguidas, la politica vuelve a los pesos de mejor.zip y la tasa de
          aprendizaje se multiplica por factor_reduccion_lr (sin bajar de lr_minimo).
        La revision ocurre al inicio de cada rollout, cuando el buffer esta vacio, para que un
        cambio de pesos no se mezcle con datos recolectados con la politica anterior.
    """

    def __init__(self, config: ConfigPPOJefes, carpeta: Path, referencia: Mapping[str, float],
                 lanzador: Lanzador = lanzar_evaluacion, reloj: Callable[[], float] = time.monotonic) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            config (ConfigPPOJefes): Parametros de la corrida y de las reglas.
            carpeta (Path): Carpeta de la corrida (runs/<nombre>).
            referencia (Mapping[str, float]): Jefe -> tasa de victoria de la imitacion.
            lanzador (Lanzador): Funcion que lanza el proceso de evaluacion (inyectable en pruebas).
            reloj (Callable): Reloj monotono en segundos (inyectable en pruebas).

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si la referencia esta vacia.

        Ejemplo:
            CallbackEvaluacionPeriodica(config, Path("runs/multijefe_ppo"), {"amlo": 0.9})
        """
        super().__init__()
        self.config = config
        self.carpeta = carpeta
        self.lanzador = lanzador
        self.reloj = reloj
        inicial = EstadoEvaluacion(referencia=dict(referencia), mejor_media=tasa_media(referencia),
                                   learning_rate=config.learning_rate)
        self.estado = EstadoEvaluacion.desde_dict(leer_json(carpeta / "checkpoints" / NOMBRE_ESTADO), inicial)
        self.estado.referencia = dict(referencia)
        self.control = ControlRegresion(config, seguidas=self.estado.seguidas)
        self.proceso: ProcesoEvaluacion | None = None
        self._snapshot: Path | None = None
        self._salida: Path | None = None
        self._paso_snapshot = 0
        self._proxima = 0.0

    def guardar_estado(self) -> None:
        """
        Escribe el estado de la evaluacion junto a los checkpoints.

        Contexto:
            Tras cada evaluacion y al cerrar el entrenamiento.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            callback.guardar_estado()
        """
        self.estado.seguidas = self.control.seguidas
        escribir_json(self.carpeta / "checkpoints" / NOMBRE_ESTADO, self.estado.a_dict())

    def _on_training_start(self) -> None:
        """
        Restaura la tasa de aprendizaje corregida y programa la siguiente evaluacion.

        Contexto:
            SB3 lo llama una vez antes del primer rollout. Una corrida sin evaluaciones previas usa el
            plazo corto de la primera evaluacion; una reanudada, el intervalo normal.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        aplicar_tasa_aprendizaje(self.model, self.estado.learning_rate)
        minutos = self.config.minutos_evaluacion if self.estado.evaluaciones else self.config.minutos_primera_evaluacion
        self._proxima = self.reloj() + minutos * 60.0

    def _on_step(self) -> bool:
        """
        No interviene en cada paso (la revision va al inicio del rollout).

        Contexto:
            SB3 lo llama tras cada paso de los entornos.

        Args:
            Ninguno.

        Returns:
            bool: Siempre True.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        return True

    def _on_rollout_start(self) -> None:
        """
        Revisa la evaluacion en marcha o lanza una nueva si ya toca.

        Contexto:
            SB3 lo llama antes de recolectar cada rollout.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        if self.proceso is not None:
            if self.proceso.poll() is None:
                return
            self._terminar_evaluacion()
        elif self.reloj() >= self._proxima:
            self._iniciar_evaluacion()

    def _iniciar_evaluacion(self) -> None:
        """
        Guarda una copia del modelo y lanza su evaluacion.

        Contexto:
            _on_rollout_start, cuando vence el plazo.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede guardar la copia.

        Ejemplo:
            (uso interno)
        """
        self._paso_snapshot = self.num_timesteps
        carpeta = self.carpeta / "evaluaciones"
        carpeta.mkdir(parents=True, exist_ok=True)
        self._snapshot = carpeta / f"modelo_{self._paso_snapshot}.zip"
        self._salida = carpeta / f"eval_{self._paso_snapshot}.json"
        self.model.save(self._snapshot)
        self.proceso = self.lanzador(self._snapshot, self._salida, self.config)
        print(f"[evaluacion] lanzada con el modelo de {self._paso_snapshot} pasos", flush=True)

    def _terminar_evaluacion(self) -> None:
        """
        Lee el resultado de la evaluacion terminada y aplica las reglas.

        Contexto:
            _on_rollout_start, cuando el proceso de evaluacion ya salio. Si fallo o no dejo
            resultado, se descarta y se reprograma sin tocar el modelo ni el contador de regresion.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        proceso, snapshot, salida = self.proceso, self._snapshot, self._salida
        self.proceso = self._snapshot = self._salida = None
        self._proxima = self.reloj() + self.config.minutos_evaluacion * 60.0
        resultado = leer_json(salida) if salida is not None else {}
        tasas = {j: float(d["tasa"]) for j, d in resultado.get("por_jefe", {}).items()}
        if proceso is None or snapshot is None or proceso.returncode != 0 or not tasas:
            print("[evaluacion] fallida o sin resultado; se descarta", flush=True)
            if snapshot is not None:
                snapshot.unlink(missing_ok=True)
            return
        self.aplicar_resultado(tasas, snapshot, self._paso_snapshot, resultado)

    def aplicar_resultado(self, tasas: Mapping[str, float], snapshot: Path, paso: int,
                          detalle: Mapping[str, Any] | None = None) -> None:
        """
        Aplica la compuerta de promocion y la regla de regresion a una evaluacion terminada.

        Contexto:
            _terminar_evaluacion y pruebas de las reglas.

        Args:
            tasas (Mapping[str, float]): Jefe -> tasa de victoria del modelo evaluado.
            snapshot (Path): Copia del modelo evaluado; se promueve a mejor.zip o se borra.
            paso (int): Pasos de entrenamiento del modelo evaluado.
            detalle (Mapping[str, Any], opcional): Resultado completo, para el historial.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si la evaluacion no incluye a todos los jefes de la referencia.

        Ejemplo:
            callback.aplicar_resultado({"amlo": 0.9}, Path("snap.zip"), 100_000)
        """
        decision = evaluar_checkpoint(tasas, self.estado.referencia, self.estado.mejor_media, self.config)
        mejor = self.carpeta / "mejor.zip"
        if decision.promover:
            self._publicar(snapshot, mejor)
            self.estado.mejor_media, self.estado.mejor_paso = decision.media, paso
            self.estado.promociones += 1
        if not decision.promover:
            snapshot.unlink(missing_ok=True)
        revertir = self.control.registrar(decision)
        if revertir:
            restaurar_pesos(self.model, mejor)
            self.estado.learning_rate = tasa_aprendizaje_corregida(self.estado.learning_rate, self.config)
            aplicar_tasa_aprendizaje(self.model, self.estado.learning_rate)
            self.estado.reversiones += 1
        registro = {
            "pasos": paso, "tasas": dict(tasas), "media": decision.media, "promovido": decision.promover,
            "regresion": decision.regresion, "revertido": revertir, "motivo": decision.motivo,
            "caidas": decision.caidas, "learning_rate": self.estado.learning_rate,
        }
        self.estado.evaluaciones.append(registro)
        self.logger.record("dm/eval/tasa_media", decision.media)
        self.logger.record("dm/eval/mejor_media", self.estado.mejor_media)
        self.logger.record("dm/eval/promovido", float(decision.promover))
        self.logger.record("dm/eval/revertido", float(revertir))
        for jefe, tasa in tasas.items():
            self.logger.record(f"dm/eval/{jefe}/tasa_victoria", tasa)
        self.guardar_estado()
        with (self.carpeta / NOMBRE_HISTORIAL).open("a", encoding="utf-8") as historial:
            historial.write(json.dumps({**registro, "detalle": dict(detalle or {})}) + "\n")
        print(f"[evaluacion] {paso} pasos: media {decision.media:.3f} ({decision.motivo})"
              f"{'; PROMOVIDO' if decision.promover else ''}{'; REVERTIDO' if revertir else ''}", flush=True)

    @staticmethod
    def _publicar(origen: Path, destino: Path) -> None:
        """
        Copia un modelo a su destino de forma atomica.

        Contexto:
            aplicar_resultado, al promover: un lector (evaluar.ps1, ver_pelea.ps1) nunca debe ver un
            mejor.zip a medio escribir.

        Args:
            origen (Path): Zip del modelo promovido.
            destino (Path): Ruta de mejor.zip.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede copiar.

        Ejemplo:
            (uso interno)
        """
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(".tmp")
        shutil.copyfile(origen, temporal)
        temporal.replace(destino)

    def _on_training_end(self) -> None:
        """
        Cierra la evaluacion en marcha y guarda el estado.

        Contexto:
            SB3 lo llama al terminar learn (por parada ordenada o por agotar los pasos).

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir el estado.

        Ejemplo:
            (llamado por SB3)
        """
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.terminate()
        self.guardar_estado()

