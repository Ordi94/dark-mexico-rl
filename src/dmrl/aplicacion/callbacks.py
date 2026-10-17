"""
Callbacks del entrenamiento: metricas propias de TensorBoard, curriculo, checkpoints, mejor
modelo y parada ordenada.

Todos leen el resumen de episodio que el entorno publica en info["episodio"] al terminar o
truncar una pelea. Los episodios cuyo resultado es "error" (caida del juego) no cuentan para
ninguna metrica de rendimiento ni para el curriculo.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from dmrl.dominio.curriculo import Curriculo

RESULTADO_ERROR = "error"


def episodios_terminados(locales: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """
    Resumenes de los episodios que terminaron en el ultimo paso de los entornos.

    Contexto:
        _on_step de cada callback; self.locals["infos"] trae un dict por entorno.

    Args:
        locales (dict[str, Any]): self.locals del callback.

    Returns:
        Iterator[dict[str, Any]]: Resumenes "episodio" presentes en la info.

    Raises:
        Ninguna.

    Ejemplo:
        list(episodios_terminados({"infos": [{"episodio": {"victoria": True}}]}))
    """
    for info in locales.get("infos", ()):
        resumen = info.get("episodio")
        if resumen is not None:
            yield resumen


class CallbackMetricas(BaseCallback):
    """
    Publica en TensorBoard las metricas del problema (prefijo dm/) al final de cada rollout.

    Contexto:
        Se pasa a MaskablePPO.learn junto a los demas callbacks.
    """

    def __init__(self, ventana: int = 100, curriculo: Curriculo | None = None, imprimir: bool = True) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            ventana (int): Episodios recientes sobre los que se promedia.
            curriculo (Curriculo, opcional): Para publicar nivel y tasa del nivel.
            imprimir (bool): Si es True escribe una linea de progreso por rollout en la consola.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            callback = CallbackMetricas(100, curriculo)
        """
        super().__init__()
        self.imprimir = imprimir
        self.ventana = ventana
        self.curriculo = curriculo
        self.recientes: deque[dict[str, Any]] = deque(maxlen=ventana)
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
        Acumula los episodios terminados en este paso.

        Contexto:
            SB3 lo llama tras cada paso de los entornos.

        Args:
            Ninguno.

        Returns:
            bool: Siempre True (no interrumpe el entrenamiento).

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
            self.recientes.append(resumen)
        return True

    def _on_rollout_end(self) -> None:
        """
        Registra las metricas de la ventana reciente y la velocidad de muestreo.

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
        registrar = self.logger.record
        transcurrido = max(time.perf_counter() - self._t0, 1e-9)
        registrar("dm/pasos_por_segundo", (self.num_timesteps - self._pasos0) / transcurrido)
        registrar("dm/episodios_total", self.episodios_total)
        registrar("dm/errores_instancia_total", self.errores_total)
        if self.curriculo is not None:
            registrar("dm/nivel_curriculo", self.curriculo.nivel)
            registrar("dm/multiplicador_vida_jefe", self.curriculo.multiplicador)
            registrar("dm/tasa_victoria_nivel", self.curriculo.tasa_victoria)
        if not self.recientes:
            return

        def media(clave: str) -> float:
            """
            Media de un campo numerico en la ventana.

            Contexto:
                _on_rollout_end.

            Args:
                clave (str): Campo del resumen de episodio.

            Returns:
                float: Media de la ventana.

            Raises:
                KeyError: Si algun resumen no trae el campo.

            Ejemplo:
                (uso interno)
            """
            return statistics.fmean(float(e[clave]) for e in self.recientes)

        completos = [e for e in self.recientes if e["fase_inicio"] == 1]
        tasa = statistics.fmean(float(e["victoria"]) for e in completos) if completos else 0.0
        registrar("dm/tasa_victoria", tasa)
        registrar("dm/tasa_victoria_global", media("victoria"))
        registrar("dm/danio_al_jefe", media("danio_hecho"))
        registrar("dm/progreso_jefe", media("progreso_jefe"))
        registrar("dm/danio_recibido", media("danio_recibido"))
        registrar("dm/parries", media("parries"))
        registrar("dm/caguamas_usadas", media("caguamas_usadas"))
        registrar("dm/caguamas_interrumpidas", media("caguamas_interrumpidas"))
        registrar("dm/fase_max", media("fase_max"))
        registrar("dm/duracion_s", media("duracion_s"))
        registrar("dm/recompensa_episodio", media("recompensa"))
        if self.imprimir:
            nivel = self.curriculo.nivel if self.curriculo is not None else 0
            print(
                f"pasos={self.num_timesteps} sps={(self.num_timesteps - self._pasos0) / transcurrido:.0f} "
                f"episodios={self.episodios_total} nivel={nivel} victoria={tasa:.2f} "
                f"danio_jefe={media('danio_hecho'):.0f} recibido={media('danio_recibido'):.0f} "
                f"parries={media('parries'):.1f} recompensa={media('recompensa'):.2f}", flush=True,
            )


class CallbackCurriculo(BaseCallback):
    """
    Alimenta el curriculo con cada episodio y publica el nivel vigente a los entornos.

    Contexto:
        El curriculo vive en el proceso principal; los entornos corren en subprocesos, asi que
        el nivel se les envia con env_method al empezar y cada vez que sube.
    """

    def __init__(self, curriculo: Curriculo) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            curriculo (Curriculo): Estado compartido del curriculo.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            callback = CallbackCurriculo(Curriculo())
        """
        super().__init__()
        self.curriculo = curriculo

    def _publicar(self) -> None:
        """
        Envia multiplicador y probabilidad de fase 2 a todos los entornos.

        Contexto:
            _on_training_start y cada subida de nivel.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        self.training_env.env_method("fijar_multiplicador", self.curriculo.multiplicador)
        self.training_env.env_method("fijar_prob_fase2", self.curriculo.config.prob_fase2)

    def _on_training_start(self) -> None:
        """
        Aplica el nivel inicial (o el restaurado de un checkpoint).

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
        self._publicar()

    def _on_step(self) -> bool:
        """
        Registra los episodios terminados y sube el nivel si corresponde.

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
                continue
            if self.curriculo.registrar_episodio(bool(resumen["victoria"]), int(resumen["fase_inicio"])):
                self._publicar()
        return True


def guardar_estado_curriculo(curriculo: Curriculo, ruta: Path) -> None:
    """
    Escribe el estado del curriculo junto al checkpoint.

    Contexto:
        CallbackCheckpoint y el cierre del entrenamiento.

    Args:
        curriculo (Curriculo): Estado a guardar.
        ruta (Path): Archivo JSON de destino.

    Returns:
        None: No retorna valor.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        guardar_estado_curriculo(curriculo, Path("checkpoints/curriculo.json"))
    """
    ruta.write_text(json.dumps(curriculo.a_dict()), encoding="utf-8")


def cargar_estado_curriculo(curriculo: Curriculo, ruta: Path) -> bool:
    """
    Restaura el curriculo desde un archivo si existe y es valido.

    Contexto:
        Reanudacion del entrenamiento.

    Args:
        curriculo (Curriculo): Curriculo a restaurar.
        ruta (Path): Archivo JSON guardado antes.

    Returns:
        bool: True si se restauro; False si el archivo falta o esta danado.

    Raises:
        Ninguna.

    Ejemplo:
        cargar_estado_curriculo(curriculo, Path("checkpoints/curriculo.json"))
    """
    try:
        curriculo.cargar(json.loads(ruta.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


class CallbackCheckpoint(BaseCallback):
    """
    Guarda el modelo y el curriculo cada cierto numero de pasos, conservando solo los ultimos.

    Contexto:
        Permite reanudar con ultimo.zip + curriculo.json. Los checkpoints numerados dan puntos
        de retorno si una mala racha degrada la politica.
    """

    def __init__(self, directorio: Path, curriculo: Curriculo, cada: int = 100_000, conservar: int = 5) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            directorio (Path): Carpeta de checkpoints (se crea si falta).
            curriculo (Curriculo): Estado a guardar junto al modelo.
            cada (int): Pasos (de todos los entornos) entre checkpoints.
            conservar (int): Cuantos checkpoints numerados se mantienen.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede crear la carpeta.

        Ejemplo:
            callback = CallbackCheckpoint(Path("runs/a/checkpoints"), curriculo)
        """
        super().__init__()
        self.directorio = directorio
        self.curriculo = curriculo
        self.cada = cada
        self.conservar = conservar
        self._proximo = cada
        directorio.mkdir(parents=True, exist_ok=True)

    def guardar(self, etiqueta: str | None = None) -> Path:
        """
        Guarda ultimo.zip, el estado del curriculo y, si se da etiqueta, un checkpoint numerado.

        Contexto:
            _on_step y el cierre del entrenamiento.

        Args:
            etiqueta (str, opcional): Sufijo del checkpoint numerado (p. ej. los pasos).

        Returns:
            Path: Ruta de ultimo.zip.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            callback.guardar("100000")
        """
        ultimo = self.directorio / "ultimo.zip"
        self.model.save(ultimo)
        guardar_estado_curriculo(self.curriculo, self.directorio / "curriculo.json")
        if etiqueta is not None:
            self.model.save(self.directorio / f"modelo_{etiqueta}.zip")
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
            self.guardar(str(self.num_timesteps))
            self._proximo = (self.num_timesteps // self.cada + 1) * self.cada
        return True


class CallbackMejorModelo(BaseCallback):
    """
    Guarda mejor.zip cuando mejora el rendimiento en el nivel de curriculo mas alto alcanzado.

    Contexto:
        El rendimiento no es comparable entre niveles (el jefe cambia de vida), asi que el
        orden es lexicografico: nivel de curriculo, tasa de victoria y progreso contra el jefe
        de los ultimos episodios del nivel.
    """

    def __init__(self, ruta: Path, curriculo: Curriculo, ventana: int = 100, minimo_episodios: int = 50) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            ruta (Path): Archivo mejor.zip de destino.
            curriculo (Curriculo): Para conocer el nivel vigente.
            ventana (int): Episodios recientes que se promedian.
            minimo_episodios (int): Episodios minimos del nivel antes de evaluar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            callback = CallbackMejorModelo(Path("mejor.zip"), curriculo)
        """
        super().__init__()
        self.ruta = ruta
        self.curriculo = curriculo
        self.minimo_episodios = minimo_episodios
        self._episodios: deque[dict[str, Any]] = deque(maxlen=ventana)
        self._nivel = curriculo.nivel
        self.mejor: tuple[int, float, float] | None = None

    def _on_step(self) -> bool:
        """
        Acumula los episodios del nivel vigente.

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
        if self.curriculo.nivel != self._nivel:
            self._nivel = self.curriculo.nivel
            self._episodios.clear()
        for resumen in episodios_terminados(self.locals):
            if resumen.get("resultado") != RESULTADO_ERROR and resumen["fase_inicio"] == 1:
                self._episodios.append(resumen)
        return True

    def _on_rollout_end(self) -> None:
        """
        Guarda mejor.zip si la ventana actual supera al mejor registrado.

        Contexto:
            SB3 lo llama al terminar cada rollout.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Si no se puede escribir.

        Ejemplo:
            (llamado por SB3)
        """
        if len(self._episodios) < self.minimo_episodios:
            return
        puntaje = (
            self._nivel,
            statistics.fmean(float(e["victoria"]) for e in self._episodios),
            statistics.fmean(float(e["progreso_jefe"]) for e in self._episodios),
        )
        if self.mejor is None or puntaje > self.mejor:
            self.mejor = puntaje
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ruta)


class CallbackParada(BaseCallback):
    """
    Detiene el entrenamiento de forma ordenada cuando aparece un archivo de parada.

    Contexto:
        detener.ps1 crea el archivo; asi el modelo se guarda y los juegos se cierran con su
        orden de cierre en lugar de matarlos.
    """

    def __init__(self, archivo: Path, cada: int = 200) -> None:
        """
        Crea el callback.

        Contexto:
            Construccion del entrenamiento.

        Args:
            archivo (Path): Archivo cuya existencia pide parar.
            cada (int): Pasos entre comprobaciones del disco.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            callback = CallbackParada(Path("runs/a/PARAR"))
        """
        super().__init__()
        self.archivo = archivo
        self.cada = cada
        self._n = 0

    def _on_step(self) -> bool:
        """
        Devuelve False (parar) si el archivo de parada existe.

        Contexto:
            SB3 lo llama tras cada paso de los entornos; False termina learn().

        Args:
            Ninguno.

        Returns:
            bool: False para parar, True para seguir.

        Raises:
            Ninguna.

        Ejemplo:
            (llamado por SB3)
        """
        self._n += 1
        if self._n % self.cada:
            return True
        return not self.archivo.exists()
