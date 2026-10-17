"""
Reglas puras del PPO multi-jefe que parte de la politica clonada del bot.

Una politica imitada ya juega al nivel del bot; PPO mal calibrado puede destruirla en las primeras
actualizaciones porque el critico no conoce el nuevo problema. Estas reglas cubren tres defensas:

* el calentamiento y la penalizacion KL hacia la politica clonada (coeficiente_kl);
* una compuerta de promocion: un checkpoint solo reemplaza al mejor si mejora la tasa de victoria
  media y ningun jefe cae mas de lo tolerado respecto a la imitacion (evaluar_checkpoint);
* una regla de regresion: si el rendimiento medio queda claramente por debajo de la imitacion en
  evaluaciones seguidas, se vuelve al ultimo mejor modelo y se baja la tasa de aprendizaje
  (ControlRegresion).

Todo es calculo sobre numeros: no abre el juego, no lee archivos y no usa torch.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Una tasa de victoria se compara con esta holgura numerica para que 0.10 de caida exacta cuente
# como tolerada pese al error de coma flotante.
EPSILON_COMPARACION = 1e-9


@dataclass(frozen=True)
class ConfigPPOJefes:
    """
    Hiperparametros y reglas de seguridad del PPO multi-jefe.

    Contexto:
        CLI entrenar-jefes, callbacks de evaluacion y el modelo con penalizacion KL.

    Atributos:
        nombre: carpeta de la corrida dentro de runs/.
        instancias: instancias del juego (se reparten entre mapas segun el numero de jefes).
        pasos_totales: techo de pasos; se puede parar antes.
        n_steps: pasos por entorno en cada rollout.
        batch_size: tamano de minibatch.
        n_epochs: pasadas por rollout.
        learning_rate: tasa de aprendizaje inicial (baja para no borrar lo clonado).
        gamma: descuento.
        gae_lambda: lambda de la ventaja generalizada.
        clip_range: recorte de PPO.
        ent_coef: coeficiente de entropia (bajo: la politica clonada ya es casi determinista).
        vf_coef: peso de la perdida del valor.
        max_grad_norm: recorte del gradiente.
        target_kl: parada anticipada de las epocas de una actualizacion si el KL aproximado la supera.
        pasos_calentamiento: pasos iniciales en que solo se ajusta la cabeza de valor.
        beta_kl: coeficiente inicial de la penalizacion KL hacia la politica clonada.
        pasos_decaimiento_kl: pasos tras el calentamiento en que beta_kl baja linealmente a cero.
        checkpoint_cada: pasos entre guardados de ultimo.zip.
        minutos_evaluacion: minutos de reloj entre evaluaciones automaticas en juego.
        minutos_primera_evaluacion: minutos hasta la primera evaluacion de una corrida que aun no tiene
            ninguna; mas corto que el intervalo normal para detectar pronto un colapso al empezar.
        episodios_evaluacion: peleas por jefe en cada evaluacion.
        caida_maxima_por_jefe: caida de tasa tolerada de cada jefe respecto a la imitacion.
        caida_clara_media: caida de la tasa media respecto a la imitacion que cuenta como regresion.
        evaluaciones_para_revertir: evaluaciones seguidas en regresion que disparan la correccion.
        factor_reduccion_lr: factor con que se multiplica la tasa de aprendizaje al corregir.
        lr_minimo: la tasa de aprendizaje nunca baja de este valor.
        semilla: semilla base.
        dispositivo: "cpu" o "cuda".
        hilos_torch: hilos de CPU de PyTorch en el proceso principal.
        trazar_golpes: si los entornos registran cada golpe recibido para publicar de que ataque muere
            el jugador (cuesta algo de velocidad por paso).
    """

    nombre: str = "multijefe_ppo"
    instancias: int = 6
    pasos_totales: int = 200_000_000
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 5
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    pasos_calentamiento: int = 200_000
    beta_kl: float = 0.3
    pasos_decaimiento_kl: int = 5_000_000
    checkpoint_cada: int = 100_000
    minutos_evaluacion: float = 45.0
    minutos_primera_evaluacion: float = 20.0
    episodios_evaluacion: int = 20
    caida_maxima_por_jefe: float = 0.10
    caida_clara_media: float = 0.10
    evaluaciones_para_revertir: int = 2
    factor_reduccion_lr: float = 0.5
    lr_minimo: float = 1e-5
    semilla: int = 0
    dispositivo: str = "cpu"
    hilos_torch: int = 2
    trazar_golpes: bool = True

    def __post_init__(self) -> None:
        """
        Valida que los parametros tengan sentido.

        Contexto:
            Construccion del dataclass; un valor absurdo en la configuracion de una corrida nocturna
            se descubriria horas despues.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si una tasa o coeficiente es no positivo/negativo, o una probabilidad o
                caida esta fuera de [0, 1], o los contadores son menores que 1.

        Ejemplo:
            ConfigPPOJefes(learning_rate=1e-4)
        """
        if self.learning_rate <= 0 or self.lr_minimo <= 0 or self.lr_minimo > self.learning_rate:
            raise ValueError("learning_rate y lr_minimo deben ser positivos y lr_minimo <= learning_rate")
        if not 0.0 < self.factor_reduccion_lr < 1.0:
            raise ValueError("factor_reduccion_lr debe estar en (0, 1)")
        if self.beta_kl < 0 or self.pasos_calentamiento < 0 or self.pasos_decaimiento_kl < 0:
            raise ValueError("beta_kl y los pasos de calentamiento/decaimiento no pueden ser negativos")
        if not (0.0 <= self.caida_maxima_por_jefe <= 1.0 and 0.0 <= self.caida_clara_media <= 1.0):
            raise ValueError("las caidas toleradas deben estar en [0, 1]")
        if self.evaluaciones_para_revertir < 1 or self.episodios_evaluacion < 1:
            raise ValueError("evaluaciones_para_revertir y episodios_evaluacion deben ser > 0")
        if self.minutos_evaluacion <= 0 or self.minutos_primera_evaluacion <= 0:
            raise ValueError("minutos_evaluacion y minutos_primera_evaluacion deben ser > 0")


def coeficiente_kl(pasos: int, config: ConfigPPOJefes) -> float:
    """
    Coeficiente de la penalizacion KL hacia la politica clonada en un instante del entrenamiento.

    Contexto:
        MaskablePPOKL.train, en cada actualizacion. Durante el calentamiento la politica esta
        congelada y la penalizacion no se usa; despues arranca en beta_kl y baja linealmente a cero
        en pasos_decaimiento_kl: PPO queda libre cuando el critico ya es fiable.

    Args:
        pasos (int): Pasos de entrenamiento acumulados.
        config (ConfigPPOJefes): Parametros de la corrida.

    Returns:
        float: Coeficiente en [0, beta_kl]; 0 durante el calentamiento y pasado el decaimiento.

    Raises:
        Ninguna.

    Ejemplo:
        coeficiente_kl(200_000, ConfigPPOJefes())  # 0.3 al terminar el calentamiento
    """
    if pasos < config.pasos_calentamiento or config.pasos_decaimiento_kl <= 0:
        return 0.0
    restante = 1.0 - (pasos - config.pasos_calentamiento) / config.pasos_decaimiento_kl
    return config.beta_kl * max(0.0, min(1.0, restante))


def en_calentamiento(pasos: int, config: ConfigPPOJefes) -> bool:
    """
    Indica si el critico aun se esta calentando con la politica congelada.

    Contexto:
        MaskablePPOKL.train.

    Args:
        pasos (int): Pasos de entrenamiento acumulados.
        config (ConfigPPOJefes): Parametros de la corrida.

    Returns:
        bool: True mientras pasos < pasos_calentamiento.

    Raises:
        Ninguna.

    Ejemplo:
        en_calentamiento(0, ConfigPPOJefes())  # True
    """
    return pasos < config.pasos_calentamiento


@dataclass(frozen=True)
class DecisionEvaluacion:
    """
    Veredicto sobre un checkpoint evaluado en juego.

    Contexto:
        Resultado de evaluar_checkpoint; lo consume el callback de evaluacion y se guarda en el
        historial de la corrida.

    Atributos:
        promover: True si el checkpoint pasa a ser el mejor.
        media: tasa de victoria media entre jefes del checkpoint.
        mejor_media: tasa media del mejor modelo vigente antes de esta evaluacion.
        caidas: por jefe, tasa de la imitacion menos tasa del checkpoint (positivo = peor).
        jefes_que_caen: jefes cuya caida supera la tolerancia.
        regresion: True si la media queda claramente por debajo de la imitacion.
        motivo: texto corto con la razon del veredicto.
    """

    promover: bool
    media: float
    mejor_media: float
    caidas: dict[str, float]
    jefes_que_caen: tuple[str, ...]
    regresion: bool
    motivo: str


def tasa_media(tasas: Mapping[str, float]) -> float:
    """
    Tasa de victoria media entre jefes (cada jefe pesa lo mismo).

    Contexto:
        Criterio de promocion: un jefe facil con muchas peleas no debe compensar a uno dificil.

    Args:
        tasas (Mapping[str, float]): Jefe -> tasa de victoria.

    Returns:
        float: Media aritmetica de las tasas.

    Raises:
        ValueError: Si no hay ningun jefe.

    Ejemplo:
        tasa_media({"amlo": 0.9, "nahual": 1.0})  # 0.95
    """
    if not tasas:
        raise ValueError("no hay tasas que promediar")
    return statistics.fmean(tasas.values())


def evaluar_checkpoint(tasas: Mapping[str, float], referencia: Mapping[str, float], mejor_media: float,
                       config: ConfigPPOJefes) -> DecisionEvaluacion:
    """
    Decide si un checkpoint evaluado reemplaza al mejor y si cuenta como regresion.

    Contexto:
        Fin de cada evaluacion automatica. Promueve solo si la tasa media supera a la del mejor
        modelo vigente Y ningun jefe cae mas de caida_maxima_por_jefe respecto a la imitacion: una
        mejora media lograda sacrificando a un jefe no es una mejora.

    Args:
        tasas (Mapping[str, float]): Jefe -> tasa de victoria del checkpoint.
        referencia (Mapping[str, float]): Jefe -> tasa de victoria de la imitacion.
        mejor_media (float): Tasa media del mejor modelo vigente.
        config (ConfigPPOJefes): Tolerancias.

    Returns:
        DecisionEvaluacion: Veredicto con el desglose por jefe.

    Raises:
        ValueError: Si el checkpoint no trae algun jefe de la referencia o no hay jefes.

    Ejemplo:
        evaluar_checkpoint({"amlo": 0.9}, {"amlo": 0.9}, 0.85, ConfigPPOJefes()).promover  # True
    """
    faltan = sorted(set(referencia) - set(tasas))
    if faltan:
        raise ValueError(f"la evaluacion no incluye los jefes {faltan}")
    media = tasa_media({j: tasas[j] for j in referencia})
    caidas = {j: referencia[j] - tasas[j] for j in referencia}
    que_caen = tuple(sorted(j for j, c in caidas.items() if c > config.caida_maxima_por_jefe + EPSILON_COMPARACION))
    regresion = media < tasa_media(referencia) - config.caida_clara_media - EPSILON_COMPARACION
    if que_caen:
        motivo = f"cae mas de {config.caida_maxima_por_jefe:.0%} en {', '.join(que_caen)}"
    elif media <= mejor_media:
        motivo = f"la media {media:.3f} no supera al mejor {mejor_media:.3f}"
    else:
        motivo = "mejora la media sin bajar ningun jefe"
    promover = not que_caen and media > mejor_media
    return DecisionEvaluacion(promover, media, mejor_media, caidas, que_caen, regresion, motivo)


@dataclass
class ControlRegresion:
    """
    Cuenta evaluaciones seguidas en regresion y dispara la correccion.

    Contexto:
        Callback de evaluacion automatica. La correccion es volver al ultimo mejor modelo y bajar la
        tasa de aprendizaje; el contador se reinicia al corregir para dar a la nueva tasa su propia
        ventana de evaluacion.

    Atributos:
        config: tolerancias y numero de evaluaciones seguidas.
        seguidas: evaluaciones consecutivas en regresion hasta ahora.
    """

    config: ConfigPPOJefes
    seguidas: int = 0
    historial: list[bool] = field(default_factory=list)

    def registrar(self, decision: DecisionEvaluacion) -> bool:
        """
        Anota una evaluacion y dice si hay que corregir.

        Contexto:
            Callback de evaluacion, tras evaluar_checkpoint.

        Args:
            decision (DecisionEvaluacion): Veredicto de la evaluacion.

        Returns:
            bool: True si ya hubo evaluaciones_para_revertir regresiones seguidas (y el contador
                se reinicia).

        Raises:
            Ninguna.

        Ejemplo:
            control.registrar(decision)
        """
        self.historial.append(decision.regresion)
        self.seguidas = self.seguidas + 1 if decision.regresion else 0
        if self.seguidas >= self.config.evaluaciones_para_revertir:
            self.seguidas = 0
            return True
        return False


def tasa_aprendizaje_corregida(actual: float, config: ConfigPPOJefes) -> float:
    """
    Tasa de aprendizaje tras una correccion por regresion.

    Contexto:
        Callback de evaluacion, al revertir al mejor modelo.

    Args:
        actual (float): Tasa vigente.
        config (ConfigPPOJefes): Factor de reduccion y minimo.

    Returns:
        float: actual * factor_reduccion_lr, sin bajar de lr_minimo.

    Raises:
        Ninguna.

    Ejemplo:
        tasa_aprendizaje_corregida(1e-4, ConfigPPOJefes())  # 5e-5
    """
    return max(config.lr_minimo, actual * config.factor_reduccion_lr)


@dataclass
class EstadoEvaluacion:
    """
    Estado persistente de la evaluacion automatica y de la correccion por regresion.

    Contexto:
        Se guarda junto a los checkpoints para que reanudar el entrenamiento conserve la tasa de
        aprendizaje corregida, el mejor modelo y el recuento de regresiones.

    Atributos:
        referencia: jefe -> tasa de victoria de la imitacion (piso de la compuerta de promocion).
        mejor_media: tasa media del mejor modelo vigente.
        mejor_paso: pasos de entrenamiento del mejor modelo (0 = la imitacion).
        learning_rate: tasa de aprendizaje vigente.
        seguidas: evaluaciones seguidas en regresion.
        promociones: checkpoints que pasaron a ser el mejor.
        reversiones: veces que se volvio al mejor modelo.
        evaluaciones: veredicto resumido de cada evaluacion (pasos, tasas, promovido, motivo).
    """

    referencia: dict[str, float]
    mejor_media: float
    mejor_paso: int = 0
    learning_rate: float = 1e-4
    seguidas: int = 0
    promociones: int = 0
    reversiones: int = 0
    evaluaciones: list[dict[str, object]] = field(default_factory=list)

    def a_dict(self) -> dict[str, object]:
        """
        Representacion serializable a JSON.

        Contexto:
            Guardado junto al checkpoint.

        Args:
            Ninguno.

        Returns:
            dict[str, object]: Todos los campos con tipos JSON.

        Raises:
            Ninguna.

        Ejemplo:
            EstadoEvaluacion({"amlo": 0.9}, 0.9).a_dict()["mejor_media"]
        """
        return {
            "referencia": dict(self.referencia), "mejor_media": self.mejor_media, "mejor_paso": self.mejor_paso,
            "learning_rate": self.learning_rate, "seguidas": self.seguidas, "promociones": self.promociones,
            "reversiones": self.reversiones, "evaluaciones": list(self.evaluaciones),
        }

    @staticmethod
    def desde_dict(datos: Mapping[str, object], por_defecto: EstadoEvaluacion) -> EstadoEvaluacion:
        """
        Reconstruye el estado desde un JSON, tolerando archivos incompletos o danados.

        Contexto:
            Reanudacion del entrenamiento: un estado ilegible no debe impedir seguir entrenando.

        Args:
            datos (Mapping[str, object]): Contenido leido del JSON (puede estar vacio).
            por_defecto (EstadoEvaluacion): Estado a usar para lo que falte o no sea valido.

        Returns:
            EstadoEvaluacion: Estado restaurado.

        Raises:
            Ninguna.

        Ejemplo:
            EstadoEvaluacion.desde_dict({}, EstadoEvaluacion({"amlo": 0.9}, 0.9))
        """
        try:
            referencia = {str(j): float(t) for j, t in dict(datos["referencia"]).items()}  # type: ignore[call-overload]
            return EstadoEvaluacion(
                referencia=referencia, mejor_media=float(datos["mejor_media"]),  # type: ignore[arg-type]
                mejor_paso=int(datos.get("mejor_paso", 0)),  # type: ignore[call-overload]
                learning_rate=float(datos.get("learning_rate", por_defecto.learning_rate)),  # type: ignore[arg-type]
                seguidas=int(datos.get("seguidas", 0)),  # type: ignore[call-overload]
                promociones=int(datos.get("promociones", 0)),  # type: ignore[call-overload]
                reversiones=int(datos.get("reversiones", 0)),  # type: ignore[call-overload]
                evaluaciones=list(datos.get("evaluaciones", [])),  # type: ignore[call-overload]
            )
        except (KeyError, TypeError, ValueError):
            return por_defecto


def metricas_de_ventana(episodios: Sequence[Mapping[str, object]], fase_normal: int,
                        maximo_ataques: int = 5) -> dict[str, float]:
    """
    Metricas de TensorBoard de un jefe a partir de sus episodios recientes.

    Contexto:
        CallbackMetricasJefes, al final de cada rollout, una vez por jefe.

    Args:
        episodios (Sequence[Mapping[str, object]]): Resumenes recientes de episodios de un jefe.
        fase_normal (int): Fase de inicio de una pelea completa contra ese jefe.
        maximo_ataques (int): Cuantos ataques mortales distintos se publican (los mas frecuentes).

    Returns:
        dict[str, float]: tasa_victoria, tasa_victoria_inicio_normal, danio_al_jefe, danio_recibido,
            parries, duracion_s, progreso_jefe, recompensa, muertes y muerte_por/<ataque> (fraccion
            de los episodios de la ventana); vacio si no hay episodios.

    Raises:
        Ninguna.

    Ejemplo:
        metricas_de_ventana([{"victoria": True, "danio_hecho": 2000.0}], 1)["tasa_victoria"]
    """
    if not episodios:
        return {}
    n = len(episodios)

    def media(clave: str) -> float:
        """
        Media de un campo numerico entre los episodios.

        Contexto:
            metricas_de_ventana.

        Args:
            clave (str): Campo del resumen.

        Returns:
            float: Media; 0.0 si ningun episodio lo trae.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        valores = [float(e[clave]) for e in episodios if clave in e]  # type: ignore[arg-type]
        return statistics.fmean(valores) if valores else 0.0

    normales = [e for e in episodios if int(e.get("fase_inicio", fase_normal)) == fase_normal]  # type: ignore[call-overload]
    metricas = {
        "tasa_victoria": sum(1 for e in episodios if e.get("victoria")) / n,
        "tasa_victoria_inicio_normal": (sum(1 for e in normales if e.get("victoria")) / len(normales)
                                        if normales else 0.0),
        "danio_al_jefe": media("danio_hecho"), "danio_recibido": media("danio_recibido"),
        "parries": media("parries"), "duracion_s": media("duracion_s"), "progreso_jefe": media("progreso_jefe"),
        "recompensa": media("recompensa"), "muertes": sum(1 for e in episodios if e.get("muerte")) / n,
    }
    conteo: dict[str, int] = {}
    for e in episodios:
        if e.get("muerte") and e.get("golpe_mortal"):
            conteo[str(e["golpe_mortal"])] = conteo.get(str(e["golpe_mortal"]), 0) + 1
    for ataque, veces in sorted(conteo.items(), key=lambda par: (-par[1], par[0]))[:maximo_ataques]:
        limpio = "".join(c if c.isalnum() else "_" for c in ataque)
        metricas[f"muerte_por/{limpio}"] = veces / n
    return metricas
