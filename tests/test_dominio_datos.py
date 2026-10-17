"""Tabla de golpes, vocabulario de montages, configuracion y especificacion de la observacion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dmrl.dominio import acciones as acc
from dmrl.dominio.configuracion import ConfigEntorno
from dmrl.dominio.observacion import (
    CAMPOS,
    TAMANO_SIMPLE,
    ApiladorObservaciones,
    ConstructorObservacion,
    especificacion_markdown,
)
from dmrl.dominio.tabla_golpes import TablaGolpes
from dmrl.dominio.vocabulario import (
    INDICE_NINGUNO,
    VOCAB_JEFE,
    VOCAB_JUGADOR,
    indice_montage,
    tamano_codificacion,
)
from dmrl.infraestructura.datos import ARCHIVO_TABLA_GOLPES, CARPETA_DATOS, cargar_tabla_golpes
from tests.support_dmrl import jefe, jugador, montage, obs

UPPERCUT = "Boss_Attack_Uppercut_InP_Montage"


def tabla_de_prueba() -> TablaGolpes:
    """
    Tabla con un golpe parable, uno no parable y un combo.

    Contexto:
        Base de las pruebas de consulta.

    Args:
        Ninguno.

    Returns:
        TablaGolpes: Tabla de tres montages.

    Raises:
        Ninguna.

    Ejemplo:
        tabla_de_prueba().montages()
    """
    return TablaGolpes.desde_dict({
        UPPERCUT: [{"pos": 0.9, "parry": True, "dano": 10}],
        "Salto": [{"pos": 0.6, "min": 0.55, "max": 0.65, "parry": False, "dano": 35}],
        "Combo": [{"pos": 1.0, "parry": True}, {"pos": 0.5, "parry": True}, {"pos": 2.0, "parry": False}],
    })


def test_tabla_ordena_los_golpes_por_posicion() -> None:
    """Los golpes de un montage quedan en orden aunque el archivo no lo este."""
    assert [g.pos for g in tabla_de_prueba().golpes("Combo")] == [0.5, 1.0, 2.0]
    assert tabla_de_prueba().montages() == (UPPERCUT, "Combo", "Salto")


def test_tabla_valores_por_defecto_de_minimo_maximo_y_dano() -> None:
    """Si faltan min, max o dano se toma la posicion y cero."""
    g = tabla_de_prueba().golpes("Combo")[0]
    assert (g.minimo, g.maximo, g.danio) == (0.5, 0.5, 0.0)


def test_proximo_golpe_aplica_el_play_rate() -> None:
    """Faltan (0.9 - 0.5) / 2.0 = 0.2 s de juego a rate 2.0."""
    p = tabla_de_prueba().proximo_golpe(UPPERCUT, 0.5, 2.0)
    assert p is not None and p.segundos == pytest.approx(0.2) and p.indice == 0


def test_proximo_golpe_ignora_los_ya_ocurridos() -> None:
    """Pasado el ultimo golpe no queda ninguno; entre dos, devuelve el siguiente."""
    tabla = tabla_de_prueba()
    assert tabla.proximo_golpe(UPPERCUT, 0.95, 1.0) is None
    siguiente = tabla.proximo_golpe("Combo", 0.7, 1.0)
    assert siguiente is not None and siguiente.golpe.pos == 1.0 and siguiente.indice == 1


def test_proximo_golpe_solo_parry_salta_los_no_parables() -> None:
    """Con solo_parry el golpe no parable del combo se omite."""
    tabla = tabla_de_prueba()
    assert tabla.proximo_golpe("Salto", 0.0, 1.0, solo_parry=True) is None
    assert tabla.proximo_golpe("Combo", 1.5, 1.0, solo_parry=True) is None
    assert tabla.proximo_golpe("Combo", 1.5, 1.0) is not None


def test_proximo_golpe_con_rate_no_positivo_usa_velocidad_uno() -> None:
    """Un play rate de 0 no divide por cero."""
    p = tabla_de_prueba().proximo_golpe(UPPERCUT, 0.4, 0.0)
    assert p is not None and p.segundos == pytest.approx(0.5)


def test_montage_desconocido_o_nulo_no_tiene_golpes() -> None:
    """Montages sin ataque (o sin montage) devuelven vacio."""
    tabla = tabla_de_prueba()
    assert tabla.golpes("ANIM_CARPET_Montage") == ()
    assert tabla.golpes(None) == ()
    assert TablaGolpes().montages() == ()


def test_la_tabla_medida_carga_y_cubre_los_ataques_conocidos() -> None:
    """El archivo data/tabla_golpes.json trae los golpes medidos del jefe."""
    tabla = cargar_tabla_golpes()
    assert UPPERCUT in tabla.montages()
    assert any(not g.parry for g in tabla.golpes("Boss_Attack_ChargedCombo_InP_Montage"))
    assert len(tabla.golpes("Anim_SwordV2_Combo03_01_RM_Montage1")) >= 5


def test_cargar_tabla_rechaza_archivos_mal_formados(tmp_path: Path) -> None:
    """Un JSON que no es objeto, o con golpes sin posicion, es un error claro."""
    malo = tmp_path / "t.json"
    malo.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_tabla_golpes(malo)
    malo.write_text(json.dumps({"M": [{"parry": True}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_tabla_golpes(malo)
    with pytest.raises(FileNotFoundError):
        cargar_tabla_golpes(tmp_path / "no_existe.json")


def test_vocabulario_codifica_ninguno_conocido_y_otro() -> None:
    """0 = sin montage, 1..n = vocabulario, n+1 = desconocido."""
    assert indice_montage(None, VOCAB_JEFE) == INDICE_NINGUNO
    assert indice_montage("", VOCAB_JEFE) == INDICE_NINGUNO
    assert indice_montage(VOCAB_JEFE[0], VOCAB_JEFE) == 1
    assert indice_montage("MontageInventado", VOCAB_JEFE) == len(VOCAB_JEFE) + 1
    assert tamano_codificacion(VOCAB_JEFE) == len(VOCAB_JEFE) + 2


def test_el_vocabulario_cubre_todos_los_montages_observados_en_el_juego() -> None:
    """Los montages medidos en 250 episodios reales estan todos en los vocabularios."""
    vistos = json.loads((CARPETA_DATOS / "vocabulario_montages.json").read_text(encoding="utf-8"))
    assert set(vistos["player"]) <= set(VOCAB_JUGADOR)
    assert set(vistos["boss"]) <= set(VOCAB_JEFE)
    assert len(set(VOCAB_JEFE)) == len(VOCAB_JEFE) and len(set(VOCAB_JUGADOR)) == len(VOCAB_JUGADOR)
    assert ARCHIVO_TABLA_GOLPES.exists()


def test_los_montages_con_golpes_estan_en_el_vocabulario_del_jefe() -> None:
    """Todo ataque medido tiene su propia posicion en el one-hot."""
    for nombre in cargar_tabla_golpes().montages():
        assert nombre in VOCAB_JEFE


def test_configuracion_deriva_segundos_y_pasos_por_decision() -> None:
    """2 frames a 30 Hz = 1/15 s; 360 s = 5400 decisiones."""
    cfg = ConfigEntorno()
    assert cfg.segundos_por_paso == pytest.approx(1 / 15)
    assert cfg.pasos_maximos == 5400
    assert cfg.a_dict()["recompensa"]["victoria"] == 10.0


def test_campos_tiene_un_nombre_unico_por_indice() -> None:
    """La especificacion y el vector tienen el mismo tamano y nombres sin repetir."""
    nombres = [n for n, _ in CAMPOS]
    assert len(nombres) == TAMANO_SIMPLE == len(set(nombres))


def test_observacion_tiene_el_tamano_y_los_rangos_de_la_especificacion() -> None:
    """Vector float32 de TAMANO_SIMPLE valores, todos en [-1, 1]."""
    v = ConstructorObservacion(tabla_de_prueba()).construir(obs(), acc.ACCION_ATAQUE, 12.0)
    assert v.dtype == np.float32 and v.shape == (TAMANO_SIMPLE,)
    assert v.min() >= -1.0 and v.max() <= 1.0


def indice(nombre: str) -> int:
    """
    Posicion de un campo en el vector.

    Contexto:
        Pruebas de la observacion.

    Args:
        nombre (str): Nombre en CAMPOS.

    Returns:
        int: Indice del campo.

    Raises:
        ValueError: Si el campo no existe.

    Ejemplo:
        indice("jugador_vida")
    """
    return [n for n, _ in CAMPOS].index(nombre)


def construir(jug: dict[str, object] | None = None, jf: dict[str, object] | None = None, dist: float = 300.0,
              previa: int = 0, segundos: float = 0.0) -> np.ndarray:
    """
    Construye un vector con la tabla de prueba.

    Contexto:
        Atajo de las pruebas de la observacion.

    Args:
        jug (dict, opcional): Bloque del jugador.
        jf (dict, opcional): Bloque del jefe.
        dist (float): Distancia.
        previa (int): Accion previa.
        segundos (float): Tiempo del episodio.

    Returns:
        np.ndarray: Vector de observacion simple.

    Raises:
        Ninguna.

    Ejemplo:
        construir(jugador(hp=50.0))
    """
    return ConstructorObservacion(tabla_de_prueba()).construir(obs(jug, jf, dist), previa, segundos)


def test_observacion_jugador_normaliza_vida_caguamas_y_poder() -> None:
    """vida/vida_max, caguamas/3 y poder 0..1."""
    v = construir(jugador(hp=40.0, caguamas=3, power=0.5))
    assert v[indice("jugador_vida")] == pytest.approx(0.4)
    assert v[indice("jugador_caguamas")] == pytest.approx(1.0)
    assert v[indice("jugador_poder")] == pytest.approx(0.5)


def test_observacion_bebida_marca_bebiendo_y_su_progreso() -> None:
    """Con el montage de la bebida a la mitad, progreso 0.5 y one-hot del montage."""
    v = construir(jugador(montage=montage("MM_Idle1_Montage", 1.0, largo=2.0)))
    assert v[indice("jugador_bebiendo")] == 1.0
    assert v[indice("jugador_progreso_bebida")] == pytest.approx(0.5)
    assert v[indice("jugador_montage_MM_Idle1_Montage")] == 1.0


def test_observacion_reaccion_a_golpe_por_prefijo_del_montage() -> None:
    """Los montages Hit_* y Hurt_* marcan reaccion a golpe."""
    for nombre in ("Anim_SwordV2_Hit_Small_F_IP_Montage", "Anim_Sword_Hurt_Big_F_Root_Montage"):
        assert construir(jugador(montage=montage(nombre, 0.1)))[indice("jugador_reaccion_golpe")] == 1.0
    assert construir()[indice("jugador_reaccion_golpe")] == 0.0


def test_observacion_montage_desconocido_va_a_la_posicion_otro() -> None:
    """Un montage fuera del vocabulario activa 'otro' y no 'ninguno'."""
    v = construir(jugador(montage=montage("Inventado", 0.1)))
    assert v[indice("jugador_montage_otro")] == 1.0
    assert v[indice("jugador_montage_ninguno")] == 0.0
    assert construir()[indice("jugador_montage_ninguno")] == 1.0


def test_observacion_jefe_fase_y_tiempo_al_proximo_golpe() -> None:
    """Uppercut con el golpe a 0.9 y montage en 0.5: faltan 0.4 s (0.4/1.5), parable."""
    v = construir(jf=jefe(montage=montage(UPPERCUT, 0.5, largo=4.0), attacking=True))
    assert v[indice("jefe_fase1")] == 1.0 and v[indice("jefe_fase2")] == 0.0
    assert v[indice("jefe_golpe_pendiente")] == 1.0
    assert v[indice("jefe_golpe_tiempo")] == pytest.approx(0.4 / 1.5)
    assert v[indice("jefe_golpe_parable")] == 1.0
    assert v[indice("jefe_parry_tiempo")] == pytest.approx(0.4 / 1.5)
    assert v[indice(f"jefe_montage_{UPPERCUT}")] == 1.0
    assert v[indice("jefe_montage_restante")] == pytest.approx(3.5 / 4.0)


def test_observacion_golpe_no_parable_no_activa_la_senal_de_parry() -> None:
    """Con un golpe no parable pendiente, el tiempo de parry queda en 1 (no hay)."""
    v = construir(jf=jefe(montage=montage("Salto", 0.2, largo=1.0)))
    assert v[indice("jefe_golpe_pendiente")] == 1.0 and v[indice("jefe_golpe_parable")] == 0.0
    assert v[indice("jefe_parry_tiempo")] == 1.0


def test_observacion_sin_golpes_pendientes_deja_los_tiempos_en_uno() -> None:
    """Sin montage del jefe no hay golpe pendiente."""
    v = construir()
    assert v[indice("jefe_golpe_pendiente")] == 0.0
    assert v[indice("jefe_golpe_tiempo")] == 1.0


def test_observacion_jefe_de_fase_2_y_aturdido() -> None:
    """Fase 2 y aturdimiento se reflejan en sus indicadores."""
    v = construir(jf=jefe(phase=1, stunned=True, hp=500.0, hp_max=1000.0))
    assert v[indice("jefe_fase2")] == 1.0 and v[indice("jefe_aturdido")] == 1.0
    assert v[indice("jefe_vida")] == pytest.approx(0.5)


def test_observacion_geometria_relativa_jugador_jefe() -> None:
    """Jugador mirando a +X con el jefe en +X: de frente (cos=1) y el jefe mirando hacia el (cos=1)."""
    v = construir(jugador(pos=[0.0, 0.0, 0.0], yaw=0.0), jefe(pos=[300.0, 0.0, 0.0], yaw=180.0), dist=300.0)
    assert v[indice("rel_jefe_cos")] == pytest.approx(1.0)
    assert v[indice("rel_jefe_sen")] == pytest.approx(0.0, abs=1e-6)
    assert v[indice("rel_mirada_cos")] == pytest.approx(1.0)
    assert v[indice("rel_distancia")] == pytest.approx(300.0 / 2000.0)


def test_observacion_jefe_a_la_derecha_y_mirando_a_otro_lado() -> None:
    """Jefe en +Y con el jugador mirando +X: seno +1; si el jefe mira en su propio +X, no lo mira."""
    v = construir(jugador(pos=[0.0, 0.0, 0.0], yaw=0.0), jefe(pos=[0.0, 300.0, 0.0], yaw=0.0))
    assert v[indice("rel_jefe_sen")] == pytest.approx(1.0)
    assert v[indice("rel_mirada_cos")] == pytest.approx(0.0, abs=1e-6)


def test_observacion_sin_bloques_queda_en_ceros_salvo_los_indicadores_ninguno() -> None:
    """Sin jugador ni jefe la geometria es cero y los montages caen en 'ninguno'."""
    v = ConstructorObservacion(TablaGolpes()).construir({}, 0, 0.0)
    assert v[indice("rel_distancia")] == 0.0 and v[indice("jugador_vida")] == 0.0
    assert v[indice("jefe_golpe_tiempo")] == 1.0


def test_observacion_accion_previa_one_hot_y_tiempo_normalizado() -> None:
    """La accion previa activa su posicion y 180 s son la mitad del episodio."""
    v = construir(previa=acc.ACCION_CAGUAMA, segundos=180.0)
    assert v[indice("accion_previa_caguama")] == 1.0 and v[indice("accion_previa_nada")] == 0.0
    assert v[indice("tiempo_episodio")] == pytest.approx(0.5)


def test_observacion_acota_valores_fuera_de_rango() -> None:
    """Una velocidad o distancia enormes se recortan a 1."""
    v = construir(jugador(vel=[5000.0, 0.0, 0.0]), dist=99999.0)
    assert v[indice("jugador_velocidad")] == 1.0 and v[indice("rel_distancia")] == 1.0


def test_apilador_llena_la_ventana_con_la_primera_y_desplaza() -> None:
    """Al iniciar repite la primera; cada agregado descarta la mas vieja."""
    a = ApiladorObservaciones(3, 3)
    primero = a.iniciar(np.array([1, 1, 1], dtype=np.float32))
    assert primero.tolist() == [1.0] * 9
    despues = a.agregar(np.array([2, 2, 2], dtype=np.float32))
    assert despues.tolist() == [1.0] * 6 + [2.0] * 3
    assert a.agregar(np.array([3, 3, 3], dtype=np.float32)).tolist() == [1.0] * 3 + [2.0] * 3 + [3.0] * 3


def test_apilador_vacio_devuelve_ceros_y_rechaza_formas_incorrectas() -> None:
    """Antes de iniciar la ventana es de ceros; una forma distinta se rechaza."""
    a = ApiladorObservaciones(2, 2)
    assert a.apilado().tolist() == [0.0] * 4
    with pytest.raises(ValueError):
        a.agregar(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        a.iniciar(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        ApiladorObservaciones(2, 0)


def test_especificacion_documenta_cada_indice() -> None:
    """El Markdown generado lista todos los indices y los vocabularios."""
    texto = especificacion_markdown(4)
    assert f"| {TAMANO_SIMPLE - 1} | tiempo_episodio |" in texto
    assert f"{TAMANO_SIMPLE * 4} valores" in texto
    assert VOCAB_JEFE[0] in texto and VOCAB_JUGADOR[0] in texto
    assert texto.count("\n| ") >= TAMANO_SIMPLE
