"""Imitacion multi-jefe: rutas, carga de lotes, resumenes, medicion de dispositivos, ajuste y publicacion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dmrl.aplicacion import imitacion_jefes as imj
from dmrl.dominio import acciones as acc
from dmrl.dominio.imitacion import ConfigAprendizaje, Demostraciones, identificador_episodio
from dmrl.dominio.observacion_v2 import tamano_simple_v2
from dmrl.infraestructura.demostraciones import guardar_lote, leer_json
from tests.support_imitacion import regla_sintetica
from tests.support_jefes import catalogo_prueba

CPU = torch.device("cpu")
APRENDIZAJE_RAPIDO = ConfigAprendizaje(epocas_maximas=2, epocas_valor=1, tamano_lote=64, paciencia=5)


def demostraciones_de_jefes(episodios: int = 6, pasos: int = 12, ronda: int = 0, semilla: int = 0,
                            inicio: int = 0) -> Demostraciones:
    """
    Demostraciones sinteticas con la dimension de la observacion v2 del catalogo de prueba.

    Contexto:
        Datos del ajuste multi-jefe; la etiqueta es siempre valida segun su mascara.

    Args:
        episodios (int): Cantidad de episodios.
        pasos (int): Pasos por episodio.
        ronda (int): Ronda de los identificadores.
        semilla (int): Semilla del generador.
        inicio (int): Indice del primer episodio dentro de la ronda.

    Returns:
        Demostraciones: Conjunto con filas contiguas por episodio.

    Raises:
        Ninguna.

    Ejemplo:
        datos = demostraciones_de_jefes(episodios=4)
    """
    azar = np.random.default_rng(semilla)
    n = episodios * pasos
    dimension = tamano_simple_v2(catalogo_prueba()) * 4
    obs = azar.uniform(-1.0, 1.0, (n, dimension)).astype(np.float32)
    etiquetas = regla_sintetica(obs)
    mascaras = azar.random((n, acc.N_ACCIONES)) > 0.3
    mascaras[np.arange(n), etiquetas] = True
    mascaras[:, acc.ACCION_NADA] = True
    ids = np.repeat([identificador_episodio(ronda, inicio + i) for i in range(episodios)], pasos).astype(np.int64)
    return Demostraciones(obs, mascaras, etiquetas, etiquetas.copy(), ids, azar.normal(size=n).astype(np.float32))


def metadatos_de(ronda: int, jefe: str, episodios: int, inicio: int = 0, **extra: Any) -> list[dict[str, Any]]:
    """
    Metadatos de episodios de un jefe como los deja la recoleccion.

    Contexto:
        Acompanan a cada lote guardado.

    Args:
        ronda (int): Ronda.
        jefe (str): Jefe de los episodios.
        episodios (int): Cantidad.
        inicio (int): Indice del primer episodio.
        **extra (Any): Campos adicionales o a sobrescribir.

    Returns:
        list[dict[str, Any]]: Un diccionario por episodio.

    Raises:
        Ninguna.

    Ejemplo:
        metadatos_de(0, "amlo", 3)
    """
    base = {"jefe": jefe, "fase_inicio": 1, "epsilon": 0.0, "pasos": 12, "victoria": True}
    return [{**base, "id": identificador_episodio(ronda, inicio + i), "indice": inicio + i, **extra}
            for i in range(episodios)]


def escribir_conjunto(carpeta: Path) -> None:
    """
    Guarda dos rondas con dos jefes: amlo en la ronda 0 y nahual en la ronda 2.

    Contexto:
        Prepara los datos en disco de las pruebas de carga y ajuste.

    Args:
        carpeta (Path): Carpeta raiz de los datos.

    Returns:
        None: No retorna valor.

    Raises:
        Ninguna.

    Ejemplo:
        escribir_conjunto(tmp_path / "datos")
    """
    guardar_lote(carpeta / "ronda_00" / "w0_00000.npz", demostraciones_de_jefes(6, ronda=0, semilla=1),
                 metadatos_de(0, "amlo", 6))
    guardar_lote(carpeta / "ronda_02" / "w0_00000.npz", demostraciones_de_jefes(6, ronda=2, semilla=2),
                 metadatos_de(2, "nahual", 6, fase_inicio=2, epsilon=0.1, fase_max=2, victoria=False))


def test_las_rutas_cuelgan_de_la_carpeta_base() -> None:
    rutas = imj.RutasImitacionJefes(Path("base"))
    assert rutas.datos == Path("base/datos") and rutas.mejor == Path("base/mejor.zip")
    assert rutas.modelo("2x256") == Path("base/modelos/2x256.zip")
    assert rutas.informe("2x256") == Path("base/modelos/2x256.json")
    assert rutas.evaluacion("bot") == Path("base/evaluaciones/bot.json")
    assert imj.RutasImitacionJefes().base == imj.CARPETA_IMITACION_JEFES


def test_cargar_conjunto_une_todas_las_rondas_y_asigna_jefe_a_cada_episodio(tmp_path: Path) -> None:
    escribir_conjunto(tmp_path)
    conjunto = imj.cargar_conjunto_jefes(tmp_path)
    assert len(conjunto.demostraciones) == 12 * 12 and len(conjunto.metadatos) == 12
    por_jefe = {j: sum(1 for v in conjunto.jefe_de_episodio.values() if v == j) for j in ("amlo", "nahual")}
    assert por_jefe == {"amlo": 6, "nahual": 6}
    assert conjunto.jefe_de_episodio[identificador_episodio(2, 0)] == "nahual"
    conjunto.demostraciones.validar()


def test_cargar_conjunto_sin_lotes_falla_indicando_la_carpeta(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no hay demostraciones"):
        imj.cargar_conjunto_jefes(tmp_path)


def test_el_resumen_de_datos_cuenta_por_jefe_ruido_victorias_y_segunda_fase() -> None:
    metadatos = (metadatos_de(0, "amlo", 3, victoria=True)
                 + metadatos_de(0, "amlo", 1, inicio=3, victoria=False, epsilon=0.2, fase_max=2)
                 + metadatos_de(0, "nahual", 2, fase_inicio=2, pasos=5))
    resumen = imj.resumen_de_datos(metadatos)
    assert list(resumen) == ["amlo", "nahual"]
    assert resumen["amlo"] == {"peleas": 4.0, "pasos": 48.0, "victorias_bot": 3.0, "con_ruido": 1.0,
                               "inicio_fase2": 0.0, "pasaron_fase2": 1.0}
    assert resumen["nahual"]["inicio_fase2"] == 2.0 and resumen["nahual"]["pasos"] == 10.0
    assert imj.resumen_de_datos([]) == {}


def test_datos_por_jefe_en_disco_coincide_con_el_resumen_de_los_metadatos(tmp_path: Path) -> None:
    escribir_conjunto(tmp_path)
    en_disco = imj.datos_por_jefe_en_disco(tmp_path)
    assert en_disco == imj.resumen_de_datos(imj.cargar_conjunto_jefes(tmp_path).metadatos)
    assert imj.datos_por_jefe_en_disco(tmp_path / "vacia") == {}


def test_config_modelo_usa_las_capas_de_la_arquitectura_y_la_observacion_v2() -> None:
    catalogo = catalogo_prueba()
    config, tamano = imj.config_modelo(catalogo, "3x256", 5)
    assert config.capas == (256, 256, 256) and config.semilla == 5
    assert tamano == tamano_simple_v2(catalogo)
    with pytest.raises(ValueError, match="arquitectura desconocida"):
        imj.config_modelo(catalogo, "nope", 0)


def test_medir_dispositivos_en_cpu_estima_segundos_por_epoca_del_conjunto_completo() -> None:
    datos = demostraciones_de_jefes(4, 10)
    tiempos = imj.medir_dispositivos(catalogo_prueba(), "2x256", datos, APRENDIZAJE_RAPIDO, pasos_muestra=16,
                                     dispositivos=("cpu",))
    assert set(tiempos) == {"cpu"} and tiempos["cpu"] > 0.0


def test_medir_dispositivos_con_muestra_mayor_que_el_conjunto_usa_todo_el_conjunto() -> None:
    datos = demostraciones_de_jefes(2, 5)
    tiempos = imj.medir_dispositivos(catalogo_prueba(), "2x256", datos, APRENDIZAJE_RAPIDO, dispositivos=("cpu",))
    assert tiempos["cpu"] > 0.0


def test_medir_dispositivos_rechaza_un_conjunto_vacio() -> None:
    vacio = demostraciones_de_jefes(2, 5)
    vacio = vacio.seleccionar(np.zeros(len(vacio), dtype=bool))
    with pytest.raises(ValueError, match="no hay datos"):
        imj.medir_dispositivos(catalogo_prueba(), "2x256", vacio, APRENDIZAJE_RAPIDO, dispositivos=("cpu",))


def test_medir_dispositivos_omite_cuda_cuando_no_esta_disponible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    datos = demostraciones_de_jefes(2, 5)
    assert imj.medir_dispositivos(catalogo_prueba(), "2x256", datos, APRENDIZAJE_RAPIDO,
                                  dispositivos=("cuda",)) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere una GPU CUDA")
def test_medir_dispositivos_mide_cuda_si_esta_disponible() -> None:
    datos = demostraciones_de_jefes(2, 8)
    tiempos = imj.medir_dispositivos(catalogo_prueba(), "2x256", datos, APRENDIZAJE_RAPIDO, pasos_muestra=16,
                                     dispositivos=("cpu", "cuda"))
    assert set(tiempos) == {"cpu", "cuda"}


def test_elegir_dispositivo_prefiere_el_mas_rapido_y_cae_a_cpu_sin_mediciones() -> None:
    assert imj.elegir_dispositivo_mas_rapido({"cpu": 30.0, "cuda": 4.0}).type == "cuda"
    assert imj.elegir_dispositivo_mas_rapido({"cpu": 3.0, "cuda": 4.0}).type == "cpu"
    assert imj.elegir_dispositivo_mas_rapido({}) == CPU


def test_entrenar_arquitectura_guarda_el_modelo_y_un_informe_con_metricas_por_jefe(tmp_path: Path) -> None:
    datos = tmp_path / "datos"
    escribir_conjunto(datos)
    conjunto = imj.cargar_conjunto_jefes(datos)
    rutas = imj.RutasImitacionJefes(tmp_path)
    mensajes: list[str] = []
    informe = imj.entrenar_arquitectura(catalogo_prueba(), conjunto, "2x256", APRENDIZAJE_RAPIDO, 0, CPU, rutas,
                                        mensajes.append)
    assert rutas.modelo("2x256").is_file() and rutas.informe("2x256").is_file()
    assert informe["arquitectura"] == "2x256" and informe["dispositivo"] == "cpu"
    assert set(informe["por_jefe"]) == {"amlo", "nahual"}
    assert informe["coincidencia_tras_recargar"] == pytest.approx(1.0)
    assert informe["por_accion"] and "nombre" in informe["por_accion"][0]
    assert leer_json(rutas.informe("2x256"))["arquitectura"] == "2x256"
    assert any("epoca 1" in m for m in mensajes) and any("precision de validacion" in m for m in mensajes)


def test_entrenar_arquitectura_usa_print_como_registro_por_defecto(tmp_path: Path,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    datos = tmp_path / "datos"
    escribir_conjunto(datos)
    conjunto = imj.cargar_conjunto_jefes(datos)
    imj.entrenar_arquitectura(catalogo_prueba(), conjunto, "2x256", ConfigAprendizaje(epocas_maximas=1,
                              epocas_valor=1, tamano_lote=64), 1, CPU, imj.RutasImitacionJefes(tmp_path))
    assert "2x256" in capsys.readouterr().out


def preparar_publicacion(tmp_path: Path) -> imj.RutasImitacionJefes:
    """
    Carpeta de una corrida con el modelo ajustado de la arquitectura 2x256.

    Contexto:
        Pruebas de publicar_modelo.

    Args:
        tmp_path (Path): Carpeta temporal.

    Returns:
        RutasImitacionJefes: Rutas de la corrida.

    Raises:
        Ninguna.

    Ejemplo:
        rutas = preparar_publicacion(tmp_path)
    """
    rutas = imj.RutasImitacionJefes(tmp_path)
    rutas.modelo("2x256").parent.mkdir(parents=True)
    rutas.modelo("2x256").write_bytes(b"pesos-del-modelo")
    return rutas


def test_publicar_modelo_copia_el_modelo_y_fija_la_referencia(tmp_path: Path) -> None:
    rutas = preparar_publicacion(tmp_path)
    evaluacion = {"por_jefe": {"amlo": {"tasa": 0.8, "juego": {"peleas": 20, "tasa_victoria": 0.8}},
                               "nahual": {"tasa": 1.0, "juego": {}}}}
    tasas = imj.publicar_modelo(rutas, "2x256", evaluacion)
    assert tasas == {"amlo": 0.8, "nahual": 1.0}
    assert rutas.mejor.read_bytes() == b"pesos-del-modelo"
    referencia = leer_json(tmp_path / "referencia.json")
    assert referencia == {"arquitectura": "2x256", "tasas": tasas, "peleas": {"amlo": 20, "nahual": 0}}
    mejor = leer_json(tmp_path / "mejor.json")
    assert mejor["por_jefe"]["amlo"]["peleas"] == 20 and mejor["tasas"] == tasas


def test_publicar_modelo_rechaza_una_evaluacion_sin_tasas_y_un_modelo_inexistente(tmp_path: Path) -> None:
    rutas = preparar_publicacion(tmp_path)
    with pytest.raises(ValueError, match="tasas"):
        imj.publicar_modelo(rutas, "2x256", {})
    with pytest.raises(ValueError):
        imj.publicar_modelo(rutas, "2x256", {"por_jefe": {}})
    with pytest.raises(FileNotFoundError, match="3x256"):
        imj.publicar_modelo(rutas, "3x256", {"por_jefe": {"amlo": {"tasa": 1.0, "juego": {}}}})
    assert not rutas.mejor.exists()
