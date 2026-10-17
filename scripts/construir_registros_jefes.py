"""
Genera los registros iniciales data/jefes/<jefe>.json a partir de las tablas medidas antes de
existir el entorno multi-jefe (data/tabla_golpes.json, data/vocabulario_montages.json y las tablas
de LVL_AMLO en probe_amlo/).

Contexto:
    Se ejecuta una vez para crear los cinco registros. Despues, scripts/medir_jefes.py produce
    mediciones nuevas (alcance, golpes del Chamuco, retardos de ataques a distancia) y
    --fusionar las incorpora a los registros sin tocar lo ya medido que la nueva medicion no cubre.

Uso:
    python scripts/construir_registros_jefes.py            # crea los registros que no existan
    python scripts/construir_registros_jefes.py --forzar   # los reescribe desde las tablas originales
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from dmrl.dominio.vocabulario import VOCAB_JEFE as VOCAB_JEFE_V1  # noqa: E402

DATOS = RAIZ / "data"
SALIDA = DATOS / "jefes"
PROBE = RAIZ / "probe_amlo"

MENORES = ["BP_ChairosOP_C", "BP_Ghoul_C", "BP_Chairos_C"]
JEFES_AMLO = ["BP_AMLOBOSS_C", "BP_Chamuco_C", "BP_CHUPACABRAS_C"]
DESPLAZAMIENTO_JUGADOR = {"x": 450.0, "y": 0.0, "z": -90.0}
POS_ARENA_AMLO = {"x": -5606.0, "y": -1800.0, "z": 11293.0}
ALCANCE_MELEE_INICIAL = 350.0
# Alcance de los ataques que llegan al jugador esté donde esté.
GLOBAL = 6000.0

REACCIONES_PARRY = ["Boss_Parried_RM_Montage", "Boss_Staggered_RM_Montage"]


def cargar(ruta: Path) -> dict[str, Any]:
    """
    Lee un JSON.

    Contexto:
        Entrada de las tablas originales.

    Args:
        ruta (Path): Archivo a leer.

    Returns:
        dict[str, Any]: Contenido.

    Raises:
        FileNotFoundError: Si no existe.

    Ejemplo:
        cargar(DATOS / "tabla_golpes.json")
    """
    return json.loads(ruta.read_text(encoding="utf-8"))


def golpes_desde_tabla(tabla: dict[str, Any], incluir: set[str] | None = None,
                       excluir: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Convierte una tabla de golpes original al formato de los registros.

    Contexto:
        Juice (data/tabla_golpes.json) y los jefes de LVL_AMLO (probe_amlo/tabla_golpes_*.json).

    Args:
        tabla (dict[str, Any]): montage -> lista de golpes originales.
        incluir (set[str], opcional): Solo estos montages.
        excluir (set[str], opcional): Se omiten estos montages.

    Returns:
        dict[str, list[dict[str, Any]]]: montage -> golpes con posicion, rango, parry, dano y alcance.

    Raises:
        Ninguna.

    Ejemplo:
        golpes_desde_tabla({"M": [{"pos": 1.0, "parry": True, "dano": 10}]})
    """
    salida: dict[str, list[dict[str, Any]]] = {}
    for montage, filas in tabla.items():
        if montage.startswith("evento:"):
            continue
        if incluir is not None and montage not in incluir:
            continue
        if excluir is not None and montage in excluir:
            continue
        # Una observacion aislada (n < 3) es ruido de medicion; se descarta salvo que sea la unica.
        firme = [f for f in filas if int(f.get("n", 0)) >= 3] or list(filas)
        salida[montage] = [
            {
                "pos": float(f["pos"]), "min": float(f.get("min", f["pos"])), "max": float(f.get("max", f["pos"])),
                "parry": bool(f["parry"]), "dano": float(f.get("dano", 0.0)), "n": int(f.get("n", 0)),
                "alcance": ALCANCE_MELEE_INICIAL, "tipo": "melee", "velocidad": 0.0,
            }
            for f in firme
        ]
    return salida


def spec_mod(identidad: str, mapa: str, clases: list[str], inicio: dict[str, Any], hooks: list[dict[str, Any]],
             purga: list[str], fase2: dict[str, Any] | None) -> dict[str, Any]:
    """
    Descripcion que el mod valida y usa para reiniciar el episodio de un jefe.

    Contexto:
        Seccion "mod" de cada registro; se envia tal cual en la orden reset.

    Args:
        identidad (str): Identificador del jefe.
        mapa (str): Mapa donde ocurre.
        clases (list[str]): Clases Blueprint admitidas como jefe.
        inicio (dict[str, Any]): Como empieza el episodio.
        hooks (list[dict[str, Any]]): Eventos de victoria, fase y ataques a distancia.
        purga (list[str]): Jefes que se destruyen en cada reinicio.
        fase2 (dict[str, Any], opcional): Jefe de la fase siguiente, para episodios que arrancan en ella.

    Returns:
        dict[str, Any]: Descripcion lista para enviar.

    Raises:
        Ninguna.

    Ejemplo:
        spec_mod("amlo", "LVL_AMLO", ["BP_AMLOBOSS_C"], {...}, [], [], None)
    """
    spec: dict[str, Any] = {
        "id": identidad, "map": mapa, "classes": clases, "purge": purga, "start": inicio, "hooks": hooks,
    }
    if mapa == "LVL_AMLO":
        spec["minors"] = MENORES
    if fase2:
        spec["phase2"] = fase2
    return spec


def registro_juice(fase2: bool) -> dict[str, Any]:
    """
    Registro de Juice (fase 1) o de su actor de fase 2.

    Contexto:
        Los datos salen de data/tabla_golpes.json y data/vocabulario_montages.json; la fase 2 usa
        los montages Ninja*.

    Args:
        fase2 (bool): True para el actor de fase 2.

    Returns:
        dict[str, Any]: Registro.

    Raises:
        FileNotFoundError: Si faltan las tablas originales.

    Ejemplo:
        registro_juice(False)["id"]  # "juice"
    """
    tabla = cargar(DATOS / "tabla_golpes.json")
    vocab = cargar(DATOS / "vocabulario_montages.json")
    montages_ninja = {m for m in tabla if m.startswith("Ninja")}
    # Montages observados en 250 episodios mas el vocabulario completo de la observacion v1 (incluye
    # montages del SDK que no aparecieron en esas corridas), para que ninguno caiga en "otro".
    montages_jefe = list(dict.fromkeys([*vocab["boss"].keys(), *VOCAB_JEFE_V1, *tabla]))
    if fase2:
        golpes = golpes_desde_tabla(tabla, incluir=montages_ninja)
        vocabulario = [m for m in montages_jefe if m.startswith("Ninja")]
    else:
        golpes = golpes_desde_tabla(tabla, excluir=montages_ninja)
        vocabulario = [m for m in montages_jefe if not m.startswith("Ninja")]
    return {
        "id": "juice_fase2" if fase2 else "juice",
        "nombre": "Juice (fase 2)" if fase2 else "Juice",
        "mapa": "LVL_Boss",
        "indice": 1 if fase2 else 0,
        "clases": ["BP_Enemy_JuicePhase2_C"] if fase2 else ["BP_Enemy_Juice_C"],
        "vida_max": 1000.0 if fase2 else 2000.0,
        "fase_avanzada": fase2,
        "reacciones": REACCIONES_PARRY,
        "alcance_defecto": ALCANCE_MELEE_INICIAL,
        "vocabulario": vocabulario,
        "golpes": golpes,
        "eventos_distancia": {},
    }


def registro_amlo() -> dict[str, Any]:
    """
    Registro de AMLO.

    Contexto:
        Tabla de probe_amlo/tabla_golpes_amlo.json; los ataques lanzados quedan como eventos.

    Args:
        Ninguno.

    Returns:
        dict[str, Any]: Registro.

    Raises:
        FileNotFoundError: Si falta la tabla.

    Ejemplo:
        registro_amlo()["mapa"]  # "LVL_AMLO"
    """
    tabla = cargar(PROBE / "tabla_golpes_amlo.json")
    golpes = golpes_desde_tabla(tabla)
    golpes.pop("A_spear_1hand04_ThrowMid01_Montage", None)
    # Katana lanzada: el montage dura 0.4 s (play rate 2.0) y el proyectil sale a 0.156 s; llega
    # 0.156 s + distancia / 4975 uu/s despues de empezar (19 impactos, de 880 a 3870 uu), es decir,
    # casi siempre cuando el montage ya termino. Se programa desde el montage, que avisa antes que el evento.
    golpes["A_spear_1hand04_ThrowMid01_Montage"] = [{
        "pos": 0.312, "min": 0.312, "max": 0.312, "parry": False, "dano": 35.0, "n": 19, "alcance": GLOBAL,
        "tipo": "proyectil", "post_montage": True, "velocidad": 4975.0,
    }]
    # Agarre a distancia (Force Choke): arrastra al jugador hacia AMLO durante el montage y lo mata
    # (dano 1e11) a los 4.2 s si no esquiva; se evita con la maniobra o con el parry (medido 3 de 3 y 1 de 1).
    golpes["Paired_ForceChoke_Att_Montage"] = [{
        "pos": 4.2, "min": 4.2, "max": 4.2, "parry": False, "dano": 100.0, "n": 6, "alcance": GLOBAL,
        "tipo": "area", "post_montage": False, "velocidad": 0.0,
    }]
    return {
        "id": "amlo", "nombre": "AMLO", "mapa": "LVL_AMLO", "indice": 2, "clases": ["BP_AMLOBOSS_C"],
        "vida_max": 2000.0, "fase_avanzada": False, "reacciones": REACCIONES_PARRY,
        "alcance_defecto": ALCANCE_MELEE_INICIAL,
        # El montage del lanzamiento no tiene golpe propio: el impacto lo da el evento de la katana.
        "ignorar": ["A_spear_1hand04_ThrowMid01_Montage", "Paired_ForceChoke_Att_Montage"],
        "vocabulario": sorted(set(golpes) | {"A_spear_1hand04_ThrowMid01_Montage", "Anim_Sword_Evade_B_Root_Montage",
                                             "AS_Mage_Powerup_Light_01_Montage", *REACCIONES_PARRY}),
        "golpes": golpes,
        # Ataques cuyo impacto no pertenece a ningun montage. Las katanas del piso caen 1.93 s despues
        # del evento (medido con el jugador quieto, scripts/medir_jefes.py) y no se evitan saliendo del
        # area (scripts/medir_esquiva.py): solo con la invulnerabilidad de la maniobra.
        "eventos_distancia": {
            "amlo_katanas_floor": {
                "impactos": [{"retardo": 1.93, "danio": 30.0}], "velocidad": 0.0, "parry": False,
                "tipo": "area", "esquivable": True, "centro": "jugador", "vigencia": 0.25,
            },
        },
    }


def registro_chamuco() -> dict[str, Any]:
    """
    Registro del Chamuco (fase 2 de AMLO).

    Contexto:
        Tabla de probe_amlo/tabla_golpes_chamuco.json; el golpe de piso con lava es un golpe de
        area no parable dentro de su montage.

    Args:
        Ninguno.

    Returns:
        dict[str, Any]: Registro.

    Raises:
        FileNotFoundError: Si falta la tabla.

    Ejemplo:
        registro_chamuco()["vida_max"]  # 1500.0
    """
    golpes = golpes_desde_tabla(cargar(PROBE / "tabla_golpes_chamuco.json"))
    # El golpe de piso alcanza al jugador a cualquier distancia (impacto en 18 de 18 eventos, incluso
    # teletransportado 1200 uu): solo se evita con la invulnerabilidad de la maniobra.
    for fila in golpes["Anim_Spear_JumpAttackA_RM_Montage"]:
        fila["tipo"] = "area"
        fila["alcance"] = GLOBAL
    return {
        "id": "chamuco", "nombre": "Chamuco", "mapa": "LVL_AMLO", "indice": 3, "clases": ["BP_Chamuco_C"],
        "vida_max": 1500.0, "fase_avanzada": True, "reacciones": REACCIONES_PARRY,
        "alcance_defecto": ALCANCE_MELEE_INICIAL,
        "vocabulario": sorted({*golpes, "Boss_Parried_RM_Montage", "Boss_Staggered_RM_Montage"}),
        "golpes": golpes,
        "eventos_distancia": {},
    }


def registro_nahual() -> dict[str, Any]:
    """
    Registro del Chupacabras (clase Nahual).

    Contexto:
        Tabla de probe_amlo/tabla_golpes_chupacabras.json; el rayo cae despues de terminar su
        montage (golpe con post_montage) y la rafaga de fuego son tres golpes no parables.

    Args:
        Ninguno.

    Returns:
        dict[str, Any]: Registro.

    Raises:
        FileNotFoundError: Si falta la tabla.

    Ejemplo:
        registro_nahual()["id"]  # "nahual"
    """
    golpes = golpes_desde_tabla(cargar(PROBE / "tabla_golpes_chupacabras.json"))
    # El rayo cae 1.216-1.233 s despues de iniciar el montage (que dura 1.18 s) sobre el jugador
    # donde este (desplazado 300 uu el impacto sigue en 93-100% de los casos). La rafaga de fuego
    # son tres proyectiles cuyo tiempo de vuelo crece con la distancia: posicion = base + d / 3050
    # uu/s (regresion con 22 rafagas, error de 0.04 s).
    for fila in golpes["AS_Mage_Ability_Spell_HighRtLight_02_Montage"]:
        fila.update({"tipo": "rayo", "alcance": GLOBAL, "post_montage": True, "velocidad": 0.0})
    golpes["AS_Mage_Elemental_Fire_Volley_01_Montage"] = [
        {"pos": base, "min": base, "max": base, "parry": False, "dano": 40.0, "n": 22, "alcance": GLOBAL,
         "tipo": "proyectil", "velocidad": 3050.0}
        for base in (1.783, 2.599, 3.568)
    ]
    return {
        "id": "nahual", "nombre": "Chupacabras (Nahual)", "mapa": "LVL_AMLO", "indice": 4,
        "clases": ["BP_CHUPACABRAS_C"], "vida_max": 2000.0, "fase_avanzada": False,
        "reacciones": ["NinjaHeadHit_02_Montage"], "alcance_defecto": ALCANCE_MELEE_INICIAL,
        "vocabulario": sorted({*golpes, "Anim_Sword_Evade_B_Root_Montage", "NinjaHeadHit_02_Montage"}),
        "golpes": golpes,
        "eventos_distancia": {},
    }


def agregar_mod(registros: dict[str, dict[str, Any]]) -> None:
    """
    Agrega a cada registro su seccion de episodio y la descripcion para el mod.

    Contexto:
        Ultimo paso de la construccion; separa los datos del jefe de como se inicia su episodio.

    Args:
        registros (dict[str, dict[str, Any]]): Registros por identificador (se modifican).

    Returns:
        None: No retorna valor.

    Raises:
        KeyError: Si falta alguno de los cinco registros.

    Ejemplo:
        agregar_mod(registros)
    """
    registros["juice"]["episodio"] = {
        "via": "juice", "start_phase": 1, "vida_fase1": 1700.0, "vida_fase2": 1000.0,
        "clases_victoria": ["BP_Enemy_JuicePhase2_C"],
        "mod": spec_mod("juice", "LVL_Boss", ["BP_Enemy_Juice_C", "BP_Enemy_JuicePhase2_C"],
                        {"kind": "legacy_fog"}, [], ["BP_Enemy_Juice_C", "BP_Enemy_JuicePhase2_C"], None),
    }
    registros["juice_fase2"]["episodio"] = {
        "via": "juice", "start_phase": 2, "vida_fase1": 0.0, "vida_fase2": 1000.0,
        "clases_victoria": ["BP_Enemy_JuicePhase2_C"],
    }
    hooks_amlo = [
        {"path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C:ENTER PHASE 2", "event": "phase2"},
        {"path": "/Game/BP_Chamuco.BP_Chamuco_C:death", "event": "victory"},
        {"path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C:Ataque Throw Kat", "event": "remote", "id": "amlo_throw_kat"},
        {"path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C:ATAQUE DISTANCE KATANAS FLOOR", "event": "remote",
         "id": "amlo_katanas_floor"},
        {"path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C:AOE", "event": "remote", "id": "amlo_aoe"},
    ]
    inicio_amlo = {"kind": "spawn_class", "class_path": "/Game/BP_AMLOBOSS.BP_AMLOBOSS_C", "pos": POS_ARENA_AMLO,
                   "player_offset": DESPLAZAMIENTO_JUGADOR}
    registros["amlo"]["episodio"] = {
        "via": "amlo", "start_phase": 1, "vida_fase1": 1600.0, "vida_fase2": 1500.0,
        "clases_victoria": ["BP_Chamuco_C"], "segundos_maximos": 600.0,
        "mod": spec_mod("amlo", "LVL_AMLO", ["BP_AMLOBOSS_C", "BP_Chamuco_C"], inicio_amlo, hooks_amlo, JEFES_AMLO,
                        {"class": "BP_Chamuco_C", "timeout": 60.0}),
    }
    registros["chamuco"]["episodio"] = {
        "via": "chamuco", "start_phase": 2, "vida_fase1": 0.0, "vida_fase2": 1500.0,
        "clases_victoria": ["BP_Chamuco_C"],
        "mod": spec_mod(
            "chamuco", "LVL_AMLO", ["BP_Chamuco_C"],
            {"kind": "spawn_class", "class_path": "/Game/BP_Chamuco.BP_Chamuco_C", "pos": POS_ARENA_AMLO, "hp": 1500,
             "player_offset": DESPLAZAMIENTO_JUGADOR},
            [{"path": "/Game/BP_Chamuco.BP_Chamuco_C:death", "event": "victory"},
             {"path": "/Game/BP_Chamuco.BP_Chamuco_C:Ataque ESPECIAL GOLPE PISO LAVA", "event": "remote",
              "id": "chamuco_lava"}], JEFES_AMLO, None),
    }
    registros["nahual"]["episodio"] = {
        "via": "nahual", "start_phase": 1, "vida_fase1": 2000.0, "vida_fase2": 0.0,
        "clases_victoria": ["BP_CHUPACABRAS_C"],
        "mod": spec_mod(
            "nahual", "LVL_AMLO", ["BP_CHUPACABRAS_C"],
            {"kind": "level_event", "level_class": "LVL_AMLO_C", "event": "SPAWN  NAHUAL",
             "boss_class": "BP_CHUPACABRAS_C", "wait_seconds": 1.0,
             "player_offset": {"x": 450.0, "y": 0.0, "z": -91.0}},
            [{"path": "/Game/BP_CHUPACABRAS.BP_CHUPACABRAS_C:death", "event": "victory"},
             {"path": "/Game/BP_CHUPACABRAS.BP_CHUPACABRAS_C:ATTACK RAYO", "event": "remote", "id": "nahual_rayo"},
             {"path": "/Game/BP_CHUPACABRAS.BP_CHUPACABRAS_C:ATTACK FIREBALL", "event": "remote",
              "id": "nahual_fireball"}], JEFES_AMLO, None),
    }


# Ajustes del bot de reglas por jefe (seccion "bot" de cada registro). El especial deja al jugador
# 2.55 s sin parry ni maniobra: solo se usa si el siguiente impacto esta a mas de 2.7 s.
AJUSTES_BOT: dict[str, dict[str, float]] = {
    "amlo": {"especial_seguro": 2.7, "pausa_bajo_vida": 0.2, "pausa_maxima": 25.0},
    "chamuco": {"especial_seguro": 2.7},
    "nahual": {"especial_seguro": 2.7},
}


def main() -> int:
    """
    Crea los registros que falten (o todos con --forzar).

    Contexto:
        Linea de comandos.

    Args:
        Ninguno.

    Returns:
        int: 0 al terminar.

    Raises:
        OSError: Si no se puede escribir.

    Ejemplo:
        python scripts/construir_registros_jefes.py
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true", help="reescribe los registros existentes")
    args = parser.parse_args()
    registros = {
        "juice": registro_juice(False), "juice_fase2": registro_juice(True), "amlo": registro_amlo(),
        "chamuco": registro_chamuco(), "nahual": registro_nahual(),
    }
    agregar_mod(registros)
    for identidad, ajustes in AJUSTES_BOT.items():
        registros[identidad]["bot"] = ajustes
    SALIDA.mkdir(parents=True, exist_ok=True)
    for identidad, registro in registros.items():
        ruta = SALIDA / f"{identidad}.json"
        if ruta.exists() and not args.forzar:
            print(f"ya existe, no se toca: {ruta.name}")
            continue
        ruta.write_text(json.dumps(registro, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"escrito {ruta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
