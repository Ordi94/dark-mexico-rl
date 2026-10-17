"""
Validacion de reinicios seguidos de un jefe sin recargar el mapa: el estado debe quedar limpio.

Contexto:
    Lanza una instancia con exec habilitado, hace N reinicios consecutivos del mismo jefe (el primero
    carga el mapa, los demas son en sitio) y entre uno y otro juega una cantidad aleatoria de pasos con
    acciones al azar para ensuciar el estado (jugador herido o muerto, jefe herido, caguamas gastadas,
    jefe en otra fase). Tras cada reinicio inspecciona el mundo y comprueba: exactamente un jugador
    vivo con vida, caguamas, poder y resistencia llenos; exactamente un jefe de la lista de clases y
    con vida completa; ningun jefe de otra pelea; ningun enemigo menor; sin pantalla de muerte ni de
    victoria; la dilatacion de tiempo pedida; guardado bloqueado. Escribe
    bench_results/reinicios_jefes_<jefe>.json.

Uso:
    python scripts/validar_reinicios_jefes.py --jefe nahual --reinicios 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import psutil

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.jefes import PerfilJefe  # noqa: E402
from dmrl.infraestructura.jefes import cargar_catalogo  # noqa: E402

LUA_ESTADO = """
local G = BRIDGE.G
local out = { players = 0, bosses = {}, minors = 0, widgets = "" }
local arr = {}
G.gs:GetAllActorsOfClass(G.world, G.character_class, arr)
for _, item in ipairs(arr) do
    local a = item:get()
    local ok, cls = pcall(G.class_name, a)
    if ok then
        if cls == "BP_Player_C" then out.players = out.players + 1 end
        if cls == "BP_ChairosOP_C" or cls == "BP_Ghoul_C" or cls == "BP_Chairos_C" then out.minors = out.minors + 1 end
        if cls == "BP_AMLOBOSS_C" or cls == "BP_Chamuco_C" or cls == "BP_CHUPACABRAS_C"
            or cls == "BP_Enemy_Juice_C" or cls == "BP_Enemy_JuicePhase2_C" then
            local dead = false
            pcall(function() dead = a["DEATH?"] == true end)
            out.bosses[#out.bosses + 1] = cls .. (dead and ":muerto" or ":vivo")
        end
    end
end
local lib = StaticFindObject("/Script/UMG.Default__WidgetBlueprintLibrary")
local wc = StaticFindObject("/Script/UMG.UserWidget")
local w, names = {}, {}
lib:GetAllWidgetsOfClass(G.world, w, wc, true)
for _, item in ipairs(w) do names[#names + 1] = G.class_name(item:get()) end
out.widgets = table.concat(names, ",")
out.dilation = G.gs:GetGlobalTimeDilation(G.world)
out.saves_unblocked = BRIDGE.guards.saves_unblocked
out.saves_blocked = BRIDGE.guards.saves_blocked
out.pauses_closed = BRIDGE.guards.pauses_closed
local p = G.pawn()
if p then
    local d = p.BPC_DamageSystem
    out.p_hp, out.p_max, out.p_dead = d.Health, d.MaxHealth, d.IsDead
    out.caguamas, out.power, out.stamina = p.Caguamas, p.Power, p.Resistencia
end
return out
"""


def comprobar(estado: dict[str, Any], perfil: PerfilJefe, reinicio: dict[str, Any]) -> list[str]:
    """
    Devuelve las anomalias del mundo tras un reinicio.

    Contexto:
        validar.

    Args:
        estado (dict[str, Any]): Resultado de LUA_ESTADO.
        perfil (PerfilJefe): Jefe del episodio.
        reinicio (dict[str, Any]): Observacion devuelta por reset.

    Returns:
        list[str]: Descripcion de cada anomalia; vacia si el estado es limpio.

    Raises:
        Ninguna.

    Ejemplo:
        comprobar(estado, perfil, obs)
    """
    problemas: list[str] = []
    if estado["players"] != 1:
        problemas.append(f"jugadores={estado['players']}")
    if estado.get("p_dead") or abs(float(estado.get("p_hp", 0)) - float(estado.get("p_max", 1))) > 1e-6:
        problemas.append(f"jugador herido {estado.get('p_hp')}/{estado.get('p_max')}")
    if estado.get("caguamas") != 3 or float(estado.get("power", 0)) < 1.0 - 1e-6 or float(estado.get("stamina", 0)) < 0.99:
        problemas.append(f"recursos caguamas={estado.get('caguamas')} power={estado.get('power')} "
                         f"resistencia={estado.get('stamina')}")
    vivos = [b for b in estado["bosses"].values() if b.endswith(":vivo")] if isinstance(estado["bosses"], dict) \
        else [b for b in estado["bosses"] if b.endswith(":vivo")]
    clases = {f"{c}:vivo" for c in perfil.clases}
    permitidas = {f"{c}:vivo" for c in (perfil.episodio.mod or {}).get("classes", [])} | clases
    if len(vivos) != 1 or vivos[0] not in permitidas:
        problemas.append(f"jefes vivos={vivos}")
    if estado["minors"]:
        problemas.append(f"menores={estado['minors']}")
    for pantalla in ("WP_LoadingScreen_C", "WP_ENDGAME_C", "WP_SKINUNLOCKED_C"):
        if pantalla in estado["widgets"]:
            problemas.append(f"pantalla residual {pantalla}")
    if estado.get("saves_unblocked"):
        problemas.append("guardado sin bloquear")
    boss = reinicio.get("boss") or {}
    if float(boss.get("hp", 0)) < 0.95 * float(boss.get("hp_max", 1)):
        problemas.append(f"jefe herido {boss.get('hp')}/{boss.get('hp_max')}")
    return problemas


def validar(jefe: str, reinicios: int, semilla: int) -> dict[str, Any]:
    """
    Hace los reinicios y devuelve las anomalias por reinicio.

    Contexto:
        main.

    Args:
        jefe (str): Jefe de episodio.
        reinicios (int): Reinicios consecutivos.
        semilla (int): Semilla de las acciones que ensucian el estado.

    Returns:
        dict[str, Any]: Tiempos, anomalias y memoria del proceso del juego.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        validar("chamuco", 5, 0)
    """
    catalogo = cargar_catalogo()
    perfil = catalogo.perfil(jefe)
    mod = dict(perfil.episodio.mod or {})
    azar = random.Random(semilla)
    cliente = BridgeClient(f"rei_{jefe}", reply_timeout=120.0)
    cliente.launch(LaunchConfig(instance_id=f"rei_{jefe}", profile="training", sim_hz=30.0, debug=True,
                                affinity=list(range(4, 28))), connect_timeout=120.0)
    proceso = psutil.Process(cliente.process.pid) if cliente.process else None
    filas: list[dict[str, Any]] = []
    memoria_inicial = proceso.memory_info().rss / 2**20 if proceso else 0.0
    try:
        for n in range(reinicios):
            t0 = time.time()
            crudo = cliente.reset(boss=mod)
            segundos = time.time() - t0
            # reset no devuelve la recarga: el primero la hace siempre y los demas son en sitio.
            estado = cliente.exec_lua(LUA_ESTADO)
            problemas = comprobar(estado, perfil, crudo)
            filas.append({"n": n, "segundos": round(segundos, 3), "problemas": problemas,
                          "mapa": crudo.get("map"), "jefe": (crudo.get("boss") or {}).get("cls"),
                          "dilatacion": estado.get("dilation"), "pantallas": estado.get("widgets")})
            for _ in range(azar.randint(30, 400)):
                validas = [i for i, ok in enumerate(acc.mascara_acciones(crudo.get("player"))) if ok]
                accion = acc.comando_de_accion(azar.choice(validas))
                crudo = cliente.step(Action(btn=accion.btn, lb=accion.lb, mx=accion.mx, my=accion.my), 2)
                if (crudo.get("player") or {}).get("dead") or (crudo.get("ev") or {}).get("victory"):
                    break
        memoria_final = proceso.memory_info().rss / 2**20 if proceso else 0.0
        final = cliente.exec_lua(LUA_ESTADO)
    finally:
        cliente.close()
    return {
        "jefe": jefe, "reinicios": len(filas), "con_problemas": sum(1 for f in filas if f["problemas"]),
        "primero_s": filas[0]["segundos"] if filas else None,
        "siguientes_medio_s": sum(f["segundos"] for f in filas[1:]) / max(1, len(filas) - 1),
        "siguientes_max_s": max((f["segundos"] for f in filas[1:]), default=0.0),
        "memoria_mb": {"inicio": round(memoria_inicial, 1), "final": round(memoria_final, 1)},
        "guardados_bloqueados": final.get("saves_blocked"), "guardados_sin_bloquear": final.get("saves_unblocked"),
        "filas": filas,
    }


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/validar_reinicios_jefes.py.

    Args:
        Ninguno.

    Returns:
        int: 0 si todos los reinicios quedaron limpios, 1 si no.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        python scripts/validar_reinicios_jefes.py --jefe amlo --reinicios 30
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefe", required=True)
    parser.add_argument("--reinicios", type=int, default=30)
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()
    resultado = validar(args.jefe, args.reinicios, args.semilla)
    (RAIZ / "bench_results" / f"reinicios_jefes_{args.jefe}.json").write_text(json.dumps(resultado, indent=1),
                                                                                encoding="utf-8")
    print(f"{args.jefe}: {resultado['reinicios']} reinicios, {resultado['con_problemas']} con problemas; "
          f"primero {resultado['primero_s']} s, siguientes {resultado['siguientes_medio_s']:.2f} s (max "
          f"{resultado['siguientes_max_s']:.2f}); memoria {resultado['memoria_mb']}; guardados bloqueados "
          f"{resultado['guardados_bloqueados']}, sin bloquear {resultado['guardados_sin_bloquear']}")
    for f in resultado["filas"]:
        if f["problemas"]:
            print("  ", f["n"], f["problemas"])
    return 1 if resultado["con_problemas"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
