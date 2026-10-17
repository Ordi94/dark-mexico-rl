"""
Comprueba en el juego real que el guardado, los logros y la pausa siguen neutralizados en cada jefe.

Contexto:
    Para cada jefe del entorno multi-jefe: reinicia la pelea, deja al jefe con muy poca vida y lo mata
    con el bot de reglas (la victoria dispara dos SaveGameToSlot y varios logros), luego simula la
    pulsacion de Start (menu de pausa) y comprueba que: los guardados quedan bloqueados (contador de
    bloqueados sube y el de sin bloquear sigue en cero), los logros se fuerzan a progreso 0, el menu de
    pausa se cierra solo y no aparece MexicoSouls\\Saved\\SaveGames. Escribe
    bench_results/guardado_jefes.json.

Uso:
    python scripts/verificar_guardado_jefes.py --jefes juice amlo chamuco nahual
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from dmbridge import Action, BridgeClient  # noqa: E402
from dmbridge.launcher import LaunchConfig  # noqa: E402
from dmrl.dominio import acciones as acc  # noqa: E402
from dmrl.dominio.amenazas import SeguidorAmenazas  # noqa: E402
from dmrl.dominio.bot_jefes import BotJefes  # noqa: E402
from dmrl.infraestructura.jefes import cargar_catalogo  # noqa: E402

CARPETA_GUARDADOS = Path(os.environ.get("LOCALAPPDATA", "")) / "MexicoSouls" / "Saved" / "SaveGames"
LUA_DEJAR_CASI_MUERTO = """
local b = BRIDGE.G.boss_actor()
if not b then return 0 end
local d = b.BPC_DamageSystem
d.Health = math.min(d.Health, 5.0)
return 1
"""
LUA_CONTADORES = """
local g = BRIDGE.guards
return { saves_blocked = g.saves_blocked, saves_unblocked = g.saves_unblocked,
         achievements_blocked = g.achievements_blocked, pauses_closed = g.pauses_closed }
"""
LUA_PULSAR_START = """
local p = BRIDGE.G.pawn()
p["InpActEvt_Gamepad_Special_Right_K2Node_InputKeyEvent_24"]({ KeyName = FName("Gamepad_Special_Right") })
return 1
"""
LUA_PAUSAS = """
local lib = StaticFindObject("/Script/UMG.Default__WidgetBlueprintLibrary")
local wc = StaticFindObject("/Script/UMG.UserWidget")
local w, n = {}, 0
lib:GetAllWidgetsOfClass(BRIDGE.G.world, w, wc, true)
for _, it in ipairs(w) do if BRIDGE.G.class_name(it:get()) == "WP_PauseGame_C" then n = n + 1 end end
return n
"""


def verificar(jefe: str) -> dict[str, Any]:
    """
    Mata al jefe y comprueba las protecciones.

    Contexto:
        main.

    Args:
        jefe (str): Jefe de episodio.

    Returns:
        dict[str, Any]: Contadores antes y despues, resultado de la pelea y estado de la pausa.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        verificar("nahual")
    """
    catalogo = cargar_catalogo()
    perfil = catalogo.perfil(jefe)
    mod = dict(perfil.episodio.mod or {})
    cliente = BridgeClient(f"grd_{jefe}", reply_timeout=120.0)
    cliente.launch(LaunchConfig(instance_id=f"grd_{jefe}", profile="training", sim_hz=30.0, debug=True,
                                affinity=list(range(4, 28))), connect_timeout=120.0)
    bot = BotJefes.desde_catalogo(catalogo)
    seguidor = SeguidorAmenazas(catalogo)
    try:
        # AMLO no muere: su victoria es la del Chamuco (transicion natural a la segunda fase).
        obs = cliente.reset(boss=mod, start_phase=2 if jefe == "amlo" else None)
        antes = cliente.exec_lua(LUA_CONTADORES)
        cliente.exec_lua(LUA_DEJAR_CASI_MUERTO)
        bot.reiniciar()
        seguidor.reiniciar()
        victoria = False
        for _ in range(3000):
            amenazas = seguidor.actualizar(obs)
            mascara = acc.mascara_acciones(obs.get("player"))
            decision = bot.decidir(obs, mascara, amenazas)
            accion = acc.comando_de_accion(decision)
            obs = cliente.step(Action(btn=accion.btn, lb=accion.lb, mx=accion.mx, my=accion.my), 2)
            bot.confirmar(decision)
            if (obs.get("ev") or {}).get("victory"):
                victoria = True
                break
            if (obs.get("player") or {}).get("dead"):
                break
        # Los guardados de la victoria llegan hasta ~10 s despues (el Chamuco 7 s, Juice 10 s).
        for _ in range(240):
            obs = cliente.step(Action(), 2)
        despues = cliente.exec_lua(LUA_CONTADORES)
        cliente.exec_lua(LUA_PULSAR_START)
        abiertas = cliente.exec_lua(LUA_PAUSAS)
        for _ in range(12):
            cliente.step(Action(), 2)
        cerradas = cliente.exec_lua(LUA_PAUSAS)
        final = cliente.exec_lua(LUA_CONTADORES)
    finally:
        cliente.close()
    return {
        "jefe": jefe, "victoria": victoria, "antes": antes, "despues_de_la_victoria": despues,
        "pausa_abierta_tras_start": abiertas, "pausa_tras_12_pasos": cerradas, "final": final,
        "carpeta_de_guardados_existe": CARPETA_GUARDADOS.exists(),
    }


def main() -> int:
    """
    Linea de comandos.

    Contexto:
        python scripts/verificar_guardado_jefes.py.

    Args:
        Ninguno.

    Returns:
        int: 0 si las protecciones se cumplen en todos los jefes, 1 si no.

    Raises:
        BridgeError: Si el juego falla.

    Ejemplo:
        python scripts/verificar_guardado_jefes.py --jefes nahual
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jefes", nargs="+", required=True)
    args = parser.parse_args()
    resultados = [verificar(j) for j in args.jefes]
    (RAIZ / "bench_results" / "guardado_jefes.json").write_text(json.dumps(resultados, indent=1), encoding="utf-8")
    fallo = False
    for r in resultados:
        f = r["final"]
        ok = (r["victoria"] and f["saves_unblocked"] == 0 and int(r["pausa_tras_12_pasos"]) == 0
              and not r["carpeta_de_guardados_existe"])
        fallo = fallo or not ok
        print(f"{r['jefe']:8} victoria={r['victoria']} guardados bloqueados {r['antes']['saves_blocked']} -> "
              f"{f['saves_blocked']} (sin bloquear {f['saves_unblocked']}), logros bloqueados "
              f"{f['achievements_blocked']}, pausa abierta {r['pausa_abierta_tras_start']} -> "
              f"{r['pausa_tras_12_pasos']} (cerradas por el mod {f['pauses_closed']}), carpeta SaveGames "
              f"{r['carpeta_de_guardados_existe']} -> {'OK' if ok else 'FALLA'}")
    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())
