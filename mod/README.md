# Mod DarkMexicoBridge

Mod de UE4SS escrito en Lua. Es la mitad del puente: dentro del juego lee el estado, aplica las acciones que manda Python, reinicia las peleas y bloquea el guardado y los logros. La otra mitad es `src/dmbridge`.

Instalación: [docs/01-instalacion.md](../docs/01-instalacion.md). Funcionamiento: [docs/03-como-funciona.md](../docs/03-como-funciona.md).

| Archivo | Responsabilidad |
|---|---|
| `Scripts/main.lua` | Punto de entrada. Solo actúa con `DMB_TRAINING=1`. Atiende las órdenes del canal. |
| `Scripts/dmb/channel.lua` | Lectura y escritura de los dos named pipes. |
| `Scripts/dmb/state.lua` | Construcción de la observación del paso. |
| `Scripts/dmb/actions.lua` | Traducción de una acción a botones y movimiento. |
| `Scripts/dmb/episode.lua`, `arena.lua`, `selection.lua`, `spec.lua`, `curriculum.lua` | Reinicio de la pelea, jefe elegido por clase, validación de la descripción del jefe, vida escalada del jefe. |
| `Scripts/dmb/events.lua`, `hooks.lua` | Ganchos de daño, parry, fase 2 y victoria. |
| `Scripts/dmb/guards.lua` | Guardado anulado, logros a 0 y cierre del menú de pausa. |
| `Scripts/dmb/keepalive.lua` | Referencias fuertes a las clases Blueprint para que no se descarguen al recargar el mapa. |
| `Scripts/dmb/game.lua`, `json.lua`, `log.lua` | Utilidades. |

Variables de entorno que lee: `DMB_TRAINING`, `DMB_ID`, `DMB_LOG_DIR`, `DMB_IDLE_TIMEOUT` y `DMB_DEBUG` (descritas al inicio de `Scripts/main.lua`).

Las pruebas de los módulos puros (`json`, `spec`, `selection`, `curriculum`) corren en Python con la biblioteca `lupa`, sin el juego: `tests/test_lua_modulos.py`.
