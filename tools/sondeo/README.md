# Herramientas de sondeo

Mods de UE4SS y utilidades en Lua que usé para medir a los jefes: qué clases existen, cuándo conecta cada golpe, qué ataques llegan de lejos. No intervienen en el entrenamiento. Para qué sirven y cómo usarlas paso a paso está en [docs/06-medir-un-jefe.md](../../docs/06-medir-un-jefe.md).

| Carpeta | Contenido |
|---|---|
| `DarkMexicoProbe/` | Mod de sondeo del mapa de Sheinbaum. Se desactiva solo cuando el proceso arranca con `DMB_TRAINING=1`. |
| `DarkMexicoProbeAmlo/` | Mod de sondeo de `LVL_AMLO`. Solo actúa con `DMP_AMLO=1`. Bloquea guardado y logros. |
| `probe_lab/` | Utilidades de medición para el sondeo de Sheinbaum. `init.lua` las carga todas. |
| `probe_amlo/` | Utilidades de medición de `LVL_AMLO`, lanzador y cierre por PID, y las tablas de golpes medidas. |

Instalación: copia la carpeta del mod que vayas a usar dentro de `ue4ss\Mods` en la carpeta del juego, igual que `DarkMexicoBridge` ([docs/01](../../docs/01-instalacion.md)). `DarkMexicoProbeAmlo` trae un `enabled.txt` vacío y UE4SS lo carga sin editar `mods.txt`. El de Sheinbaum necesita la línea `DarkMexicoProbe : 1` en `mods.txt`.

Variables de entorno:

| Variable | Uso |
|---|---|
| `DMPROBE_DIR` | Carpeta donde los mods escriben sus registros y leen `cmd.lua`. Si no existe, usan la carpeta de trabajo del juego. `lanzar_sondeo.ps1` la fija en `tools\sondeo\salida`. |
| `DMPROBE_LAB` | Carpeta de `probe_lab`, para que `init.lua` encuentre sus archivos. |
| `DMP_AMLO` | `1` activa `DarkMexicoProbeAmlo`. |
| `DARKMEXICO_DIR` | Carpeta del juego, para los lanzadores. |

Los sondeos no deben correr con una partida tuya abierta. `DarkMexicoProbeAmlo` bloquea el guardado y los logros; el de Sheinbaum no, así que úsalo solo en una instancia de prueba.
