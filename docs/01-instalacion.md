# Instalación

Esta guía deja el juego listo para que Python lo controle. Si algo no coincide con lo que ves, casi siempre es la versión de UE4SS o la ruta del juego.

## Lo que vas a modificar

Dos cosas dentro de la carpeta del juego, y nada más:

- `dwmapi.dll` y la carpeta `ue4ss\`, que es UE4SS.
- `ue4ss\Mods\DarkMexicoBridge\`, el mod de este repositorio, y una línea en `ue4ss\Mods\mods.txt`.

No toco paks, el ejecutable ni ninguna DLL del juego, así que desinstalar es borrar lo anterior.

## 1. Comprar el juego y ubicar la carpeta

Compra Dark México en Steam: <https://store.steampowered.com/app/4346760/>. El proyecto no incluye ni descarga archivos del juego.

La carpeta de instalación suele ser:

```
C:\Program Files (x86)\Steam\steamapps\common\DarkMexico
```

El ejecutable que interesa está en `MexicoSouls\Binaries\Win64\MexicoSouls-Win64-Shipping.exe`. Esa carpeta `Win64` es donde van los archivos de UE4SS.

El código busca la carpeta del juego en este orden:

1. La variable de entorno `DARKMEXICO_DIR`, si la defines.
2. El registro de Windows (la ruta de Steam) y las bibliotecas que aparecen en `steamapps\libraryfolders.vdf`.
3. La ruta normal de Steam, solo para que el mensaje de error diga dónde buscó.

Si el juego está en otro disco y Steam no lo encuentra, define la variable antes de usar cualquier script:

```powershell
$env:DARKMEXICO_DIR = "D:\Juegos\Steam\steamapps\common\DarkMexico"
```

Para dejarla fija: `setx DARKMEXICO_DIR "D:\Juegos\Steam\steamapps\common\DarkMexico"` y abre una consola nueva.

## 2. Instalar UE4SS v3.0.1-1176

Uso exactamente esta versión. UE4SS cambia seguido y las versiones nuevas modifican la API de Lua, así que no garantizo que otra funcione.

La etiqueta `experimental-latest` ya apunta a una compilación más nueva. La que usé está en el archivo de versiones experimentales:

- Página: <https://github.com/UE4SS-RE/RE-UE4SS/releases/tag/experimental>
- Archivo: `UE4SS_v3.0.1-1176-g1ba38db4.zip` (8 741 395 bytes, el que no lleva el prefijo `zDEV-`)
- Descarga directa: <https://github.com/UE4SS-RE/RE-UE4SS/releases/download/experimental/UE4SS_v3.0.1-1176-g1ba38db4.zip>
- SHA-256: `e615ee7a2c24344ea5aeef85183d6cfd550f4f6352225976679f9fefffd0b9a5`

Comprueba el hash antes de descomprimir:

```powershell
Get-FileHash .\UE4SS_v3.0.1-1176-g1ba38db4.zip -Algorithm SHA256
```

Descomprime el zip. Dentro hay `dwmapi.dll` y la carpeta `ue4ss`. Copia las dos cosas a:

```
<carpeta del juego>\MexicoSouls\Binaries\Win64\
```

Con el juego cerrado. El resultado debe verse así:

```
Win64\
  MexicoSouls-Win64-Shipping.exe
  dwmapi.dll          <- nuevo
  ue4ss\              <- nuevo
    UE4SS.dll
    UE4SS-settings.ini
    Mods\
```

UE4SS se inyecta con `dwmapi.dll`, que Windows carga antes que la del sistema. Por eso se carga también cuando juegas normal. Si prefieres no mezclar, renombra `dwmapi.dll` a `dwmapi.dll.off` mientras juegas y devuélvele el nombre para entrenar.

## 3. Instalar el mod DarkMexicoBridge

Copia la carpeta `mod\DarkMexicoBridge` de este repositorio dentro de `ue4ss\Mods`:

```powershell
$mods = "<carpeta del juego>\MexicoSouls\Binaries\Win64\ue4ss\Mods"
Copy-Item -Recurse .\mod\DarkMexicoBridge $mods
```

Después abre `ue4ss\Mods\mods.txt` y agrega esta línea antes del bloque `Keybinds`:

```
DarkMexicoBridge : 1
```

Si tu `mods.txt` tiene la línea `; Built-in keybinds, do not move up!`, deja la línea nueva por encima de ese comentario.

El mod no hace nada si el proceso no arranca con la variable `DMB_TRAINING=1`. Los scripts del proyecto la ponen solos. Cuando abres el juego desde Steam, el mod imprime `inactivo` en `UE4SS.log` y se queda quieto.

Con `DMB_TRAINING=1` el mod sí cambia dos cosas, y las dos son a propósito:

- Anula `SaveGameToSlot`, para que las instancias de entrenamiento no escriban tu partida guardada.
- Pone el progreso de logros en 0, para que no se desbloquee nada en tu cuenta.

Aun así, las instancias escriben en `%LOCALAPPDATA%\MexicoSouls\Saved` (un `CrashReportClient.ini` por arranque, la caché de pipelines y `GameUserSettings.ini`). Haz una copia antes de entrenar, en una carpeta que se llame `backup_savegames_<dd-MM-yyyy_HH-mm-ss>` dentro del repositorio (está en `.gitignore`):

```powershell
$marca = Get-Date -Format "dd-MM-yyyy_HH-mm-ss"
Copy-Item -Recurse "$env:LOCALAPPDATA\MexicoSouls\Saved" ".\backup_savegames_$marca\Saved"
```

Cuando termines, `.\scripts\verificar_saved.ps1` compara por SHA-256 la carpeta actual con la copia y `-Restaurar` la deja igual.

## 4. Python 3.12 y el proyecto

Necesitas Python 3.12 (el proyecto exige `>=3.12,<3.13`). Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Esto instala `dmbridge` (el cliente del puente) y `dmrl` (entorno, bot, imitación y PPO) en modo editable, más las herramientas de desarrollo. La instalación baja PyTorch, unos 200 MB. Si quieres que PyTorch use tu GPU en el ajuste por imitación, instala antes la versión con CUDA desde su índice:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

El PPO corre en CPU aunque tengas GPU. Los motivos están en [docs/04-entrena-tu-ia.md](04-entrena-tu-ia.md).

## 5. Verificar con una prueba de humo

Cierra el juego si lo tenías abierto y ejecuta:

```powershell
.\.venv\Scripts\python.exe scripts\bridge_smoke.py
```

Lanza una instancia sin ventana, hace `reset`, ejecuta acciones al azar durante 120 s de juego y revisa que el estado sea coherente (cada paso avanza los frames pedidos, la vida está acotada, las posiciones son finitas). Termina con código 0 si todo pasa. Imprime pasos por segundo y latencia: en mi equipo, un paso de 2 frames tarda unos 2 ms.

Si falla:

| Síntoma | Causa probable |
|---|---|
| `No se encontro el juego en ...` | La ruta no es la de Steam: define `DARKMEXICO_DIR`. |
| El juego abre y se cierra a los 2 minutos sin que Python conecte | El mod no está cargado. Revisa `mods.txt` y `ue4ss\UE4SS.log`. |
| El juego se cierra al arrancar | Versión de UE4SS distinta de la v3.0.1-1176, o una actualización del juego. |
| `[Bridge] inactivo: falta DMB_TRAINING=1` en el registro | Normal si abriste el juego a mano. Los scripts lo definen solos. |

La prueba comprobó que guardado y logros quedan bloqueados: no debe aparecer `Saved\SaveGames\savegame.sav`. Para comprobarlo aparte: `python scripts\check_guards.py`.

## Desinstalar

Para quitar solo el puente: borra `ue4ss\Mods\DarkMexicoBridge` y la línea `DarkMexicoBridge : 1` de `mods.txt`.

Para quitarlo todo: borra `dwmapi.dll` y la carpeta `ue4ss` de `Win64`. Después, si quieres comprobar el estado de tus datos, usa `.\scripts\verificar_saved.ps1` con tu copia de seguridad.
