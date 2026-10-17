# Medir un jefe y agregar uno nuevo

Un jefe nuevo no se agrega editando un número. Hay que medirlo con el juego abierto, escribir su registro y volver a entrenar. Esta guía explica cómo medí los cuatro que hay, con las herramientas de `tools/sondeo`, y qué pasos siguen.

Calcula varios días de trabajo. Lo más lento es entender el juego, no el código.

## Por qué hay que medirlo

La red no recibe un número que diga "viene un golpe". Recibe el tiempo que falta para el siguiente impacto, si se puede parar, y el daño, y esos datos salen de una tabla por animación (montage) del jefe: en qué segundo de la animación conecta cada golpe, si es parable, cuánto daña y a qué distancia llega. Esa tabla es la que hace jugable el parry. Sin ella tienes que esperar a que la red lo descubra sola, y mi corrida de PPO desde cero terminó en 0% de victorias.

Cada jefe es un archivo en `data/jefes/<id>.json`:

| Campo | Contenido |
|---|---|
| `id`, `nombre`, `mapa`, `indice` | Identificador, mapa del juego (`LVL_Boss` o `LVL_AMLO`) y posición en el one-hot de identidad. |
| `clases` | Clases Blueprint de este actor (`BP_AMLOBOSS_C`). La identidad se deduce de la clase, nunca por una bandera como `IS BOSS?`: los NPC amistosos del mapa de AMLO también la tienen en `true`. |
| `vida_max`, `fase_avanzada` | Vida máxima del actor y si es la segunda fase de una pelea. |
| `vocabulario` | Montages del jefe que la observación reconoce (cualquier otro cae en "otro"). |
| `golpes` | Por montage, la lista de golpes: `pos` (segundo de la animación), `parry`, `dano`, `alcance`, `tipo` (`melee`, `area`, ...), `velocidad` si es un proyectil y `n` (cuántas veces lo medí). |
| `eventos_distancia` | Ataques que no pertenecen a una animación (katanas del piso, rayo): retardo desde el evento hasta cada impacto. |
| `episodio` | Cómo reinicia el mod la pelea: cómo se genera al jefe, dónde se coloca al jugador, qué hooks avisan fase 2 y victoria, qué enemigos menores se eliminan, tiempo máximo. |

Los registros iniciales los creó `scripts/construir_registros_jefes.py` a partir de las primeras tablas (`tools/sondeo/probe_amlo/tabla_golpes_*.json` y `data/tabla_golpes.json`). Para un jefe nuevo conviene copiar uno de los existentes y medir.

## Las herramientas de sondeo

Están en `tools/sondeo` y son mods de UE4SS más utilidades en Lua que se cargan desde el juego. No son parte del entrenamiento. Se instalan igual que `DarkMexicoBridge` y se activan con variables de entorno, así que no hacen nada en un arranque normal.

| Carpeta | Para qué sirve |
|---|---|
| `DarkMexicoProbe` | Mod de sondeo del mapa de Sheinbaum. Registra el mapa, detecta la aparición de las clases que vigilas, vuelca propiedades y funciones de cada clase a archivos y tiene un canal de comandos por archivo. |
| `DarkMexicoProbeAmlo` | Lo mismo para `LVL_AMLO`. Se activa con `DMP_AMLO=1`. Incluye las protecciones de guardado y de logros. |
| `probe_lab` | Utilidades de medición en Lua para el sondeo de Sheinbaum: bucle por frame, tareas, hooks de daño, bot de parry, teletransporte, dilatación del tiempo. |
| `probe_amlo` | Las mismas utilidades para `LVL_AMLO` (`lab.lua`, `hooks.lua`, `bot.lua`), los lanzadores `lanzar_sondeo.ps1` y `cerrar_sondeo.ps1`, y las tablas de golpes medidas. |

El canal de comandos funciona así: el mod vigila `cmd.lua` en la carpeta que indique la variable `DMPROBE_DIR`. Cuando el archivo cambia, lo ejecuta en el hilo del juego y escribe el resultado en `cmd_result.log`. Puedes experimentar sin reiniciar el juego: editas `cmd.lua`, guardas y lees la respuesta. El lanzador usa `tools\sondeo\salida` si no defines la variable.

```powershell
# Una instancia para sondear LVL_AMLO, con el mod activo y a paso fijo
.\tools\sondeo\probe_amlo\lanzar_sondeo.ps1 -Activo -PasoFijo
# ...editar tools\sondeo\salida\cmd.lua y leer cmd_result.log...
.\tools\sondeo\probe_amlo\cerrar_sondeo.ps1
```

`cerrar_sondeo.ps1` cierra solo el proceso cuyo PID guardó el lanzador. Nunca cierres por nombre: matarías cualquier otra instancia del juego.

## Paso 1. Ver qué hay en el mapa

1. Instala el mod de sondeo que corresponda (copia la carpeta a `ue4ss\Mods`). `DarkMexicoProbeAmlo` trae un `enabled.txt` vacío y UE4SS lo carga sin tocar `mods.txt`.
2. Lanza una instancia con el mod activo y vuelca los actores del mapa y las clases candidatas.
3. De cada clase de jefe anota: vida, variables de estado (`Is Stunned`, `Attacking`), funciones (`ENTER PHASE 2`, `death`), si hereda de la misma clase base que los jefes ya conocidos.
4. Averigua cómo empieza la pelea. En el mapa de Sheinbaum hay que caminar hasta una niebla y pulsar A. En el de AMLO el jefe se puede generar directamente con `BeginDeferredActorSpawnFromClass`, sin niebla.
5. Averigua cómo termina: qué evento marca la victoria y si hay una segunda fase que es un actor distinto.

Cuidado con una cosa: cada victoria y cada activación de portal llaman a `SaveGameToSlot`. Las protecciones de los mods de sondeo lo anulan. No sondees sin ellas.

## Paso 2. Medir golpes

`scripts/medir_jefes.py` hace casi todo. Con una instancia con `exec` habilitado reinicia la pelea con registro de golpes y deja al jugador quieto, así que el jefe recorre todo su repertorio. Para que la medición no termine con la muerte cura al jugador cuando baja de 75 de vida. Cada cierto tiempo cambia la distancia entre los dos y la fracción de vida del jefe, porque algunos ataques (teletransporte, katanas, rayo) solo aparecen a distancia o con poca vida.

```powershell
.\.venv\Scripts\python.exe scripts\medir_jefes.py --jefe chamuco --segundos 900
.\.venv\Scripts\python.exe scripts\medir_jefes.py --jefe amlo --segundos 1200 --fusionar
```

El resultado queda en `bench_results\medicion_<jefe>.json`. Con `--fusionar` se incorpora a `data/jefes/<jefe>.json` sin tocar lo ya medido que la nueva medición no cubre.

Cómo se obtiene la posición de un golpe: se avanza de a 1 frame (1/30 s, la mayor resolución del puente) y, cuando el jugador recibe daño, se registra el montage del jefe y su posición antes y después del paso. La posición del golpe es el centro de ese intervalo. Los golpes se agrupan por cercanía, lo que promedia el error. La resolución es de ±0.017 s de montage. Que un golpe se pueda parar sale de un indicador del propio juego.

Algunos ejemplos de lo que salió, para que tengas una idea:

- `Uppercut` de Sheinbaum: golpe a los 0.883 s, parable, 10 de daño.
- `ChargedCombo`: tres golpes parables de 25 (a 0.717, 1.217 y 1.85 s) y uno no parable de 40 a los 2.85 s.
- La descarga de lanzas de Sheinbaum, `Combo03_01_RM_Montage1`, son 7 golpes no parables de 40, uno cada segundo. Los últimos salen de pocas observaciones, porque casi siempre mataban al jugador antes.

Para los ataques lanzados (el rayo del Chupacabras, la katana lanzada de AMLO) el golpe no está en la animación. Mido el retardo entre el evento y el impacto con el jugador quieto: el rayo cae 1.233 s después del evento, sin desviación en 64 impactos; la katana lanzada tarda `0.156 s + distancia / 4975 unidades por segundo`.

## Paso 3. Qué se esquiva y cómo

`scripts/medir_esquiva.py` teletransporta al jugador un desplazamiento al azar justo después del evento de un ataque a distancia y cuenta si el ataque llega. Sirve para saber si basta con salir del área o si solo la invulnerabilidad de la maniobra salva. En mis jefes casi nada se evita saliendo del área (el golpe de piso del Chamuco conecta el 100% aunque te muevas 1 200 unidades), y por eso la defensa es rodar en el instante del impacto. La medición usa teletransporte, así que no dice si el jugador puede correr tanto en ese tiempo.

## Paso 4. Escribir el registro y el reinicio

1. Crea `data/jefes/<id>.json` con los campos de la tabla de arriba. El campo `episodio.mod` es el que valida `mod/DarkMexicoBridge/Scripts/dmb/spec.lua` con listas de valores admitidos (`id`, `map`, `classes`, `purge`, `start`, `hooks`, `minors`, `phase2`). Un campo desconocido se rechaza.
2. Si el reinicio en sitio no basta, el mod tiene `dmb/arena.lua` (destruir enemigos menores, regenerar al jugador con `Respawn_Player`, generar al jefe) y `dmb/selection.lua` (elegir al jefe por clase).
3. Comprueba que el reinicio sea limpio: 30 reinicios seguidos con el estado inicial idéntico (vida 100/100, 3 caguamas, jefe con su vida completa) y sin crecimiento de memoria. `scripts/validar_reinicios_jefes.py` hace esa comprobación. Cada 150 episodios el entorno recarga el mapa entero para acotar la memoria.

## Paso 5. Validar antes de entrenar

Con el juego real, en este orden:

| Comprobación | Script |
|---|---|
| El entorno cumple `check_env` de Gymnasium | `python -m dmrl verificar` |
| Reinicios limpios, observación finita y en rango, recompensa coherente con la vida del jefe | `scripts/validar_entorno_jefes.py` |
| La máscara de acciones corresponde a lo que el juego acepta | `scripts/validar_mascara_jefes.py` |
| Guardado y logros bloqueados | `scripts/check_guards.py` |

Después, escribe o ajusta un bot de reglas para ese jefe (`src/dmrl/dominio/bot_jefes.py`; una sola implementación parametrizada por el registro). Si el bot no gana al menos el 30% de las peleas (el mínimo que exigí), la tabla o las reglas fallan y no tiene sentido pasar a entrenar. Los cuatro jefes que hay lo superan con holgura.

## Paso 6. Volver a entrenar

El tamaño de la observación v2 se deriva del catálogo: el one-hot de identidad tiene un valor por jefe y el vocabulario de montages es la unión de los de todos. Un jefe nuevo cambia el tamaño de la entrada, y los modelos publicados dejan de cargar. Hay que recolectar demostraciones de todos los jefes, ajustar la imitación y repetir el PPO ([docs/04](04-entrena-tu-ia.md)). Calcula la noche del PPO más las horas de recolección. No probé ningún atajo para conservar los modelos existentes.

## Riesgos conocidos

- Todo esto depende de nombres internos de clases, funciones y animaciones. Una actualización del juego puede renombrarlos y hay que volver a medir.
- Los golpes que llegan después de terminar la animación (rayo, katanas, golpe de piso) no están en la tabla del montage. Van como `post_montage` o como eventos.
- El juego detiene la animación unas décimas de segundo al conectar golpes, de modo que el tiempo de cada golpe se recalcula en cada observación y no desde una hora de inicio fija.
- Un hook de `TakeDamage` que toque al causante sin comprobar `IsValid()` tumba el juego cuando el causante es nulo (el área de las katanas de AMLO y el rayo llegan así).
