# Dark México RL

[![CI](https://github.com/Ordi94/dark-mexico-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/Ordi94/dark-mexico-rl/actions/workflows/ci.yml)

[English](README.en.md)

Una red neuronal que aprendió a pelear contra los jefes de [Dark México](https://store.steampowered.com/app/4346760/), un juego de acción para un jugador de NorDreams. Primero imita a un bot de reglas que escribí a mano y después mejora sola con PPO. Quería ver si una red podía ganarle a un jefe que yo nunca vencí.

Este proyecto es educativo, no oficial y no tiene relación con NorDreams. Necesitas el juego comprado. El repositorio no incluye ningún archivo del juego.

## Demo

![Panel de neuronas en vivo durante una pelea contra Sheinbaum](docs/img/panel.gif)

Catorce segundos del panel completo en una pelea real contra Sheinbaum, en su segunda fase (el panel muestra "Juice", el nombre interno del jefe): las neuronas de las dos capas de la política, la probabilidad de cada acción, las entradas que más pesaron en la decisión y la confianza V(s) de los últimos 20 s, con una marca verde en cada parry. Se explica en [docs/02-ver-pelear.md](docs/02-ver-pelear.md).

![La IA peleando contra Sheinbaum con el panel de neuronas](docs/img/pelea.gif)

Quince segundos de una pelea real contra Sheinbaum, en su segunda fase: a la izquierda el juego, a la derecha el panel. Se ve cómo la red espera casi siempre y, cuando llega el golpe, la barra de Parry sube al 100% y la IA hace el parry.

## Resultados

Cien peleas por jefe, jefe al 100% de vida, primera fase de cada jefe, política determinista. Todas las cifras salen del juego real, no de una simulación.

| Jefe | Bot de reglas (50 peleas) | Imitación (80 peleas) | PPO (100 peleas) |
|---|---|---|---|
| Sheinbaum (`juice`) | 66% | 76% | 92% |
| AMLO + Chamuco (seguidos) | 100% | 100% | 92% |
| Chamuco | 100% | 100% | 99% |
| Chupacabras | 98% | 100% | 100% |
| Media | 91.0% | 94.0% | 95.75% |

En el código y en los comandos Sheinbaum se llama `juice`, por el nombre de su Blueprint en el juego.

Con 100 peleas una tasa tiene unos 5 puntos de incertidumbre, así que la mejora de la media sobre la imitación está dentro del ruido. Lo que sí cambió es Sheinbaum: la imitación nunca pasó de 76-79% y el PPO llegó a 92%. AMLO bajó de 100% a 92%, y lo cuento en [docs/05-lo-que-aprendimos.md](docs/05-lo-que-aprendimos.md), junto con lo que no funcionó: PPO desde cero terminó con 0% de victorias, y tres rondas de DAgger no mejoraron nada.

Los modelos entrenados no están en el repositorio. Se descargan de la sección Releases (instrucciones en [docs/02-ver-pelear.md](docs/02-ver-pelear.md)).

## Inicio rápido

1. Compra Dark México en Steam e instálalo.
2. Instala UE4SS v3.0.1-1176 y el mod `DarkMexicoBridge` en la carpeta del juego: [docs/01-instalacion.md](docs/01-instalacion.md).
3. Crea un entorno con Python 3.12 e instala el proyecto:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```
4. Comprueba que el puente funciona con el juego real:
   ```powershell
   .\.venv\Scripts\python.exe scripts\bridge_smoke.py
   ```
5. Descarga `darkmexico-multijefe-ppo.zip` de Releases y guárdalo como `runs\multijefe_ppo\mejor.zip`.
6. Mira pelear al modelo con el panel:
   ```powershell
   .\scripts\ver_pelea.ps1 -Multijefe -Jefe amlo -Panel
   ```

Si el juego no está en la ruta normal de Steam, define `DARKMEXICO_DIR` con su carpeta de instalación.

## Cómo funciona

Un mod de UE4SS escrito en Lua vive dentro del juego. Lee el estado del jugador y del jefe, aplica las acciones de la red y avanza el reloj del juego a pasos fijos de 1/30 s. Se comunica con Python por dos named pipes de Windows y el juego se queda esperando la siguiente orden, así que la simulación solo avanza cuando Python decide. Sin ventana (`-nullrhi`) una instancia corre unas 34 veces más rápido que en tiempo real. La red ve 640 números (160 por instante, los últimos 4 apilados), elige entre 22 acciones y el juego le tapa las que no puede ejecutar en ese momento. Los detalles están en [docs/03-como-funciona.md](docs/03-como-funciona.md) y hay una [guía interactiva](docs/guia/index.html) para quien nunca ha visto aprendizaje por refuerzo.

## Ruta de aprendizaje

| Qué quieres hacer | Tiempo | Documento |
|---|---|---|
| Ver pelear a un modelo ya entrenado | 15 minutos | [docs/02-ver-pelear.md](docs/02-ver-pelear.md) |
| Entender el bot de reglas y las demostraciones | una tarde | [docs/04-entrena-tu-ia.md](docs/04-entrena-tu-ia.md) |
| Entrenar la imitación | unas 2 horas | [docs/04-entrena-tu-ia.md](docs/04-entrena-tu-ia.md) |
| Mejorar con PPO | una noche | [docs/04-entrena-tu-ia.md](docs/04-entrena-tu-ia.md) |
| Agregar un jefe nuevo | varios días | [docs/06-medir-un-jefe.md](docs/06-medir-un-jefe.md) |

La historia completa, con los números y los errores, está en [docs/05-lo-que-aprendimos.md](docs/05-lo-que-aprendimos.md).

## Requisitos

- Windows 10 u 11. El puente usa named pipes de Windows y el juego es solo para Windows.
- Dark México comprado en Steam. Probé la build 23613382. Una actualización del juego puede cambiar nombres internos de clases y animaciones, y entonces el mod deja de funcionar hasta que se vuelvan a medir.
- Python 3.12.
- Unos 8 núcleos para entrenar. Con 6 instancias del juego y 2 hilos de PyTorch obtuve unos 450 pasos por segundo en un i7-14700KF. No lo probé con menos núcleos.
- 1 GB de RAM por instancia del juego sin ventana.
- La GPU es opcional. Solo acelera el ajuste por imitación (4 s por época contra 9 s en CPU). El PPO corre en CPU porque con lotes tan pequeños es más rápido.

## Cómo lo hice

Construí este proyecto con ayuda de [Claude Code](https://claude.com/claude-code), el asistente de programación de Anthropic. Buena parte del código, las pruebas y la documentación los escribió el asistente. Mi parte fue decidir qué construir, probar cada etapa en el juego, notar cuando algo no cuadraba (al principio el mod movía la cámara y el personaje se quedaba atrás) y escoger el rumbo: imitar primero al bot, entrenar a todos los jefes con una sola red y no seguir entrenando cuando ya no mejoraba.

También lo usé para aprender. La [guía interactiva](docs/guia/index.html) salió de mis propias preguntas sobre softmax, Tanh y PPO. Si estás empezando, te recomiendo lo mismo: pregunta todo lo que no entiendas y comprueba cada cosa en el juego real, porque ahí aparecieron casi todos los errores.

## Créditos y aviso

Dark México es de NorDreams: <https://store.steampowered.com/app/4346760/>. Este proyecto no está afiliado a NorDreams ni a Steam y no cuenta con su respaldo. Es solo para un jugador, no sirve para jugar en línea y no incluye archivos del juego. Los mods se instalan en tu copia comprada.

El mod usa [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS) (versión v3.0.1-1176), que no se redistribuye aquí. El entrenamiento usa [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), [Gymnasium](https://gymnasium.farama.org/) y [PyTorch](https://pytorch.org/).

## Licencia

MIT, ver [LICENSE](LICENSE). Cubre el código de este repositorio y no el juego.
