# Entrena tu propia IA

El camino que funcionó tiene tres etapas: un bot de reglas escrito a mano pelea y deja demostraciones, una red copia al bot, y PPO mejora a esa red. Saltarse la primera cuesta una noche de cómputo y deja la red en 0% de victorias (ver [docs/05](05-lo-que-aprendimos.md)).

Todos los comandos se ejecutan desde la raíz del repositorio, en PowerShell, con la instalación de [docs/01](01-instalacion.md) lista. Los tiempos son los míos: un i7-14700KF (20 núcleos), 128 GB de RAM y una RTX 3060, con la máquina libre. El propio entrenamiento ocupa la máquina, así que haz una copia de `Saved` antes (ver docs/01) y no juegues otra cosa mientras corre.

Los scripts con la opción `-Multijefe` usan la observación v2 y los cuatro jefes. Sin esa opción siguen disponibles los de la primera versión (solo Sheinbaum, que en los comandos se llama `juice`; observación v1), que dejé porque sus pruebas cubren código compartido.

## 1. El bot de reglas

El bot (`src/dmrl/dominio/bot_reglas.py` y `bot_jefes.py`) lee la misma observación que la red y aplica reglas con las tablas de golpes. No usa información que la red no tenga. Sus reglas contra Sheinbaum:

- Parry cuando faltan 0.262 s o menos para un golpe que se puede parar.
- Rodar de lado cuando faltan 0.20 s o menos para uno que no se puede parar.
- Contraataque cuando el jefe está aturdido.
- Beber una caguama con la vida bajo 50, sin golpes pendientes y a 450 unidades o más del jefe.
- Atacar en las ventanas sin golpe en 1.2 s y con estamina de 0.4 o más.

Mídelo antes de gastar cómputo en nada más:

```powershell
.\scripts\evaluar.ps1 -Multijefe -Politica bot -Jefe amlo -Episodios 20
```

Con 50 peleas por jefe el bot saca 66% contra Sheinbaum, 100% contra AMLO, 100% contra el Chamuco y 98% contra el Chupacabras. Es el techo de la imitación: una red que copia al bot no pasa de lo que el bot sabe hacer.

## 2. Recolectar demostraciones

El bot juega peleas completas y cada estado se guarda con la acción que el bot habría elegido. Tres decisiones que cambian el resultado:

- **Ruido DART.** La mitad de las peleas de cada jefe se juegan con una probabilidad por paso (entre 0.05 y 0.20) de ejecutar una acción válida al azar, pero la etiqueta guardada sigue siendo la del bot. Así la red ve cómo se recupera de situaciones donde el bot nunca estaría.
- **Segunda fase.** El 20% de las peleas de Sheinbaum y de AMLO empieza en la segunda fase, para que haya datos de ella sin esperar a que el bot llegue.
- **Etiquetas honestas.** La marca de "golpe atendido" solo se da si la acción se ejecutó de verdad. Antes de corregirlo, con un parry inminente el bot etiquetaba "parry" en el 94% de los estados sin ruido y el 75% con ruido. Ahora son 99% y 100%.

Sheinbaum y los otros tres jefes viven en mapas distintos. Lo hice en dos tandas con las cuatro instancias en el mismo mapa:

```powershell
.\scripts\recolectar_jefes.ps1 -Jefes juice -Episodios 320 -Instancias 4 -Nombre multijefe_imitacion -Ronda 1 -Semilla 100 -MaxMinutos 20
.\scripts\recolectar_jefes.ps1 -Jefes amlo,chamuco,nahual -Episodios 320 -Instancias 4 -Nombre multijefe_imitacion -Ronda 0 -Semilla 0 -MaxMinutos 75
```

Salieron 1 280 peleas y 2 414 963 pasos en 74 minutos (Sheinbaum, 9 min; los tres de AMLO, 65 min). Ocupan 5.9 GB en `runs\multijefe_imitacion\datos`. `-MaxMinutos` corta con las peleas intercaladas por jefe, de modo que todos quedan con la misma cantidad. El script se puede relanzar: omite lo ya guardado.

## 3. Imitación

Una red `MaskablePPO` se ajusta con aprendizaje supervisado a las acciones del bot. Es la misma red que usará el PPO, para que el archivo se pueda cargar directamente.

| Decisión | Valor |
|---|---|
| Pérdida | Entropía cruzada con logits enmascarados: una acción prohibida recibe probabilidad cero. |
| Partición | Por episodio y estratificada por jefe, 15% a validación. Pasos consecutivos de una pelea son casi idénticos; si repartes pasos sueltos, la validación se contamina. |
| Regularización | AdamW con decaimiento de pesos de 1e-2, parada temprana con paciencia 6, tasa a la mitad si la validación se estanca. |
| Pesos de clase | Moderados: (mediana / frecuencia) elevado a 0.5, acotado entre 1/3 y 3. Parry, caguama y especial son raros y deciden la pelea. |
| Cabeza de valor | Se ajusta aparte a los retornos descontados y no toca la política. Un crítico al azar deshace la política clonada en las primeras actualizaciones de PPO. |

Primero mide qué dispositivo conviene. En mi máquina la GPU gana por 2.1 veces (4.1 s por época contra 8.6 s en CPU), porque aquí el ajuste no está dentro del bucle del juego y el lote es grande:

```powershell
.\.venv\Scripts\python.exe -m dmrl imitar-jefes --accion dispositivos
.\.venv\Scripts\python.exe -m dmrl imitar-jefes --accion entrenar --arquitectura 2x256 --dispositivo auto
```

Probé tres arquitecturas (`2x256`, `2x512`, `3x256`). El ajuste completo de una tarda unos 4 minutos. Las tres llegaron a 99.6-99.7% de precisión en validación y a 95-97% de victorias medias en 40 peleas por jefe; la capacidad no era el cuello de botella. Elegí `2x256` por ser la más pequeña y porque empató dentro del ruido: 94.0% con 80 peleas por jefe contra 93.8% de `3x256`.

Evalúa en juego y publica el modelo. La publicación copia el modelo a `mejor.zip` y escribe `referencia.json`, que más adelante sirve de piso al PPO:

```powershell
.\.venv\Scripts\python.exe -m dmrl evaluar-modelo-jefes --modelo runs\multijefe_imitacion\modelos\2x256.zip --episodios 40 --instancias 6 --semilla 1000 --prefijo evi --salida runs\multijefe_imitacion\evaluaciones\2x256.json
.\.venv\Scripts\python.exe -m dmrl imitar-jefes --accion publicar --arquitectura 2x256 --evaluacion 2x256
```

Lo que obtuve: Sheinbaum 76% (60 de 79), AMLO, Chamuco y Chupacabras 100% (80 de 80 cada uno).

## 4. PPO desde la imitación

```powershell
.\scripts\entrenar.ps1 -Multijefe
```

Parte de `runs\multijefe_imitacion\mejor.zip`, usa 6 instancias del juego (2 en el mapa de Sheinbaum y 4 en el de AMLO), abre TensorBoard en <http://localhost:6006> y se reanuda solo desde `runs\multijefe_ppo\checkpoints\ultimo.zip`. Para detenerlo con orden: `.\scripts\detener.ps1 -Multijefe` (unos 7 s, guarda el modelo y cierra los juegos).

Para una corrida de toda la noche que sobreviva al cierre de la consola, usa `.\scripts\iniciar_ppo_desacoplado.ps1`. Lanzado desde un PowerShell oculto, el entrenamiento cayó a la mitad de pasos por segundo (250 contra 450). El proceso y sus juegos terminaron en los núcleos de eficiencia del procesador; con `cmd.exe` como padre se reparten bien. No encontré una causa más fina.

Una imitación ya juega como el bot, y PPO mal calibrado la destruye en las primeras actualizaciones. Las defensas:

| Defensa | Cómo |
|---|---|
| Calentamiento del crítico | Los primeros 200 000 pasos solo se entrena la cabeza de valor. La política tiene gradiente nulo. |
| Penalización KL | Después se suma `beta * KL(política clonada || política actual)`. `beta` empieza en 0.3 y baja linealmente a cero en 5 millones de pasos. |
| Pasos pequeños | Tasa de aprendizaje 1e-4 constante, recorte 0.1, `ent_coef` 0.001, `target_kl` 0.02. |
| Compuerta de promoción | `mejor.zip` solo cambia si un checkpoint evaluado en juego mejora la media y ningún jefe cae más de 10 puntos respecto a la imitación. |
| Regresión | Si la media queda 10 puntos por debajo de la referencia en 2 evaluaciones seguidas, se vuelve a los pesos de `mejor.zip` y la tasa de aprendizaje se reduce a la mitad. |

La implementación está en `src/dmrl/aplicacion/ppo_kl.py` (`MaskablePPOKL` copia el cuerpo de `MaskablePPO.train` de sb3-contrib 2.9.0 con esos tres cambios) y `callbacks_jefes.py`.

El resto de los parámetros: `n_steps` 1024 por entorno (6 144 por actualización), minilotes de 512, 5 épocas, `gamma` 0.995, `gae_lambda` 0.95, CPU con 2 hilos de torch. En CPU porque cada paso mueve un lote diminuto: medí 437 pasos/s en CPU contra 358 en la GPU.

Cada 45 minutos (la primera a los 20) un proceso aparte juega 20 peleas por jefe con una copia del modelo, sin detener el entrenamiento. Con 20 peleas una tasa tiene unos 10 puntos de incertidumbre, así que esa evaluación decide rápido, no con precisión. Cuando quieras una cifra para publicar, detén el entrenamiento y evalúa con 100 peleas por jefe:

```powershell
.\scripts\detener.ps1 -Multijefe
.\scripts\evaluar.ps1 -Multijefe -Jefe todos -Episodios 100
```

Mi mejor modelo salió del checkpoint de 5 669 712 pasos, unas 4 horas de reloj a unos 450 pasos por segundo. La corrida terminó en 6.7 millones de pasos y ninguna evaluación posterior lo superó, porque la última evaluación automática fue justo la de ese checkpoint.

## 5. Mirar el entrenamiento

Abre TensorBoard en <http://localhost:6006>, pestaña Scalars:

| Grupo | Qué mirar |
|---|---|
| `dm/<jefe>/tasa_victoria` | Ventana de 50 peleas por jefe. Incluye peleas con política muestreada, así que sale más baja que la evaluación determinista. |
| `dm/` | `tasa_victoria_media`, `pasos_por_segundo` (esperado 420-475), `errores_instancia_total` (esperado 0), `kl_referencia`, `beta_kl`. |
| `dm/eval/` | `tasa_media`, `mejor_media`, `promovido`, `revertido` de cada evaluación automática. |
| `train/` | `explained_variance` (cerca de 1 es bueno), `entropy_loss` (en valor absoluto, 3.09 sería azar uniforme entre 22 acciones), `approx_kl`. |

Dos cosas que salen anormales y no lo son: `entropy_loss` cerca de 0 (la política clonada es casi determinista) y `kl_referencia` pequeño al principio (la política casi no se movió). Si después de unas horas la media no sube, el siguiente paso es probar con `--beta-kl 0`, que suelta la penalización. Yo no tuve que hacerlo.

Un resumen en una pantalla: `.\.venv\Scripts\python.exe scripts\resumen_ppo_jefes.py`.

## Riesgos que conviene conocer

- Los seis juegos avanzan a la vez. Un juego lento o una caída frena a todos. Las caídas se cuentan en `dm/errores_instancia_total` y el entorno relanza la instancia.
- El entrenamiento no se reanuda tras reiniciar Windows. El último checkpoint tiene como máximo 100 000 pasos (3-4 minutos).
- La evaluación automática compite por CPU durante unos 10 minutos con sus 4 instancias.
- Si cambias el vocabulario o el orden de la observación, los modelos ya entrenados dejan de servir.
