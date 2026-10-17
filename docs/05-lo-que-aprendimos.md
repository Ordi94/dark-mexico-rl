# Lo que aprendimos

La versión corta: PPO desde cero fracasó, un bot de reglas hecho a mano sí ganaba, una red que lo copiaba igualaba al bot, y PPO partiendo de esa red lo superó contra uno de los jefes. Fueron unos días, y la mayor parte se fue en problemas de ingeniería y no de aprendizaje.

## Resultados, sin adornos

Todo en peleas completas contra el juego real, jefe al 100% de vida, política determinista. Los intervalos son de Wilson al 95%.

| Etapa | Contra qué | Peleas | Victorias |
|---|---|---|---|
| PPO desde cero, 991 mil pasos (43 min) | Sheinbaum | 40 | 0%, daño medio al jefe 393 de 2 700 |
| Bot de reglas | Sheinbaum | 50 | 72% |
| Bot de reglas | Sheinbaum | 150 | 76% (IC 69-82%) |
| Imitación v1 (solo Sheinbaum) | Sheinbaum | 150 | 79% (IC 71-84%) |
| Tres rondas de DAgger sobre la v1 | Sheinbaum | 150 cada una | 78%, 69%, 67% |
| Bot de reglas, cuatro jefes | Sheinbaum / AMLO / Chamuco / Chupacabras | 50 | 66% / 100% / 100% / 98% |
| Imitación multi-jefe | Sheinbaum / AMLO / Chamuco / Chupacabras | 80 | 76% / 100% / 100% / 100% |
| PPO desde la imitación, checkpoint de 5.67 M pasos | Sheinbaum / AMLO / Chamuco / Chupacabras | 100 | 92% / 92% / 99% / 100% |

La media del PPO es 95.75% contra 94.0% de la imitación. Con 100 peleas esa diferencia no es significativa. La mejora que sí es real está en Sheinbaum: de 76% a 92%. Los intervalos de Sheinbaum (66-84% y 85-96%) casi no se tocan.

AMLO empeoró: de 100% a 92%. Perdió 7 de esas 8 peleas en su segunda fase (el Chamuco) y una por tiempo. No investigué por qué. Mi sospecha es que PPO, al arreglar Sheinbaum, movió la política en una zona que afecta al Chamuco, pero es una sospecha. Con una compuerta de promoción que solo mira la media y una tolerancia de 10 puntos por jefe, un 100% a 92% pasa.

## PPO desde cero: 0%

Lo primero que hice fue lo obvio: MaskablePPO con la máscara de acciones, una red de 2 capas de 256, recompensa por daño, parry y victoria, y 4 instancias del juego. Corrió 43 minutos, 991 mil pasos, a unos 385 pasos por segundo.

La recompensa subió de -7.7 a -1.1. El daño medio al jefe pasó de 66 a 218 por episodio. Las bebidas interrumpidas bajaron de 1.5 a 0.2 por pelea, o sea que aprendió cuándo no beber. Y con eso, contra el jefe al 100% de vida, ganó 0 de 40 peleas.

Lo que no aprendió fue parar golpes. El bot hace 24 parries por pelea y la red 1.5. El parry exige apretar LB con anticipación: la ventana se abre unos 0.17 s después de pulsar y dura unos 0.12 s. Mi explicación es que con acciones al azar eso casi nunca ocurre, y sin parries la red no recibe ninguna señal de que existen. Una exploración mejor quizá lo habría encontrado, pero estimé de 50 a 200 millones de pasos para llegar a ganar con regularidad, entre 1.5 y 6 días seguidos de cómputo. No lo intenté.

## Un bot de reglas ganaba, y eso cambió el plan

Si el juego es determinista en sus tiempos, un programa que lea el montage del jefe y apriete LB 0.235 s antes del impacto debería parar. Lo escribí y acertó 12 de 12 parries. Con las tablas de golpes medidas (ver [docs/06](06-medir-un-jefe.md)) ganó el 72% de las peleas contra Sheinbaum. Que un bot de reglas hiciera lo que la red no podía decía qué hacer: usarlo de maestro.

La imitación es aprendizaje supervisado. Recolecté 600 peleas del bot, con ruido y con arranques en fase 2, y ajusté la misma red a sus decisiones. En validación (episodios que la red nunca vio) coincide con el bot el 99.5% de las veces. En juego saca 79% contra 76% del bot, con intervalos que se solapan: es una copia fiel, no una mejora.

## DAgger: no mejoró

DAgger deja que la red juegue y que el bot etiquete lo que ella visita, para corregir los estados donde se equivoca. Hice tres rondas de 300 peleas. Los resultados en juego fueron 79%, 78%, 69% y 67%, cada uno con 150 peleas. Empeoró o no cambió.

Mi explicación es que la red ya coincidía con el bot en el 99.5% de los estados, así que había poco que corregir y el techo es el propio bot. Más datos de un bot que gana el 76% no producen una red que gane más. Elegí la ronda 0 por su desempeño en juego y no por su pérdida. Eso también importa: el ganador de la selección con 50 peleas sacó 82% y al confirmar con 100 peleas nuevas sacó 77%. La diferencia es suerte, y con 50 peleas la suerte vale 12 puntos.

## PPO desde la imitación: sí funcionó, con cuidado

La imitación ya juega al nivel del bot. Si le pones PPO sin cuidado lo primero que hace es destruirla: el crítico se ajustó a los retornos del bot y no a los de la política que PPO va modificando, así que las ventajas no significan nada. Por eso hay calentamiento del crítico, penalización KL contra la política clonada y una compuerta de promoción (detalle en [docs/04](04-entrena-tu-ia.md)).

La compuerta evita que un mal sorteo estropee el modelo, pero también rechaza buenos modelos por mala suerte. Ejemplo real: la primera evaluación (1.17 millones de pasos, 20 peleas por jefe) dio 13 de 20 contra Sheinbaum, una caída de 10.9 puntos con respecto a la imitación, apenas por encima de la tolerancia de 10. No se promovió. El modelo no era peor; con 20 peleas la incertidumbre es de ±10 puntos. Las siguientes evaluaciones sí se promovieron:

| Pasos | Sheinbaum | AMLO | Chamuco | Chupacabras | Media (20 peleas por jefe) |
|---|---|---|---|---|---|
| 1 172 304 | 65% | 100% | 100% | 100% | 91.3% (no promovido) |
| 2 696 016 | 80% | 100% | 100% | 100% | 95.0% |
| 4 195 152 | 100% | 95% | 100% | 100% | 98.75% |
| 5 669 712 | 100% | 100% | 100% | 100% | 100% |

Con 20 peleas el último checkpoint salió perfecto. Al evaluarlo con 100 peleas por jefe: 92%, 92%, 99% y 100%, media 95.75%. Es probable que ese 100% haya sido en parte suerte. No publico el 100%.

## Problemas que me comieron más tiempo que el aprendizaje

### El juego se caía al recargar

Con reinicios cada pocos segundos (una política al azar muere rápido), el juego se caía con `EXCEPTION_ACCESS_VIOLATION` dentro de `UE4SS.dll` entre cada 13 y cada 46 reinicios. Algunas veces el reinicio quedaba atascado sin encontrar al jugador. Abrí los minidumps con cdb y vi la pila dentro de `FFieldClassVariant::GetFName`. Medí las direcciones de las clases Blueprint antes y después de cada `OpenLevel(LVL_Boss)`: cambiaban en 12 de 12 reinicios. El recolector de basura descarga todas las clases Blueprint del juego y las vuelve a cargar, y UE4SS acababa leyendo propiedades de una clase ya liberada.

La solución es dejar referencias fuertes a las 114 clases Blueprint de `/Game` desde un objeto que el motor no recolecta. Con eso hubo 1 409 reinicios seguidos sin una caída, y de paso el reinicio bajó de 0.8 s a 0.15-0.26 s porque ya no recarga recursos.

### El roll que iba hacia atrás

El jugador ruedaba siempre hacia atrás aunque yo pidiera otra dirección. El juego usa la dirección del último vector de entrada que consumió, no el del mismo frame. Si botón y movimiento llegan en el mismo frame, gana el vector anterior y el personaje rueda hacia atrás. Lo comprobé con las 8 direcciones: 8 de 8 hacia atrás. Con el movimiento un frame antes, rueda en la dirección pedida. El mod ahora manda el movimiento primero y dispara el botón en el frame siguiente. El primer bot sobrevivía pero no ganaba.

### La barra de poder que valía 0.9999999999999999

El ataque especial exige `Power >= 1.0` exacto. La barra sube 0.1 por golpe, y diez sumas de 0.1 dan 0.9999999999999999 en punto flotante. El JSON del mod redondeaba a 9 cifras y mostraba `1`. El primer bot gastaba el 36% de sus decisiones en un especial que el juego rechazaba. Con la comparación exacta hecha en el mod (`power_ready`), el bot pasó de ganar 0 de 12 peleas a ganar 10 de 12.

### Un golpe con causante nulo tumbaba el juego

Un hook de `TakeDamage` llamaba un método sobre el causante del golpe. El área de las katanas de AMLO y el rayo del Chupacabras llegan con causante nulo, y esa llamada tumbaba el juego con un acceso de memoria inválido dentro de UE4SS. `pcall` no ayuda porque la falla ocurre en código nativo. Ahora se comprueba `IsValid()` antes de tocar al causante.

### El protocolo cambió los números

El bot sacaba 88% contra Sheinbaum cuando lo medí con el entorno multi-jefe y 66% después, sin que el bot hubiera cambiado. Lo que cambió fue el protocolo: al principio el entorno sorteaba la segunda fase en el 20% de las peleas, y el bot las gana casi siempre. Ahora todas las cifras de este documento usan la primera fase de cada jefe de forma explícita. Si comparas con otros números, comprueba el protocolo.

### Los núcleos de eficiencia

El PPO corría a 450 pasos por segundo lanzado desde una consola normal y a 250 lanzado desde un PowerShell oculto. Las diferencias de CPU lo explicaban: el proceso y sus juegos acababan en los núcleos de eficiencia del procesador (CPUs 16 a 27 al 80%, los de rendimiento casi libres). Con `cmd.exe` como padre la carga se reparte. Perdí unas horas de entrenamiento antes de verlo. No sé por qué el padre cambia dónde cae el planificador; solo que ocurre. `scripts\iniciar_ppo_desacoplado.ps1` lo evita.

### Otros que valen una línea

- La animación se detiene unas décimas de segundo al conectar golpes. Si calculas el momento del siguiente golpe con una hora de inicio fija, se desfasa hasta 0.1 s y el parry falla. Se recalcula en cada observación.
- AMLO tiene un agarre a distancia (`Paired_ForceChoke`) que mata a los 4.2 s si estás a más de 1 500 unidades y AMLO tiene menos de la mitad de vida. Se evita rodando en el momento justo; sin eso, 2 de 2 muertes de AMLO eran ese agarre.
- Si golpeas a AMLO sin pausa lo dejas aturdido para siempre y no se transforma en el Chamuco. El 18% de las peleas llegaba al límite de tiempo. El bot ahora se detiene a defender cuando AMLO baja del 20%.
- Después de pasar de AMLO al Chamuco el enfoque (lock-on) queda fijo en un objetivo que ya no existe, y el personaje golpea al aire. El entorno lo suelta y lo vuelve a fijar.

## Limitaciones

- Cuatro jefes. El juego tiene más.
- No hay generalización a jefes nuevos. La observación incluye la identidad del jefe y la tabla de golpes; un jefe nuevo exige medirlo.
- La red depende de la tabla de golpes, que se midió una vez. Una actualización del juego que cambie tiempos o nombres de animaciones la invalida.
- El portal del Chupacabras y el acertijo de las estatuas no se ejercitan. La red empieza cada pelea ya frente al jefe.
- Las cifras son de una sola corrida de PPO y una sola semilla de entrenamiento. No sé cuánto varían entre corridas.
