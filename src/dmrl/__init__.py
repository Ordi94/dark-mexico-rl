"""
Entrenamiento por refuerzo contra el jefe de Dark Mexico.

Capas (las dependencias apuntan siempre hacia el dominio):
    dominio          reglas puras sin entrada/salida: observacion, acciones y mascara,
                     recompensa, curriculo, tabla de golpes y bot de reglas.
    infraestructura  acceso al juego (paquete dmbridge) y a los procesos del sistema.
    aplicacion       entorno Gymnasium, entrenamiento MaskablePPO y evaluacion.
    entrada          linea de comandos (python -m dmrl).
"""
