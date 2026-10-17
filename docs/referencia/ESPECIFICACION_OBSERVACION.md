# Especificacion de la observacion

Observacion simple: 112 valores float32 (indices 0..111). La politica recibe las ultimas 4 concatenadas (la mas vieja primero): 448 valores.

Todos los valores estan en [0, 1] salvo los cosenos y senos, que estan en [-1, 1]. Un bloque ausente (por ejemplo sin jefe durante una carga) se codifica en cero.

| Indice | Nombre | Significado |
|---|---|---|
| 0 | jugador_vida | vida / vida maxima |
| 1 | jugador_resistencia | Resistencia (estamina), 0..1 |
| 2 | jugador_caguamas | caguamas restantes / 3 |
| 3 | jugador_poder | barra de poder del especial, 0..1 |
| 4 | jugador_bloquea | IsBlocking |
| 5 | jugador_bebiendo | montage de la bebida activo |
| 6 | jugador_progreso_bebida | posicion / duracion del montage de la bebida (0 si no bebe) |
| 7 | jugador_reaccion_golpe | montage de reaccion a golpe (Hit_* o Hurt_*) |
| 8 | jugador_rueda | IsRolling |
| 9 | jugador_ventana_parry | Is Within Parry Window |
| 10 | jugador_ataca | Is Attacking |
| 11 | jugador_puede_atacar | CanAttack |
| 12 | jugador_puede_rodar | CanRoll |
| 13 | jugador_puede_curar | Can Heal |
| 14 | jugador_enfocado | TargetLock? |
| 15 | jugador_velocidad | velocidad horizontal / 800 uu/s |
| 16 | jugador_montage_tiempo | posicion / duracion del montage activo del jugador |
| 17 | jugador_montage_ninguno | one-hot del montage del jugador |
| 18 | jugador_montage_Anim_SwordV2_Combo01_02_RM_Montage | one-hot del montage del jugador |
| 19 | jugador_montage_Anim_Sword_Attack_Slash05_Root_Montage | one-hot del montage del jugador |
| 20 | jugador_montage_Anim_SwordV2_Combo02_01_RM_Montage | one-hot del montage del jugador |
| 21 | jugador_montage_Anim_SwordV2_Combo03_04_RM_Montage | one-hot del montage del jugador |
| 22 | jugador_montage_Anim_SwordV2_Combo03_05_RM_Montage | one-hot del montage del jugador |
| 23 | jugador_montage_NinjaCutDownKick_Montage | one-hot del montage del jugador |
| 24 | jugador_montage_NinjaBlockIntro_Montage1 | one-hot del montage del jugador |
| 25 | jugador_montage_ALS_N_LandRoll_F_Montage | one-hot del montage del jugador |
| 26 | jugador_montage_MM_Idle1_Montage | one-hot del montage del jugador |
| 27 | jugador_montage_Anim_SwordV2_Walk_F_RM_Montage | one-hot del montage del jugador |
| 28 | jugador_montage_Anim_Sword_Hurt_Big_F_Root_Montage | one-hot del montage del jugador |
| 29 | jugador_montage_Anim_SwordV2_Hit_Big_F_IP_Montage | one-hot del montage del jugador |
| 30 | jugador_montage_Anim_SwordV2_Hit_Middle_F_IP_Montage | one-hot del montage del jugador |
| 31 | jugador_montage_Anim_SwordV2_Hit_Middle_L_IP_Montage | one-hot del montage del jugador |
| 32 | jugador_montage_Anim_SwordV2_Hit_Small_F_IP_Montage | one-hot del montage del jugador |
| 33 | jugador_montage_NinjaRLegEvade_Montage | one-hot del montage del jugador |
| 34 | jugador_montage_otro | one-hot del montage del jugador |
| 35 | jefe_vida | vida / vida maxima de su fase (la escala del curriculo ya aplicada) |
| 36 | jefe_fase1 | one-hot de fase: fase 1 |
| 37 | jefe_fase2 | one-hot de fase: fase 2 |
| 38 | jefe_aturdido | Is Stunned (tras un parry) |
| 39 | jefe_ataca | Attacking (no es fiable durante algunos combos) |
| 40 | jefe_velocidad | velocidad horizontal / 800 uu/s |
| 41 | jefe_montage_tiempo | posicion / duracion del montage activo del jefe |
| 42 | jefe_montage_restante | segundos de juego que faltan para terminar el montage / 4, tope 1 |
| 43 | jefe_montage_rate | play rate del montage / 2 |
| 44 | jefe_golpe_pendiente | 1 si queda algun golpe por delante en el montage (tabla medida) |
| 45 | jefe_golpe_tiempo | segundos hasta el proximo golpe / 1.5, tope 1 (1 si no hay) |
| 46 | jefe_golpe_parable | el proximo golpe se puede parar |
| 47 | jefe_parry_tiempo | segundos hasta el proximo golpe parable / 1.5, tope 1 (1 si no hay) |
| 48 | jefe_montage_ninguno | one-hot del montage del jefe |
| 49 | jefe_montage_ANIM_CARPET_Montage | one-hot del montage del jefe |
| 50 | jefe_montage_Boss_Parried_RM_Montage | one-hot del montage del jefe |
| 51 | jefe_montage_Boss_Staggered_RM_Montage | one-hot del montage del jefe |
| 52 | jefe_montage_Boss_Attack_Uppercut_InP_Montage | one-hot del montage del jefe |
| 53 | jefe_montage_Boss_Attack_ChargedCombo_InP_Montage | one-hot del montage del jefe |
| 54 | jefe_montage_Boss_Attack_ChargedCombo_InP_Montage1 | one-hot del montage del jefe |
| 55 | jefe_montage_Boss_Attack_SwingAndSlam_InP_Montage | one-hot del montage del jefe |
| 56 | jefe_montage_Boss_Attack_Running_Simple_InP_Montage | one-hot del montage del jefe |
| 57 | jefe_montage_Boss_Attack_HandAndSwordSwing_InP_Montage | one-hot del montage del jefe |
| 58 | jefe_montage_Anim_SwordV2_Combo03_01_RM_Montage1 | one-hot del montage del jefe |
| 59 | jefe_montage_Anim_SwordV2_Combo03_01_RM_Montage2 | one-hot del montage del jefe |
| 60 | jefe_montage_Anim_SwordV2_Combo04_IP_Montage | one-hot del montage del jefe |
| 61 | jefe_montage_Anim_Sword_Evade_B_Root_Montage | one-hot del montage del jefe |
| 62 | jefe_montage_Anim_Sword_Air_Attack1_InPlace_Montage | one-hot del montage del jefe |
| 63 | jefe_montage_Anim_Sword_Air_Attack2_InPlace_Montage | one-hot del montage del jefe |
| 64 | jefe_montage_Anim_Sword_Air_ComboAttack_InPlace_Montage | one-hot del montage del jefe |
| 65 | jefe_montage_Anim_Sword_Attack_Stab01_Root_Cargada | one-hot del montage del jefe |
| 66 | jefe_montage_Anim_Spear_JumpAttackA_RM_Montage | one-hot del montage del jefe |
| 67 | jefe_montage_A_spear_1hand04_ThrowMid01_Montage | one-hot del montage del jefe |
| 68 | jefe_montage_Anim_SwordV2_Hit_Middle_F_RM_Montage | one-hot del montage del jefe |
| 69 | jefe_montage_Anim_SwordV2_Hit_Small_L_RM_Montage | one-hot del montage del jefe |
| 70 | jefe_montage_Anim_SwordV2_Hit_Small_R_RM_Montage | one-hot del montage del jefe |
| 71 | jefe_montage_NinjaBackEvade_Montage | one-hot del montage del jefe |
| 72 | jefe_montage_NinjaFrontEvade_Montage | one-hot del montage del jefe |
| 73 | jefe_montage_NinjaLeftHook_Individual | one-hot del montage del jefe |
| 74 | jefe_montage_NinjaLeftHook_Montage | one-hot del montage del jefe |
| 75 | jefe_montage_NinjaLowRightThaiKick_Montage | one-hot del montage del jefe |
| 76 | jefe_montage_NinjaObliqueKick_Montage | one-hot del montage del jefe |
| 77 | jefe_montage_NinjaRightCross_Montage | one-hot del montage del jefe |
| 78 | jefe_montage_NinjaRightUppercut_Montage1 | one-hot del montage del jefe |
| 79 | jefe_montage_NinjaSpinningBackfist_Montage | one-hot del montage del jefe |
| 80 | jefe_montage_NinjaStraightKick_Montage | one-hot del montage del jefe |
| 81 | jefe_montage_NinjaWheelKick_Montage | one-hot del montage del jefe |
| 82 | jefe_montage_NinjaWheelKick_Montage1 | one-hot del montage del jefe |
| 83 | jefe_montage_otro | one-hot del montage del jefe |
| 84 | rel_distancia | distancia 2D al jefe / 2000, tope 1 |
| 85 | rel_jefe_cos | coseno del angulo del jefe en el marco del jugador (0 = de frente) |
| 86 | rel_jefe_sen | seno del angulo del jefe en el marco del jugador (+ = a su derecha) |
| 87 | rel_mirada_cos | coseno del angulo entre la mirada del jefe y la direccion hacia el jugador (1 = lo mira) |
| 88 | rel_mirada_sen | seno de ese angulo |
| 89 | accion_previa_nada | one-hot de la accion anterior |
| 90 | accion_previa_mover_0 | one-hot de la accion anterior |
| 91 | accion_previa_mover_45 | one-hot de la accion anterior |
| 92 | accion_previa_mover_90 | one-hot de la accion anterior |
| 93 | accion_previa_mover_135 | one-hot de la accion anterior |
| 94 | accion_previa_mover_180 | one-hot de la accion anterior |
| 95 | accion_previa_mover_225 | one-hot de la accion anterior |
| 96 | accion_previa_mover_270 | one-hot de la accion anterior |
| 97 | accion_previa_mover_315 | one-hot de la accion anterior |
| 98 | accion_previa_roll_0 | one-hot de la accion anterior |
| 99 | accion_previa_roll_45 | one-hot de la accion anterior |
| 100 | accion_previa_roll_90 | one-hot de la accion anterior |
| 101 | accion_previa_roll_135 | one-hot de la accion anterior |
| 102 | accion_previa_roll_180 | one-hot de la accion anterior |
| 103 | accion_previa_roll_225 | one-hot de la accion anterior |
| 104 | accion_previa_roll_270 | one-hot de la accion anterior |
| 105 | accion_previa_roll_315 | one-hot de la accion anterior |
| 106 | accion_previa_ataque | one-hot de la accion anterior |
| 107 | accion_previa_especial | one-hot de la accion anterior |
| 108 | accion_previa_parry | one-hot de la accion anterior |
| 109 | accion_previa_bloqueo | one-hot de la accion anterior |
| 110 | accion_previa_caguama | one-hot de la accion anterior |
| 111 | tiempo_episodio | tiempo transcurrido / duracion maxima del episodio |

Vocabularios de montages (nombre exacto del juego):

- Jugador (18 posiciones): ninguno, Anim_SwordV2_Combo01_02_RM_Montage, Anim_Sword_Attack_Slash05_Root_Montage, Anim_SwordV2_Combo02_01_RM_Montage, Anim_SwordV2_Combo03_04_RM_Montage, Anim_SwordV2_Combo03_05_RM_Montage, NinjaCutDownKick_Montage, NinjaBlockIntro_Montage1, ALS_N_LandRoll_F_Montage, MM_Idle1_Montage, Anim_SwordV2_Walk_F_RM_Montage, Anim_Sword_Hurt_Big_F_Root_Montage, Anim_SwordV2_Hit_Big_F_IP_Montage, Anim_SwordV2_Hit_Middle_F_IP_Montage, Anim_SwordV2_Hit_Middle_L_IP_Montage, Anim_SwordV2_Hit_Small_F_IP_Montage, NinjaRLegEvade_Montage, otro
- Jefe (36 posiciones): ninguno, ANIM_CARPET_Montage, Boss_Parried_RM_Montage, Boss_Staggered_RM_Montage, Boss_Attack_Uppercut_InP_Montage, Boss_Attack_ChargedCombo_InP_Montage, Boss_Attack_ChargedCombo_InP_Montage1, Boss_Attack_SwingAndSlam_InP_Montage, Boss_Attack_Running_Simple_InP_Montage, Boss_Attack_HandAndSwordSwing_InP_Montage, Anim_SwordV2_Combo03_01_RM_Montage1, Anim_SwordV2_Combo03_01_RM_Montage2, Anim_SwordV2_Combo04_IP_Montage, Anim_Sword_Evade_B_Root_Montage, Anim_Sword_Air_Attack1_InPlace_Montage, Anim_Sword_Air_Attack2_InPlace_Montage, Anim_Sword_Air_ComboAttack_InPlace_Montage, Anim_Sword_Attack_Stab01_Root_Cargada, Anim_Spear_JumpAttackA_RM_Montage, A_spear_1hand04_ThrowMid01_Montage, Anim_SwordV2_Hit_Middle_F_RM_Montage, Anim_SwordV2_Hit_Small_L_RM_Montage, Anim_SwordV2_Hit_Small_R_RM_Montage, NinjaBackEvade_Montage, NinjaFrontEvade_Montage, NinjaLeftHook_Individual, NinjaLeftHook_Montage, NinjaLowRightThaiKick_Montage, NinjaObliqueKick_Montage, NinjaRightCross_Montage, NinjaRightUppercut_Montage1, NinjaSpinningBackfist_Montage, NinjaStraightKick_Montage, NinjaWheelKick_Montage, NinjaWheelKick_Montage1, otro
