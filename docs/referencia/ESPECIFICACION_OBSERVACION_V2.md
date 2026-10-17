# Especificacion de la observacion v2 (multi-jefe)

Observacion simple: 160 valores float32 (indices 0..159). La politica recibe las ultimas 4 concatenadas (la mas vieja primero): 640 valores.

Todos los valores estan en [0, 1] salvo los cosenos y senos, que estan en [-1, 1]. Un bloque ausente (por ejemplo sin jefe durante una carga) se codifica en cero. La observacion v1 (docs/ESPECIFICACION_OBSERVACION.md) no cambia: los modelos entrenados con ella siguen cargando.

Bloques: jugador (35, igual que la v1), identidad y estado del jefe, montage del jefe (vocabulario union), amenaza generica (tabla del jefe), amenaza a distancia, relativo jugador-jefe, accion previa y tiempo de episodio.

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
| 35 | jefe_id_juice | one-hot de la identidad del actor jefe (por su clase Blueprint) |
| 36 | jefe_id_juice_fase2 | one-hot de la identidad del actor jefe (por su clase Blueprint) |
| 37 | jefe_id_amlo | one-hot de la identidad del actor jefe (por su clase Blueprint) |
| 38 | jefe_id_chamuco | one-hot de la identidad del actor jefe (por su clase Blueprint) |
| 39 | jefe_id_nahual | one-hot de la identidad del actor jefe (por su clase Blueprint) |
| 40 | jefe_vida | vida / vida maxima del actor |
| 41 | jefe_fase_avanzada | 1 si el actor es la segunda fase de su pelea (juice_fase2, chamuco) |
| 42 | jefe_aturdido | Is Stunned (tras un parry) |
| 43 | jefe_ataca | Attacking (no es fiable durante algunos combos) |
| 44 | jefe_teletransportando | Is Teleporting (solo AMLO y el Chupacabras; 0 en los demas) |
| 45 | jefe_velocidad | velocidad horizontal / 800 uu/s |
| 46 | jefe_montage_tiempo | posicion / duracion del montage activo del jefe |
| 47 | jefe_montage_restante | segundos de juego que faltan para terminar el montage / 4, tope 1 |
| 48 | jefe_montage_rate | play rate del montage / 2 |
| 49 | jefe_montage_ninguno | one-hot del montage del jefe (vocabulario union) |
| 50 | jefe_montage_ANIM_CARPET_Montage | one-hot del montage del jefe (vocabulario union) |
| 51 | jefe_montage_A_spear_1hand04_ThrowMid01_Montage | one-hot del montage del jefe (vocabulario union) |
| 52 | jefe_montage_Anim_Spear_JumpAttackA_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 53 | jefe_montage_Anim_SwordV2_Combo03_01_RM_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 54 | jefe_montage_Anim_SwordV2_Combo03_01_RM_Montage2 | one-hot del montage del jefe (vocabulario union) |
| 55 | jefe_montage_Anim_SwordV2_Combo04_IP_Montage | one-hot del montage del jefe (vocabulario union) |
| 56 | jefe_montage_Anim_SwordV2_Hit_Middle_F_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 57 | jefe_montage_Anim_SwordV2_Hit_Small_L_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 58 | jefe_montage_Anim_SwordV2_Hit_Small_R_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 59 | jefe_montage_Anim_Sword_Air_Attack1_InPlace_Montage | one-hot del montage del jefe (vocabulario union) |
| 60 | jefe_montage_Anim_Sword_Air_Attack2_InPlace_Montage | one-hot del montage del jefe (vocabulario union) |
| 61 | jefe_montage_Anim_Sword_Air_ComboAttack_InPlace_Montage | one-hot del montage del jefe (vocabulario union) |
| 62 | jefe_montage_Anim_Sword_Attack_Stab01_Root_Cargada | one-hot del montage del jefe (vocabulario union) |
| 63 | jefe_montage_Anim_Sword_Evade_B_Root_Montage | one-hot del montage del jefe (vocabulario union) |
| 64 | jefe_montage_Boss_Attack_ChargedCombo_InP_Montage | one-hot del montage del jefe (vocabulario union) |
| 65 | jefe_montage_Boss_Attack_ChargedCombo_InP_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 66 | jefe_montage_Boss_Attack_HandAndSwordSwing_InP_Montage | one-hot del montage del jefe (vocabulario union) |
| 67 | jefe_montage_Boss_Attack_Running_Simple_InP_Montage | one-hot del montage del jefe (vocabulario union) |
| 68 | jefe_montage_Boss_Attack_SwingAndSlam_InP_Montage | one-hot del montage del jefe (vocabulario union) |
| 69 | jefe_montage_Boss_Attack_Uppercut_InP_Montage | one-hot del montage del jefe (vocabulario union) |
| 70 | jefe_montage_Boss_Parried_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 71 | jefe_montage_Boss_Staggered_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 72 | jefe_montage_NinjaBackEvade_Montage | one-hot del montage del jefe (vocabulario union) |
| 73 | jefe_montage_NinjaSpinningBackfist_Montage | one-hot del montage del jefe (vocabulario union) |
| 74 | jefe_montage_NinjaWheelKick_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 75 | jefe_montage_NinjaStraightKick_Montage | one-hot del montage del jefe (vocabulario union) |
| 76 | jefe_montage_NinjaObliqueKick_Montage | one-hot del montage del jefe (vocabulario union) |
| 77 | jefe_montage_NinjaLowRightThaiKick_Montage | one-hot del montage del jefe (vocabulario union) |
| 78 | jefe_montage_NinjaRightCross_Montage | one-hot del montage del jefe (vocabulario union) |
| 79 | jefe_montage_NinjaFrontEvade_Montage | one-hot del montage del jefe (vocabulario union) |
| 80 | jefe_montage_NinjaWheelKick_Montage | one-hot del montage del jefe (vocabulario union) |
| 81 | jefe_montage_NinjaLeftHook_Individual | one-hot del montage del jefe (vocabulario union) |
| 82 | jefe_montage_NinjaLeftHook_Montage | one-hot del montage del jefe (vocabulario union) |
| 83 | jefe_montage_NinjaRightUppercut_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 84 | jefe_montage_AS_Mage_Powerup_Light_01_Montage | one-hot del montage del jefe (vocabulario union) |
| 85 | jefe_montage_Anim_CK3_ComboA_01_IP_Montage | one-hot del montage del jefe (vocabulario union) |
| 86 | jefe_montage_Anim_CK3_ComboA_01_RM_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 87 | jefe_montage_Anim_CK3_ComboA_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 88 | jefe_montage_Anim_CK3_ComboA_RM_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 89 | jefe_montage_Anim_CK3_ComboB_03_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 90 | jefe_montage_Anim_CK3_ComboB_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 91 | jefe_montage_Anim_CK3_ComboC_01_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 92 | jefe_montage_Anim_CK3_ComboC_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 93 | jefe_montage_Anim_CK3_ComboD_02_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 94 | jefe_montage_Anim_CK3_ComboE_01_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 95 | jefe_montage_Anim_Sword_Attack_Slash04_Root_Montage | one-hot del montage del jefe (vocabulario union) |
| 96 | jefe_montage_NinjaLeftEvade1_Montage | one-hot del montage del jefe (vocabulario union) |
| 97 | jefe_montage_NinjaRightEvade1_Montage | one-hot del montage del jefe (vocabulario union) |
| 98 | jefe_montage_Paired_ForceChoke_Att_Montage | one-hot del montage del jefe (vocabulario union) |
| 99 | jefe_montage_1HM_AerialSwing01_2HitCombo_Montage | one-hot del montage del jefe (vocabulario union) |
| 100 | jefe_montage_1HM_AerialSwing01_Forward_Montage | one-hot del montage del jefe (vocabulario union) |
| 101 | jefe_montage_Anim_CK3_ComboE_01_RM_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 102 | jefe_montage_Anim_SwordV2_Combo02_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 103 | jefe_montage_Anim_SwordV2_Slash_RM_Montage | one-hot del montage del jefe (vocabulario union) |
| 104 | jefe_montage_Anim_Sword_Attack_Slash03_Root_Montage | one-hot del montage del jefe (vocabulario union) |
| 105 | jefe_montage_Anim_Sword_Attack_Slash03_Root_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 106 | jefe_montage_Boss_Attack_Uppercut_InP_Montage1 | one-hot del montage del jefe (vocabulario union) |
| 107 | jefe_montage_Boss_Attack_Uppercut_InP_Montage2 | one-hot del montage del jefe (vocabulario union) |
| 108 | jefe_montage_AS_Mage_Ability_Spell_HighRtLight_02_Montage | one-hot del montage del jefe (vocabulario union) |
| 109 | jefe_montage_AS_Mage_Elemental_Fire_Volley_01_Montage | one-hot del montage del jefe (vocabulario union) |
| 110 | jefe_montage_Anim_Whisper_Attack1_Montage | one-hot del montage del jefe (vocabulario union) |
| 111 | jefe_montage_Anim_Whisper_Attack2_Montage | one-hot del montage del jefe (vocabulario union) |
| 112 | jefe_montage_Anim_Whisper_Attack3_Montage | one-hot del montage del jefe (vocabulario union) |
| 113 | jefe_montage_NinjaHeadHit_02_Montage | one-hot del montage del jefe (vocabulario union) |
| 114 | jefe_montage_otro | one-hot del montage del jefe (vocabulario union) |
| 115 | amenaza_hay | 1 si hay algun impacto pendiente (montage o evento) |
| 116 | amenaza_golpe_tiempo | segundos hasta el proximo impacto / 1.5, tope 1 (1 si no hay) |
| 117 | amenaza_golpe_parable | el proximo impacto se puede parar |
| 118 | amenaza_golpe_danio | dano del proximo impacto / 50, tope 1 |
| 119 | amenaza_golpes_pendientes | impactos que quedan en el ataque en curso / 5, tope 1 |
| 120 | amenaza_en_alcance | el jugador esta dentro del alcance del proximo impacto |
| 121 | amenaza_parry_tiempo | segundos hasta el proximo impacto parable / 1.5, tope 1 (1 si no hay) |
| 122 | amenaza_esquivar_tiempo | segundos hasta el proximo impacto no parable / 1.5, tope 1 (1 si no hay) |
| 123 | remota_activa | 1 si hay un ataque a distancia pendiente (proyectil, area, rayo o evento) |
| 124 | remota_tiempo | segundos al impacto de la amenaza a distancia / 2.5, tope 1 (1 si no hay) |
| 125 | remota_esquivable | se evita con la invulnerabilidad de la maniobra |
| 126 | remota_parable | se puede parar |
| 127 | remota_danio | dano de la amenaza a distancia / 50, tope 1 |
| 128 | remota_origen_distancia | distancia al origen del ataque / 1000, tope 1 (0 si cae sobre el jugador) |
| 129 | remota_origen_cos | coseno del angulo del origen en el marco del jugador (0 = de frente) |
| 130 | remota_origen_sen | seno de ese angulo (+ = a su derecha) |
| 131 | remota_sobre_jugador | el ataque cae sobre la posicion del jugador (area, rayo) |
| 132 | rel_distancia | distancia 2D al jefe / 4000, tope 1 |
| 133 | rel_jefe_cos | coseno del angulo del jefe en el marco del jugador (0 = de frente) |
| 134 | rel_jefe_sen | seno del angulo del jefe en el marco del jugador (+ = a su derecha) |
| 135 | rel_mirada_cos | coseno del angulo entre la mirada del jefe y la direccion hacia el jugador (1 = lo mira) |
| 136 | rel_mirada_sen | seno de ese angulo |
| 137 | accion_previa_nada | one-hot de la accion anterior |
| 138 | accion_previa_mover_0 | one-hot de la accion anterior |
| 139 | accion_previa_mover_45 | one-hot de la accion anterior |
| 140 | accion_previa_mover_90 | one-hot de la accion anterior |
| 141 | accion_previa_mover_135 | one-hot de la accion anterior |
| 142 | accion_previa_mover_180 | one-hot de la accion anterior |
| 143 | accion_previa_mover_225 | one-hot de la accion anterior |
| 144 | accion_previa_mover_270 | one-hot de la accion anterior |
| 145 | accion_previa_mover_315 | one-hot de la accion anterior |
| 146 | accion_previa_roll_0 | one-hot de la accion anterior |
| 147 | accion_previa_roll_45 | one-hot de la accion anterior |
| 148 | accion_previa_roll_90 | one-hot de la accion anterior |
| 149 | accion_previa_roll_135 | one-hot de la accion anterior |
| 150 | accion_previa_roll_180 | one-hot de la accion anterior |
| 151 | accion_previa_roll_225 | one-hot de la accion anterior |
| 152 | accion_previa_roll_270 | one-hot de la accion anterior |
| 153 | accion_previa_roll_315 | one-hot de la accion anterior |
| 154 | accion_previa_ataque | one-hot de la accion anterior |
| 155 | accion_previa_especial | one-hot de la accion anterior |
| 156 | accion_previa_parry | one-hot de la accion anterior |
| 157 | accion_previa_bloqueo | one-hot de la accion anterior |
| 158 | accion_previa_caguama | one-hot de la accion anterior |
| 159 | tiempo_episodio | tiempo transcurrido / duracion maxima del episodio |

Identidades (orden del one-hot): 0 juice, 1 juice_fase2, 2 amlo, 3 chamuco, 4 nahual

Vocabulario union de montages del jefe (66 posiciones, nombre exacto del juego): ninguno, ANIM_CARPET_Montage, A_spear_1hand04_ThrowMid01_Montage, Anim_Spear_JumpAttackA_RM_Montage, Anim_SwordV2_Combo03_01_RM_Montage1, Anim_SwordV2_Combo03_01_RM_Montage2, Anim_SwordV2_Combo04_IP_Montage, Anim_SwordV2_Hit_Middle_F_RM_Montage, Anim_SwordV2_Hit_Small_L_RM_Montage, Anim_SwordV2_Hit_Small_R_RM_Montage, Anim_Sword_Air_Attack1_InPlace_Montage, Anim_Sword_Air_Attack2_InPlace_Montage, Anim_Sword_Air_ComboAttack_InPlace_Montage, Anim_Sword_Attack_Stab01_Root_Cargada, Anim_Sword_Evade_B_Root_Montage, Boss_Attack_ChargedCombo_InP_Montage, Boss_Attack_ChargedCombo_InP_Montage1, Boss_Attack_HandAndSwordSwing_InP_Montage, Boss_Attack_Running_Simple_InP_Montage, Boss_Attack_SwingAndSlam_InP_Montage, Boss_Attack_Uppercut_InP_Montage, Boss_Parried_RM_Montage, Boss_Staggered_RM_Montage, NinjaBackEvade_Montage, NinjaSpinningBackfist_Montage, NinjaWheelKick_Montage1, NinjaStraightKick_Montage, NinjaObliqueKick_Montage, NinjaLowRightThaiKick_Montage, NinjaRightCross_Montage, NinjaFrontEvade_Montage, NinjaWheelKick_Montage, NinjaLeftHook_Individual, NinjaLeftHook_Montage, NinjaRightUppercut_Montage1, AS_Mage_Powerup_Light_01_Montage, Anim_CK3_ComboA_01_IP_Montage, Anim_CK3_ComboA_01_RM_Montage1, Anim_CK3_ComboA_RM_Montage, Anim_CK3_ComboA_RM_Montage1, Anim_CK3_ComboB_03_RM_Montage, Anim_CK3_ComboB_RM_Montage, Anim_CK3_ComboC_01_RM_Montage, Anim_CK3_ComboC_RM_Montage, Anim_CK3_ComboD_02_RM_Montage, Anim_CK3_ComboE_01_RM_Montage, Anim_Sword_Attack_Slash04_Root_Montage, NinjaLeftEvade1_Montage, NinjaRightEvade1_Montage, Paired_ForceChoke_Att_Montage, 1HM_AerialSwing01_2HitCombo_Montage, 1HM_AerialSwing01_Forward_Montage, Anim_CK3_ComboE_01_RM_Montage1, Anim_SwordV2_Combo02_RM_Montage, Anim_SwordV2_Slash_RM_Montage, Anim_Sword_Attack_Slash03_Root_Montage, Anim_Sword_Attack_Slash03_Root_Montage1, Boss_Attack_Uppercut_InP_Montage1, Boss_Attack_Uppercut_InP_Montage2, AS_Mage_Ability_Spell_HighRtLight_02_Montage, AS_Mage_Elemental_Fire_Volley_01_Montage, Anim_Whisper_Attack1_Montage, Anim_Whisper_Attack2_Montage, Anim_Whisper_Attack3_Montage, NinjaHeadHit_02_Montage, otro
