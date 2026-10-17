"""
Vocabulario fijo de montages del jugador y del jefe para la observacion one-hot.

Origen: montages observados en 250 episodios reales (data/vocabulario_montages.json, medidos
con scripts/medir_golpes.py) mas los montages de combate de los volcados del SDK
(sdk/UE4SS_ObjectDump.txt) que no aparecieron en esas corridas. Cualquier otro nombre se
codifica como "otro/desconocido". El orden es parte de la especificacion: cambiarlo invalida
los modelos entrenados.
"""

from __future__ import annotations

VOCAB_JUGADOR: tuple[str, ...] = (
    "Anim_SwordV2_Combo01_02_RM_Montage",
    "Anim_Sword_Attack_Slash05_Root_Montage",
    "Anim_SwordV2_Combo02_01_RM_Montage",
    "Anim_SwordV2_Combo03_04_RM_Montage",
    "Anim_SwordV2_Combo03_05_RM_Montage",
    "NinjaCutDownKick_Montage",
    "NinjaBlockIntro_Montage1",
    "ALS_N_LandRoll_F_Montage",
    "MM_Idle1_Montage",
    "Anim_SwordV2_Walk_F_RM_Montage",
    "Anim_Sword_Hurt_Big_F_Root_Montage",
    "Anim_SwordV2_Hit_Big_F_IP_Montage",
    "Anim_SwordV2_Hit_Middle_F_IP_Montage",
    "Anim_SwordV2_Hit_Middle_L_IP_Montage",
    "Anim_SwordV2_Hit_Small_F_IP_Montage",
    "NinjaRLegEvade_Montage",
)

VOCAB_JEFE: tuple[str, ...] = (
    "ANIM_CARPET_Montage",
    "Boss_Parried_RM_Montage",
    "Boss_Staggered_RM_Montage",
    "Boss_Attack_Uppercut_InP_Montage",
    "Boss_Attack_ChargedCombo_InP_Montage",
    "Boss_Attack_ChargedCombo_InP_Montage1",
    "Boss_Attack_SwingAndSlam_InP_Montage",
    "Boss_Attack_Running_Simple_InP_Montage",
    "Boss_Attack_HandAndSwordSwing_InP_Montage",
    "Anim_SwordV2_Combo03_01_RM_Montage1",
    "Anim_SwordV2_Combo03_01_RM_Montage2",
    "Anim_SwordV2_Combo04_IP_Montage",
    "Anim_Sword_Evade_B_Root_Montage",
    "Anim_Sword_Air_Attack1_InPlace_Montage",
    "Anim_Sword_Air_Attack2_InPlace_Montage",
    "Anim_Sword_Air_ComboAttack_InPlace_Montage",
    "Anim_Sword_Attack_Stab01_Root_Cargada",
    "Anim_Spear_JumpAttackA_RM_Montage",
    "A_spear_1hand04_ThrowMid01_Montage",
    "Anim_SwordV2_Hit_Middle_F_RM_Montage",
    "Anim_SwordV2_Hit_Small_L_RM_Montage",
    "Anim_SwordV2_Hit_Small_R_RM_Montage",
    "NinjaBackEvade_Montage",
    "NinjaFrontEvade_Montage",
    "NinjaLeftHook_Individual",
    "NinjaLeftHook_Montage",
    "NinjaLowRightThaiKick_Montage",
    "NinjaObliqueKick_Montage",
    "NinjaRightCross_Montage",
    "NinjaRightUppercut_Montage1",
    "NinjaSpinningBackfist_Montage",
    "NinjaStraightKick_Montage",
    "NinjaWheelKick_Montage",
    "NinjaWheelKick_Montage1",
)

INDICE_NINGUNO = 0


def tamano_codificacion(vocab: tuple[str, ...]) -> int:
    """
    Longitud del one-hot de un vocabulario.

    Contexto:
        ConstructorObservacion; incluye las posiciones "ninguno" y "otro".

    Args:
        vocab (tuple[str, ...]): Vocabulario de montages.

    Returns:
        int: len(vocab) + 2.

    Raises:
        Ninguna.

    Ejemplo:
        tamano_codificacion(("A", "B"))  # 4
    """
    return len(vocab) + 2


def indice_montage(nombre: str | None, vocab: tuple[str, ...]) -> int:
    """
    Posicion de un montage en el one-hot de su vocabulario.

    Contexto:
        ConstructorObservacion, una vez por entidad y por paso.

    Args:
        nombre (str | None): Nombre del montage activo; None si no hay.
        vocab (tuple[str, ...]): Vocabulario de montages.

    Returns:
        int: 0 si no hay montage, 1..n si esta en el vocabulario, n+1 si es desconocido.

    Raises:
        Ninguna.

    Ejemplo:
        indice_montage("A", ("A", "B"))  # 1
    """
    if not nombre:
        return INDICE_NINGUNO
    try:
        return vocab.index(nombre) + 1
    except ValueError:
        return len(vocab) + 1
