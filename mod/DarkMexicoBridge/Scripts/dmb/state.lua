-- Construccion de la observacion que se envia a Python en cada paso.
--
-- Unidades: distancias en unidades de Unreal (1 uu = 1 cm), angulos en grados, tiempos en
-- segundos de juego salvo "wall" (segundos de reloj real desde que arranco el proceso),
-- velocidades en uu/s. Las propiedades Blueprint se leen con su nombre exacto (algunas llevan
-- espacios o signos, como "Is Within Parry Window" o "DEATH?").
--
-- Cada entidad se lee dentro de un solo pcall: si el actor se destruye a mitad de la lectura
-- la entidad sale como nula en vez de propagar el error al bucle del juego.

local G = require("dmb.game")

local S = {}

-- Nombre y duracion de cada montage, por direccion del objeto: convertir el FName a texto en
-- cada paso es la operacion mas cara de la lectura y los montages del combate son pocos. Los
-- montages se recargan con cada OpenLevel (y sus direcciones se reutilizan), asi que la cache
-- se vacia cada vez que cambia el mundo.
local montage_cache = {}
local cache_gen = -1

local function montage_info(ch)
    if cache_gen ~= G.world_gen then
        montage_cache = {}
        cache_gen = G.world_gen
    end
    local ai = ch.Mesh:GetAnimInstance()
    if not G.valid(ai) then return nil end
    local m = ai:GetCurrentActiveMontage()
    if not G.valid(m) then return nil end
    local addr = m:GetAddress()
    local info = montage_cache[addr]
    if not info then
        local okl, len = pcall(function() return m:GetPlayLength() end)
        info = { name = m:GetFName():ToString(), len = okl and len or -1.0 }
        montage_cache[addr] = info
    end
    return {
        name = info.name,
        pos = ai:Montage_GetPosition(m),
        rate = ai:Montage_GetPlayRate(m),
        len = info.len,
    }
end

--- Vacia la cache de montages (los objetos pueden reciclarse tras cambiar de mapa).
function S.reset_cache()
    montage_cache = {}
end

local function actor_kinematics(a, out)
    local loc = a:K2_GetActorLocation()
    local rot = a:K2_GetActorRotation()
    local vel = a:GetVelocity()
    out.pos = { loc.X, loc.Y, loc.Z }
    out.yaw = rot.Yaw
    out.vel = { vel.X, vel.Y, vel.Z }
end

local function read_player(p, pc)
    local dmg = p.BPC_DamageSystem
    local out = {
        hp = dmg.Health,
        hp_max = dmg.MaxHealth,
        dead = dmg.IsDead,
        blocking = dmg.IsBlocking,
        invincible = dmg.IsInvincible,
        stamina = p.Resistencia,
        power = p.Power,
        -- La barra sube de 0.1 en 0.1 y diez sumas dan 0.9999999999999999: el juego exige Power >= 1.0
        -- exacto para el especial, pero el JSON redondea a 9 cifras y mostraria 1. Se envia el booleano.
        power_ready = p.Power >= 1.0,
        caguamas = p.Caguamas,
        attacking = p["Is Attacking"],
        combo = p["Combo Counter"],
        rolling = p.IsRolling,
        parry_window = p["Is Within Parry Window"],
        can_heal = p["Can Heal"],
        can_attack = p.CanAttack,
        can_roll = p.CanRoll,
        target_lock = p["TargetLock?"],
    }
    actor_kinematics(p, out)
    if pc then out.control_yaw = pc:GetControlRotation().Yaw end
    out.montage = montage_info(p)
    -- La bebida de caguama es MM_Idle1_Montage: durante 2.15 s solo se puede caminar.
    out.drinking = out.montage ~= nil and out.montage.name == "MM_Idle1_Montage"
    return out
end

local function read_boss(b)
    local dmg = b.BPC_DamageSystem
    local out = {
        cls = G.class_name(b),
        hp = dmg.Health,
        hp_max = dmg.MaxHealth,
        phase = b.Phase,
        attacking = b.Attacking,
        stunned = b["Is Stunned"],
        dead = b["DEATH?"],
        parry_count = b["Parry Count"],
    }
    -- Solo AMLO y el Chupacabras se teletransportan; en las demas clases la propiedad no existe.
    local tp = b["Is Teleporting"]
    if type(tp) == "boolean" then out.teleporting = tp end
    actor_kinematics(b, out)
    out.montage = montage_info(b)
    return out
end

--- Observacion completa. events es la tabla de eventos acumulados desde el paso anterior.
function S.build(events, extra)
    local st = {
        frame = G.frame,
        map = G.map,
        stable = G.stable(),
        wall = os.clock(),
    }
    if st.stable then
        st.t = G.time_seconds()
        st.dt = G.delta_seconds()
        local pc = G.player_controller()
        local p = G.pawn()
        if p then
            local ok, r = pcall(read_player, p, pc)
            if ok then st.player = r else st.player_err = tostring(r) end
        end
        local b = G.boss_actor()
        if b then
            local ok, r = pcall(read_boss, b)
            if ok then st.boss = r else st.boss_err = tostring(r) end
        end
        if st.player and st.boss then
            local dx = st.boss.pos[1] - st.player.pos[1]
            local dy = st.boss.pos[2] - st.player.pos[2]
            st.dist = math.sqrt(dx * dx + dy * dy)
        end
    end
    st.ev = events
    if extra then
        for k, v in pairs(extra) do st[k] = v end
    end
    return st
end

return S
