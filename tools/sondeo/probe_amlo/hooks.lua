-- Hooks de registro (solo lectura) para LVL_AMLO. Los hooks sobre funciones Blueprint se
-- ejecutan despues del cuerpo; las funciones solo existen con su clase cargada, por eso cada
-- ruta se reintenta con AL.hooks_pump() hasta que la clase aparece (por ejemplo el
-- Chupacabras, que no existe al abrir el mapa).

local AMOUNT = "Amount_7_7BB9AB76446D18C11D94DA96B85DE23C"
local CAN_PARRY = "CanBeParried_19_AFBED70341F685CD452A27A21910B6DA"
local CAN_BLOCK = "CanBeBlocked_17_4C6668814D8CA17DBBD811BABFADD2E2"
local RESP = "DamageResponse_29_41EA7A94421239AA10EDC88D425DDFD8"
local DTYPE = "DamageType_28_51CE5BD24DF048CBB042D6B0D2FCCC81"
local INV = "SouldDamageInvincible_15_368158B94AB020DAFB1C788FED906C66"

AL.wanted_hooks = AL.wanted_hooks or {}

--- Personaje responsable de un golpe: el propio causante si es un personaje; si es un
--- proyectil o un area de efecto, su Instigator u Owner.
local function attacker_of(c)
    if not AL.valid(c) then return nil end
    local okm = pcall(function() return c.Mesh:GetAnimInstance() end)
    if okm and AL.get(c, "BPC_DamageSystem") then return c end
    local inst = nil
    pcall(function() inst = c:GetInstigator() end)
    if AL.valid(inst) then return inst end
    local own = nil
    pcall(function() own = c:GetOwner() end)
    if AL.valid(own) then return own end
    return nil
end

local function take_damage(ctx, causer, info)
    local comp = ctx:get()
    local victim = comp:GetOwner()
    local c = causer:get()
    local i = info:get()
    local att = attacker_of(c)
    local am, ap, ar, al = AL.montage(att)
    local vm, vp = AL.montage(victim)
    local p = AL.pl()
    local line = string.format(
        "HIT v=%s c=%s att=%s amt=%s parry=%s block=%s resp=%s type=%s inv=%s hp_v=%.1f am=%s ap=%.3f ar=%.2f al=%.3f vm=%s vp=%.3f d=%.0f blk=%s pw=%s pinv=%s",
        AL.cls(victim), AL.cls(c), AL.cls(att), tostring(i[AMOUNT]), tostring(i[CAN_PARRY]), tostring(i[CAN_BLOCK]),
        tostring(i[RESP]), tostring(i[DTYPE]), tostring(i[INV]), comp.Health, am, ap, ar, al, vm, vp,
        AL.dist2d(victim, att or c), tostring(p and AL.get(p.BPC_DamageSystem, "IsBlocking")),
        tostring(AL.get(p, "Is Within Parry Window")), tostring(p and AL.get(p.BPC_DamageSystem, "IsInvincible")))
    AL.log(line)
    if AL.on_hit then pcall(AL.on_hit, victim, c, att, i[AMOUNT], i[CAN_PARRY], am, ap) end
end

function AL.want_hook(path, fn)
    if AL.hooked and AL.hooked[path] then return end
    AL.wanted_hooks[path] = fn
end

function AL.hooks_pump()
    for path, fn in pairs(AL.wanted_hooks) do
        if AL.hooked and AL.hooked[path] then
            AL.wanted_hooks[path] = nil
        else
            local ok = pcall(RegisterHook, path, function(...)
                local okc, err = pcall(fn, ...)
                if not okc then AL.log("hook err " .. path .. " " .. tostring(err)) end
            end)
            if ok then
                AL.hooked = AL.hooked or {}
                AL.hooked[path] = true
                AL.wanted_hooks[path] = nil
                AL.log("hook activo " .. path)
            end
        end
    end
end

local function logger(label)
    return function(ctx, a)
        local who = "?"
        pcall(function() who = AL.name(ctx:get()) end)
        local extra = ""
        pcall(function()
            local v = a and a:get()
            if type(v) == "userdata" and v.IsValid then extra = AL.name(v) elseif v ~= nil then extra = tostring(v) end
        end)
        local m, mp = "?", 0
        pcall(function() m, mp = AL.montage(ctx:get()) end)
        AL.log(string.format("EV %s self=%s arg=%s mont=%s@%.3f", label, who, extra, tostring(m), mp))
    end
end

AL.want_hook("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:TakeDamage", take_damage)

local FUNCS = {
    ["/Game/BP_AMLOBOSS.BP_AMLOBOSS_C"] = { "ENTER PHASE 2", "death", "Teleport", "Teleport END", "Ataque Invocar Chairos", "INSTAKILL PLAYER", "AOE",
        "ATAQUE PAIRED", "EVADE AMLOS", "Ataque Throw Kat", "ATAQUE DISTANCE KATANAS FLOOR", "AtaqueParrysAMLO", "Ataques Individual Melee",
        "Ataques Individual Cargado", "Ataques en COMBO", "Ataque AIRE", "Emotiza", "DEATHENEM", "ATTACK", "ATTACKEND" },
    ["/Game/BP_CHUPACABRAS.BP_CHUPACABRAS_C"] = { "death", "Teleport", "Teleport END", "EVADE CHUPA", "ATTACK RAYO", "ATTACK FIREBALL", "Ataque PG", "RespawnEnemy", "ATTACK", "ATTACKEND", "DamageResponse" },
    ["/Game/BP_Chamuco.BP_Chamuco_C"] = { "death", "ATAQUE BASICO", "Ataque ESPECIAL GOLPE PISO LAVA", "ATAQUES COMBO", "ATAQUES CARGADO", "ATTACK", "ATTACKEND" },
    ["/Game/BP_EnemyBase.BP_EnemyBase_C"] = { "AddParryCount", "death", "INSTAKILL PLAYER" },
    ["/Game/BP_Player.BP_Player_C"] = { "death", "Parry Attack", "AMLO DEFEATED", "CHUPACABRAS DEFEATED", "TERCER BOSS DEFEATED", "LOGRO NAHUAL", "LOGRO CHAMUCO", "LOGRO LA CHINGADA", "Parry Window ON" },
    ["/Game/Gothic_Cathedral/Maps/LVL_AMLO.LVL_AMLO_C"] = { "StartFight", "SPAWN  NAHUAL", "PlayerDeath", "CustomEvent", "CustomEvent_0", "CustomEvent_1", "UNLOCKED KATANA" },
    ["/Game/InteraccionMex/BP_FogAMLO.BP_FogAMLO_C"] = { "InteractOnServer", "death", "Reset", "CustomEvent" },
    ["/Game/InteraccionMex/BP_TELEPORT.BP_TELEPORT_C"] = { "ACTIVATE PORTAL", "BndEvt__BP_TELEPORT_Box_K2Node_ComponentBoundEvent_0_ComponentBeginOverlapSignature__DelegateSignature" },
    ["/Game/InteraccionMex/BP_STATUES_ENDING.BP_STATUES_ENDING_C"] = { "ACTIVATE PORTAL", "InteractOnServer" },
    ["/Game/ThirdPerson/Blueprints/BP_MexicoGameMode.BP_MexicoGameMode_C"] = { "Respawn_Player" },
    ["/Game/BP_GhoulSpawner.BP_GhoulSpawner_C"] = { "StartSpawner", "Completed", "Player Death" },
}
for cls, list in pairs(FUNCS) do
    local short = cls:match("%.([^.]+)$")
    for _, f in ipairs(list) do AL.want_hook(cls .. ":" .. f, logger(short .. ":" .. f)) end
end
AL.want_hook("/Script/Engine.GameplayStatics:OpenLevel", function(ctx, w, name) AL.log("EV OpenLevel " .. tostring(name:get():ToString())) end)

AL.hooks_pump()
if not AL.hooks_task then
    AL.hooks_task = true
    AL.add("hooks_pump", function(c)
        if c.n % 30 == 0 then AL.hooks_pump() end
        return false
    end)
end
local pend = 0
for _ in pairs(AL.wanted_hooks) do pend = pend + 1 end
return "hooks pendientes=" .. pend
