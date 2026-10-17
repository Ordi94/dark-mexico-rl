-- Eventos de combate acumulados entre dos pasos del agente.
--
-- Algunos hechos no se pueden reconstruir con fiabilidad comparando dos observaciones: un
-- parry exitoso (el golpe no quita vida, igual que un bloqueo), varios golpes dentro del
-- mismo paso o la muerte del actor de fase 2 (que se destruye despues). Se registran con
-- hooks sobre los eventos Blueprint del juego y se envian con la siguiente observacion.
--
-- Los hooks Blueprint de UE4SS corren despues del cuerpo de la funcion. Durante una carga de
-- mapa se ignoran: los eventos que dispara la destruccion del mundo anterior no pertenecen al
-- episodio y leer actores en ese momento puede tumbar el proceso.

local G = require("dmb.game")
local L = require("dmb.log")
local H = require("dmb.hooks")
local K = require("dmb.keepalive")

local E = {}

local AMOUNT = "Amount_7_7BB9AB76446D18C11D94DA96B85DE23C"
local CAN_PARRY = "CanBeParried_19_AFBED70341F685CD452A27A21910B6DA"
local RESPONSE = "DamageResponse_29_41EA7A94421239AA10EDC88D425DDFD8"

local acc = nil

-- Con trace_hits el acumulador guarda cada golpe con su contexto (quien, con que montage y en que
-- posicion, a que distancia y cuando): es la fuente de las tablas de golpes por jefe.
E.trace_hits = false
-- Hooks dinamicos ya registrados, por ruta (cada ruta se engancha una sola vez en el proceso).
local dynamic_installed = {}
local remote_ids = {}

--- Reinicia el acumulador (al empezar cada paso o episodio).
function E.clear()
    acc = {
        dmg_taken = 0.0,
        hits_taken = 0,
        dmg_dealt = 0.0,
        hits_dealt = 0,
        parries = 0,
        phase2 = false,
        player_death = false,
        victory = false,
        taken = {},
        dealt = {},
        remote = {},
        hits = {},
    }
end
E.clear()

--- Devuelve los eventos acumulados y empieza un acumulador nuevo.
function E.take()
    local out = acc
    -- Las listas de golpes individuales solo viajan si hubo golpes en el paso.
    if #out.taken == 0 then out.taken = nil end
    if #out.dealt == 0 then out.dealt = nil end
    if #out.remote == 0 then out.remote = nil end
    if #out.hits == 0 then out.hits = nil end
    E.clear()
    return out
end

local function guarded(fn)
    return function(...)
        if G.loading then return end
        local ok, err = pcall(fn, ...)
        if not ok then L.info("evento con error: " .. tostring(err)) end
    end
end

local function montage_of(ch)
    local ai = ch.Mesh:GetAnimInstance()
    local m = ai:GetCurrentActiveMontage()
    if m and m:IsValid() then
        return m:GetFName():ToString(), ai:Montage_GetPosition(m), ai:Montage_GetPlayRate(m)
    end
    return nil, 0.0, 1.0
end

--- Personaje responsable de un golpe: el propio causante si tiene malla animada; si es un
--- proyectil o un area, su Instigator u Owner.
local function attacker_of(causer)
    local ok, ch = pcall(function()
        if causer.Mesh and causer.Mesh:GetAnimInstance():IsValid() then return causer end
        return nil
    end)
    if ok and ch then return ch end
    local inst = nil
    pcall(function() inst = causer:GetInstigator() end)
    if G.valid(inst) then return inst end
    pcall(function() inst = causer:GetOwner() end)
    if G.valid(inst) then return inst end
    return nil
end

--- Registra un golpe recibido por el jugador con su contexto (solo con E.trace_hits).
function E.trace_hit(causer_param, victim, amount, info)
    local causer = causer_param:get()
    local rec = { t = G.time_seconds(), amount = amount, parry = info[CAN_PARRY] == true }
    pcall(function() rec.resp = info[RESPONSE] end)
    -- El area de las katanas y otros ataques a distancia llegan con el causante ya destruido
    -- (nulo): llamar un metodo sobre un objeto nulo tumba el proceso dentro de UE4SS, y pcall
    -- no lo evita porque el fallo ocurre en codigo nativo. Se comprueba antes de tocarlo.
    if G.valid(causer) then
        pcall(function() rec.causer = causer:GetClass():GetFName():ToString() end)
        local att = attacker_of(causer)
        if att and G.valid(att) then
            pcall(function()
                rec.att = att:GetClass():GetFName():ToString()
                local name, pos, rate = montage_of(att)
                rec.am, rec.ap, rec.ar = name, pos, rate
                local la, lv = att:K2_GetActorLocation(), victim:K2_GetActorLocation()
                rec.d = math.sqrt((la.X - lv.X) ^ 2 + (la.Y - lv.Y) ^ 2)
            end)
        end
    else
        rec.causer = "nulo"
    end
    pcall(function()
        local name, pos = montage_of(victim)
        rec.vm, rec.vp = name, pos
        rec.blocking = victim.BPC_DamageSystem.IsBlocking
        rec.window = victim["Is Within Parry Window"]
    end)
    acc.hits[#acc.hits + 1] = rec
end

--- Registra un evento de ataque a distancia con la posicion de ambos en ese instante.
local function record_remote(id)
    local rec = { id = id, t = G.time_seconds() }
    pcall(function()
        local p = G.pawn()
        local l = p:K2_GetActorLocation()
        rec.px, rec.py, rec.pz = l.X, l.Y, l.Z
        rec.pyaw = p:K2_GetActorRotation().Yaw
    end)
    pcall(function()
        local b = G.boss_actor()
        local l = b:K2_GetActorLocation()
        rec.bx, rec.by, rec.bz = l.X, l.Y, l.Z
        rec.byaw = b:K2_GetActorRotation().Yaw
    end)
    acc.remote[#acc.remote + 1] = rec
end

--- Engancha los eventos que describe un jefe (muerte que cuenta como victoria, paso de fase y
--- ataques a distancia). Cada ruta se registra una sola vez en la vida del proceso: la funcion
--- Blueprint existe solo con su clase cargada, asi que el registro lo reintenta dmb.hooks.
function E.configure(spec)
    E.trace_hits = spec.trace_hits == true
    for _, h in ipairs(spec.hooks or {}) do
        if not dynamic_installed[h.path] then
            dynamic_installed[h.path] = true
            local event, id = h.event, h.id
            H.want(h.path, guarded(function()
                if event == "victory" then
                    acc.victory = true
                    E.victory_seen = true
                    K.need_sweep = true
                elseif event == "phase2" then
                    acc.phase2 = true
                    K.need_sweep = true
                else
                    record_remote(id)
                end
            end))
        end
    end
end

function E.install()
    local hook = H.want
    hook("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:TakeDamage", guarded(function(ctx, causer, info)
        local comp = ctx:get()
        local owner = comp:GetOwner()
        local i = info:get()
        local amount = i[AMOUNT] or 0.0
        local cls = owner:GetClass():GetFName():ToString()
        if cls == "BP_Player_C" then
            acc.dmg_taken = acc.dmg_taken + amount
            acc.hits_taken = acc.hits_taken + 1
            acc.taken[#acc.taken + 1] = { amount, i[CAN_PARRY] == true }
            if E.trace_hits then E.trace_hit(causer, owner, amount, i) end
        else
            acc.dmg_dealt = acc.dmg_dealt + amount
            acc.hits_dealt = acc.hits_dealt + 1
            acc.dealt[#acc.dealt + 1] = amount
        end
    end))
    hook("/Game/BP_Player.BP_Player_C:Parry Attack", guarded(function()
        acc.parries = acc.parries + 1
    end))
    hook("/Game/BP_Player.BP_Player_C:death", guarded(function()
        acc.player_death = true
    end))
    hook("/Game/BP_Enemy_Juice.BP_Enemy_Juice_C:ENTER PHASE 2", guarded(function()
        acc.phase2 = true
        -- La fase 2 y la pantalla final cargan clases nuevas: anclarlas en el proximo reinicio.
        K.need_sweep = true
    end))
    hook("/Game/BP_Enemy_JuicePhase2.BP_Enemy_JuicePhase2_C:death", guarded(function()
        acc.victory = true
        E.victory_seen = true
        K.need_sweep = true
    end))
end

return E
