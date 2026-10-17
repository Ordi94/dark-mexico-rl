-- Reinicio de episodio para varios jefes, descrito por datos (ver dmb.spec).
--
-- Cada instancia del juego queda fija a un mapa. Con el mapa ya cargado el episodio se reinicia
-- en sitio, sin OpenLevel (que en LVL_AMLO cuesta 2.6-4.3 s y descarga las clases Blueprint):
--   1. se destruyen los enemigos menores y todos los jefes del mapa;
--   2. el jugador se regenera con la misma funcion que usa el juego al reaparecer
--      (BP_MexicoGameMode_C:Respawn_Player), se destruye el pawn anterior (con eso se cancela el
--      temporizador de 10 s de la pantalla de muerte, que de otro modo regeneraria al jugador y
--      al jefe a mitad del episodio) y se quita la pantalla de carga de la muerte;
--   3. se genera al jefe en sitio (o se llama al evento del nivel que lo genera) y se coloca al
--      jugador a una distancia fija, mirandolo.
-- La recarga completa del mapa queda como respaldo: se usa en el primer episodio, cuando el
-- cliente la pide (reinicio periodico) o si el reinicio en sitio no consigue dejar la pelea
-- activa. Un jefe cuyo arranque es el flujo real de la niebla (kind = "legacy_fog", Juice) se
-- delega a dmb.episode sin cambios.
--
-- Los plazos se miden en tiempo de juego, con un tope en reloj real por si el juego se atasca.

local G = require("dmb.game")
local L = require("dmb.log")
local A = require("dmb.actions")
local S = require("dmb.state")
local K = require("dmb.keepalive")
local C = require("dmb.curriculum")
local E = require("dmb.events")
local R = require("dmb.episode")

local Ar = {}

Ar.WALL_TIMEOUT = 60.0
Ar.RESPAWN_TIMEOUT_FRAMES = 150
Ar.BOSS_TIMEOUT = 15.0
Ar.MAX_ATTEMPTS = 3
Ar.PHASE2_FAR_DISTANCE = 650.0
Ar.PHASE2_DISTANCE = 450.0
Ar.PHASE2_REPLACE_EVERY = 0.5
-- Widgets que deja el juego tras la muerte o la victoria y que no pertenecen al combate. El HUD
-- (WP_MainHUD_C) lo crea el pawn nuevo y no se toca.
Ar.STALE_WIDGETS = { WP_LoadingScreen_C = true, WP_ENDGAME_C = true, WP_SKINUNLOCKED_C = true }

local st = nil
local delegate = false

local function valid(o)
    return G.valid(o)
end

local function montage_name(ch)
    local ok, name = pcall(function()
        local ai = ch.Mesh:GetAnimInstance()
        local m = ai:GetCurrentActiveMontage()
        if m and m:IsValid() then return m:GetFName():ToString() end
        return nil
    end)
    return ok and name or nil
end

local function boss_active(b)
    if not b then return false end
    local ok, attacking = pcall(function() return b.Attacking end)
    return montage_name(b) ~= nil or (ok and attacking == true)
end

local function player_alive(p)
    local ok, alive = pcall(function()
        local d = p.BPC_DamageSystem
        return d.IsDead == false and d.Health > 0
    end)
    return ok and alive
end

--- Destruye todos los personajes del mundo cuya clase esta en la lista.
local function destroy_classes(classes)
    local n = 0
    for _, e in ipairs(G.actors_of(classes)) do
        if pcall(function() e:K2_DestroyActor() end) then n = n + 1 end
    end
    return n
end

--- Destruye todo BP_Player_C salvo el pawn indicado (el pawn muerto de la reaparicion).
local function destroy_other_players(keep)
    local n = 0
    local arr = {}
    pcall(function() G.gs:GetAllActorsOfClass(G.world, G.character_class, arr) end)
    local keep_addr = keep:GetAddress()
    for _, item in ipairs(arr) do
        local a = item:get()
        if valid(a) then
            local ok, cls = pcall(G.class_name, a)
            if ok and cls == "BP_Player_C" and a:GetAddress() ~= keep_addr then
                if pcall(function() a:K2_DestroyActor() end) then n = n + 1 end
            end
        end
    end
    return n
end

--- Quita de pantalla los widgets residuales de la muerte o la victoria.
local function remove_stale_widgets()
    local lib = StaticFindObject("/Script/UMG.Default__WidgetBlueprintLibrary")
    local cls = StaticFindObject("/Script/UMG.UserWidget")
    if not (valid(lib) and valid(cls)) then return 0 end
    local arr, n = {}, 0
    pcall(function() lib:GetAllWidgetsOfClass(G.world, arr, cls, true) end)
    for _, item in ipairs(arr) do
        local w = item:get()
        local ok, name = pcall(G.class_name, w)
        if ok and Ar.STALE_WIDGETS[name] then
            if pcall(function() w:RemoveFromParent() end) then n = n + 1 end
        end
    end
    return n
end

local function spawn_class(path, x, y, z)
    local cls = StaticFindObject(path)
    if not valid(cls) then return nil end
    local tr = { Rotation = { X = 0, Y = 0, Z = 0, W = 1 }, Translation = { X = x, Y = y, Z = z }, Scale3D = { X = 1, Y = 1, Z = 1 } }
    -- Colision 1 = AlwaysSpawn: el jefe nace aunque haya un resto del anterior en el punto.
    local actor = G.gs:BeginDeferredActorSpawnFromClass(G.world, cls, tr, 1, nil, 0)
    if not valid(actor) then return nil end
    G.gs:FinishSpawningActor(actor, tr, 0)
    return actor
end

--- Coloca al jugador en target + offset, mirando al punto target, y alinea la camara.
local function place_player(p, tx, ty, tz, off)
    local px, py, pz = tx + off.x, ty + off.y, tz + off.z
    p:K2_SetActorLocation({ X = px, Y = py, Z = pz }, false, {}, true)
    local yaw = math.deg(math.atan(ty - py, tx - px))
    pcall(function() p:K2_SetActorRotation({ Pitch = 0, Yaw = yaw, Roll = 0 }, true) end)
    pcall(function() G.player_controller():SetControlRotation({ Pitch = 0, Yaw = yaw, Roll = 0 }) end)
end

local function boss_xyz(b)
    local l = b:K2_GetActorLocation()
    return l.X, l.Y, l.Z
end

local function open_map()
    st.attempts = st.attempts + 1
    st.step = "open"
    st.gen_before = G.world_gen
    st.fresh = true
    A.reset()
    S.reset_cache()
    K.before_reload()
    L.info("reinicio multi-jefe: OpenLevel " .. st.spec.map .. " intento " .. st.attempts)
    G.open_level(st.spec.map)
end

--- Reintento escalonado: el primer fallo se repite en sitio; el segundo recarga el mapa; el
--- tercero se informa al cliente.
local function fail_or_retry(reason)
    L.info("reinicio multi-jefe: " .. reason)
    if st.attempts >= Ar.MAX_ATTEMPTS then
        local msg = reason .. " tras " .. st.attempts .. " intentos"
        st = nil
        return false, msg
    end
    if st.attempts == 1 then
        st.attempts = 2
        st.step, st.fresh, st.waited = "prepare", false, 0
        return nil
    end
    open_map()
    return nil
end

--- Empieza un reinicio. opts: spec (ya validada), start_phase (1|2), reload (bool).
function Ar.start(opts)
    local spec = opts.spec
    G.set_boss_classes(spec.classes)
    E.configure(spec)
    delegate = spec.start.kind == "legacy_fog"
    if delegate then
        R.start({ start_phase = opts.start_phase })
        return
    end
    st = { spec = spec, start_phase = opts.start_phase or 1, wall0 = os.clock(), attempts = 0, waited = 0 }
    if opts.reload or G.map ~= spec.map or not G.stable() then
        open_map()
    else
        st.attempts, st.step, st.fresh = 1, "prepare", false
    end
end

function Ar.active()
    return delegate or st ~= nil
end

local function tick_open()
    if G.world_gen == st.gen_before or not G.stable() or G.map ~= st.spec.map or not G.pawn() then return nil end
    st.step, st.waited = "prepare", 0
    return nil
end

local function tick_prepare()
    if not G.stable() then return nil end
    local p = G.pawn()
    if not p then
        st.waited = st.waited + 1
        if st.waited > 300 then return fail_or_retry("sin pawn en la preparacion") end
        return nil
    end
    if K.need_sweep then pcall(K.sweep) end
    local spec = st.spec
    destroy_classes(spec.minors)
    destroy_classes(spec.purge)
    A.reset()
    S.reset_cache()
    if st.fresh then
        st.step = "spawn"
        return nil
    end
    st.old_addr = p:GetAddress()
    st.frames = 0
    local ok, err = pcall(function() G.gs:GetGameMode(G.world):Respawn_Player() end)
    if not ok then return fail_or_retry("Respawn_Player fallo: " .. tostring(err)) end
    st.step = "respawn"
    return nil
end

local function tick_respawn()
    st.frames = st.frames + 1
    local p = G.pawn()
    if p and p:GetAddress() ~= st.old_addr and player_alive(p) then
        destroy_other_players(p)
        remove_stale_widgets()
        st.step = "spawn"
        return nil
    end
    if st.frames > Ar.RESPAWN_TIMEOUT_FRAMES then return fail_or_retry("el jugador no reaparecio") end
    return nil
end

local function tick_spawn()
    local p = G.pawn()
    if not p then return fail_or_retry("pawn perdido antes de generar al jefe") end
    local spec, start = st.spec, st.spec.start
    G.apply_dilation()
    E.clear()
    st.t_spawn = G.time_seconds()
    if start.kind == "spawn_class" then
        local b = spawn_class(start.class_path, start.pos.x, start.pos.y, start.pos.z)
        if not b then return fail_or_retry("no se pudo generar " .. start.class_path) end
        if start.hp then
            pcall(function()
                b.BPC_DamageSystem.MaxHealth = start.hp
                b.BPC_DamageSystem.Health = start.hp
            end)
        end
        place_player(p, start.pos.x, start.pos.y, start.pos.z, start.player_offset)
        st.step = "wait_boss"
        return nil
    end
    local lvl = FindFirstOf(start.level_class)
    if not valid(lvl) then return fail_or_retry("no hay script de nivel " .. start.level_class) end
    local ok, err = pcall(function() lvl[start.event](lvl) end)
    if not ok then return fail_or_retry("evento del nivel fallo: " .. tostring(err)) end
    st.step = "await_boss"
    return nil
end

local function tick_await_boss()
    local p, b = G.pawn(), G.boss_actor()
    if not p then return fail_or_retry("pawn perdido esperando al jefe") end
    if b then
        local x, y, z = boss_xyz(b)
        place_player(p, x, y, z, st.spec.start.player_offset)
        st.step = "wait_boss"
        return nil
    end
    if G.time_seconds() - st.t_spawn > (st.spec.start.wait_seconds or 1.0) + 5.0 then
        return fail_or_retry("el evento del nivel no genero al jefe")
    end
    return nil
end

--- Mantiene al jugador junto al jefe mientras se prepara la fase 2 (el jefe se aleja).
local function keep_near(p, b, now)
    if now - (st.t_place or -1e9) < Ar.PHASE2_REPLACE_EVERY then return end
    local far = true
    pcall(function()
        local lp, lb = p:K2_GetActorLocation(), b:K2_GetActorLocation()
        far = math.sqrt((lp.X - lb.X) ^ 2 + (lp.Y - lb.Y) ^ 2) > Ar.PHASE2_FAR_DISTANCE
    end)
    if far then
        local x, y, z = boss_xyz(b)
        local off = st.spec.start.player_offset
        local n = math.sqrt(off.x * off.x + off.y * off.y)
        if n < 1.0 then off = { x = Ar.PHASE2_DISTANCE, y = 0.0, z = off.z } else
            off = { x = off.x / n * Ar.PHASE2_DISTANCE, y = off.y / n * Ar.PHASE2_DISTANCE, z = off.z }
        end
        place_player(p, x, y, z, off)
        st.t_place = now
    end
end

local function tick_wait_boss()
    local p, b = G.pawn(), G.boss_actor()
    if not p then return fail_or_retry("pawn perdido durante el reinicio") end
    local now = G.time_seconds()
    if boss_active(b) then
        L.info(string.format("reinicio multi-jefe listo: %s activo a %.2f s de juego (intentos=%d, %.2f s reales)",
            st.spec.id, now - st.t_spawn, st.attempts, os.clock() - st.wall0))
        if st.start_phase == 2 and st.spec.phase2 then
            C.set_player_invincible(p, true)
            C.trigger_phase2(b)
            st.step, st.t_p2, st.t_place = "phase2", now, now
            return nil
        end
        st = nil
        return true
    end
    if now - st.t_spawn > Ar.BOSS_TIMEOUT then return fail_or_retry("el jefe no se activo") end
    return nil
end

local function tick_phase2()
    local p, b = G.pawn(), G.boss_actor()
    if not p then return fail_or_retry("pawn perdido en la fase 2") end
    local now = G.time_seconds()
    local want = st.spec.phase2
    if b then keep_near(p, b, now) end
    local is_p2 = b and G.class_name(b) == want.class
    if is_p2 and boss_active(b) then
        C.set_player_invincible(p, false)
        L.info(string.format("reinicio fase 2 listo: %s a %.2f s de juego", want.class, now - st.t_p2))
        st = nil
        return true
    end
    if now - st.t_p2 > (want.timeout or 60.0) then
        C.set_player_invincible(p, false)
        return fail_or_retry("el jefe de fase 2 no se activo")
    end
    return nil
end

local STEPS = {
    open = tick_open, prepare = tick_prepare, respawn = tick_respawn, spawn = tick_spawn,
    await_boss = tick_await_boss, wait_boss = tick_wait_boss, phase2 = tick_phase2,
}

--- Avanza el reinicio un frame. Devuelve nil mientras sigue, true al quedar listo o false,
--- mensaje si no se pudo.
function Ar.tick()
    if delegate then
        local r, e = R.tick()
        if r ~= nil then delegate = false end
        return r, e
    end
    if not st then return false, "sin reinicio en curso" end
    if os.clock() - st.wall0 > Ar.WALL_TIMEOUT then
        st = nil
        return false, "tope de tiempo real del reinicio"
    end
    local step = STEPS[st.step]
    if not step then
        local msg = "paso desconocido " .. tostring(st.step)
        st = nil
        return false, msg
    end
    return step()
end

return Ar
