-- Reinicio de episodio: deja la pelea contra el jefe recien empezada.
--
-- Reproduce el flujo real del juego, medido como el mas rapido y deterministico:
--   OpenLevel(LVL_Boss) -> esperar mundo nuevo estable con pawn y jefe -> teleport frente a la
--   niebla (47, -2260, 380) -> boton A (interactuar) -> el jugador cruza la niebla y el jefe
--   arranca su primer montage (ANIM_CARPET) unos 2 s de juego despues del A.
-- El episodio se da por listo cuando el jefe tiene un montage activo.
--
-- Con start_phase = 2 el episodio arranca en la fase 2: tras activarse el jefe se deja su vida
-- bajo el umbral de la transformacion (misma razon que usa el juego), se coloca al jugador a
-- R.PHASE2_DISTANCE del jefe y se espera a que el actor de fase 2 aparezca y actue. Mientras
-- tanto el jugador es invulnerable (bandera IsInvincible; su vida no se modifica) para que el
-- jefe de fase 1 no lo mate durante la transformacion.
--
-- Los plazos se miden en tiempo de juego (independientes de la velocidad de simulacion) y hay
-- un tope en reloj real por si el juego queda atascado en la carga.

local G = require("dmb.game")
local L = require("dmb.log")
local A = require("dmb.actions")
local S = require("dmb.state")
local K = require("dmb.keepalive")
local C = require("dmb.curriculum")

local R = {}

R.MAP = "LVL_Boss"
R.FOG_FRONT = { X = 47.0, Y = -2260.0, Z = 380.0 }
-- Tiempo de juego entre el teleport y el A: el trazo de interaccion necesita al menos un tick
-- con el pawn ya frente a la niebla para enfocar el area.
R.INTERACT_DELAY = 0.12
-- Si tras este tiempo el jugador no reprodujo la caminata de entrada, se repite el A.
R.INTERACT_RETRY = 1.0
R.MAX_INTERACTS = 3
-- Tiempo de juego maximo desde el A hasta que el jefe se activa.
R.BOSS_TIMEOUT = 10.0
-- Tope en reloj real del reinicio completo y numero de OpenLevel antes de rendirse.
R.WALL_TIMEOUT = 60.0
R.MAX_ATTEMPTS = 3
-- Distancia a la que se coloca al jugador del jefe para forzar la transformacion y tiempo de
-- juego maximo hasta que el jefe de fase 2 actua (medido: 7-27 s).
R.PHASE2_DISTANCE = 450.0
R.PHASE2_FAR_DISTANCE = 650.0
R.PHASE2_REPLACE_EVERY = 0.5
R.ARENA_CENTER = { X = 39.0, Y = 1458.0 }
R.PHASE2_TIMEOUT = 60.0
R.PHASE2_CLASS = "BP_Enemy_JuicePhase2_C"

local st = nil

--- Empieza un reinicio. La carga del mapa ocurre en el siguiente tick del motor.
function R.start(opts)
    st = { phase = "open", attempts = 0, wall0 = os.clock(), start_phase = (opts and opts.start_phase) or 1 }
    R._open()
end

function R._open()
    st.attempts = st.attempts + 1
    st.phase = "open"
    st.gen_before = G.world_gen
    st.interacts = 0
    A.reset()
    S.reset_cache()
    -- Antes de descargar el mundo se anclan las clases Blueprint ya cargadas (ver keepalive).
    K.before_reload()
    L.info("reinicio: OpenLevel " .. R.MAP .. " intento " .. st.attempts)
    G.open_level(R.MAP)
end

function R.active()
    return st ~= nil
end

local function boss_montage(b)
    local ok, name = pcall(function()
        local ai = b.Mesh:GetAnimInstance()
        local m = ai:GetCurrentActiveMontage()
        if m and m:IsValid() then return m:GetFName():ToString() end
        return nil
    end)
    return ok and name or nil
end

local function player_montage(p)
    local ok, name = pcall(function()
        local ai = p.Mesh:GetAnimInstance()
        local m = ai:GetCurrentActiveMontage()
        if m and m:IsValid() then return m:GetFName():ToString() end
        return nil
    end)
    return ok and name or nil
end

local function fail_or_retry(reason)
    L.info("reinicio: " .. reason)
    if st.attempts >= R.MAX_ATTEMPTS then
        local msg = reason .. " tras " .. st.attempts .. " intentos"
        st = nil
        return false, msg
    end
    R._open()
    return nil
end

--- Coloca al jugador a dist uu del jefe, del lado del centro de la arena (donde empieza la
--- pelea), para no sacarlo del escenario.
function R._place_player_near(p, b, dist)
    local loc = b:K2_GetActorLocation()
    local dx, dy = R.ARENA_CENTER.X - loc.X, R.ARENA_CENTER.Y - loc.Y
    local d = math.sqrt(dx * dx + dy * dy)
    if d < 50.0 then dx, dy, d = 0.0, -1.0, 1.0 end
    p:K2_SetActorLocation({ X = loc.X + dx / d * dist, Y = loc.Y + dy / d * dist, Z = loc.Z }, false, {}, true)
end

--- Pasa a la preparacion de la fase 2: invulnerabilidad, jugador junto al jefe y vida del jefe
--- bajo el umbral de la transformacion.
function R._prepare_phase2(p, b, now)
    C.set_player_invincible(p, true)
    R._place_player_near(p, b, R.PHASE2_DISTANCE)
    C.trigger_phase2(b)
    st.phase = "phase2"
    st.t_p2 = now
    st.t_place = now
end

--- Avanza el reinicio un frame. Devuelve nil mientras sigue, true al quedar listo o
--- false, mensaje si no se pudo.
function R.tick()
    if not st then return false, "sin reinicio en curso" end
    if os.clock() - st.wall0 > R.WALL_TIMEOUT then
        st = nil
        return false, "tope de tiempo real del reinicio"
    end
    if st.phase == "open" then
        st.waited = (st.waited or 0) + 1
        local p, b = nil, nil
        local ready = G.world_gen ~= st.gen_before and G.stable() and G.map == R.MAP
        if ready then p, b = G.pawn(), G.boss_actor() end
        if not (p and b) then
            -- Diagnostico de una espera larga: que condicion falta.
            if st.waited % 300 == 0 then
                local pc = G.player_controller()
                local raw = "nil"
                pcall(function()
                    local rp = pc and pc.Pawn
                    raw = (rp and rp:IsValid()) and G.class_name(rp) or tostring(rp)
                end)
                L.info(string.format("reinicio esperando: gen=%d/%d loading=%s settle=%d map=%s pc=%s pawn=%s (%s) jefe=%s",
                    G.world_gen, st.gen_before, tostring(G.loading), G.settle, tostring(G.map), tostring(pc ~= nil),
                    tostring(p ~= nil), raw, tostring(b ~= nil)))
            end
            return nil
        end
        G.apply_dilation()
        p:K2_SetActorLocation(R.FOG_FRONT, false, {}, true)
        st.t_tp = G.time_seconds()
        st.phase = "interact"
        return nil
    end
    local p = G.pawn()
    if not p then return fail_or_retry("pawn perdido durante el reinicio") end
    local now = G.time_seconds()
    if st.phase == "interact" then
        if now - st.t_tp < R.INTERACT_DELAY then return nil end
        A.fire(p, A.INTERACT)
        st.interacts = st.interacts + 1
        st.t_a = now
        st.t_a_first = st.t_a_first or now
        st.phase = "wait_boss"
        return nil
    end
    if st.phase == "wait_boss" then
        local b = G.boss_actor()
        if b and boss_montage(b) then
            L.info(string.format("reinicio listo: jefe activo a %.2f s de juego del primer A (A=%d, intentos=%d, %.2f s reales)",
                now - st.t_a_first, st.interacts, st.attempts, os.clock() - st.wall0))
            if st.start_phase == 2 then
                R._prepare_phase2(p, b, now)
                return nil
            end
            st = nil
            return true
        end
        if now - st.t_a > R.INTERACT_RETRY and player_montage(p) == nil and st.interacts < R.MAX_INTERACTS then
            st.phase = "interact"
            st.t_tp = now - R.INTERACT_DELAY
            return nil
        end
        if now - st.t_a_first > R.BOSS_TIMEOUT then
            return fail_or_retry("el jefe no se activo")
        end
        return nil
    end
    if st.phase == "phase2" then
        local b = G.boss_actor()
        local is_p2 = b and G.class_name(b) == R.PHASE2_CLASS
        -- La transformacion del jefe de fase 1 y el despertar del de fase 2 ocurren cuando el
        -- jugador esta cerca (medido: el jefe se aleja tras bajarle la vida y espera): se
        -- vuelve a colocar al jugador junto al jefe cada vez que se separan mas de
        -- R.PHASE2_FAR_DISTANCE, como maximo cada R.PHASE2_REPLACE_EVERY segundos de juego.
        if b and now - (st.t_place or -1e9) >= R.PHASE2_REPLACE_EVERY then
            local far = true
            pcall(function()
                local lp, lb = p:K2_GetActorLocation(), b:K2_GetActorLocation()
                far = math.sqrt((lp.X - lb.X) ^ 2 + (lp.Y - lb.Y) ^ 2) > R.PHASE2_FAR_DISTANCE
            end)
            if far then
                R._place_player_near(p, b, R.PHASE2_DISTANCE)
                st.t_place = now
            end
        end
        if is_p2 and (boss_montage(b) or b.Attacking) then
            C.set_player_invincible(p, false)
            L.info(string.format("reinicio fase 2 listo: %.2f s de juego, %.2f s reales", now - st.t_p2, os.clock() - st.wall0))
            st = nil
            return true
        end
        if now - st.t_p2 > R.PHASE2_TIMEOUT then
            C.set_player_invincible(p, false)
            local cls = b and G.class_name(b) or "ninguno"
            local dist = -1.0
            pcall(function()
                local lp, lb = p:K2_GetActorLocation(), b:K2_GetActorLocation()
                dist = math.sqrt((lp.X - lb.X) ^ 2 + (lp.Y - lb.Y) ^ 2)
            end)
            L.info(string.format("fase 2: jefe=%s dist=%.0f", cls, dist))
            return fail_or_retry("el jefe de fase 2 no se activo")
        end
        return nil
    end
    return false, "fase desconocida " .. tostring(st.phase)
end

return R
