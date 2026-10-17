-- DarkMexicoBridge
-- Puente de entrenamiento entre el juego y un proceso Python, en modo lockstep.
--
-- Solo se activa si el proceso del juego arranca con la variable de entorno DMB_TRAINING=1;
-- sin ella el mod no registra nada y el juego normal del usuario no cambia.
--
-- Variables de entorno:
--   DMB_TRAINING=1      activa el puente.
--   DMB_ID              identificador de la instancia (nombre de los pipes); por defecto "0".
--   DMB_LOG_DIR         carpeta del registro propio bridge_<id>.log (opcional).
--   DMB_IDLE_TIMEOUT    segundos reales sin cliente antes de cerrar el juego (defecto 120).
--   DMB_DEBUG=1         habilita la orden "exec" (ejecutar Lua arbitrario; solo pruebas).
--
-- Modelo de ejecucion: un callback por frame en el hilo del juego. Tras responder a una orden
-- que necesita frames (step, reset), el frame sigue normalmente; cuando el paso termina se
-- envia la observacion y el hilo del juego queda bloqueado leyendo la siguiente orden. La
-- simulacion avanza solo cuando Python lo pide, asi que no importa cuanto tarde Python en
-- decidir (sincronia en tiempo de simulacion).

if os.getenv("DMB_TRAINING") ~= "1" then
    print("[Bridge] inactivo: falta DMB_TRAINING=1\n")
    return
end

local json = require("dmb.json")
local L = require("dmb.log")
local G = require("dmb.game")
local H = require("dmb.hooks")
local Channel = require("dmb.channel")
local S = require("dmb.state")
local A = require("dmb.actions")
local E = require("dmb.events")
local R = require("dmb.episode")
local Arena = require("dmb.arena")
local Spec = require("dmb.spec")
local guards = require("dmb.guards")
local K = require("dmb.keepalive")
local C = require("dmb.curriculum")

local PROTO_VERSION = 1
local MOD_VERSION = "1.2.0"
local MAX_STEP_FRAMES = 600
local CONNECT_EVERY_FRAMES = 10

local function sanitize_id(v)
    if v and v:match("^[%w_%-]+$") and #v <= 32 then return v end
    return "0"
end

local ID = sanitize_id(os.getenv("DMB_ID"))
local DEBUG = os.getenv("DMB_DEBUG") == "1"
local IDLE_TIMEOUT = math.max(5.0, tonumber(os.getenv("DMB_IDLE_TIMEOUT") or "") or 120.0)

L.init(os.getenv("DMB_LOG_DIR"), ID)
L.info(string.format("DarkMexicoBridge %s activo id=%s debug=%s idle_timeout=%.0f %s", MOD_VERSION, ID, tostring(DEBUG), IDLE_TIMEOUT, _VERSION))

G.init_statics()
guards.install()
E.install()

-- Con estos callbacks el mod deja de tocar actores desde el inicio de la carga del mapa; sin
-- ellos (si la version de UE4SS no los ofrece) queda la deteccion por cambio de UWorld.
local function register_first(names, fn)
    local last_err = "ninguna funcion disponible"
    for _, n in ipairs(names) do
        local reg = rawget(_G, n)
        if type(reg) == "function" then
            local ok, err = pcall(reg, fn)
            if ok then return n end
            last_err = n .. ": " .. tostring(err)
        end
    end
    return nil, last_err
end
local pre_name, pre_err = register_first({ "RegisterLoadMapPreHook", "RegisterLoadMapPreCallback" }, function() G.on_load_map_pre() end)
local post_name, post_err = register_first({ "RegisterLoadMapPostHook", "RegisterLoadMapPostCallback" }, function() G.on_load_map_post() end)
L.info("LoadMap pre=" .. tostring(pre_name or pre_err) .. " post=" .. tostring(post_name or post_err))

local chan = Channel.new(ID)
local mode = "idle"
-- Motor del reinicio vigente: dmb.episode (protocolo original, Juice) o dmb.arena (varios jefes).
local reset_driver = R
local remaining = 0
local last_client = os.clock()

-- Bridge expuesto a la orden exec (solo con DMB_DEBUG=1) para mediciones.
BRIDGE = { C = C, G = G, S = S, A = A, E = E, R = R, Arena = Arena, H = H, K = K, guards = guards, json = json, L = L, chan = chan }

local function reply(t)
    t.v = PROTO_VERSION
    return chan:send(json.encode(t))
end

local function guard_summary()
    return {
        achievements_blocked = guards.achievements_blocked,
        saves_blocked = guards.saves_blocked,
        saves_unblocked = guards.saves_unblocked,
        pauses_closed = guards.pauses_closed,
        hooks_pending = H.pending_count(),
        essential = guards.essential_active(),
        classes_anchored = K.count,
    }
end

local function send_state(op, extra)
    local ex = extra or {}
    ex.op = op
    if ex.ok == nil then ex.ok = true end
    return reply(S.build(E.take(), ex))
end

-- Comandos de consola admitidos (lista de admitidos: rendimiento y calidad grafica).
local CONSOLE_ALLOWED = { "t.MaxFPS ", "r.VSync ", "r.ScreenPercentage ", "sg.", "r.SetRes ", "a.URO.", "stat " }

local function console_allowed(cmd)
    if type(cmd) ~= "string" or #cmd > 80 or not cmd:match("^[%w%._ %-]+$") then return false end
    for _, p in ipairs(CONSOLE_ALLOWED) do
        if cmd:sub(1, #p) == p then return true end
    end
    return false
end

-- Cada manejador devuelve true si la orden necesita que el juego avance frames (el bucle de
-- servicio termina y el frame continua); false si ya respondio y hay que leer otra orden.
local HANDLERS = {}

function HANDLERS.hello(msg)
    if msg.proto ~= PROTO_VERSION then
        reply({ op = "hello", ok = false, err = "version de protocolo " .. tostring(msg.proto) .. " no soportada; se espera " .. PROTO_VERSION })
        return false
    end
    reply({ op = "hello", ok = true, id = ID, mod = MOD_VERSION, lua = _VERSION, debug = DEBUG,
        map = G.map, frame = G.frame, guards = guard_summary() })
    return false
end

function HANDLERS.state(msg)
    send_state("state", { guards = guard_summary() })
    return false
end

function HANDLERS.step(msg)
    local n = msg.n or 1
    if math.type(n) ~= "integer" or n < 1 or n > MAX_STEP_FRAMES then
        reply({ op = "step", ok = false, err = "n debe ser entero entre 1 y " .. MAX_STEP_FRAMES })
        return false
    end
    local act, err = A.parse(msg.a)
    if not act then
        reply({ op = "step", ok = false, err = err })
        return false
    end
    if G.stable() then
        local ok, e = pcall(A.begin, act, n)
        if not ok then L.info("accion con error: " .. tostring(e)) end
    end
    remaining = n
    mode = "step"
    return true
end

function HANDLERS.reset(msg)
    if not guards.essential_active() then
        reply({ op = "reset", ok = false, err = "protecciones de logros/guardado inactivas", guards = guard_summary() })
        return false
    end
    -- Dificultad del episodio (curriculo). Lista de admitidos: lo que no sea un multiplicador
    -- en rango o la fase 1/2 se rechaza antes de tocar el juego.
    local mult = msg.boss_hp_mult
    if mult == nil then mult = 1.0 end
    local start_phase = msg.start_phase
    if start_phase == nil then start_phase = 1 end
    if not C.set_multiplier(mult) then
        reply({ op = "reset", ok = false, err = "boss_hp_mult debe ser un numero en [" .. C.MIN_MULT .. ", " .. C.MAX_MULT .. "]" })
        return false
    end
    if start_phase ~= 1 and start_phase ~= 2 then
        reply({ op = "reset", ok = false, err = "start_phase debe ser 1 o 2" })
        return false
    end
    -- Descripcion del jefe (varios jefes). Sin ella se conserva el flujo original de Juice.
    local spec = nil
    if msg.boss ~= nil then
        local why
        spec, why = Spec.validate(msg.boss)
        if not spec then
            reply({ op = "reset", ok = false, err = "boss no admitido: " .. tostring(why) })
            return false
        end
        if msg.reload ~= nil and type(msg.reload) ~= "boolean" then
            reply({ op = "reset", ok = false, err = "reload debe ser booleano" })
            return false
        end
    end
    E.clear()
    E.victory_seen = false
    if spec then
        H.throttle = spec.map == "LVL_AMLO"
        reset_driver = Arena
        Arena.start({ spec = spec, start_phase = start_phase, reload = msg.reload == true })
    else
        G.set_boss_classes(nil)
        E.trace_hits = false
        H.throttle = false
        reset_driver = R
        R.start({ start_phase = start_phase })
    end
    mode = "reset"
    return true
end

function HANDLERS.config(msg)
    local applied = {}
    if msg.dilation ~= nil then
        local d = msg.dilation
        if type(d) ~= "number" or d < 0.1 or d > 20 then
            reply({ op = "config", ok = false, err = "dilation fuera de [0.1, 20]" })
            return false
        end
        -- La dilatacion vive en el WorldSettings del nivel y OpenLevel la devuelve a 1: se
        -- guarda y el reinicio la vuelve a aplicar.
        G.dilation = d
        G.apply_dilation()
        applied.dilation = G.gs:GetGlobalTimeDilation(G.world)
    end
    if msg.anim_tick_option ~= nil then
        local o = msg.anim_tick_option
        if o ~= false and (math.type(o) ~= "integer" or o < 0 or o > 3) then
            reply({ op = "config", ok = false, err = "anim_tick_option debe ser 0..3 o false" })
            return false
        end
        G.anim_tick_option = (o ~= false) and o or nil
        applied.anim_tick_option = o
    end
    if msg.console ~= nil then
        local cmds = type(msg.console) == "table" and msg.console or { msg.console }
        for _, c in ipairs(cmds) do
            if not console_allowed(c) then
                reply({ op = "config", ok = false, err = "comando de consola no admitido: " .. tostring(c) })
                return false
            end
        end
        for _, c in ipairs(cmds) do G.console(c) end
        applied.console = #cmds
    end
    reply({ op = "config", ok = true, applied = applied })
    return false
end

function HANDLERS.exec(msg)
    if not DEBUG then
        reply({ op = "exec", ok = false, err = "exec solo con DMB_DEBUG=1" })
        return false
    end
    local fn, err = load(msg.lua or "", "exec", "t", _ENV)
    if not fn then
        reply({ op = "exec", ok = false, err = tostring(err) })
        return false
    end
    local ok, res = xpcall(fn, debug.traceback)
    if ok and type(res) ~= "table" then res = tostring(res) end
    reply({ op = "exec", ok = ok, result = ok and res or nil, err = (not ok) and tostring(res) or nil })
    return false
end

function HANDLERS.close(msg)
    reply({ op = "close", ok = true })
    L.info("cierre pedido por el cliente")
    chan:close()
    mode = "idle"
    G.quit()
    return true
end

local function on_disconnect()
    L.info("cliente desconectado")
    chan:close()
    mode = "idle"
    last_client = os.clock()
end

--- Atiende ordenes hasta que una necesite frames. Bloquea el hilo del juego.
local function serve()
    while true do
        local line = chan:recv()
        if not line then
            on_disconnect()
            return
        end
        local msg, err = json.decode(line)
        if type(msg) ~= "table" then
            reply({ op = "error", ok = false, err = "json invalido: " .. tostring(err) })
        else
            local h = HANDLERS[msg.op]
            if not h then
                reply({ op = "error", ok = false, err = "op desconocida: " .. tostring(msg.op) })
            else
                local ok, needs_frames = pcall(h, msg)
                if not ok then
                    reply({ op = tostring(msg.op), ok = false, err = tostring(needs_frames) })
                elseif needs_frames then
                    if not chan:connected() then on_disconnect() end
                    return
                end
            end
        end
        if not chan:connected() then
            on_disconnect()
            return
        end
    end
end

local function idle_frame()
    if G.frame % CONNECT_EVERY_FRAMES == 0 and chan:connect() then
        L.info("cliente conectado")
        mode = "await"
        return true
    end
    if os.clock() - last_client > IDLE_TIMEOUT then
        L.info(string.format("sin cliente durante %.0f s: se cierra el juego", IDLE_TIMEOUT))
        last_client = os.clock()
        G.quit()
    end
    return false
end

local function on_frame()
    G.track()
    H.note_world(G.world_gen)
    if G.stable() then
        H.pump()
        G.keep_dilation()
        guards.tick()
        C.tick()
    end
    if mode == "idle" then
        if not idle_frame() then return end
    elseif mode == "step" then
        remaining = remaining - 1
        if remaining > 0 then
            if G.stable() then
                local ok, e = pcall(A.move_frame)
                if not ok then L.info("movimiento con error: " .. tostring(e)) end
            end
            return
        end
        if not send_state("step") then
            on_disconnect()
            return
        end
        mode = "await"
    elseif mode == "reset" then
        local r, e = reset_driver.tick()
        if r == nil then return end
        E.clear()
        if not send_state("reset", { ok = r, err = e, guards = guard_summary() }) then
            on_disconnect()
            return
        end
        mode = "await"
    end
    if mode == "await" then serve() end
end

LoopInGameThreadAfterFrames(1, function()
    local ok, err = pcall(on_frame)
    if not ok then L.info("error en el frame: " .. tostring(err)) end
end)
