-- Registro diferido de hooks de UE4SS.
--
-- RegisterHook necesita que la UFunction exista. Las funciones nativas existen desde el
-- arranque, pero las de Blueprint (BP_Player, BPC_DamageSystem, el jefe) solo cuando su clase
-- se carga, es decir al abrir LVL_Boss. Cada hook pedido queda pendiente y pump() lo
-- reintenta en los frames estables hasta que se registra; una vez registrado sobrevive a los
-- OpenLevel porque la clase sigue cargada.

local L = require("dmb.log")

local H = {}

local pending = {}
H.installed = 0
local active = {}

--- Pide un hook. Se intenta de inmediato y, si la funcion aun no existe, en cada pump().
function H.want(path, fn)
    pending[#pending + 1] = { path = path, fn = fn, logged = false }
    H.pump(true)
end

-- Tras un cambio de mundo se reintenta en cada frame (las clases acaban de cargarse y el hook
-- debe quedar listo antes del primer golpe); despues solo cada PUMP_EVERY frames, porque los
-- hooks de un jefe que no pertenece al mapa actual no se podran registrar nunca y reintentarlos
-- en cada frame es trabajo perdido. Solo se activa (H.throttle) en los mapas de varios jefes; en
-- LVL_Boss se reintenta en cada frame como siempre.
H.throttle = false
H.FAST_FRAMES = 180
H.PUMP_EVERY = 30
local frames_since_world = 0
local last_gen = -1

--- Informa del contador de mundos del juego para dosificar los reintentos.
function H.note_world(gen)
    if gen ~= last_gen then
        last_gen = gen
        frames_since_world = 0
    else
        frames_since_world = frames_since_world + 1
    end
end

--- Reintenta los hooks pendientes. Barato cuando no queda ninguno.
function H.pump(force)
    if #pending == 0 then return end
    if H.throttle and not force and frames_since_world > H.FAST_FRAMES and frames_since_world % H.PUMP_EVERY ~= 0 then return end
    local keep = {}
    for _, h in ipairs(pending) do
        local ok, err = pcall(RegisterHook, h.path, h.fn)
        if ok then
            H.installed = H.installed + 1
            active[h.path] = true
            L.info("hook activo " .. h.path)
        else
            if not h.logged then
                -- Solo la primera linea del error: el resto es la traza de Lua.
                L.info("hook pendiente " .. h.path .. ": " .. (tostring(err):match("[^\r\n]*") or "?"))
                h.logged = true
            end
            keep[#keep + 1] = h
        end
    end
    pending = keep
end

--- true si el hook de esa ruta ya esta registrado.
function H.is_active(path)
    return active[path] == true
end

function H.pending_count()
    return #pending
end

return H
