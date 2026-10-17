-- Validacion de la descripcion de jefe que envia el cliente en la orden reset.
--
-- Modulo puro (sin API de Unreal). La descripcion sale de data/jefes/<jefe>.json (seccion
-- "mod") y es la unica fuente de verdad de mapa, clases y arranque del episodio; el mod no
-- conoce jefes por nombre. Todo valor externo se valida con listas de admitidos: un mapa, un
-- tipo de arranque o una ruta fuera de lo admitido se rechaza antes de tocar el juego.

local M = {}

M.MAPS = { LVL_Boss = true, LVL_AMLO = true }
M.START_KINDS = { legacy_fog = true, spawn_class = true, level_event = true }
M.HOOK_EVENTS = { victory = true, phase2 = true, remote = true }
M.MAX_LIST = 32

local function is_name(s)
    return type(s) == "string" and #s <= 64 and s:match("^[%w_]+$") ~= nil
end

-- Ruta de objeto de Unreal: /Game/Carpeta/Archivo.Clase_C o con funcion tras dos puntos.
local function is_path(s)
    if type(s) ~= "string" or #s > 200 then return false end
    return s:match("^/Game/[%w_/]+%.[%w_]+$") ~= nil or s:match("^/Game/[%w_/]+%.[%w_]+:[%w_ ]+$") ~= nil
end

local function is_number(v, lo, hi)
    return type(v) == "number" and v == v and v >= lo and v <= hi
end

local function name_list(list, field)
    if type(list) ~= "table" or #list == 0 or #list > M.MAX_LIST then
        return nil, field .. " debe ser una lista de 1 a " .. M.MAX_LIST .. " nombres"
    end
    for _, n in ipairs(list) do
        if not is_name(n) then return nil, field .. " contiene un nombre no admitido" end
    end
    return list
end

local function vec3(v, field)
    if type(v) ~= "table" or not (is_number(v.x, -1e6, 1e6) and is_number(v.y, -1e6, 1e6) and is_number(v.z, -1e6, 1e6)) then
        return nil, field .. " debe ser {x, y, z} numerico"
    end
    return v
end

local function check_start(start)
    if type(start) ~= "table" or not M.START_KINDS[start.kind] then
        return nil, "start.kind no admitido"
    end
    if start.kind == "legacy_fog" then return start end
    local ok, err = vec3(start.player_offset, "start.player_offset")
    if not ok then return nil, err end
    if start.kind == "spawn_class" then
        if not is_path(start.class_path) then return nil, "start.class_path no es una ruta admitida" end
        ok, err = vec3(start.pos, "start.pos")
        if not ok then return nil, err end
        if start.hp ~= nil and not is_number(start.hp, 1, 100000) then return nil, "start.hp fuera de rango" end
    else
        if not is_name(start.level_class) then return nil, "start.level_class no admitido" end
        if type(start.event) ~= "string" or #start.event > 64 or not start.event:match("^[%w_ ]+$") then
            return nil, "start.event no admitido"
        end
        if not is_name(start.boss_class) then return nil, "start.boss_class no admitido" end
        if not is_number(start.wait_seconds or 1, 0, 10) then return nil, "start.wait_seconds fuera de rango" end
    end
    return start
end

local function check_hooks(hooks)
    if hooks == nil then return {} end
    if type(hooks) ~= "table" or #hooks > M.MAX_LIST then return nil, "hooks debe ser una lista corta" end
    for _, h in ipairs(hooks) do
        if type(h) ~= "table" or not is_path(h.path) or not M.HOOK_EVENTS[h.event] then
            return nil, "hook no admitido"
        end
        if h.event == "remote" and not is_name(h.id) then return nil, "hook remoto sin id valido" end
    end
    return hooks
end

--- Valida la descripcion y devuelve una copia normalizada, o nil y el motivo del rechazo.
function M.validate(spec)
    if type(spec) ~= "table" then return nil, "boss debe ser un objeto" end
    if not is_name(spec.id) then return nil, "boss.id no admitido" end
    if not M.MAPS[spec.map] then return nil, "boss.map no admitido: " .. tostring(spec.map) end
    local classes, err = name_list(spec.classes, "boss.classes")
    if not classes then return nil, err end
    local purge
    purge, err = name_list(spec.purge or spec.classes, "boss.purge")
    if not purge then return nil, err end
    local minors = {}
    if spec.minors ~= nil then
        minors, err = name_list(spec.minors, "boss.minors")
        if not minors then return nil, err end
    end
    local start
    start, err = check_start(spec.start)
    if not start then return nil, err end
    local hooks
    hooks, err = check_hooks(spec.hooks)
    if not hooks then return nil, err end
    local phase2 = spec.phase2
    if phase2 ~= nil then
        if type(phase2) ~= "table" or not is_name(phase2.class) or not is_number(phase2.timeout or 60, 5, 300) then
            return nil, "boss.phase2 no admitido"
        end
    end
    return {
        id = spec.id, map = spec.map, classes = classes, purge = purge, minors = minors, start = start,
        hooks = hooks, phase2 = phase2, trace_hits = spec.trace_hits == true,
    }
end

return M
