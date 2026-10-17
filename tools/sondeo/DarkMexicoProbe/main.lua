-- DarkMexicoProbe
-- Mod de sondeo: registra el mapa actual, detecta la aparicion de las clases Blueprint
-- relevantes del jefe y del jugador, vuelca sus propiedades/funciones y expone un canal de
-- comandos por archivo para poder experimentar sin teclado ni relanzar el juego.

-- En modo entrenamiento (DMB_TRAINING=1, instancias lanzadas por DarkMexicoBridge) el sondeo
-- no se carga: su canal por archivo y su registro son compartidos por todas las instancias y
-- su sondeo periodico de clases agrega trabajo al hilo del juego.
if os.getenv("DMB_TRAINING") == "1" then
    print("[Probe] desactivado en modo entrenamiento" .. string.char(10))
    return
end

local UEHelpers = require("UEHelpers")

local OUT_DIR = (os.getenv("DMPROBE_DIR") or ".") .. "\\"
local LOG_PATH = OUT_DIR .. "probe.log"
local CMD_PATH = OUT_DIR .. "cmd.lua"
local CMD_RESULT_PATH = OUT_DIR .. "cmd_result.log"

-- Clases cuyas instancias se vigilan. Se usa el nombre corto que espera FindAllOf.
local WATCHED_CLASSES = {
    "BP_Player_C",
    "BP_AMLOBOSS_C",
    "BP_Enemy_Juice_C",
    "BP_Enemy_JuicePhase2_C",
    "BP_EnemyBase_C",
    "BP_MexicoGameInstance_C",
    "Gamemode_C",
}

-- Instancias ya volcadas (por direccion) para no repetir volcados completos.
local dumped_instances = {}
local last_map = nil

--- Escribe una linea en el log propio. Abre y cierra el archivo en cada llamada para que
--- el contenido quede en disco aunque el juego termine de forma abrupta.
local function log(msg)
    local f = io.open(LOG_PATH, "a")
    if f then
        f:write(os.date("%H:%M:%S") .. " " .. tostring(msg) .. "\n")
        f:close()
    end
    print("[Probe] " .. tostring(msg) .. "\n")
end

--- Convierte cualquier valor devuelto por UE4SS a texto legible sin lanzar errores.
local function fmt(v, depth)
    depth = depth or 0
    local t = type(v)
    if t == "nil" then return "nil" end
    if t == "number" or t == "boolean" or t == "string" then return tostring(v) end
    if t == "userdata" then
        local ok, r = pcall(function()
            if v.IsValid and not v:IsValid() then return "<invalid>" end
            if v.GetFullName then return "obj:" .. v:GetFullName() end
            if v.ToString then return v:ToString() end
            if v.get then
                local g = v:get()
                return "param:" .. tostring(g)
            end
            return nil
        end)
        if ok and r then return r end
        -- Estructuras simples (FVector, FRotator): se prueban campos habituales.
        local parts = {}
        for _, k in ipairs({ "X", "Y", "Z", "Pitch", "Yaw", "Roll" }) do
            local okf, val = pcall(function() return v[k] end)
            if okf and type(val) == "number" then parts[#parts + 1] = k .. "=" .. string.format("%.3f", val) end
        end
        if #parts > 0 then return "{" .. table.concat(parts, ",") .. "}" end
        return "<" .. tostring(v) .. ">"
    end
    return "<" .. t .. ">"
end

--- Nombre de tipo de una propiedad (clase de FProperty), p. ej. FloatProperty.
local function prop_type(p)
    local ok, r = pcall(function() return p:GetClass():GetFName():ToString() end)
    return ok and r or "?"
end

local function prop_name(p)
    local ok, r = pcall(function() return p:GetFName():ToString() end)
    return ok and r or "?"
end

--- Recorre una clase y todas sus superclases ejecutando cb(clase).
local function walk_supers(cls, cb)
    local guard = 0
    while cls and guard < 40 do
        local ok_v, valid = pcall(function() return cls:IsValid() end)
        if not (ok_v and valid) then break end
        cb(cls)
        local ok, sup = pcall(function() return cls:GetSuperStruct() end)
        if not ok then break end
        cls = sup
        guard = guard + 1
    end
end

--- Vuelca propiedades y funciones (con sus parametros) de una clase y su jerarquia.
local function dump_class(cls, path)
    local f = io.open(path, "w")
    if not f then return end
    walk_supers(cls, function(c)
        f:write("=== " .. c:GetFullName() .. "\n")
        pcall(function()
            c:ForEachProperty(function(p)
                f:write(string.format("  PROP %s : %s\n", prop_name(p), prop_type(p)))
            end)
        end)
        pcall(function()
            c:ForEachFunction(function(fn)
                local params = {}
                pcall(function()
                    fn:ForEachProperty(function(p)
                        params[#params + 1] = prop_name(p) .. ":" .. prop_type(p)
                    end)
                end)
                f:write(string.format("  FUNC %s(%s)\n", fn:GetFName():ToString(), table.concat(params, ", ")))
            end)
        end)
    end)
    f:close()
end

--- Vuelca el valor actual de todas las propiedades de una instancia.
local function dump_instance(obj, path)
    local f = io.open(path, "w")
    if not f then return end
    f:write("INSTANCE " .. obj:GetFullName() .. "\n")
    walk_supers(obj:GetClass(), function(c)
        f:write("=== " .. c:GetFullName() .. "\n")
        pcall(function()
            c:ForEachProperty(function(p)
                local n = prop_name(p)
                local ok, v = pcall(function() return obj[n] end)
                f:write(string.format("  %s : %s = %s\n", n, prop_type(p), ok and fmt(v) or "<err>"))
            end)
        end)
    end)
    f:close()
end

--- Nombre corto del mapa activo, a partir del nombre completo del UWorld.
local function current_map()
    local ok, r = pcall(function()
        local w = UEHelpers.GetWorld()
        if w and w:IsValid() then return w:GetFName():ToString() end
        return nil
    end)
    return ok and r or nil
end

--- Revisa cada clase vigilada y, si hay instancias nuevas, vuelca clase e instancia.
local function scan_instances()
    for _, cname in ipairs(WATCHED_CLASSES) do
        local ok, list = pcall(FindAllOf, cname)
        if ok and list then
            for _, obj in ipairs(list) do
                local okv, addr, full = pcall(function()
                    if obj:IsValid() then return obj:GetAddress(), obj:GetFullName() end
                end)
                -- La clave combina direccion y nombre: tras un OpenLevel el asignador suele
                -- reutilizar la misma direccion para el actor nuevo y, con solo la direccion,
                -- la instancia nueva no se registraria.
                local key = okv and addr and (tostring(addr) .. "|" .. tostring(full)) or nil
                if key and not dumped_instances[key] then
                    dumped_instances[key] = true
                    log("INSTANCIA NUEVA " .. cname .. " -> " .. full)
                    -- Los objetos CDO/skeleton se omiten: no representan una instancia viva.
                    -- Con P.auto_dump=false solo se registra la aparicion: el volcado completo
                    -- tarda varios milisegundos y mete tirones en mediciones de tiempo.
                    if P.auto_dump and not string.find(full, "Default__", 1, true) and not string.find(full, "SKEL_", 1, true) then
                        local tag = cname .. "_" .. string.format("%X", addr)
                        pcall(dump_class, obj:GetClass(), OUT_DIR .. "class_" .. cname .. ".txt")
                        pcall(dump_instance, obj, OUT_DIR .. "inst_" .. tag .. ".txt")
                    end
                end
            end
        end
    end
end

-- Entorno para el canal de comandos: expone utilidades de este mod.
P = {
    log = log, fmt = fmt, dump_class = dump_class, dump_instance = dump_instance,
    prop_name = prop_name, prop_type = prop_type, walk_supers = walk_supers,
    UEHelpers = UEHelpers, OUT_DIR = OUT_DIR, scan = scan_instances,
}

local last_cmd = nil

-- Al arrancar se toma como ya ejecutado lo que haya en cmd.lua: ese archivo conserva el ultimo
-- comando de la sesion anterior (por ejemplo un OpenLevel o un "quit") y, sin esta marca, se
-- volveria a ejecutar solo al iniciar el juego.
do
    local f = io.open(CMD_PATH, "r")
    if f then
        last_cmd = f:read("*a")
        f:close()
    end
end

--- Ejecuta cmd.lua cuando su contenido cambia; el resultado va a cmd_result.log.
--- Permite iterar experimentos sin relanzar el juego ni usar el teclado.
local function poll_command()
    local f = io.open(CMD_PATH, "r")
    if not f then return end
    local src = f:read("*a")
    f:close()
    if src == last_cmd or src == "" then return end
    last_cmd = src
    local out = io.open(CMD_RESULT_PATH, "w")
    if not out then
        log("no se pudo abrir " .. CMD_RESULT_PATH)
        return
    end
    local fn, err = load(src, "cmd", "t", _ENV)
    if not fn then
        out:write("LOAD ERROR: " .. tostring(err) .. "\n")
    else
        local ok, res = xpcall(fn, debug.traceback)
        out:write((ok and "OK: " or "ERROR: ") .. tostring(res) .. "\n")
    end
    out:close()
end

log("DarkMexicoProbe cargado. Version UE4SS: " .. tostring(UE4SS and UE4SS.GetVersion and select(1, UE4SS.GetVersion()) or "?"))

--- Registra los bucles periodicos del mod (vigilancia de mapa/instancias y canal de comandos).
--- Se expone en P para poder purgar con ClearAllDelayedActions() los bucles de experimentos
--- que un comando haya dejado vivos y volver a levantar solo los del mod.
local function start_loops()
    LoopInGameThreadWithDelay(1000, function()
        local m = current_map()
        if m ~= last_map then
            log("MAPA: " .. tostring(m))
            last_map = m
        end
        pcall(scan_instances)
    end)

    LoopInGameThreadWithDelay(500, function()
        local ok, err = pcall(poll_command)
        if not ok then log("poll_command error: " .. tostring(err)) end
    end)
end

P.start_loops = start_loops
-- El volcado automatico de cada instancia nueva viene desactivado: recorre y lee todas las
-- propiedades del actor justo cuando el mapa se esta cargando, y una lectura de un puntero de
-- objeto aun no inicializado (o que apunta al mundo anterior) tumbo el juego con un
-- EXCEPTION_ACCESS_VIOLATION dentro de UE4SS tras un OpenLevel. Para volcar una clase o una
-- instancia concreta usar P.dump_class / P.dump_instance desde cmd.lua con el mapa ya estable.
P.auto_dump = false
start_loops()

-- Guarda de logros: el juego desbloquea logros de Steam desde Blueprint (primera muerte,
-- parry perfecto, segunda fase, derrotar jefes...) a traves de
-- AchievementWriteCallbackProxy:WriteProgress. Durante pruebas y entrenamiento esos eventos
-- ocurren miles de veces y desbloquearian logros en la cuenta real del jugador, lo que no se
-- puede deshacer. El pre-hook baja el progreso a 0: con progreso menor a 100 Steam no marca el
-- logro como desbloqueado. Se registra cada intento para poder auditarlo.
local ACHIEVEMENT_FN = "/Script/OnlineSubsystemUtils.AchievementWriteCallbackProxy:WriteProgress"
P.block_achievements = true
local achievement_hooked = false

local function hook_achievements()
    if achievement_hooked then return true end
    local ok, err = pcall(function()
        RegisterHook(ACHIEVEMENT_FN, function(ctx, world, pc, name, progress, tag)
            local n = "?"
            pcall(function() n = name:get():ToString() end)
            if P.block_achievements then
                local okset = pcall(function() progress:set(0.0) end)
                log("LOGRO bloqueado: " .. n .. " (progreso forzado a 0: " .. tostring(okset) .. ")")
            else
                log("LOGRO permitido: " .. n)
            end
        end)
    end)
    if ok then
        achievement_hooked = true
        log("guarda de logros activa")
    else
        log("guarda de logros pendiente: " .. tostring(err))
    end
    return achievement_hooked
end

-- La funcion nativa existe desde el arranque del motor, pero se reintenta cada segundo por si
-- el modulo OnlineSubsystemUtils aun no estuviera cargado cuando arranca el mod.
if not hook_achievements() then
    local handle
    handle = LoopInGameThreadWithDelay(1000, function()
        if hook_achievements() and handle then CancelDelayedAction(handle) end
    end)
end
