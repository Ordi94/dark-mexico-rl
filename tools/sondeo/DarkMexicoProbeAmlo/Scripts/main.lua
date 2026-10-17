-- DarkMexicoProbeAmlo
-- Mod de sondeo para LVL_AMLO: canal de comandos por archivo (probe_amlo\cmd.lua) y
-- protecciones de partida guardada y logros.
--
-- Solo se activa si el proceso arranca con la variable de entorno DMP_AMLO=1. UE4SS carga
-- todos los mods de la carpeta Mods al arrancar cualquier instancia del juego, incluidas las
-- del entrenamiento; sin la variable el mod termina aqui y no registra hooks, bucles ni
-- archivos, asi que esas instancias no cambian.
if os.getenv("DMP_AMLO") ~= "1" then
    print("[ProbeAmlo] inactivo: falta DMP_AMLO=1\n")
    return
end

local UEHelpers = require("UEHelpers")

local OUT_DIR = (os.getenv("DMPROBE_DIR") or ".") .. "\\"
local LOG_PATH = OUT_DIR .. "amlo.log"
local CMD_PATH = OUT_DIR .. "cmd.lua"
local CMD_RESULT_PATH = OUT_DIR .. "cmd_result.log"

--- Escribe una linea en amlo.log. Abre y cierra el archivo en cada llamada para que el
--- contenido quede en disco aunque el juego termine de forma abrupta. No se usa UE4SS.log:
--- lo trunca cada instancia que arranca, incluidas las del entrenamiento.
local function log(msg)
    local f = io.open(LOG_PATH, "a")
    if f then
        f:write(string.format("%s %.3f %s\n", os.date("%H:%M:%S"), os.clock(), tostring(msg)))
        f:close()
    end
end

AM = {
    log = log,
    OUT_DIR = OUT_DIR,
    UEHelpers = UEHelpers,
    saves_blocked = 0,
    saves_unblocked = 0,
    achievements_blocked = 0,
    save_guard = false,
    achievement_guard = false,
}

-- Proteccion de la partida guardada. Las victorias contra jefes llaman SaveGameToSlot y
-- escribirian MexicoSouls\Saved, que comparten todas las instancias y la partida real del
-- usuario. RegisterHook sobre una funcion nativa corre antes del cuerpo: se anula el objeto a
-- guardar y SaveGameToSlot vuelve sin escribir porque exige un objeto valido.
local SAVE_FN = "/Script/Engine.GameplayStatics:SaveGameToSlot"
local ok_save, err_save = pcall(RegisterHook, SAVE_FN, function(ctx, obj, slot)
    local s = "?"
    pcall(function() s = slot:get():ToString() end)
    local ok = pcall(function() obj:set(nil) end)
    local now_nil = false
    pcall(function()
        local v = obj:get()
        now_nil = (v == nil) or not v:IsValid()
    end)
    if ok and now_nil then
        AM.saves_blocked = AM.saves_blocked + 1
    else
        AM.saves_unblocked = AM.saves_unblocked + 1
    end
    log("GUARDADO slot=" .. s .. " anulado=" .. tostring(ok and now_nil))
end)
AM.save_guard = ok_save
log("guarda de guardado " .. (ok_save and "activa" or ("FALLO " .. tostring(err_save))))

-- Proteccion de logros: el progreso se fuerza a 0 para que Steam no marque nada. Los logros
-- se disparan desde Blueprint (parry, beber, derrotar jefes) y no se pueden deshacer.
local ACH_FN = "/Script/OnlineSubsystemUtils.AchievementWriteCallbackProxy:WriteProgress"
local function hook_achievements()
    if AM.achievement_guard then return true end
    local ok = pcall(RegisterHook, ACH_FN, function(ctx, world, pc, name, progress, tag)
        local n = "?"
        pcall(function() n = name:get():ToString() end)
        local okset = pcall(function() progress:set(0.0) end)
        AM.achievements_blocked = AM.achievements_blocked + 1
        log("LOGRO bloqueado " .. n .. " ok=" .. tostring(okset))
    end)
    AM.achievement_guard = ok
    return ok
end
if not hook_achievements() then
    local handle
    handle = LoopInGameThreadWithDelay(1000, function()
        if hook_achievements() and handle then CancelDelayedAction(handle) end
    end)
end
log("guarda de logros " .. (AM.achievement_guard and "activa" or "pendiente"))

-- Canal de comandos: ejecuta probe_amlo\cmd.lua en el hilo del juego cuando cambia su
-- contenido. Lo presente al arrancar se marca como ya ejecutado para que un comando de la
-- sesion anterior (OpenLevel, salir) no se repita solo.
local last_cmd = nil
do
    local f = io.open(CMD_PATH, "r")
    if f then
        last_cmd = f:read("*a")
        f:close()
    end
end

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

LoopInGameThreadWithDelay(250, function()
    local ok, err = pcall(poll_command)
    if not ok then log("poll_command error: " .. tostring(err)) end
end)

log("DarkMexicoProbeAmlo activo")
