-- Mantiene cargadas las clases Blueprint del juego entre reinicios.
--
-- Problema medido: cada OpenLevel(LVL_Boss) descarga todas las clases Blueprint del juego
-- (BP_Player_C, BP_Enemy_Juice_C, BPC_DamageSystem_C, montages, widgets...) y las vuelve a
-- cargar en direcciones nuevas, a veces reutilizando direcciones de objetos ya liberados. Con
-- reinicios cada pocos segundos, UE4SS termina resolviendo una propiedad o una funcion sobre
-- memoria liberada y el juego cae con EXCEPTION_ACCESS_VIOLATION dentro de UE4SS (11 caidas
-- en 13-46 reinicios por instancia), o una lectura de propiedad devuelve nada y el reinicio
-- no encuentra al jefe o al pawn.
--
-- Solucion: un ObjectLibrary colgado de Engine.GameSingleton (el motor es raiz del recolector
-- de basura y ese puntero esta vacio en este juego) con referencias fuertes a todas las clases
-- Blueprint de /Game, excepto las de los mapas (la clase del script de nivel pertenece al
-- paquete del mundo y debe descargarse con el). Con eso las clases no se descargan, sus
-- direcciones no cambian y el reinicio ademas es mas rapido (no recarga recursos).
--
-- El barrido (ForEachUObject, ~90 ms) se hace antes de un OpenLevel, cuando LVL_Boss lleva
-- tiempo cargado y estable, y se repite solo si en el episodio aparecio contenido nuevo
-- (fase 2, pantalla final).

local G = require("dmb.game")
local L = require("dmb.log")

local K = {}

K.holder = nil
K.count = 0
K.sweeps = 0
K.need_sweep = true
K.error = nil

local CLASS_KINDS = {
    BlueprintGeneratedClass = true,
    AnimBlueprintGeneratedClass = true,
    WidgetBlueprintGeneratedClass = true,
}

local added = {}

local function holder()
    if K.holder and G.valid(K.holder) then return K.holder end
    local eng = FindFirstOf("GameEngine")
    if not G.valid(eng) then return nil end
    local current = eng.GameSingleton
    if G.valid(current) then
        if current:GetClass():GetFName():ToString() ~= "ObjectLibrary" then
            K.error = "Engine.GameSingleton ya esta en uso por el juego"
            return nil
        end
        K.holder = current
        return current
    end
    local gi = FindFirstOf("GameInstance")
    local lib = StaticConstructObject(StaticFindObject("/Script/Engine.ObjectLibrary"), gi)
    if not G.valid(lib) then
        K.error = "no se pudo crear el ObjectLibrary"
        return nil
    end
    eng.GameSingleton = lib
    K.holder = lib
    return lib
end

local function keep(full)
    if not full:find(" /Game/", 1, true) then return false end
    if full:find("/Game/LVL_", 1, true) or full:find("/Maps/", 1, true) then return false end
    if full:find("SKEL_", 1, true) or full:find("REINST_", 1, true) or full:find("TRASHCLASS", 1, true) then return false end
    return true
end

-- Mapas de combate en los que se ancla (los dos mapas de jefes del juego).
local COMBAT_MAPS = { LVL_Boss = true, LVL_AMLO = true }

--- Agrega al ancla las clases Blueprint cargadas que aun no tenga. Solo en un mapa de combate
--- estable.
function K.sweep()
    if not (G.stable() and COMBAT_MAPS[G.map]) then return false end
    local lib = holder()
    if not lib then
        L.info("anclaje de clases no disponible: " .. tostring(K.error))
        return false
    end
    local t0 = os.clock()
    local picked = {}
    ForEachUObject(function(obj)
        local ok, cn = pcall(function() return obj:GetClass():GetFName():ToString() end)
        if ok and CLASS_KINDS[cn] then picked[#picked + 1] = obj end
    end)
    local objs = lib.Objects
    local new = 0
    for _, cls in ipairs(picked) do
        local ok, full = pcall(function() return cls:GetFullName() end)
        if ok and not added[full] and keep(full) then
            objs[objs:GetArrayNum() + 1] = cls
            added[full] = true
            new = new + 1
        end
    end
    K.count = objs:GetArrayNum()
    K.sweeps = K.sweeps + 1
    K.need_sweep = false
    L.info(string.format("clases ancladas: %d nuevas, %d en total (%.0f ms)", new, K.count, (os.clock() - t0) * 1000))
    return true
end

--- Barre si hace falta. Se llama justo antes de cada OpenLevel del reinicio.
function K.before_reload()
    if K.need_sweep then pcall(K.sweep) end
end

return K
