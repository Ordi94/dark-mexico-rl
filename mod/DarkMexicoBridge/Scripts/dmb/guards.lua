-- Protecciones del modo entrenamiento.
--
-- 1. Logros: el juego desbloquea logros de Steam desde Blueprint (parry perfecto, beber,
--    primera muerte, derrotar al jefe) mediante AchievementWriteCallbackProxy:WriteProgress.
--    En entrenamiento esos eventos se repiten miles de veces y desbloquearian logros reales,
--    lo que no se puede deshacer. El pre-hook baja el progreso a 0 (Steam no marca nada).
-- 2. Partida guardada: la victoria llama dos veces a SaveGameToSlot("savegame") y escribiria
--    la partida del usuario (y todas las instancias comparten MexicoSouls\Saved). El pre-hook
--    anula el objeto a guardar: SaveGameToSlot con objeto nulo devuelve false sin tocar el
--    disco.
-- 3. Pausa: Start y Escape crean un WP_PauseGame_C que no detiene el juego pero roba el foco
--    de la interfaz. Si aparece uno (por ejemplo por una pulsacion fisica con la ventana
--    enfocada) se cierra con el manejador de su boton Reanudar.
--    No se usa NotifyOnNewObject: su callback puede ejecutarse desde el hilo de carga
--    asincrona mientras el hilo del juego corre Lua, y el estado Lua no es seguro entre hilos.

local G = require("dmb.game")
local L = require("dmb.log")
local H = require("dmb.hooks")

local M = {}

M.achievements_blocked = 0
M.saves_blocked = 0
M.saves_unblocked = 0
M.pauses_closed = 0
M.PAUSE_SWEEP_FRAMES = 600

local pause_check_frames = 0

local ESSENTIAL = {
    "/Script/OnlineSubsystemUtils.AchievementWriteCallbackProxy:WriteProgress",
    "/Script/Engine.GameplayStatics:SaveGameToSlot",
}

--- true si las protecciones de logros y partida guardada estan activas. Sin ellas el mod no
--- inicia episodios: una victoria sin proteccion escribiria la partida del usuario.
function M.essential_active()
    for _, p in ipairs(ESSENTIAL) do
        if not H.is_active(p) then return false end
    end
    return true
end

--- SaveGameToSlot recibe el objeto a guardar como primer parametro; con el objeto nulo
--- termina sin escribir el archivo. (AsyncSaveGameToSlot no es una UFunction en este motor y
--- el juego no la usa: los guardados medidos pasan todos por SaveGameToSlot.)
local function block_save(ctx, obj, slot)
    local s = "?"
    pcall(function() s = slot:get():ToString() end)
    local ok, err = pcall(function() obj:set(nil) end)
    local now_nil = false
    pcall(function()
        local v = obj:get()
        now_nil = (v == nil) or not v:IsValid()
    end)
    if ok and now_nil then
        M.saves_blocked = M.saves_blocked + 1
    else
        M.saves_unblocked = M.saves_unblocked + 1
    end
    L.info("guardado slot=" .. s .. " anulado=" .. tostring(ok and now_nil) .. (ok and "" or (" " .. tostring(err))))
end

local function block_achievement(ctx, world, pc, name, progress, tag)
    local n = "?"
    pcall(function() n = name:get():ToString() end)
    local ok = pcall(function() progress:set(0.0) end)
    M.achievements_blocked = M.achievements_blocked + 1
    L.info("logro bloqueado: " .. n .. " progreso=0 ok=" .. tostring(ok))
end

--- Registra las protecciones.
function M.install()
    H.want(ESSENTIAL[1], block_achievement)
    H.want(ESSENTIAL[2], block_save)
    -- El widget de pausa se crea dentro del evento del boton (Start del mando o Escape) y se
    -- agrega al viewport ahi mismo; el hook corre despues del cuerpo y deja pendiente la
    -- revision para el siguiente frame.
    for _, evt in ipairs({ "InpActEvt_Gamepad_Special_Right_K2Node_InputKeyEvent_24", "InpActEvt_Escape_K2Node_InputKeyEvent_7" }) do
        H.want("/Game/BP_Player.BP_Player_C:" .. evt, function() pause_check_frames = 5 end)
    end
end

local widget_lib, widget_class = nil, nil

--- Menus de pausa de primer nivel del mundo actual. Se pide la clase nativa UserWidget y se
--- filtra por nombre: la clase Blueprint WP_PauseGame_C se recarga en cada OpenLevel y no se
--- puede guardar (ver game.lua).
local function pause_widgets()
    if not G.valid(widget_lib) then widget_lib = StaticFindObject("/Script/UMG.Default__WidgetBlueprintLibrary") end
    if not G.valid(widget_class) then widget_class = StaticFindObject("/Script/UMG.UserWidget") end
    local out = {}
    if not (G.valid(widget_lib) and G.valid(widget_class)) then return out end
    local arr = {}
    pcall(function() widget_lib:GetAllWidgetsOfClass(G.world, arr, widget_class, true) end)
    for _, item in ipairs(arr) do
        local w = item:get()
        local ok, name = pcall(G.class_name, w)
        if ok and name == "WP_PauseGame_C" then out[#out + 1] = w end
    end
    return out
end

--- Cierra los menus de pausa visibles. Se llama una vez por frame (casi siempre sin trabajo).
function M.tick()
    -- Ademas de los hooks, una revision de respaldo cada PAUSE_SWEEP_FRAMES frames cubre
    -- cualquier otra via de apertura del menu.
    if G.frame % M.PAUSE_SWEEP_FRAMES == 0 then pause_check_frames = math.max(pause_check_frames, 1) end
    if pause_check_frames <= 0 or not G.stable() then return end
    pause_check_frames = pause_check_frames - 1
    for _, w in ipairs(pause_widgets()) do
        pcall(function()
            if w:IsValid() and w:IsInViewport() then
                w["BndEvt__WP_MainMenu_BT_StartGame_K2Node_ComponentBoundEvent_0_OnButtonClickedEvent__DelegateSignature"](w)
                M.pauses_closed = M.pauses_closed + 1
                L.info("menu de pausa cerrado")
            end
        end)
    end
end

return M
