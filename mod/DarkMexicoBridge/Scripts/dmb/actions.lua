-- Acciones del agente sobre el BP_Player.
--
-- Las acciones se ejecutan llamando directamente a los eventos de entrada del Blueprint
-- (InpActEvt_*) con su FKey, igual que haria el mando: el juego aplica sus propias
-- condiciones (CanAttack, CanRoll, Power, vida llena, Resistencia). No se envian pulsaciones
-- del sistema operativo, asi que el usuario puede seguir usando la PC.
--
-- Tipos de entrada:
--  * Botones de un disparo (btn): ataque, especial, maniobra, caguama y enfoque. Se pulsan
--    una vez al inicio del paso.
--  * Botones mantenidos (lb, lt): bloqueo y correr. Se envia el evento de pulsar o soltar
--    solo cuando el estado pedido cambia respecto al paso anterior.
--  * Movimiento (mx, my en [-1, 1]): se aplica en cada frame del paso. Con mf="boss" el eje
--    my apunta hacia el jefe y mx a su derecha (independiente de la camara); con mf="cam" se
--    usa el evento Move del Blueprint, relativo a la rotacion de control, como el stick.
-- Start, Escape, la cruceta y el boton A no se exponen: abren menus o el selector de armas.

local G = require("dmb.game")

local A = {}

local function key(name) return { KeyName = FName(name) } end

-- Evento del Blueprint y nombre de la tecla del mando para cada boton.
local BUTTONS = {
    rb = { "InpActEvt_Gamepad_RightShoulder_K2Node_InputKeyEvent_33", "Gamepad_RightShoulder" },
    y = { "InpActEvt_Gamepad_FaceButton_Top_K2Node_InputKeyEvent_26", "Gamepad_FaceButton_Top" },
    b = { "InpActEvt_Gamepad_FaceButton_Right_K2Node_InputKeyEvent_32", "Gamepad_FaceButton_Right" },
    x = { "InpActEvt_Gamepad_FaceButton_Left_K2Node_InputKeyEvent_19", "Gamepad_FaceButton_Left" },
    rs = { "InpActEvt_Gamepad_RightThumbstick_K2Node_InputKeyEvent_29", "Gamepad_RightThumbstick" },
}
local LB_DOWN = { "InpActEvt_Gamepad_LeftShoulder_K2Node_InputKeyEvent_20", "Gamepad_LeftShoulder" }
local LB_UP = { "InpActEvt_Gamepad_LeftShoulder_K2Node_InputKeyEvent_21", "Gamepad_LeftShoulder" }
local LT_DOWN = { "InpActEvt_Gamepad_LeftTrigger_K2Node_InputKeyEvent_30", "Gamepad_LeftTrigger" }
local LT_UP = { "InpActEvt_Gamepad_LeftTrigger_K2Node_InputKeyEvent_31", "Gamepad_LeftTrigger" }
-- Interactuar: solo lo usa el reinicio para atravesar la niebla del jefe.
A.INTERACT = { "InpActEvt_Gamepad_FaceButton_Bottom_K2Node_InputKeyEvent_17", "Gamepad_FaceButton_Bottom" }

A.BUTTON_NAMES = { "rb", "y", "b", "x", "rs" }

-- Estado de los botones mantenidos y accion vigente.
local held = { lb = false, lt = false }
local current = { mx = 0.0, my = 0.0, mf = "boss" }
-- Boton diferido al segundo frame del paso (ver A.begin).
local deferred_btn = nil

--- Llama un evento de entrada del pawn.
function A.fire(pawn, entry)
    pawn[entry[1]](key(entry[2]))
end

local function clamp(v)
    if type(v) ~= "number" or v ~= v then return 0.0 end
    if v > 1.0 then return 1.0 end
    if v < -1.0 then return -1.0 end
    return v
end

--- Valida una accion recibida. Devuelve la accion normalizada o nil, mensaje.
--- Lista de admitidos: cualquier boton o marco desconocido se rechaza en lugar de ignorarse.
function A.parse(a)
    if a == nil then a = {} end
    if type(a) ~= "table" then return nil, "a debe ser un objeto" end
    local btn = a.btn
    if btn ~= nil and not BUTTONS[btn] then return nil, "btn desconocido: " .. tostring(btn) end
    local mf = a.mf or "boss"
    if mf ~= "boss" and mf ~= "cam" then return nil, "mf debe ser boss o cam" end
    if a.lb ~= nil and type(a.lb) ~= "boolean" then return nil, "lb debe ser booleano" end
    if a.lt ~= nil and type(a.lt) ~= "boolean" then return nil, "lt debe ser booleano" end
    return { btn = btn, lb = a.lb == true, lt = a.lt == true, mx = clamp(a.mx or 0.0), my = clamp(a.my or 0.0), mf = mf }
end

--- Olvida los botones mantenidos (el pawn nuevo de un reinicio nace sin nada pulsado).
function A.reset()
    held.lb, held.lt = false, false
    current = { mx = 0.0, my = 0.0, mf = "boss" }
    deferred_btn = nil
end

--- Aplica la parte instantanea de la accion al inicio del paso. Devuelve true si se pudo.
--- n es el numero de frames del paso.
---
--- La maniobra (roll) toma su direccion del ultimo vector de entrada de movimiento que el
--- personaje consumio, no del que se aplica en el mismo frame: si el boton y el movimiento
--- van juntos el personaje rueda siempre hacia atras (medido en las 8 direcciones). Por eso,
--- cuando la maniobra trae direccion y el paso tiene mas de un frame, se aplica primero el
--- movimiento y el boton se dispara en el frame siguiente.
function A.begin(act, n)
    current = act
    deferred_btn = nil
    local p = G.pawn()
    if not p then return false end
    if act.lb ~= held.lb then
        A.fire(p, act.lb and LB_DOWN or LB_UP)
        held.lb = act.lb
    end
    if act.lt ~= held.lt then
        A.fire(p, act.lt and LT_DOWN or LT_UP)
        held.lt = act.lt
    end
    local directional = act.btn == "b" and (act.mx ~= 0.0 or act.my ~= 0.0)
    if directional and (n or 1) > 1 then
        A.move_frame()
        deferred_btn = act.btn
        return true
    end
    if act.btn then A.fire(p, BUTTONS[act.btn]) end
    A.move_frame()
    return true
end

--- Aplica el movimiento de la accion vigente en el frame actual.
function A.move_frame()
    local mx, my = current.mx, current.my
    if mx == 0.0 and my == 0.0 then return end
    local p = G.pawn()
    if not p then return end
    if deferred_btn then
        local btn = deferred_btn
        deferred_btn = nil
        A.fire(p, BUTTONS[btn])
    end
    if current.mf == "cam" then
        p:Move(mx, my)
        return
    end
    local b = G.boss_actor()
    if not b then return end
    local lp, lb = p:K2_GetActorLocation(), b:K2_GetActorLocation()
    local fx, fy = lb.X - lp.X, lb.Y - lp.Y
    local d = math.sqrt(fx * fx + fy * fy)
    if d < 1.0 then return end
    fx, fy = fx / d, fy / d
    -- En Unreal X es adelante, Y derecha y Z arriba: la derecha de (fx, fy) es (-fy, fx).
    local wx, wy = fx * my - fy * mx, fy * my + fx * mx
    p:AddMovementInput({ X = wx, Y = wy, Z = 0.0 }, 1.0, false)
end

return A
