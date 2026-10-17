-- lab.lua
-- Utilidades de medicion para validar el puente con el juego. Se carga desde cmd.lua con
-- dofile(); define la tabla global LAB. Todo se ejecuta en el hilo del juego.
--
-- Principios:
--  * Nunca se guardan referencias a actores entre frames sin volver a validarlas: un
--    OpenLevel destruye el mundo y deja punteros colgando.
--  * Las tareas por frame corren en un unico bucle LoopInGameThreadAfterFrames(1); cada
--    tarea esta protegida con pcall para que un error no mate el bucle en silencio.
--  * Los registros van a lab.log y a UE4SS.log (print), cuya marca de tiempo es de alta
--    resolucion y sirve para medir duraciones en tiempo real.

local U = P.UEHelpers

LAB = LAB or {}
LAB.LOG_PATH = P.OUT_DIR .. "lab.log"
LAB.tasks = LAB.tasks or {}
LAB.frame = LAB.frame or 0

--- Escribe una linea en lab.log y en UE4SS.log.
function LAB.log(msg)
    local f = io.open(LAB.LOG_PATH, "a")
    local line = string.format("%s f=%d %s", os.date("%H:%M:%S"), LAB.frame, tostring(msg))
    if f then f:write(line .. "\n"); f:close() end
    print("[LAB] " .. line .. "\n")
end

local function valid(o)
    if o == nil then return false end
    local ok, r = pcall(function() return o:IsValid() end)
    return ok and r
end
LAB.valid = valid

--- Primer objeto valido de la clase que no sea CDO; nil si no hay.
function LAB.first(cls)
    local ok, list = pcall(FindAllOf, cls)
    if not ok or not list then return nil end
    for _, o in ipairs(list) do
        if valid(o) then
            local n = o:GetFullName()
            if not n:find("Default__", 1, true) and not n:find("SKEL_", 1, true) then return o end
        end
    end
    return nil
end

function LAB.all(cls)
    local out = {}
    local ok, list = pcall(FindAllOf, cls)
    if ok and list then
        for _, o in ipairs(list) do
            if valid(o) and not o:GetFullName():find("Default__", 1, true) then out[#out + 1] = o end
        end
    end
    return out
end

function LAB.pc()
    local pc = U.GetPlayerController()
    if valid(pc) then return pc end
    return nil
end

--- Pawn poseido por el PlayerController (no el primer BP_Player_C que aparezca).
function LAB.pl()
    local pc = LAB.pc()
    if not pc then return nil end
    local ok, p = pcall(function() return pc:K2_GetPawn() end)
    if ok and valid(p) then return p end
    return nil
end

--- Jefe activo: BP_EnemyBase_C (incluye subclases) con IS BOSS? = true y controlado por
--- su IA. Tras la muerte del jugador el juego genera un jefe nuevo y el anterior queda sin
--- controlador en (0,0,0) hasta que se destruye; por eso no basta con el primero que aparezca.
function LAB.boss()
    local fallback = nil
    for _, e in ipairs(LAB.all("BP_EnemyBase_C")) do
        local ok, b = pcall(function() return e["IS BOSS?"] end)
        if ok and b then
            local okc, ctrl = pcall(function() return e.Controller end)
            if okc and valid(ctrl) then return e end
            fallback = fallback or e
        end
    end
    return fallback or LAB.first("BP_EnemyBase_C")
end

function LAB.world() return U.GetWorld() end
function LAB.gs() return U.GetGameplayStatics() end

function LAB.name(o)
    if not valid(o) then return "nil" end
    local ok, r = pcall(function() return o:GetFName():ToString() end)
    return ok and r or "?"
end

function LAB.cls(o)
    if not valid(o) then return "nil" end
    local ok, r = pcall(function() return o:GetClass():GetFName():ToString() end)
    return ok and r or "?"
end

function LAB.v3(v)
    if v == nil then return "nil" end
    local ok, r = pcall(function() return string.format("%.0f,%.0f,%.0f", v.X, v.Y, v.Z) end)
    return ok and r or "?"
end

function LAB.loc(a)
    if not valid(a) then return nil end
    local ok, v = pcall(function() return a:K2_GetActorLocation() end)
    if ok then return v end
    return nil
end

function LAB.dist2d(a, b)
    local la, lb = LAB.loc(a), LAB.loc(b)
    if not la or not lb then return -1 end
    local dx, dy = la.X - lb.X, la.Y - lb.Y
    return math.sqrt(dx * dx + dy * dy)
end

function LAB.tgame()
    local ok, t = pcall(function() return LAB.gs():GetTimeSeconds(LAB.world()) end)
    return ok and t or -1
end

function LAB.treal()
    local ok, t = pcall(function() return LAB.gs():GetRealTimeSeconds(LAB.world()) end)
    return ok and t or -1
end

--- Montage activo de un personaje: nombre y posicion (segundos dentro del montage).
function LAB.montage(ch)
    if not valid(ch) then return "nil", 0 end
    local ok, name, pos = pcall(function()
        local ai = ch.Mesh:GetAnimInstance()
        if not valid(ai) then return "noanim", 0 end
        local m = ai:GetCurrentActiveMontage()
        if not valid(m) then return "none", 0 end
        return m:GetFName():ToString(), ai:Montage_GetPosition(m)
    end)
    if ok then return name, pos end
    return "err", 0
end

function LAB.hp(ch)
    if not valid(ch) then return -1 end
    local ok, h = pcall(function() return ch.BPC_DamageSystem.Health end)
    return ok and h or -1
end

function LAB.get(o, prop)
    if not valid(o) then return nil end
    local ok, v = pcall(function() return o[prop] end)
    if ok then return v end
    return nil
end

--- Instantanea compacta del combate en una linea.
function LAB.snap()
    local p, b, pc = LAB.pl(), LAB.boss(), LAB.pc()
    local pm, pp = LAB.montage(p)
    local bm, bp = LAB.montage(b)
    local vt = pc and LAB.name(pc:GetViewTarget()) or "nil"
    local cam = "nil"
    pcall(function() cam = LAB.v3(pc.PlayerCameraManager:GetCameraLocation()) end)
    local rot = "nil"
    pcall(function() local r = pc:GetControlRotation(); rot = string.format("%.0f", r.Yaw) end)
    return string.format(
        "tg=%.2f P[%s hp=%.0f res=%.2f cag=%s mont=%s@%.2f blk=%s pw=%s roll=%s atk=%s lock=%s] B[%s hp=%.0f mont=%s@%.2f att=%s stun=%s ph=%s dead=%s] d=%.0f vt=%s cam=%s yaw=%s",
        LAB.tgame(), LAB.v3(LAB.loc(p)), LAB.hp(p), tonumber(LAB.get(p, "Resistencia")) or -1,
        tostring(LAB.get(p, "Caguamas")), pm, pp,
        tostring(p and LAB.get(p.BPC_DamageSystem, "IsBlocking")), tostring(LAB.get(p, "Is Within Parry Window")),
        tostring(LAB.get(p, "IsRolling")), tostring(LAB.get(p, "Is Attacking")), tostring(LAB.get(p, "TargetLock?")),
        LAB.v3(LAB.loc(b)), LAB.hp(b), bm, bp, tostring(LAB.get(b, "Attacking")), tostring(LAB.get(b, "Is Stunned")),
        tostring(LAB.get(b, "Phase")), tostring(LAB.get(b, "DEATH?")), LAB.dist2d(p, b), vt, cam, rot)
end

--- Llama un evento de entrada del BP_Player con su FKey (p. ej. "Gamepad_RightShoulder").
function LAB.key(evt, keyname)
    local p = LAB.pl()
    if not p then return false, "sin pawn" end
    return pcall(function() p[evt]({ KeyName = FName(keyname) }) end)
end

-- Eventos del BP_Player por boton del mando (nombre del evento, nombre de la tecla).
LAB.BTN = {
    RB = { "InpActEvt_Gamepad_RightShoulder_K2Node_InputKeyEvent_33", "Gamepad_RightShoulder" },
    LB_DOWN = { "InpActEvt_Gamepad_LeftShoulder_K2Node_InputKeyEvent_20", "Gamepad_LeftShoulder" },
    LB_UP = { "InpActEvt_Gamepad_LeftShoulder_K2Node_InputKeyEvent_21", "Gamepad_LeftShoulder" },
    LT_A = { "InpActEvt_Gamepad_LeftTrigger_K2Node_InputKeyEvent_30", "Gamepad_LeftTrigger" },
    LT_B = { "InpActEvt_Gamepad_LeftTrigger_K2Node_InputKeyEvent_31", "Gamepad_LeftTrigger" },
    RS = { "InpActEvt_Gamepad_RightThumbstick_K2Node_InputKeyEvent_29", "Gamepad_RightThumbstick" },
    X = { "InpActEvt_Gamepad_FaceButton_Left_K2Node_InputKeyEvent_19", "Gamepad_FaceButton_Left" },
    Y = { "InpActEvt_Gamepad_FaceButton_Top_K2Node_InputKeyEvent_26", "Gamepad_FaceButton_Top" },
    B = { "InpActEvt_Gamepad_FaceButton_Right_K2Node_InputKeyEvent_32", "Gamepad_FaceButton_Right" },
    A17 = { "InpActEvt_Gamepad_FaceButton_Bottom_K2Node_InputKeyEvent_17", "Gamepad_FaceButton_Bottom" },
    A45 = { "InpActEvt_Gamepad_FaceButton_Bottom_K2Node_InputKeyEvent_45", "Gamepad_FaceButton_Bottom" },
    A53 = { "InpActEvt_Gamepad_FaceButton_Bottom_K2Node_InputKeyEvent_53", "Gamepad_FaceButton_Bottom" },
    START = { "InpActEvt_Gamepad_Special_Right_K2Node_InputKeyEvent_24", "Gamepad_Special_Right" },
    UP3 = { "InpActEvt_Gamepad_DPad_Up_K2Node_InputKeyEvent_3", "Gamepad_DPad_Up" },
    DOWN2 = { "InpActEvt_Gamepad_DPad_Down_K2Node_InputKeyEvent_2", "Gamepad_DPad_Down" },
}

function LAB.btn(id)
    local b = LAB.BTN[id]
    if not b then return false, "boton desconocido " .. tostring(id) end
    return LAB.key(b[1], b[2])
end

--- Agrega una tarea por frame. fn(ctx) devuelve true para terminar.
--- ctx: n (frames desde el inicio), t0 (tiempo de juego al inicio), data (tabla libre).
--- Si se llama desde dentro de una tarea, se encola y entra al terminar el frame actual
--- (la lista se reconstruye en cada tick y la perderia).
function LAB.add(name, fn)
    local t = { name = name, fn = fn, ctx = { n = 0, t0 = LAB.tgame(), data = {} } }
    if LAB._in_tick then
        LAB._pending = LAB._pending or {}
        LAB._pending[#LAB._pending + 1] = t
    else
        LAB.tasks[#LAB.tasks + 1] = t
    end
    LAB.log("TAREA+ " .. name)
end

--- Borra todas las tareas. Si se llama desde dentro de una tarea, el borrado se aplica al final
--- del frame (la lista se reconstruye en cada tick); las tareas agregadas despues del borrado
--- en ese mismo frame se conservan.
function LAB.clear()
    if LAB._in_tick then
        LAB._cleared = true
        LAB._pending = nil
    else
        LAB.tasks = {}
        LAB._pending = nil
    end
    LAB.log("TAREAS borradas")
end

function LAB.tick()
    LAB.frame = LAB.frame + 1
    -- Durante y justo despues de un cambio de mapa no se toca ningun actor: el mundo anterior
    -- se esta destruyendo y leerlo puede tumbar el proceso. Se esperan unos frames.
    -- Se compara la direccion del UWorld y no su nombre: al recargar el mismo mapa el nombre
    -- no cambia. Tras LAB.open las tareas esperan a que el mundo sea otro objeto.
    local okw, wkey = pcall(function() local w = LAB.world(); return w:GetFullName() .. "@" .. tostring(w:GetAddress()) end)
    wkey = okw and wkey or "?"
    if wkey ~= LAB._last_world then
        LAB._last_world = wkey
        LAB._settle = 3
        LAB._await_world = nil
        LAB.log("mundo nuevo: " .. wkey)
    end
    if LAB._await_world then return end
    if LAB._settle and LAB._settle > 0 then
        LAB._settle = LAB._settle - 1
        return
    end
    LAB._in_tick = true
    local keep = {}
    for _, t in ipairs(LAB.tasks) do
        t.ctx.n = t.ctx.n + 1
        local ok, done = pcall(t.fn, t.ctx)
        if not ok then
            LAB.log("TAREA ERROR " .. t.name .. ": " .. tostring(done))
        elseif done then
            LAB.log("TAREA fin " .. t.name .. " frames=" .. t.ctx.n)
        else
            keep[#keep + 1] = t
        end
    end
    LAB._in_tick = false
    if LAB._cleared then
        keep = {}
        LAB._cleared = false
    end
    if LAB._pending then
        for _, t in ipairs(LAB._pending) do keep[#keep + 1] = t end
        LAB._pending = nil
    end
    LAB.tasks = keep
end

if not LAB.loop_handle then
    LAB.loop_handle = LoopInGameThreadAfterFrames(1, function() pcall(LAB.tick) end)
    LAB.log("bucle por frame iniciado")
end

--- Tarea de muestreo: registra LAB.snap() cada 'every' frames durante 'frames' frames.
function LAB.watch(label, frames, every)
    every = every or 1
    LAB.add("watch " .. label, function(c)
        if c.n % every == 0 then LAB.log(label .. " " .. LAB.snap()) end
        return c.n >= frames
    end)
end

--- Abre un mapa por nombre.
function LAB.open(map)
    LAB.log("OpenLevel " .. map)
    LAB._await_world = true
    LAB.gs():OpenLevel(LAB.world(), FName(map), true, "")
end

function LAB.dilation(x)
    LAB.gs():SetGlobalTimeDilation(LAB.world(), x)
    LAB.log("dilation=" .. tostring(LAB.gs():GetGlobalTimeDilation(LAB.world())))
end

--- Grabador de cambios: cada frame compara un conjunto de campos y registra solo lo que
--- cambia. Permite reconstruir la linea de tiempo exacta (muerte, golpes, montages, fase)
--- sin inundar el registro.
function LAB.rec_fields()
    local p, b, pc = LAB.pl(), LAB.boss(), LAB.pc()
    local pm = LAB.montage(p)
    local bm = LAB.montage(b)
    return {
        pawn = LAB.name(p),
        vt = pc and LAB.name(pc:GetViewTarget()) or "nil",
        php = string.format("%.1f", LAB.hp(p)),
        pdead = tostring(p and LAB.get(p.BPC_DamageSystem, "IsDead")),
        pmont = pm,
        pblk = tostring(p and LAB.get(p.BPC_DamageSystem, "IsBlocking")),
        ppw = tostring(LAB.get(p, "Is Within Parry Window")),
        pcag = tostring(LAB.get(p, "Caguamas")),
        boss = LAB.name(b),
        bhp = string.format("%.1f", LAB.hp(b)),
        bmont = bm,
        batt = tostring(LAB.get(b, "Attacking")),
        bstun = tostring(LAB.get(b, "Is Stunned")),
        bph = tostring(LAB.get(b, "Phase")),
        bdead = tostring(LAB.get(b, "DEATH?")),
        bparry = tostring(LAB.get(b, "Parry Count")),
        map = (function() local ok, r = pcall(function() return LAB.world():GetFName():ToString() end); return ok and r or "?" end)(),
    }
end

function LAB.rec_start(label)
    LAB.rec_on = true
    LAB.rec_prev = {}
    LAB.add("rec " .. (label or ""), function(c)
        if not LAB.rec_on then return true end
        local ok, cur = pcall(LAB.rec_fields)
        if not ok then return false end
        local diff = {}
        for k, v in pairs(cur) do
            if LAB.rec_prev[k] ~= v then diff[#diff + 1] = k .. "=" .. v end
        end
        if #diff > 0 then
            table.sort(diff)
            LAB.log(string.format("REC tg=%.3f %s", LAB.tgame(), table.concat(diff, " ")))
        end
        LAB.rec_prev = cur
    end)
end

function LAB.rec_stop() LAB.rec_on = false end

LAB.log("lab.lua cargado")
return "lab ok"
