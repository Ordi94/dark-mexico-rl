-- lab.lua (sondeo de LVL_AMLO)
-- Utilidades de medicion cargadas desde probe_amlo\cmd.lua con dofile(). Define la tabla
-- global AL. Todo corre en el hilo del juego.
--
-- Principios (los mismos que hicieron estable la medicion de LVL_Boss):
--  * Ninguna referencia a actores sobrevive entre frames sin revalidarla: un OpenLevel destruye
--    el mundo y deja punteros colgando.
--  * Las tareas por frame corren en un unico LoopInGameThreadAfterFrames(1) protegido con pcall.
--  * Durante un cambio de mundo y 3 frames despues no se toca ningun actor.

local U = AM.UEHelpers

AL = AL or {}
AL.LOG_PATH = AM.OUT_DIR .. "lab.log"
AL.tasks = AL.tasks or {}
AL.frame = AL.frame or 0

function AL.log(msg)
    local f = io.open(AL.LOG_PATH, "a")
    local line = string.format("%s f=%d tg=%.3f %s", os.date("%H:%M:%S"), AL.frame, AL.tgame and AL.tgame() or -1, tostring(msg))
    if f then f:write(line .. "\n"); f:close() end
end

local function valid(o)
    if o == nil then return false end
    local ok, r = pcall(function() return o:IsValid() end)
    return ok and r
end
AL.valid = valid

function AL.all(cls)
    local out = {}
    local ok, list = pcall(FindAllOf, cls)
    if ok and list then
        for _, o in ipairs(list) do
            if valid(o) then
                local n = o:GetFullName()
                if not n:find("Default__", 1, true) and not n:find("SKEL_", 1, true) then out[#out + 1] = o end
            end
        end
    end
    return out
end
function AL.first(cls) return AL.all(cls)[1] end

function AL.world() return U.GetWorld() end
function AL.gs() return U.GetGameplayStatics() end
function AL.pc()
    local pc = U.GetPlayerController()
    if valid(pc) then return pc end
end
function AL.pl()
    local pc = AL.pc()
    if not pc then return nil end
    local ok, p = pcall(function() return pc:K2_GetPawn() end)
    if ok and valid(p) then return p end
end

function AL.name(o)
    if not valid(o) then return "nil" end
    local ok, r = pcall(function() return o:GetFName():ToString() end)
    return ok and r or "?"
end
function AL.cls(o)
    if not valid(o) then return "nil" end
    local ok, r = pcall(function() return o:GetClass():GetFName():ToString() end)
    return ok and r or "?"
end
function AL.v3(v)
    if v == nil then return "nil" end
    local ok, r = pcall(function() return string.format("%.0f,%.0f,%.0f", v.X, v.Y, v.Z) end)
    return ok and r or "?"
end
function AL.loc(a)
    if not valid(a) then return nil end
    local ok, v = pcall(function() return a:K2_GetActorLocation() end)
    if ok then return v end
end
function AL.dist2d(a, b)
    local la, lb = AL.loc(a), AL.loc(b)
    if not la or not lb then return -1 end
    local dx, dy = la.X - lb.X, la.Y - lb.Y
    return math.sqrt(dx * dx + dy * dy)
end
function AL.tgame()
    local ok, t = pcall(function() return AL.gs():GetTimeSeconds(AL.world()) end)
    return ok and t or -1
end
function AL.get(o, prop)
    if not valid(o) then return nil end
    local ok, v = pcall(function() return o[prop] end)
    if ok then return v end
end
function AL.map()
    local ok, r = pcall(function() return AL.world():GetFName():ToString() end)
    return ok and r or "?"
end

--- Montage activo: nombre, posicion, play rate y duracion.
function AL.montage(ch)
    if not valid(ch) then return "nil", 0, 1, 0 end
    local ok, name, pos, rate, len = pcall(function()
        local ai = ch.Mesh:GetAnimInstance()
        if not valid(ai) then return "noanim", 0, 1, 0 end
        local m = ai:GetCurrentActiveMontage()
        if not valid(m) then return "none", 0, 1, 0 end
        local l = 0
        pcall(function() l = m:GetPlayLength() end)
        return m:GetFName():ToString(), ai:Montage_GetPosition(m), ai:Montage_GetPlayRate(m), l
    end)
    if ok then return name, pos, rate, len end
    return "err", 0, 1, 0
end

function AL.hp(ch)
    if not valid(ch) then return -1 end
    local ok, h = pcall(function() return ch.BPC_DamageSystem.Health end)
    return ok and h or -1
end
function AL.hpmax(ch)
    if not valid(ch) then return -1 end
    local ok, h = pcall(function() return ch.BPC_DamageSystem.MaxHealth end)
    return ok and h or -1
end

--- Enemigos vivos del mundo (BP_EnemyBase_C y subclases).
function AL.enemies() return AL.all("BP_EnemyBase_C") end

--- Primer enemigo vivo con controlador de la clase pedida (nombre de clase corto).
function AL.find_enemy(clsname)
    local fb = nil
    for _, e in ipairs(AL.enemies()) do
        if AL.cls(e) == clsname then
            local okc, ctrl = pcall(function() return e.Controller end)
            if okc and valid(ctrl) and not AL.get(e, "DEATH?") then return e end
            fb = fb or e
        end
    end
    return fb
end

function AL.key(evt, keyname)
    local p = AL.pl()
    if not p then return false, "sin pawn" end
    return pcall(function() p[evt]({ KeyName = FName(keyname) }) end)
end

AL.BTN = {
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
}
function AL.btn(id)
    local b = AL.BTN[id]
    if not b then return false, "boton desconocido " .. tostring(id) end
    return AL.key(b[1], b[2])
end

--- Tarea por frame: fn(ctx) devuelve true para terminar. ctx: n, t0, data.
function AL.add(name, fn)
    local t = { name = name, fn = fn, ctx = { n = 0, t0 = AL.tgame(), data = {} } }
    if AL._in_tick then
        AL._pending = AL._pending or {}
        AL._pending[#AL._pending + 1] = t
    else
        AL.tasks[#AL.tasks + 1] = t
    end
    AL.log("TAREA+ " .. name)
end
function AL.clear()
    if AL._in_tick then AL._cleared = true; AL._pending = nil else AL.tasks = {}; AL._pending = nil end
    AL.log("TAREAS borradas")
end

function AL.tick()
    AL.frame = AL.frame + 1
    local okw, wkey = pcall(function() local w = AL.world(); return w:GetFullName() .. "@" .. tostring(w:GetAddress()) end)
    wkey = okw and wkey or "?"
    if wkey ~= AL._last_world then
        AL._last_world = wkey
        AL._settle = 3
        AL._await_world = nil
        AL.log("mundo nuevo: " .. wkey)
        if AL.on_new_world then pcall(AL.on_new_world) end
    end
    if AL._await_world then return end
    if AL._settle and AL._settle > 0 then
        AL._settle = AL._settle - 1
        if AL._settle == 0 and AL.dilation_wanted then
            pcall(function() AL.gs():SetGlobalTimeDilation(AL.world(), AL.dilation_wanted) end)
        end
        return
    end
    AL._in_tick = true
    local keep = {}
    for _, t in ipairs(AL.tasks) do
        t.ctx.n = t.ctx.n + 1
        local ok, done = pcall(t.fn, t.ctx)
        if not ok then
            AL.log("TAREA ERROR " .. t.name .. ": " .. tostring(done))
        elseif done then
            AL.log("TAREA fin " .. t.name .. " frames=" .. t.ctx.n)
        else
            keep[#keep + 1] = t
        end
    end
    AL._in_tick = false
    if AL._cleared then keep = {}; AL._cleared = false end
    if AL._pending then
        for _, t in ipairs(AL._pending) do keep[#keep + 1] = t end
        AL._pending = nil
    end
    AL.tasks = keep
end

if not AL.loop_handle then
    AL.loop_handle = LoopInGameThreadAfterFrames(1, function() pcall(AL.tick) end)
    AL.log("bucle por frame iniciado")
end

function AL.open(map)
    AL.log("OpenLevel " .. map)
    AL._await_world = true
    AL.gs():OpenLevel(AL.world(), FName(map), true, "")
end

--- Dilatacion global; se reaplica sola tras cada cambio de mundo (OpenLevel la reinicia).
function AL.dilation(x)
    AL.dilation_wanted = x
    AL.gs():SetGlobalTimeDilation(AL.world(), x)
    return AL.gs():GetGlobalTimeDilation(AL.world())
end

--- Inventario de actores del mundo: cuenta por clase y posiciones de las clases filtradas.
function AL.inventory(path, pattern)
    local f = io.open(path, "w")
    local counts, order = {}, {}
    local arr = {}
    local actor_cls = StaticFindObject("/Script/Engine.Actor")
    AL.gs():GetAllActorsOfClass(AL.world(), actor_cls, arr)
    for _, item in ipairs(arr) do
        local a = item:get()
        local c = AL.cls(a)
        if not counts[c] then counts[c] = 0; order[#order + 1] = c end
        counts[c] = counts[c] + 1
        if pattern and c:find(pattern) then
            f:write(string.format("ACTOR %s %s loc=%s\n", c, AL.name(a), AL.v3(AL.loc(a))))
        end
    end
    table.sort(order)
    for _, c in ipairs(order) do f:write(string.format("CLASS %5d %s\n", counts[c], c)) end
    f:close()
    return #arr
end

function AL.hook(path, fn)
    AL.hooked = AL.hooked or {}
    if AL.hooked[path] then return true end
    local ok, e = pcall(RegisterHook, path, function(...)
        local okc, err = pcall(fn, ...)
        if not okc then AL.log("hook err " .. path .. " " .. tostring(err)) end
    end)
    if ok then AL.hooked[path] = true end
    AL.log("hook " .. path .. " ok=" .. tostring(ok) .. (ok and "" or (" " .. tostring(e):match("[^\r\n]*"))))
    return ok
end

AL.log("lab.lua cargado")

--- Vuelca propiedades y funciones (con parametros) de una clase y su jerarquia hasta 'stop'.
function AL.dump_class(cls, path, stop)
    local f = io.open(path, "w")
    local guard = 0
    while valid(cls) and guard < 40 do
        local cname = cls:GetFName():ToString()
        if stop and cname == stop then break end
        f:write("=== " .. cls:GetFullName() .. "\n")
        pcall(function()
            cls:ForEachProperty(function(p)
                local okt, t = pcall(function() return p:GetClass():GetFName():ToString() end)
                f:write(string.format("  PROP %s : %s\n", p:GetFName():ToString(), okt and t or "?"))
            end)
        end)
        pcall(function()
            cls:ForEachFunction(function(fn)
                local params = {}
                pcall(function() fn:ForEachProperty(function(p) params[#params + 1] = p:GetFName():ToString() end) end)
                f:write(string.format("  FUNC %s(%s)\n", fn:GetFName():ToString(), table.concat(params, ", ")))
            end)
        end)
        local ok, sup = pcall(function() return cls:GetSuperStruct() end)
        if not ok then break end
        cls = sup
        guard = guard + 1
    end
    f:close()
end

--- Lee el valor de propiedades simples de un objeto (numeros, booleanos, nombres, objetos).
function AL.props(obj, names)
    local out = {}
    for _, k in ipairs(names) do
        local ok, v = pcall(function()
            local x = obj[k]
            if type(x) == "userdata" then
                if x.IsValid and x.GetFullName then
                    if not x:IsValid() then return "nil" end
                    return x:GetFName():ToString()
                end
                if x.ToString then return x:ToString() end
                return tostring(x)
            end
            return x
        end)
        out[#out + 1] = k .. "=" .. tostring(ok and v or "err")
    end
    return table.concat(out, " ")
end

--- Jefe objetivo de las mediciones (clase corta); AL.boss() lo resuelve cada vez.
AL.boss_cls = AL.boss_cls or "BP_AMLOBOSS_C"
function AL.boss() return AL.find_enemy(AL.boss_cls) end

--- Grabador de cambios por frame del jugador y del jefe objetivo.
function AL.rec_fields()
    local p, b = AL.pl(), AL.boss()
    local pm = AL.montage(p)
    local bm = AL.montage(b)
    return {
        pawn = AL.name(p),
        php = string.format("%.0f", AL.hp(p)),
        pdead = tostring(p and AL.get(p.BPC_DamageSystem, "IsDead")),
        pmont = pm,
        boss = AL.name(b),
        bhp = string.format("%.0f", AL.hp(b)),
        bmont = bm,
        bstun = tostring(AL.get(b, "Is Stunned")),
        bph = (function() local v = AL.get(b, "Phase"); return type(v) == "number" and tostring(v) or "-" end)(),
        bdead = tostring(AL.get(b, "DEATH?")),
        btp = (function() local v = AL.get(b, "Is Teleporting"); return type(v) == "boolean" and tostring(v) or "-" end)(),
        map = AL.map(),
    }
end
function AL.rec_start(label)
    AL.rec_on = true
    AL.rec_prev = {}
    AL.add("rec " .. (label or ""), function(c)
        if not AL.rec_on then return true end
        local ok, cur = pcall(AL.rec_fields)
        if not ok then return false end
        local diff = {}
        for k, v in pairs(cur) do
            if AL.rec_prev[k] ~= v then diff[#diff + 1] = k .. "=" .. v end
        end
        if #diff > 0 then
            table.sort(diff)
            local p, b = AL.pl(), AL.boss()
            AL.log(string.format("REC %s pP=%s pB=%s d=%.0f", table.concat(diff, " "), AL.v3(AL.loc(p)), AL.v3(AL.loc(b)), AL.dist2d(p, b)))
        end
        AL.rec_prev = cur
    end)
end
function AL.rec_stop() AL.rec_on = false end

--- Teletransporta al jugador (sin barrido, con teleport fisico).
function AL.tp(x, y, z, yaw)
    local p = AL.pl()
    p:K2_SetActorLocation({ X = x, Y = y, Z = z }, false, {}, true)
    if yaw then
        pcall(function() p:K2_SetActorRotation({ Pitch = 0, Yaw = yaw, Roll = 0 }, true) end)
        pcall(function() AL.pc():SetControlRotation({ Pitch = 0, Yaw = yaw, Roll = 0 }) end)
    end
    return AL.v3(AL.loc(p))
end

--- Enemigos cercanos a un punto: clase, nombre, distancia, vida.
function AL.near(pt, radius)
    local out = {}
    for _, e in ipairs(AL.enemies()) do
        local l = AL.loc(e)
        if l then
            local d = math.sqrt((l.X - pt.X) ^ 2 + (l.Y - pt.Y) ^ 2)
            if d <= radius then out[#out + 1] = string.format("%s(%.0f hp=%.0f dead=%s)", AL.name(e), d, AL.hp(e), tostring(AL.get(e, "DEATH?"))) end
        end
    end
    return table.concat(out, " ")
end
return "lab ok"
