-- heal.lua
-- Medicion de la curacion con caguama (boton X) en pelea real: el jugador recibe dano del jefe
-- sin escribir su vida, se aleja, bebe y se registra cada frame lo que pasa durante la
-- animacion. Requiere lab.lua cargado (tabla LAB). Todo corre dentro del bucle por frame de
-- LAB, en el hilo del juego.

HL = HL or {}

local function fmt(...) return string.format(...) end

-- Los escenarios se expresan como listas de pasos y no como corrutinas: las funciones C de
-- UE4SS (print, llamadas a UFunction) fallan dentro de una corrutina porque el hilo Lua de la
-- corrutina no esta registrado en UE4SS. Cada paso es fn(s) y se llama una vez por frame;
-- devuelve true para pasar al siguiente, "abort" para terminar el escenario. s.n cuenta los
-- frames del paso y s.data es compartido por todo el escenario.

--- Ejecuta una lista de pasos (se aplana: un elemento puede ser otra lista de pasos).
function HL.run(name, steps)
    local flat = {}
    local function add(x)
        if type(x) == "function" then flat[#flat + 1] = x else for _, y in ipairs(x) do add(y) end end
    end
    add(steps)
    local i, n, data = 1, 0, {}
    LAB.add(name, function()
        local f = flat[i]
        if not f then return true end
        n = n + 1
        -- Un jugador muerto invalida la medicion (y el juego aun le deja beber): se corta aqui.
        local p = LAB.pl()
        -- Con varias peleas en la lista se salta al arranque de la siguiente.
        if p and not data.allow_dead and not HL.FIGHT_START[f] and LAB.get(p.BPC_DamageSystem, "IsDead") then
            LAB.log("ESCENARIO " .. name .. " cortado: jugador muerto | " .. HL.line())
            HL.trace_left = 0
            local j = i + 1
            while flat[j] and not HL.FIGHT_START[flat[j]] do j = j + 1 end
            if not flat[j] then return true end
            i, n = j, 0
            return false
        end
        local r = f({ n = n, data = data })
        if r == "abort" then LAB.log("ESCENARIO " .. name .. " abortado en paso " .. i); return true end
        if r then i = i + 1; n = 0 end
        return flat[i] == nil
    end)
end

local S = {}
HL.S = S
HL.FIGHT_START = setmetatable({}, { __mode = "k" })

function S.wait(k) return function(s) return s.n >= k end end
function S.call(fn) return function(s) fn(s); return true end end

--- Espera hasta que pred(s) sea verdadero; si pasan maxf frames, aborta.
function S.until_(pred, maxf, label)
    return function(s)
        local ok, r = pcall(pred, s)
        if ok and r then return true end
        if s.n >= (maxf or 600) then LAB.log("TIMEOUT " .. (label or "")); return "abort" end
        return false
    end
end

--- Montage activo del jugador con su velocidad de reproduccion.
local function pmontage(p)
    local name, pos, rate = "none", 0, 0
    pcall(function()
        local ai = p.Mesh:GetAnimInstance()
        local m = ai:GetCurrentActiveMontage()
        if m and m:IsValid() then
            name, pos, rate = m:GetFName():ToString(), ai:Montage_GetPosition(m), ai:Montage_GetPlayRate(m)
        end
    end)
    return name, pos, rate
end

local function speed2d(a)
    local ok, s = pcall(function() local v = a:GetVelocity(); return math.sqrt(v.X * v.X + v.Y * v.Y) end)
    return ok and s or -1
end

--- Estado completo del jugador y del jefe en una linea; dt es relativo a la ultima pulsacion de X.
function HL.line()
    local p, b = LAB.pl(), LAB.boss()
    if not p then return "sin pawn" end
    local m, pos, rate = pmontage(p)
    local bm, bpos = LAB.montage(b)
    local dmg = p.BPC_DamageSystem
    return fmt("tg=%.3f dt=%.3f hp=%.1f cag=%s pow=%.2f res=%.2f mont=%s@%.3f r=%.2f spd=%.0f maxw=%.0f canheal=%s canatk=%s canroll=%s roll=%s atk=%s blk=%s inv=%s dead=%s loc=%s B[%s@%.3f att=%s hp=%.0f] d=%.0f",
        LAB.tgame(), LAB.tgame() - (HL.t0 or 0), LAB.hp(p), tostring(p.Caguamas), tonumber(p.Power) or -1,
        tonumber(p.Resistencia) or -1, m, pos, rate, speed2d(p), p.CharacterMovement.MaxWalkSpeed,
        tostring(p["Can Heal"]), tostring(p.CanAttack), tostring(p.CanRoll), tostring(p.IsRolling),
        tostring(p["Is Attacking"]), tostring(dmg.IsBlocking), tostring(dmg.IsInvincible), tostring(dmg.IsDead),
        LAB.v3(LAB.loc(p)), bm, bpos, tostring(LAB.get(b, "Attacking")), LAB.hp(b), LAB.dist2d(p, b))
end

-- Traza por frame: mientras HL.trace_left > 0 registra HL.line() cada frame con la etiqueta activa.
if not HL.trace_task then
    HL.trace_task = true
    HL.trace_left = 0
    LAB.add("heal_trace", function()
        if HL.trace_left > 0 then
            HL.trace_left = HL.trace_left - 1
            LAB.log("TR[" .. (HL.tag or "") .. "] " .. HL.line())
        end
        return false
    end)
end

function HL.trace(tag, frames)
    HL.tag = tag
    HL.trace_left = frames
end

--- Captura de pantalla del motor (se guarda en Saved\Screenshots); queda en el frame pedido.
function HL.shot(label)
    local ok, err = pcall(function()
        local ksl = StaticFindObject("/Script/Engine.Default__KismetSystemLibrary")
        ksl:ExecuteConsoleCommand(LAB.world(), "shot", LAB.pc())
    end)
    LAB.log(fmt("SHOT %s tg=%.3f dt=%.3f ok=%s %s", label, LAB.tgame(), LAB.tgame() - (HL.t0 or 0), tostring(ok), ok and "" or tostring(err)))
end

-- Hooks de solo lectura sobre la logica de curacion del BP_Player. Se registran una vez por
-- proceso: UE4SS no permite quitar hooks de Blueprint de forma fiable y duplicarlos
-- duplicaria las lineas del registro.
if not HL.hooks_done then
    HL.hooks_done = true
    local function H(path, fn)
        local ok, e = pcall(RegisterHook, path, function(...)
            -- Al cambiar de mapa el motor interrumpe los montages del mundo que se destruye y
            -- dispara estos manejadores; leer el pawn en ese momento tumbo el juego con un
            -- acceso a memoria invalida. Mientras LAB espera o asienta el mundo nuevo no se
            -- registra nada.
            if LAB._await_world or (LAB._settle or 0) > 0 then return end
            local okc, err = pcall(fn, ...)
            if not okc then LAB.log("hook err " .. path .. " " .. tostring(err)) end
        end)
        LAB.log("hook " .. path .. " ok=" .. tostring(ok) .. (ok and "" or (" " .. tostring(e))))
    end
    local function st() return LAB.pl() and HL.line() or "" end
    for _, f in ipairs({ "Reset Cure", "RESET CURE 2", "chelita", "ResetSpeed", "CAGUAMA CHARGE", "ADD POWER",
        "LOGRO DRINK BEER", "CancelAnimation", "Cooldown Vigor", "StartRecoveringVigor", "Stop Recovering Vigor",
        "InpActEvt_Gamepad_FaceButton_Left_K2Node_InputKeyEvent_19" }) do
        H("/Game/BP_Player.BP_Player_C:" .. f, function() LAB.log("HOOK Player:" .. f .. " | " .. st()) end)
    end
    H("/Game/BP_Player.BP_Player_C:Heal", function(ctx, amount)
        LAB.log("HOOK Player:Heal amount=" .. tostring(amount:get()) .. " | " .. st())
    end)
    H("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:Heal", function(ctx, amount)
        LAB.log("HOOK DamageSystem:Heal amount=" .. tostring(amount:get()) .. " | " .. st())
    end)
    H("/Game/BP_Player.BP_Player_C:Damage Response", function(ctx, resp, amount)
        LAB.log("HOOK Player:Damage Response resp=" .. tostring(resp:get()) .. " amount=" .. tostring(amount:get()) .. " | " .. st())
    end)
    -- Manejadores de los montages reproducidos con PlayMontage: el sufijo identifica el nodo y
    -- NotifyName dice que notify del montage disparo la logica.
    local guids = { "A5A3D59345F98B78FB02C58B72F94860", "A40910B44288BF68998835A2F73D7933", "119B0ACB4F6CB702EECD2198A0388025",
        "677C6DF34975A06AA1246F844037BDF0", "FC06EA7C44BCF147F789529371880EF3", "C52A75CB4AF557E851C6E794463E61BE",
        "211799354D7D6CEFF2F0B59E6ADB8B9B", "56E795924D497FC8E0977EA319E5C348", "CBC0647B475A541F432B6DA2D297F552",
        "E9640CC14EFFDF58BFA65C82A9BC4101", "4114109C41A5E8973AC96E85B8079E6B", "FEA4C7FB46559B2B6F2E949615ECCE79",
        "684CBA2249C87FAE38F29B8AD7A76921", "D65BB30942533140D6433091332F29CD", "7453984C486FF5834C5843AB95A49300" }
    for _, g in ipairs(guids) do
        for _, kind in ipairs({ "OnNotifyBegin_", "OnNotifyEnd_", "OnInterrupted_", "OnBlendOut_", "OnCompleted_" }) do
            local fname = kind .. g
            H("/Game/BP_Player.BP_Player_C:" .. fname, function(ctx, nn)
                local m = pmontage(LAB.pl())
                if m == "MM_Idle1_Montage" or HL.trace_left > 0 then
                    LAB.log(fmt("HOOK %s%s notify=%s | %s", kind, g:sub(1, 6), tostring(nn:get():ToString()), st()))
                end
            end)
        end
    end
    -- Golpes recibidos por el jugador, con la posicion de su propio montage en ese momento.
    H("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:TakeDamage", function(ctx, causer, info)
        local comp = ctx:get()
        if LAB.cls(comp:GetOwner()) ~= "BP_Player_C" then return end
        local i = info:get()
        local amt = -1
        pcall(function() amt = i["Amount_7_7BB9AB76446D18C11D94DA96B85DE23C"] end)
        HL.hits = (HL.hits or 0) + 1
        LAB.log(fmt("GOLPE amount=%s | %s", tostring(amt), st()))
    end)
end

--- Pasos que arrancan la pelea por el flujo real: OpenLevel, teleport frente a la niebla y A.
function S.fight()
    local first = S.call(function() LAB.open("LVL_Boss") end)
    HL.FIGHT_START[first] = true
    return {
        first,
        S.until_(function() return LAB.pl() and LAB.boss() end, 900, "pawn y jefe"),
        S.wait(3),
        S.call(function() LAB.pl():K2_SetActorLocation({ X = 47, Y = -2260, Z = 380 }, false, {}, true) end),
        S.wait(7),
        S.call(function() LAB.btn("A17"); LAB.log("FIGHT A pulsado") end),
        S.until_(function() local m = LAB.montage(LAB.boss()); return m ~= "none" and m ~= "nil" end, 600, "jefe activo"),
        S.call(function() LAB.log("FIGHT jefe activo | " .. HL.line()) end),
    }
end

local function dir_to(p, b, sign)
    local lp, lb = LAB.loc(p), LAB.loc(b)
    local dx, dy = lb.X - lp.X, lb.Y - lp.Y
    local d = math.sqrt(dx * dx + dy * dy)
    if d < 1 then return { X = 0, Y = -1, Z = 0 } end
    return { X = sign * dx / d, Y = sign * dy / d, Z = 0 }
end
HL.dir_to = dir_to

--- Se acerca al jefe y espera, sin defenderse, hasta que la vida baje de below.
--- Aborta si el jugador muere o no recibe golpe en maxf frames.
function S.get_hit(below, maxf)
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if not p or not b then return false end
        local hp = LAB.hp(p)
        if p.BPC_DamageSystem.IsDead then LAB.log("GET_HIT jugador muerto"); return "abort" end
        if hp < (below or 100) - 0.01 then
            LAB.log(fmt("GET_HIT vida %.1f < %.1f", hp, below or 100))
            return true
        end
        if s.n > (maxf or 2400) then LAB.log("GET_HIT sin golpe a tiempo"); return "abort" end
        if LAB.dist2d(p, b) > 170 then p:AddMovementInput(dir_to(p, b, 1), 1.0, false) end
        return false
    end
end

--- Camina alejandose del jefe durante k frames.
function S.retreat(k)
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if p and b then p:AddMovementInput(dir_to(p, b, -1), 1.0, false) end
        return s.n >= k
    end
end

--- Espera a que el jugador no tenga montage activo (no pulsar X en medio de otra accion).
function S.wait_free(maxf)
    return S.until_(function() return pmontage(LAB.pl()) == "none" end, maxf or 300, "jugador libre")
end

--- Pulsa X, fija el origen de tiempo dt y activa la traza por frame.
function HL.drink(tag, frames)
    HL.t0 = LAB.tgame()
    LAB.log("DRINK " .. tag .. " ANTES | " .. HL.line())
    HL.trace(tag, frames or 180)
    LAB.btn("X")
end
function S.drink(tag, frames) return S.call(function() HL.drink(tag, frames) end) end

--- Constantes del montage de beber y del juego leidas de los objetos cargados.
function HL.info()
    local out = {}
    pcall(function()
        local m = StaticFindObject("/Game/Characters/Mannequins/Anims/Unarmed/MM_Idle1_Montage.MM_Idle1_Montage")
        out[#out + 1] = fmt("MM_Idle1_Montage len=%.3f rate=%.2f blendIn=%s blendOut=%s blendOutTrig=%s",
            m:GetPlayLength(), m.RateScale, tostring(pcall(function() return m.BlendIn.BlendTime end) and m.BlendIn.BlendTime),
            tostring(pcall(function() return m.BlendOut.BlendTime end) and m.BlendOut.BlendTime), tostring(m.BlendOutTriggerTime))
        local ok, err = pcall(function()
            local n = m.Notifies
            out[#out + 1] = "notifies=" .. tostring(#n)
            for i = 1, #n do
                local e = n[i]
                local parts = {}
                for _, k in ipairs({ "NotifyName", "TriggerTimeOffset", "EndTriggerTimeOffset", "Duration", "LinkValue", "TriggerWeightThreshold" }) do
                    local okk, v = pcall(function() local x = e[k]; if type(x) == "userdata" and x.ToString then return x:ToString() end; return x end)
                    parts[#parts + 1] = k .. "=" .. (okk and tostring(v) or "?")
                end
                pcall(function() parts[#parts + 1] = "Notify=" .. e.Notify:GetFullName() end)
                pcall(function() parts[#parts + 1] = "NotifyNameField=" .. e.Notify.NotifyName:ToString() end)
                out[#out + 1] = table.concat(parts, " ")
            end
        end)
        if not ok then out[#out + 1] = "notifies err " .. tostring(err) end
    end)
    pcall(function()
        local gi = P.UEHelpers.GetGameInstance()
        out[#out + 1] = "GI CaguamasMaximas=" .. tostring(gi["CaguamasMáximas"])
    end)
    pcall(function()
        local p = LAB.pl()
        out[#out + 1] = fmt("player Caguamas=%s DrinkBeers=%s CanHeal=%s Power=%s MaxHealth=%s", tostring(p.Caguamas), tostring(p.DrinkBeers),
            tostring(p["Can Heal"]), tostring(p.Power), tostring(p.BPC_DamageSystem.MaxHealth))
    end)
    local s = table.concat(out, "\n")
    LAB.log("INFO " .. s)
    return s
end


--- Rueda alejandose del jefe en cuanto CanRoll lo permite: tras el primer golpe de un combo
--- es la unica forma de no comerse los siguientes (el combo cargado suma 115 de dano).
function S.roll_away(maxf)
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if not p or not b then return false end
        p:AddMovementInput(HL.dir_to(p, b, -1), 1.0, false)
        if p.CanRoll then
            LAB.btn("B")
            LAB.log("ROLL_AWAY tras " .. s.n .. " frames | " .. HL.line())
            return true
        end
        return s.n >= (maxf or 90)
    end
end

--- Pasos de un ensayo completo: asegurar vida < 100 con dano real, alejarse, esperar a estar
--- libre, beber y, durante 200 frames, opcionalmente seguir caminando (move=true) y ejecutar
--- action() en el frame at (contado desde la pulsacion de X).
function HL.trial(tag, opts)
    opts = opts or {}
    -- Con sprint=true la huida se hace corriendo (LT, 800 uu/s, mas rapido que el jefe al
    -- caminar) y se suelta LT unos frames antes de beber: mientras se corre Can Heal es false.
    local flee = opts.sprint and {
        S.roll_away(),
        S.call(function() LAB.btn("LT_A") end),
        S.retreat(opts.retreat or 180),
        S.call(function() LAB.btn("LT_B") end),
        S.wait(3),
    } or { S.retreat(opts.retreat or 150) }
    return {
        S.get_hit(opts.below or 100),
        flee,
        S.wait_free(300),
        S.drink(tag, opts.frames or 200),
        function(s)
            local p, b = LAB.pl(), LAB.boss()
            if opts.move and p and b then p:AddMovementInput(HL.dir_to(p, b, -1), 1.0, false) end
            if opts.spam_x and s.n % opts.spam_x == 0 then LAB.btn("X") end
            if opts.action and s.n == (opts.at or 18) then
                LAB.log(fmt("ACCION %s en dt=%.3f | %s", opts.label or "?", LAB.tgame() - HL.t0, HL.line()))
                opts.action()
            end
            return s.n >= (opts.frames or 200)
        end,
        S.call(function() LAB.log("TRIAL " .. tag .. " FIN | " .. HL.line()) end),
    }
end

-- Primer impacto de cada ataque del jefe en fase 1, en segundos de montage (medido con hooks de
-- dano). Sirve para pulsar X de modo que el golpe caiga en un instante elegido de la bebida.
HL.FIRST_IMPACT = {
    Boss_Attack_Uppercut_InP_Montage = 0.895,
    Boss_Attack_ChargedCombo_InP_Montage = 0.713,
    Boss_Attack_ChargedCombo_InP_Montage1 = 0.713,
    Boss_Attack_SwingAndSlam_InP_Montage = 1.844,
    Anim_Sword_Air_ComboAttack_InPlace_Montage = 0.85,
    Anim_Sword_Air_Attack1_InPlace_Montage = 0.558,
    Anim_SwordV2_Combo03_01_RM_Montage2 = 0.685,
}

--- Junto al jefe, espera un ataque conocido y pulsa X cuando faltan hit_at segundos (de juego)
--- para su primer impacto: el golpe deberia llegar hit_at segundos despues de empezar a beber.
function S.drink_into_hit(tag, hit_at, frames)
    local st = { fired = false, left = frames or 200 }
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if not p or not b then return false end
        if st.fired then
            st.left = st.left - 1
            return st.left <= 0
        end
        if s.n > 3600 then LAB.log("DRINK_INTO_HIT sin ocasion"); return "abort" end
        local d = LAB.dist2d(p, b)
        local bm, bpos = LAB.montage(b)
        local imp = HL.FIRST_IMPACT[bm]
        if not imp and d > 150 then p:AddMovementInput(HL.dir_to(p, b, 1), 1.0, false) end
        if imp and pmontage(p) == "none" then
            local rate = 1.0
            pcall(function() local ai = b.Mesh:GetAnimInstance(); rate = ai:Montage_GetPlayRate(ai:GetCurrentActiveMontage()) end)
            local togo = (imp - bpos) / rate
            if togo > 0 and togo <= hit_at then
                LAB.log(fmt("DRINK_INTO_HIT %s impacto esperado en %.3f s (%s@%.3f r=%.1f)", tag, togo, bm, bpos, rate))
                HL.drink(tag, frames or 200)
                st.fired = true
            end
        end
        return false
    end
end

-- Todos los impactos conocidos de cada ataque del jefe en fase 1 (segundos de montage).
HL.IMPACTS = {
    Boss_Attack_Uppercut_InP_Montage = { 0.895 },
    Boss_Attack_ChargedCombo_InP_Montage = { 0.713, 1.19, 1.87, 2.84 },
    Boss_Attack_ChargedCombo_InP_Montage1 = { 0.713, 1.19, 1.87, 2.84 },
    Boss_Attack_SwingAndSlam_InP_Montage = { 1.844, 4.175 },
    Anim_Sword_Air_ComboAttack_InPlace_Montage = { 0.85, 2.07 },
    Anim_Sword_Air_Attack1_InPlace_Montage = { 0.558 },
    Anim_SwordV2_Combo03_01_RM_Montage2 = { 0.685 },
}

--- Junto al jefe, pulsa X cuando el siguiente impacto del ataque en curso llegara entre lo y
--- hi segundos de juego despues (sirve para que el golpe caiga despues del notify de curacion).
function S.drink_window(tag, lo, hi, frames)
    local st = { fired = false, left = frames or 200 }
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if not p or not b then return false end
        if st.fired then
            st.left = st.left - 1
            return st.left <= 0
        end
        if s.n > 3600 then LAB.log("DRINK_WINDOW sin ocasion"); return "abort" end
        local bm, bpos = LAB.montage(b)
        local list = HL.IMPACTS[bm]
        if not list and LAB.dist2d(p, b) > 150 then p:AddMovementInput(HL.dir_to(p, b, 1), 1.0, false) end
        if list and p["Can Heal"] and LAB.hp(p) < 100 then
            local rate = 1.0
            pcall(function() local ai = b.Mesh:GetAnimInstance(); rate = ai:Montage_GetPlayRate(ai:GetCurrentActiveMontage()) end)
            for _, imp in ipairs(list) do
                if imp > bpos then
                    local togo = (imp - bpos) / rate
                    if togo >= lo and togo <= hi then
                        LAB.log(fmt("DRINK_WINDOW %s impacto esperado en %.3f s (%s@%.3f r=%.1f)", tag, togo, bm, bpos, rate))
                        HL.drink(tag, frames or 200)
                        st.fired = true
                    end
                    break
                end
            end
        end
        return false
    end
end

--- Huye del jefe (corriendo) hasta que se cumpla una condicion de seguridad: jefe sin atacar
--- (Attacking=false) y a mas de min_d uu; entonces suelta LT y bebe. Mide si esa regla basta
--- para terminar la animacion sin recibir golpes.
function S.drink_when_safe(tag, min_d, frames)
    local st = { phase = 0, left = frames or 200 }
    return function(s)
        local p, b = LAB.pl(), LAB.boss()
        if not p or not b then return false end
        if st.phase == 2 then
            st.left = st.left - 1
            return st.left <= 0
        end
        if s.n > 1800 then LAB.log("DRINK_WHEN_SAFE sin ocasion"); return "abort" end
        local d = LAB.dist2d(p, b)
        local att = LAB.get(b, "Attacking")
        if st.phase == 0 then
            if s.n == 1 then LAB.btn("LT_A") end
            if not att and d > min_d then
                LAB.btn("LT_B")
                st.phase = 1
                st.release = s.n
            else
                p:AddMovementInput(HL.dir_to(p, b, -1), 1.0, false)
            end
        elseif st.phase == 1 and s.n >= st.release + 2 and p["Can Heal"] and pmontage(p) == "none" then
            LAB.log(fmt("DRINK_WHEN_SAFE %s d=%.0f", tag, d))
            HL.drink(tag, frames or 200)
            st.phase = 2
        end
        return false
    end
end

LAB.log("heal.lua cargado")
return "heal ok"
