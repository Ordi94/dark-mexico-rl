-- Bot de medicion: mantiene al jugador junto al jefe objetivo (AL.boss_cls) para que el jefe
-- recorra su repertorio y cada golpe quede registrado por el hook de TakeDamage.
-- Opciones:
--   frames      duracion de la tarea
--   heal_below  si la vida del jugador baja de este valor se escribe a 100 (solo medicion)
--   attack      cada cuantos frames pulsar RB estando a menos de 220 uu (nil = no ataca)
--   parry       tabla {montage = {pos_impacto,...}}: pulsa LB 'lead*rate' antes de cada impacto
--   lead        anticipacion del parry en segundos de montage (defecto 0.235, medido con Juice)
--   keep_dist   distancia a la que se acerca (defecto 180)
--   lock        pulsar RS para enfocar al jefe

function AL.fightbot(opts)
    opts = opts or {}
    local frames = opts.frames or 3600
    local lead = opts.lead or 0.235
    local st = { pm = "", pp = 0, release_at = -1 }
    AL.bot_stats = { parry_tries = 0 }
    AL.add("fightbot", function(c)
        local p, b = AL.pl(), AL.boss()
        if not p or not b then return c.n > frames end
        if AL.get(p.BPC_DamageSystem, "IsDead") then return c.n > frames end
        local hp = AL.hp(p)
        if opts.heal_below and hp < opts.heal_below and hp > 0 then p.BPC_DamageSystem.Health = 100.0 end
        local d = AL.dist2d(p, b)
        local bm, bp, br = AL.montage(b)
        if opts.parry then
            local hits = opts.parry[bm]
            if hits and bm == st.pm then
                for _, t in ipairs(hits) do
                    local trig = t - lead * br
                    if st.pp < trig and bp >= trig then
                        AL.btn("LB_DOWN")
                        st.release_at = c.n + (opts.hold or 25)
                        AL.bot_stats.parry_tries = AL.bot_stats.parry_tries + 1
                        AL.log(string.format("PARRYBOT LB %s pos=%.3f rate=%.2f impacto=%.3f d=%.0f", bm, bp, br, t, d))
                    end
                end
            end
            if st.release_at == c.n then AL.btn("LB_UP") end
        end
        st.pm, st.pp = bm, bp
        if d > (opts.keep_dist or 180) then
            local lp, lb = AL.loc(p), AL.loc(b)
            p:AddMovementInput({ X = (lb.X - lp.X) / d, Y = (lb.Y - lp.Y) / d, Z = 0 }, 1.0, false)
        end
        if opts.lock and c.n % 60 == 0 and not AL.get(p, "TargetLock?") and d < 450 then AL.btn("RS") end
        if opts.attack and d < 220 and c.n % opts.attack == 0 then AL.btn("RB") end
        return c.n > frames
    end)
end

--- Lleva al jugador frente a la niebla de AMLO, camina hasta que el area de interaccion
--- queda enfocada y pulsa A (flujo real). Al terminar llama on_done(frames_hasta_A).
function AL.start_amlo(on_done)
    AL.tp(-1600, -1762, 11250, 180)
    AL.add("start_amlo", function(c)
        local p = AL.pl()
        if not p then return c.n > 600 end
        local it = AL.get(p, "AC_InteractionTrace")
        local fa = AL.get(it, "FocusedInteractionArea")
        if AL.valid(fa) then
            AL.btn("A17")
            AL.log("START_AMLO A pulsado en " .. AL.v3(AL.loc(p)) .. " frames=" .. c.n)
            if on_done then on_done(c.n) end
            return true
        end
        p:AddMovementInput({ X = -1, Y = 0, Z = 0 }, 1.0, false)
        return c.n > 600
    end)
end
return "bot ok"
