-- Arranque deterministico de la pelea: teleport frente a la niebla + boton A (flujo real).
function LAB.start_fight(on_ready)
  LAB.open("LVL_Boss")
  LAB.add("start_fight", function(c)
    local p = LAB.pl()
    if c.n == 5 and p then
      p:K2_SetActorLocation({X=47, Y=-2260, Z=380}, false, {}, true)
    end
    if c.n == 12 then LAB.btn("A17"); LAB.T_A = LAB.tgame(); LAB.log("A pulsado en la niebla") end
    if c.n > 12 then
      local bm = LAB.montage(LAB.boss())
      if bm ~= "none" and bm ~= "nil" then
        LAB.log(string.format("jefe activo: %s a %.2f s del A", bm, LAB.tgame() - LAB.T_A))
        if on_ready then on_ready() end
        return true
      end
    end
    return c.n > 600
  end)
end
-- Pelea automatica simple: acercarse al jefe y atacar; registra danos. heal_below mantiene vivo al jugador.
function LAB.autofight(frames, opts)
  opts = opts or {}
  LAB.add("autofight", function(c)
    local p, b = LAB.pl(), LAB.boss()
    if not p or not b then return c.n > frames end
    local d = LAB.dist2d(p, b)
    if opts.heal_below and LAB.hp(p) < opts.heal_below and LAB.hp(p) > 0 then
      LAB.log("autofight: reponiendo vida del jugador " .. LAB.hp(p) .. " -> 100 (solo para seguir midiendo)")
      p.BPC_DamageSystem.Health = 100.0
    end
    if d > (opts.range or 170) then
      local lp, lb = LAB.loc(p), LAB.loc(b)
      local dx, dy = lb.X - lp.X, lb.Y - lp.Y
      p:AddMovementInput({X = dx / d, Y = dy / d, Z = 0}, 1.0, false)
    else
      if opts.lock and not p["TargetLock?"] and c.n % 30 == 0 then LAB.btn("RS") end
      if c.n % (opts.every or 12) == 0 then LAB.btn(opts.button or "RB") end
    end
    return c.n > frames
  end)
end
return "fight ok"
