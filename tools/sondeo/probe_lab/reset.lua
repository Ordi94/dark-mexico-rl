-- Prueba de reinicio: repite OpenLevel(LVL_Boss) varias veces, ensuciando el estado antes de cada
-- reinicio (dano, caguama, muerte, fase 2, victoria) y verificando que el estado inicial vuelve
-- a ser identico. Las duraciones en tiempo real se leen de las marcas de UE4SS.log.
function LAB.clean_state()
  local p, b = LAB.pl(), LAB.boss()
  local fog = LAB.first("BP_FogBoss_C")
  local nb = #LAB.all("BP_EnemyBase_C")
  local pm = LAB.montage(p)
  return string.format("P[%s hp=%.0f/%.0f cag=%s res=%.2f pow=%s dead=%s mont=%s maxw=%.0f lock=%s] B[%s %s hp=%.0f/%.0f ph=%s dead=%s parry=%s] enemigos=%d fog=%s gi_juice=%s",
    LAB.v3(LAB.loc(p)), LAB.hp(p), p.BPC_DamageSystem.MaxHealth, tostring(p.Caguamas), p.Resistencia, tostring(p.Power), tostring(p.BPC_DamageSystem.IsDead), pm, p.CharacterMovement.MaxWalkSpeed, tostring(p["TargetLock?"]),
    LAB.cls(b), LAB.v3(LAB.loc(b)), LAB.hp(b), b.BPC_DamageSystem.MaxHealth, tostring(b.Phase), tostring(b["DEATH?"]), tostring(b["Parry Count"]), nb, LAB.name(fog),
    tostring(P.UEHelpers.GetGameInstance()["New Param Juice Defeated?"]))
end

-- dirty: funcion que ensucia el estado antes del reinicio siguiente (puede ser nil).
function LAB.reset_reps(dirties)
  local i = 0
  local function rep()
    i = i + 1
    if i > #dirties then LAB.log("RESET fin de repeticiones"); return end
    LAB.log("RESET rep " .. i .. " OpenLevel")
    LAB.open("LVL_Boss")
    LAB.add("rep" .. i, function(c)
      local p, b = LAB.pl(), LAB.boss()
      if c.n == 1 then return false end
      if not c.data.ready then
        if p and b then
          c.data.ready = true
          c.data.nready = c.n
          LAB.log("RESET rep " .. i .. " LISTO frames=" .. c.n .. " tg=" .. string.format("%.3f", LAB.tgame()) .. " " .. LAB.clean_state())
        end
        return c.n > 600
      end
      if c.n == c.data.nready + 3 then
        p:K2_SetActorLocation({X=47, Y=-2260, Z=380}, false, {}, true)
      elseif c.n == c.data.nready + 8 then
        LAB.btn("A17"); c.data.ta = LAB.tgame()
        LAB.log("RESET rep " .. i .. " A pulsado")
      elseif c.data.ta and not c.data.attack then
        local bm = LAB.montage(b)
        if bm ~= "none" and bm ~= "nil" then
          c.data.attack = true
          LAB.log(string.format("RESET rep %d PRIMER MONTAGE DEL JEFE %s a %.2f s (juego) del A", i, bm, LAB.tgame() - c.data.ta))
          local d = dirties[i]
          if d and d[2] then
            LAB.log("RESET rep " .. i .. " ensuciando: " .. d[1])
            d[2](function() rep() end)
          else
            rep()
          end
          return true
        end
      end
      return c.n > 1200
    end)
  end
  rep()
end
return "reset ok"
