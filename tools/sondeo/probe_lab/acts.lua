-- Prueba de acciones: OpenLevel limpio (sin activar al jefe), luego cada accion con medicion.
function LAB.pstate()
  local p = LAB.pl()
  local m, pos = LAB.montage(p)
  local cm = p.CharacterMovement
  return string.format("loc=%s yaw=%.0f hp=%.1f res=%.3f cag=%s pow=%s mont=%s@%.2f spd=%.0f maxw=%.0f blk=%s pw=%s roll=%s canroll=%s atk=%s combo=%s canatk=%s lock=%s canheal=%s sprint=%s",
    LAB.v3(LAB.loc(p)), p:K2_GetActorRotation().Yaw, LAB.hp(p), p.Resistencia, tostring(p.Caguamas), tostring(p.Power), m, pos,
    (function() local v=p:GetVelocity(); return math.sqrt(v.X*v.X+v.Y*v.Y) end)(), cm.MaxWalkSpeed,
    tostring(p.BPC_DamageSystem.IsBlocking), tostring(p["Is Within Parry Window"]), tostring(p.IsRolling), tostring(p.CanRoll),
    tostring(p["Is Attacking"]), tostring(p["Combo Counter"]), tostring(p.CanAttack), tostring(p["TargetLock?"]), tostring(p["Can Heal"]), tostring(p["Can Sprint"]))
end
-- Secuencia: lista de {nombre, funcion de accion, frames a observar, cada}
function LAB.seq(steps)
  local i = 0
  local function nextstep()
    i = i + 1
    local s = steps[i]
    if not s then LAB.log("SEQ fin"); return end
    LAB.add("paso " .. s[1], function(c)
      if c.n == 1 then
        LAB.log("== " .. s[1] .. " ANTES " .. LAB.pstate())
        local ok, e = pcall(s[2])
        if not ok then LAB.log("   error accion: " .. tostring(e)) end
      elseif c.n % (s[4] or 6) == 0 then
        LAB.log(string.format("   %s +%d %s", s[1], c.n, LAB.pstate()))
      end
      if c.n >= (s[3] or 90) then nextstep(); return true end
    end)
  end
  nextstep()
end
return "acts ok"
