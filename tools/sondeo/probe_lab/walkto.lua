-- LAB.walkto(x, y, tol, label, on_done): camina con AddMovementInput hacia (x,y) en el plano.
function LAB.walkto(tx, ty, tol, label, on_done, maxf)
  maxf = maxf or 1800
  LAB.add("walkto " .. label, function(c)
    local p = LAB.pl()
    if not p then return true end
    local l = LAB.loc(p)
    local dx, dy = tx - l.X, ty - l.Y
    local d = math.sqrt(dx*dx + dy*dy)
    if c.n % (LAB.walk_every or 20) == 1 then
      local it = p.AC_InteractionTrace
      LAB.log(string.format("%s d=%.0f %s focusArea=%s tracing=%s prog=%s", label, d, LAB.snap(), LAB.name(it.FocusedInteractionArea), tostring(it.IsTracingInteractionArea), tostring(it.InteractionProgress)))
    end
    if d < tol or c.n > maxf then
      LAB.log(string.format("%s LLEGO d=%.0f loc=%s frames=%d", label, d, LAB.v3(l), c.n))
      if on_done then on_done() end
      return true
    end
    p:AddMovementInput({X = dx / d, Y = dy / d, Z = 0}, 1.0, false)
  end)
end
return "walkto ok"
