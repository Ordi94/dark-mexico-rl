-- Despachador: un hook extra que delega en LAB.on_td (redefinible sin volver a registrar).
if not LAB.hook_td2 then
  LAB.hook_td2 = true
  RegisterHook("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:TakeDamage", function(ctx, causer, info)
    if LAB.on_td then pcall(LAB.on_td, ctx:get(), causer:get(), info:get()) end
  end)
  RegisterHook("/Game/BP_Player.BP_Player_C:Parry Window ON", function(ctx) if LAB.on_pw then pcall(LAB.on_pw, true) end end)
  RegisterHook("/Game/BP_Player.BP_Player_C:Parry Window OFF", function(ctx) if LAB.on_pw then pcall(LAB.on_pw, false) end end)
end
LAB.hits = LAB.hits or {}
function LAB.on_td(comp, causer, info)
  local owner = comp:GetOwner()
  if LAB.cls(owner) == "BP_Player_C" then
    local bm, bp = LAB.montage(LAB.boss())
    local amt = info["Amount_7_7BB9AB76446D18C11D94DA96B85DE23C"]
    LAB.hits[#LAB.hits + 1] = { m = bm, p = bp, a = amt }
    LAB.log(string.format("GOLPE AL JUGADOR montage=%s pos=%.3f amount=%.1f blk=%s", bm, bp, amt, tostring(comp.IsBlocking)))
  end
end
function LAB.on_pw(on)
  local p = LAB.pl()
  local m, pos = LAB.montage(p)
  LAB.log(string.format("PARRY WINDOW %s tg=%.3f mont=%s@%.3f", on and "ON" or "OFF", LAB.tgame(), m, pos))
end
return "hooks2"
