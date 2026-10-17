-- Hooks de registro de combate (solo lectura). Se registran una sola vez por proceso.
if not LAB.hooks_done then
  LAB.hooks_done = true
  local function H(path, fn)
    local ok, e = pcall(RegisterHook, path, function(...) local okc, err = pcall(fn, ...) if not okc then LAB.log("hook err " .. path .. " " .. tostring(err)) end end)
    LAB.log("hook " .. path .. " ok=" .. tostring(ok) .. (ok and "" or (" " .. tostring(e))))
  end
  local AM = "Amount_7_7BB9AB76446D18C11D94DA96B85DE23C"
  H("/Game/ParaMex/BPC_DamageSystem.BPC_DamageSystem_C:TakeDamage", function(ctx, causer, info)
    local comp = ctx:get()
    local owner = comp:GetOwner()
    local i = info:get()
    local amt, cbp, cbb, resp = -1, "?", "?", "?"
    pcall(function() amt = i[AM] end)
    pcall(function() cbp = tostring(i["CanBeParried_19_AFBED70341F685CD452A27A21910B6DA"]) end)
    pcall(function() cbb = tostring(i["CanBeBlocked_17_4C6668814D8CA17DBBD811BABFADD2E2"]) end)
    pcall(function() resp = tostring(i["DamageResponse_29_41EA7A94421239AA10EDC88D425DDFD8"]) end)
    LAB.log(string.format("HOOK TakeDamage a=%s causer=%s amount=%s parryable=%s blockable=%s resp=%s hp_antes=%.1f blk=%s inv=%s pw=%s",
      LAB.cls(owner), LAB.cls(causer:get()), tostring(amt), cbp, cbb, resp, comp.Health, tostring(comp.IsBlocking), tostring(comp.IsInvincible),
      tostring(LAB.get(LAB.pl(), "Is Within Parry Window"))))
  end)
  for _, f in ipairs({"Parry Window ON", "Parry Window OFF", "Parry Attack", "bLocked", "AddParryCount", "Turn To Parry", "ADD POWER", "death", "Damage Response"}) do
    H("/Game/BP_Player.BP_Player_C:" .. f, function(ctx, a, b, c) LAB.log("HOOK Player:" .. f .. " " .. tostring(a and pcall(function() return a:get() end) and tostring(a:get()) or "") .. " " .. tostring(b and b.get and tostring(b:get()) or "")) end)
  end
  H("/Game/BP_Player.BP_Player_C:Damage Numbers WIdget Holder", function(ctx, crit, dmg) LAB.log("HOOK DamageNumbers crit=" .. tostring(crit:get()) .. " dmg=" .. tostring(dmg:get())) end)
  for _, f in ipairs({"AddParryCount", "death"}) do
    H("/Game/BP_EnemyBase.BP_EnemyBase_C:" .. f, function(ctx) LAB.log("HOOK Enemy:" .. f .. " " .. LAB.name(ctx:get())) end)
  end
  for _, f in ipairs({"ENTER PHASE 2", "death", "RespawnEnemy", "Ataque PARRYS"}) do
    H("/Game/BP_Enemy_Juice.BP_Enemy_Juice_C:" .. f, function(ctx) LAB.log("HOOK Juice:" .. f .. " " .. LAB.name(ctx:get())) end)
  end
  H("/Game/BP_Enemy_JuicePhase2.BP_Enemy_JuicePhase2_C:death", function(ctx) LAB.log("HOOK JuicePhase2:death " .. LAB.name(ctx:get())) end)
  H("/Game/ThirdPerson/Blueprints/BP_MexicoGameMode.BP_MexicoGameMode_C:Respawn_Player", function(ctx) LAB.log("HOOK GameMode:Respawn_Player") end)
  H("/Script/Engine.GameplayStatics:SaveGameToSlot", function(ctx, obj, slot) LAB.log("HOOK SaveGameToSlot slot=" .. tostring(slot:get():ToString())) end)
  H("/Script/Engine.GameplayStatics:OpenLevel", function(ctx, w, name) LAB.log("HOOK OpenLevel " .. tostring(name:get():ToString())) end)
end
return "hooks"
