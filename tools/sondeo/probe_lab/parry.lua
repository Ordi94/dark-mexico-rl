-- Bot de parry: lee el montage del jefe cada frame y pulsa LB 'lead' segundos antes del impacto.
LAB.PARRY_HITS = {
  Boss_Attack_Uppercut_InP_Montage = {0.895},
  Boss_Attack_SwingAndSlam_InP_Montage = {1.844},
  Boss_Attack_ChargedCombo_InP_Montage = {0.713, 1.19, 1.87},
  Boss_Attack_ChargedCombo_InP_Montage1 = {0.713, 1.18, 1.90},
}
function LAB.parrybot(frames, lead, opts)
  opts = opts or {}
  local st = { pm = "", pp = 0, release_at = -1 }
  LAB.add("parrybot", function(c)
    local p, b = LAB.pl(), LAB.boss()
    if not p or not b then return c.n > frames end
    if LAB.hp(p) < (opts.heal_below or 45) and LAB.hp(p) > 0 then p.BPC_DamageSystem.Health = 100.0 end
    local d = LAB.dist2d(p, b)
    local bm, bp = LAB.montage(b)
    local hits = LAB.PARRY_HITS[bm]
    local rate = 1.0
    if hits then pcall(function() local ai = b.Mesh:GetAnimInstance(); rate = ai:Montage_GetPlayRate(ai:GetCurrentActiveMontage()) end) end
    if hits and bm == st.pm then
      for _, t in ipairs(hits) do
        local trig = t - lead * rate
        if st.pp < trig and bp >= trig then
          LAB.btn("LB_DOWN")
          st.release_at = c.n + (opts.hold or 25)
          LAB.parry_tries = (LAB.parry_tries or 0) + 1
          LAB.log(string.format("PARRYBOT LB en %s pos=%.3f rate=%.2f (impacto esperado %.3f)", bm, bp, rate, t))
        end
      end
    end
    if st.release_at == c.n then LAB.btn("LB_UP") end
    st.pm, st.pp = bm, bp
    if d > 230 and not hits then
      local lp, lb = LAB.loc(p), LAB.loc(b)
      p:AddMovementInput({X = (lb.X - lp.X) / d, Y = (lb.Y - lp.Y) / d, Z = 0}, 1.0, false)
    end
    if opts.attack and d < 200 and not LAB.get(b, "Attacking") and c.n % 30 == 0 then LAB.btn("RB") end
    return c.n > frames
  end)
end
return "parry ok"
