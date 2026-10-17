function LAB.pausestate()
  local w = LAB.all("WP_PauseGame_C")
  local inv = {}
  for _, x in ipairs(w) do inv[#inv+1] = LAB.name(x) .. (x:IsInViewport() and "[vp]" or "") end
  local pc = LAB.pc()
  return string.format("pauseW={%s} paused=%s dil=%.2f cursor=%s tg=%.2f", table.concat(inv, ","), tostring(LAB.gs():IsGamePaused(LAB.world())), LAB.gs():GetGlobalTimeDilation(LAB.world()), tostring(pc.bShowMouseCursor), LAB.tgame())
end
-- Cierra cualquier menu de pausa abierto llamando al manejador de su boton "Reanudar".
function LAB.unpause()
  local n = 0
  for _, x in ipairs(LAB.all("WP_PauseGame_C")) do
    if x:IsInViewport() then
      x["BndEvt__WP_MainMenu_BT_StartGame_K2Node_ComponentBoundEvent_0_OnButtonClickedEvent__DelegateSignature"](x)
      n = n + 1
    end
  end
  return n
end
return "pause ok"
