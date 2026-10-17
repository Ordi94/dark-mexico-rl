-- Prueba de time dilation: para cada factor arranca una pelea limpia, aplica el factor y corre
-- el bot de parry con ataques durante el mismo tiempo de juego; las metricas se cuentan en el log.
function LAB.dil_test(factors, game_seconds)
  local i = 0
  local function nxt()
    i = i + 1
    local k = factors[i]
    if not k then LAB.dilation(1.0); LAB.log("DIL fin"); return end
    LAB.start_fight(function()
      LAB.dilation(k)
      LAB.log(string.format("DIL %s INICIO tg=%.2f", tostring(k), LAB.tgame()))
      local frames_est = 0
      local t0 = LAB.tgame()
      LAB.parrybot(1000000, 0.235, {heal_below = 60, attack = true})
      LAB.add("dil vigilante", function(c)
        if LAB.tgame() - t0 >= game_seconds then
          LAB.log(string.format("DIL %s FIN frames=%d tg=%.2f", tostring(k), c.n, LAB.tgame()))
          LAB.clear()
          LAB.dilation(1.0)
          nxt()
          return true
        end
      end)
    end)
  end
  nxt()
end
return "dil ok"
