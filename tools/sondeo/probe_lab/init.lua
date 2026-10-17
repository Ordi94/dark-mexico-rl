-- Carga todas las utilidades de medicion en orden. Uso desde cmd.lua:
--   return dofile(os.getenv("DMPROBE_LAB") .. "/init.lua")
local D = os.getenv("DMPROBE_LAB") .. "/"
for _, f in ipairs({ "lab.lua", "walkto.lua", "acts.lua", "fight.lua", "hooks.lua", "hooks2.lua", "parry.lua", "pause.lua", "reset.lua", "dilation.lua", "heal.lua" }) do
    dofile(D .. f)
end
return "init ok"
