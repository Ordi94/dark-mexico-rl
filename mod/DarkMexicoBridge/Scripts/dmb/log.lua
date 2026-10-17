-- Registro propio de cada instancia de entrenamiento.
--
-- UE4SS.log lo comparten todas las instancias que arrancan desde la misma carpeta del juego
-- (sus lineas se intercalan), asi que cada instancia escribe ademas en su propio archivo
-- bridge_<id>.log dentro de DMB_LOG_DIR. Se abre en modo append y se vacia en cada linea para
-- que el contenido sobreviva a un cierre abrupto del proceso.

local M = {}

local file = nil
local prefix = "[Bridge] "

--- Prepara el archivo de registro. dir puede ser nil: entonces solo se usa UE4SS.log.
function M.init(dir, id)
    prefix = "[Bridge " .. tostring(id) .. "] "
    if dir and dir ~= "" then
        local sep = dir:sub(-1)
        if sep ~= "\\" and sep ~= "/" then dir = dir .. "\\" end
        file = io.open(dir .. "bridge_" .. tostring(id) .. ".log", "a")
        if file then file:setvbuf("line") end
    end
end

--- Escribe una linea con marca de tiempo de pared (segundos desde que arranco el proceso).
function M.info(msg)
    local line = string.format("%s t=%.3f %s", os.date("%H:%M:%S"), os.clock(), tostring(msg))
    if file then
        file:write(line, "\n")
        file:flush()
    end
    print(prefix .. line .. "\n")
end

return M
