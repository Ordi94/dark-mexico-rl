-- Canal con el proceso Python mediante dos named pipes de Windows.
--
-- Python es el servidor: crea \\.\pipe\darkmexico_<id>_g2c (juego -> Python) y
-- \\.\pipe\darkmexico_<id>_c2g (Python -> juego). El mod los abre como cliente con io.open,
-- que en Windows termina en CreateFile y acepta rutas de pipe sin ninguna biblioteca extra.
--
-- Se usan dos pipes y no uno duplex porque el CRT de MSVC no permite alternar lectura y
-- escritura sobre el mismo FILE* sin un fseek intermedio, y fseek no es valido en un pipe:
-- cada FILE* se usa en un solo sentido. Ambos se abren "r+b" (OPEN_EXISTING) porque el modo
-- "w" pediria CREATE_ALWAYS, que no aplica a un pipe.
--
-- La lectura es bloqueante a proposito (modo lockstep): el hilo del juego espera la
-- siguiente orden. Si el proceso Python muere, Windows cierra su extremo y la lectura
-- devuelve EOF de inmediato; el canal pasa a desconectado y el juego deja de esperar.

local Channel = {}
Channel.__index = Channel

--- Crea el canal para el identificador de instancia dado (sin conectar).
function Channel.new(id)
    local base = "\\\\.\\pipe\\darkmexico_" .. tostring(id)
    return setmetatable({
        id = id,
        path_tx = base .. "_g2c",
        path_rx = base .. "_c2g",
        tx = nil,
        rx = nil,
        bytes_out = 0,
        bytes_in = 0,
    }, Channel)
end

function Channel:connected()
    return self.tx ~= nil and self.rx ~= nil
end

--- Intenta abrir ambos pipes. Falla rapido si el servidor no existe o esta ocupado.
function Channel:connect()
    if self:connected() then return true end
    local tx = io.open(self.path_tx, "r+b")
    if not tx then return false end
    local rx = io.open(self.path_rx, "r+b")
    if not rx then
        tx:close()
        return false
    end
    tx:setvbuf("full", 65536)
    self.tx, self.rx = tx, rx
    return true
end

function Channel:close()
    if self.tx then pcall(function() self.tx:close() end) end
    if self.rx then pcall(function() self.rx:close() end) end
    self.tx, self.rx = nil, nil
end

--- Envia una linea completa. Devuelve false (y cierra) si el otro extremo ya no existe.
function Channel:send(line)
    if not self.tx then return false end
    local ok = self.tx:write(line, "\n")
    if ok then ok = self.tx:flush() end
    if not ok then
        self:close()
        return false
    end
    self.bytes_out = self.bytes_out + #line + 1
    return true
end

--- Lee una linea (bloqueante). Devuelve nil si el otro extremo se cerro.
function Channel:recv()
    if not self.rx then return nil end
    local line = self.rx:read("l")
    if line == nil then
        self:close()
        return nil
    end
    self.bytes_in = self.bytes_in + #line + 1
    return line
end

return Channel
