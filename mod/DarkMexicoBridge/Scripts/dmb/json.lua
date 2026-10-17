-- Codificador y decodificador JSON minimo para el protocolo del puente.
--
-- El Lua de UE4SS no trae biblioteca JSON y el protocolo solo necesita objetos planos con
-- numeros, cadenas, booleanos, listas y objetos anidados poco profundos. Se implementa aqui
-- para no depender de modulos externos que habria que distribuir junto al juego.

local M = {}

local fmt, concat, byte = string.format, table.concat, string.byte
local mtype = math.type

--- Marca para codificar una tabla vacia como lista ([]) en lugar de objeto ({}).
M.EMPTY_ARRAY = setmetatable({}, { __name = "json.empty_array" })

local ESCAPES = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function encode_string(s)
    local escaped = s:gsub('[%c"\\]', function(c)
        return ESCAPES[c] or fmt("\\u%04x", byte(c))
    end)
    return '"' .. escaped .. '"'
end

local function encode_number(n)
    if mtype(n) == "integer" then return fmt("%d", n) end
    -- NaN e infinito no existen en JSON: se envian como null para no romper al lector.
    if n ~= n or n == math.huge or n == -math.huge then return "null" end
    return fmt("%.9g", n)
end

local encode_value

local function is_array(t)
    -- Una tabla vacia se codifica como objeto; para una lista vacia usar M.EMPTY_ARRAY.
    return t == M.EMPTY_ARRAY or t[1] ~= nil
end

local function encode_table(t, out)
    if is_array(t) then
        out[#out + 1] = "["
        for i = 1, #t do
            if i > 1 then out[#out + 1] = "," end
            encode_value(t[i], out)
        end
        out[#out + 1] = "]"
        return
    end
    out[#out + 1] = "{"
    local first = true
    for k, v in pairs(t) do
        if not first then out[#out + 1] = "," end
        first = false
        out[#out + 1] = encode_string(tostring(k))
        out[#out + 1] = ":"
        encode_value(v, out)
    end
    out[#out + 1] = "}"
end

encode_value = function(v, out)
    local tv = type(v)
    if tv == "nil" then
        out[#out + 1] = "null"
    elseif tv == "boolean" then
        out[#out + 1] = v and "true" or "false"
    elseif tv == "number" then
        out[#out + 1] = encode_number(v)
    elseif tv == "string" then
        out[#out + 1] = encode_string(v)
    elseif tv == "table" then
        encode_table(v, out)
    else
        out[#out + 1] = encode_string(tostring(v))
    end
end

--- Codifica un valor Lua a una cadena JSON de una sola linea.
function M.encode(v)
    local out = {}
    encode_value(v, out)
    return concat(out)
end

-- Decodificador recursivo descendente. Rechaza entradas mal formadas con error() para que el
-- llamador responda con un mensaje de error en vez de ejecutar un comando a medias.

local decode_value

local function skip_ws(s, i)
    local _, e = s:find("^[ \t\r\n]*", i)
    return e + 1
end

local function decode_string(s, i)
    local out = {}
    local j = i + 1
    while true do
        local c = s:sub(j, j)
        if c == "" then error("cadena sin cerrar en la posicion " .. i, 0) end
        if c == '"' then return concat(out), j + 1 end
        if c == "\\" then
            local e = s:sub(j + 1, j + 1)
            local map = { ['"'] = '"', ['\\'] = '\\', ['/'] = '/', b = '\b', f = '\f', n = '\n', r = '\r', t = '\t' }
            if map[e] then
                out[#out + 1] = map[e]
                j = j + 2
            elseif e == "u" then
                local hex = s:sub(j + 2, j + 5)
                local code = tonumber(hex, 16)
                if not code then error("escape unicode invalido en la posicion " .. j, 0) end
                out[#out + 1] = utf8.char(code)
                j = j + 6
            else
                error("escape invalido en la posicion " .. j, 0)
            end
        else
            out[#out + 1] = c
            j = j + 1
        end
    end
end

local function decode_number(s, i)
    local num = s:match("^-?%d+%.?%d*[eE]?[-+]?%d*", i)
    if not num or num == "" then error("numero invalido en la posicion " .. i, 0) end
    local v = tonumber(num)
    if v == nil then error("numero invalido en la posicion " .. i, 0) end
    return v, i + #num
end

local function decode_array(s, i)
    local arr = {}
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "]" then return arr, i + 1 end
    while true do
        local v
        v, i = decode_value(s, i)
        arr[#arr + 1] = v
        i = skip_ws(s, i)
        local c = s:sub(i, i)
        if c == "]" then return arr, i + 1 end
        if c ~= "," then error("se esperaba ',' o ']' en la posicion " .. i, 0) end
        i = skip_ws(s, i + 1)
    end
end

local function decode_object(s, i)
    local obj = {}
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == "}" then return obj, i + 1 end
    while true do
        if s:sub(i, i) ~= '"' then error("se esperaba una clave en la posicion " .. i, 0) end
        local k
        k, i = decode_string(s, i)
        i = skip_ws(s, i)
        if s:sub(i, i) ~= ":" then error("se esperaba ':' en la posicion " .. i, 0) end
        local v
        v, i = decode_value(s, skip_ws(s, i + 1))
        obj[k] = v
        i = skip_ws(s, i)
        local c = s:sub(i, i)
        if c == "}" then return obj, i + 1 end
        if c ~= "," then error("se esperaba ',' o '}' en la posicion " .. i, 0) end
        i = skip_ws(s, i + 1)
    end
end

decode_value = function(s, i)
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == "{" then return decode_object(s, i) end
    if c == "[" then return decode_array(s, i) end
    if c == '"' then return decode_string(s, i) end
    if s:sub(i, i + 3) == "true" then return true, i + 4 end
    if s:sub(i, i + 4) == "false" then return false, i + 5 end
    if s:sub(i, i + 3) == "null" then return nil, i + 4 end
    return decode_number(s, i)
end

--- Decodifica una cadena JSON. Devuelve valor o nil, mensaje de error.
function M.decode(s)
    local ok, v, i = pcall(decode_value, s, 1)
    if not ok then return nil, v end
    i = skip_ws(s, i)
    if i <= #s then return nil, "texto sobrante en la posicion " .. i end
    return v
end

return M
