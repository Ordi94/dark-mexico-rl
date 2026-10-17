-- Eleccion del jefe activo entre los personajes del mundo.
--
-- Modulo puro (sin API de Unreal) para poder comprobarlo fuera del juego. El jefe se elige por
-- su CLASE, nunca por la bandera "IS BOSS?": en LVL_AMLO los cinco NPC amistosos tienen esa
-- bandera en true y controlador propio, asi que una seleccion por bandera podria tomar a un
-- aliado invulnerable (999999 de vida) como si fuera el rival.

local M = {}

--- Conjunto {clase = true} a partir de una lista de nombres.
function M.class_set(classes)
    local set = {}
    for _, name in ipairs(classes or {}) do set[name] = true end
    return set
end

--- Indice del candidato que es el jefe activo, o nil.
--- candidates: lista de {cls = nombre de clase, controlled = bool, dead = bool}.
--- Prioridad: de la lista de clases, con controlador y vivo; si no hay, de la lista y con
--- controlador aunque este muerto (la observacion de la muerte); si no, cualquiera de la lista
--- sin controlador (transicion de fase: el actor viejo ya no tiene IA y el nuevo todavia no).
function M.pick(candidates, classes)
    local set = M.class_set(classes)
    local alive, controlled, any = nil, nil, nil
    for i, c in ipairs(candidates) do
        if set[c.cls] then
            if c.controlled and not c.dead and not alive then alive = i end
            if c.controlled and not controlled then controlled = i end
            if not any then any = i end
        end
    end
    return alive or controlled or any
end

return M
