-- Dificultad ajustable del jefe para el aprendizaje progresivo.
--
-- Escala la vida del jefe (Health y MaxHealth a la vez, para que la razon vida/maximo que
-- usan los decoradores del arbol de comportamiento no cambie: la fase 2 se activa por debajo
-- del 15% de la vida maxima, medido con vida maxima 600). Nunca se toca la vida del jugador.
--
-- El juego genera un actor nuevo para la fase 2 con su vida completa (1000), asi que la
-- escala se reaplica a cada actor jefe distinto que aparezca. Cada actor se escala una sola
-- vez: la marca es su nombre completo, que incluye un sufijo numerico unico por instancia
-- (las direcciones se reutilizan entre reinicios y no sirven de clave).

local G = require("dmb.game")

local C = {}

C.MIN_MULT = 0.05
C.MAX_MULT = 1.0
-- Fraccion de la vida maxima a la que se deja al jefe para forzar la transicion a fase 2.
C.PHASE2_TRIGGER_FRACTION = 0.1

C.mult = 1.0
local scaled_name = nil

--- Fija el multiplicador de vida para los jefes que aparezcan desde ahora.
--- Devuelve true si el valor es valido.
function C.set_multiplier(m)
    if type(m) ~= "number" or m ~= m or m < C.MIN_MULT or m > C.MAX_MULT then return false end
    C.mult = m
    scaled_name = nil
    return true
end

--- Escala la vida del jefe actual si todavia no se hizo. Se compara el nombre completo (unico
--- por instancia) en cada frame: una direccion liberada puede ocupar el actor de fase 2.
function C.tick()
    if C.mult == 1.0 then return end
    local b = G.boss_actor()
    if not b then return end
    local ok, name = pcall(function() return b:GetFullName() end)
    if not ok or name == scaled_name then return end
    scaled_name = name
    pcall(function()
        local d = b.BPC_DamageSystem
        d.MaxHealth = d.MaxHealth * C.mult
        d.Health = d.Health * C.mult
    end)
end

--- Deja al jefe por debajo del umbral de la fase 2 (con la misma razon que usa el juego).
function C.trigger_phase2(b)
    local d = b.BPC_DamageSystem
    d.Health = d.MaxHealth * C.PHASE2_TRIGGER_FRACTION
end

--- Marca de invulnerabilidad del jugador durante la preparacion de un episodio en fase 2.
--- No modifica la vida: solo evita que el jefe de fase 1 lo golpee mientras se transforma.
function C.set_player_invincible(p, on)
    pcall(function() p.BPC_DamageSystem.IsInvincible = on end)
end

return C
