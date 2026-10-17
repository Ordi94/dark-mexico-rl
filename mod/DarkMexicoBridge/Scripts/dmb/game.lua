-- Acceso a los objetos del juego: motor, mundo, controlador, pawn y jefe activo.
--
-- Reglas que evitan las caidas observadas:
--  * Ninguna referencia a un actor sobrevive al frame en que se obtuvo. El controlador, el pawn
--    y el jefe se vuelven a pedir al motor en cada frame (la cache dura solo ese frame). Un
--    actor destruido puede quedar liberado y su memoria reutilizada; leer una propiedad a
--    traves de una referencia vieja tumbo el proceso dentro de UE4SS con un
--    EXCEPTION_ACCESS_VIOLATION (con varias instancias reiniciando cada pocos segundos).
--  * Cada OpenLevel descarga y vuelve a cargar las clases Blueprint del juego (BP_Player_C,
--    BP_Enemy_Juice_C, BPC_DamageSystem_C... cambian de direccion en cada reinicio, medido).
--    Por eso nunca se guarda una clase Blueprint ni nada indexado por su direccion: solo
--    objetos nativos que viven todo el proceso (motor, objetos por defecto de bibliotecas y
--    clases nativas como Character).
--  * Los actores se buscan con GameplayStatics:GetAllActorsOfClass(Character), que recorre
--    solo los actores vivos del mundo actual. FindAllOf recorre el arreglo global de objetos y
--    justo despues de un OpenLevel puede devolver actores (y clases) del mundo anterior que el
--    recolector de basura aun esta destruyendo.
--  * Mientras se carga un mapa (entre LoadMapPre y LoadMapPost) y durante unos frames
--    despues, el mundo se considera inestable y no se toca ningun actor.

local Sel = require("dmb.selection")

local G = {}

-- Frames de espera tras detectar un mundo nuevo antes de leer actores.
G.SETTLE_FRAMES = 3

G.frame = 0
G.loading = false
G.settle = 0
G.world = nil
G.world_addr = nil
G.map = "?"
G.engine = nil
G.gs = nil
G.ksl = nil
G.character_class = nil
-- Contador de mundos vistos: el UWorld nuevo puede reutilizar la direccion del anterior al
-- recargar el mismo mapa, asi que la direccion sola no basta para saber que hubo recarga.
G.world_gen = 0
G.force_refresh = false
-- Si no es nil, valor de VisibilityBasedAnimTickOption que se fuerza en las mallas del jugador
-- y del jefe (0 = AlwaysTickPoseAndRefreshBones). Sin renderizado (-nullrhi) ninguna malla se
-- considera visible; con el valor del juego (1 = AlwaysTickPose) los golpes medidos fueron los
-- mismos que con ventana, asi que es una opcion de seguridad y no viene activa.
G.anim_tick_option = nil
-- Dilatacion de tiempo pedida por el cliente (nil = la del nivel).
G.dilation = nil
-- Clases admitidas como jefe (lista de nombres). Con nil se usa la bandera IS BOSS? como en la
-- primera version del protocolo; con una lista el jefe se elige solo por clase (ver selection).
G.boss_classes = nil

-- Cache valida solo durante el frame G.frame.
local cache = { frame = -1 }
local forced_meshes = {}

local function valid(o)
    if o == nil then return false end
    local ok, r = pcall(function() return o:IsValid() end)
    return ok and r == true
end
G.valid = valid

--- Objetos nativos que viven todo el proceso: objetos por defecto de bibliotecas y la clase
--- Character (padre nativo del jugador y de los enemigos).
function G.init_statics()
    G.gs = StaticFindObject("/Script/Engine.Default__GameplayStatics")
    G.ksl = StaticFindObject("/Script/Engine.Default__KismetSystemLibrary")
    G.character_class = StaticFindObject("/Script/Engine.Character")
end

local function engine()
    if not valid(G.engine) then
        G.engine = FindFirstOf("GameEngine")
        if not valid(G.engine) then G.engine = FindFirstOf("Engine") end
    end
    return G.engine
end

local function frame_cache()
    if cache.frame ~= G.frame then cache = { frame = G.frame } end
    return cache
end

--- Descarta todo lo asociado al mundo anterior.
function G.invalidate()
    cache = { frame = -1 }
    forced_meshes = {}
end

--- Llamado al empezar una carga de mapa: desde aqui no se toca ningun actor.
function G.on_load_map_pre()
    G.loading = true
    G.invalidate()
end

function G.on_load_map_post()
    G.loading = false
    G.force_refresh = true
end

--- Se llama una vez por frame: detecta el cambio de mundo comparando la direccion del UWorld
--- y los avisos de LoadMap (al recargar el mismo mapa la direccion puede repetirse).
function G.track()
    G.frame = G.frame + 1
    local w = nil
    local ok = pcall(function()
        local e = engine()
        if e and e.GameViewport and valid(e.GameViewport) then
            w = e.GameViewport.World
        end
    end)
    local addr = nil
    if ok and valid(w) then addr = w:GetAddress() end
    if addr ~= G.world_addr or G.force_refresh then
        G.force_refresh = false
        G.world_gen = G.world_gen + 1
        G.world_addr = addr
        G.world = addr and w or nil
        G.invalidate()
        G.settle = G.SETTLE_FRAMES
        G.map = "?"
        if G.world then
            pcall(function() G.map = G.world:GetFName():ToString() end)
        end
        return true
    end
    G.world = addr and w or nil
    if G.settle > 0 then G.settle = G.settle - 1 end
    return false
end

--- true cuando es seguro leer y modificar actores en este frame.
function G.stable()
    return (not G.loading) and G.settle == 0 and G.world ~= nil
end

--- Nombre de la clase de un objeto. Sin cache: las clases Blueprint se recargan en cada
--- OpenLevel y una direccion vieja puede quedar ocupada por otra clase.
function G.class_name(o)
    return o:GetClass():GetFName():ToString()
end

function G.is_player_pawn(p)
    local ok, name = pcall(G.class_name, p)
    return ok and name == "BP_Player_C"
end

--- Aplica G.anim_tick_option a la malla de un personaje (una vez por malla y mundo).
function G.apply_anim_option(ch)
    if G.anim_tick_option == nil or ch == nil then return end
    pcall(function()
        local mesh = ch.Mesh
        local addr = mesh:GetAddress()
        if forced_meshes[addr] == G.anim_tick_option then return end
        mesh.VisibilityBasedAnimTickOption = G.anim_tick_option
        forced_meshes[addr] = G.anim_tick_option
    end)
end

function G.player_controller()
    if not G.stable() then return nil end
    local c = frame_cache()
    if c.pc == nil then
        local pc = nil
        pcall(function() pc = G.gs:GetPlayerController(G.world, 0) end)
        c.pc = valid(pc) and pc or false
    end
    return c.pc or nil
end

--- Pawn poseido por el jugador, solo si es BP_Player_C (en el menu principal es otra clase).
function G.pawn()
    local c = frame_cache()
    if c.pawn == nil then
        c.pawn = false
        local pc = G.player_controller()
        if pc then
            local p = nil
            pcall(function() p = pc.Pawn end)
            if valid(p) and G.is_player_pawn(p) then
                G.apply_anim_option(p)
                c.pawn = p
            end
        end
    end
    return c.pawn or nil
end

local function has_controller(e)
    local ok, ctrl = pcall(function() return e.Controller end)
    return ok and valid(ctrl)
end

local function is_boss(e)
    local ok, b = pcall(function() return e["IS BOSS?"] end)
    return ok and b == true
end

--- Enemigos vivos del mundo actual: personajes (Character) con la propiedad IS BOSS?
--- definida, es decir BP_EnemyBase_C y sus subclases.
function G.enemies()
    local out = {}
    if not G.stable() then return out end
    local c = frame_cache()
    if c.enemies then return c.enemies end
    local arr = {}
    local ok = pcall(function() G.gs:GetAllActorsOfClass(G.world, G.character_class, arr) end)
    if ok then
        for _, item in ipairs(arr) do
            local e = item:get()
            if valid(e) then
                local okb, b = pcall(function() return e["IS BOSS?"] end)
                if okb and b ~= nil then out[#out + 1] = e end
            end
        end
    end
    c.enemies = out
    return out
end

--- Fija las clases de jefe admitidas (lista de nombres) o nil para volver a la bandera IS BOSS?.
function G.set_boss_classes(classes)
    G.boss_classes = classes
    cache = { frame = -1 }
end

local function is_dead(e)
    local ok, d = pcall(function() return e["DEATH?"] end)
    return ok and d == true
end

--- Jefe activo. Sin lista de clases: BP_EnemyBase_C con IS BOSS? y controlado por su IA (primera
--- version del protocolo). Con lista: el personaje de una de esas clases elegido por
--- dmb.selection, sin mirar IS BOSS?. En ambos casos, tras la muerte del jugador o el paso a la
--- fase 2 queda otro actor jefe sin controlador hasta que se destruye; si no hay jefe
--- controlado (transicion de fase) se devuelve el que haya.
function G.boss_actor()
    local c = frame_cache()
    if c.boss == nil then
        c.boss = false
        if G.boss_classes then
            local list, cands = G.enemies(), {}
            for i, e in ipairs(list) do
                local okc, cls = pcall(G.class_name, e)
                cands[i] = { cls = okc and cls or "?", controlled = has_controller(e), dead = is_dead(e) }
            end
            local idx = Sel.pick(cands, G.boss_classes)
            if idx then c.boss = list[idx] end
        else
            local fallback = nil
            for _, e in ipairs(G.enemies()) do
                if is_boss(e) then
                    if has_controller(e) then
                        c.boss = e
                        break
                    end
                    fallback = fallback or e
                end
            end
            if not c.boss and fallback then c.boss = fallback end
        end
        if c.boss then G.apply_anim_option(c.boss) end
    end
    return c.boss or nil
end

--- Todos los personajes del mundo cuya clase esta en la lista (nombres).
function G.actors_of(classes)
    local set, out = Sel.class_set(classes), {}
    for _, e in ipairs(G.enemies()) do
        local ok, cls = pcall(G.class_name, e)
        if ok and set[cls] then out[#out + 1] = e end
    end
    return out
end

function G.time_seconds()
    local ok, t = pcall(function() return G.gs:GetTimeSeconds(G.world) end)
    return ok and t or -1.0
end

function G.delta_seconds()
    local ok, t = pcall(function() return G.gs:GetWorldDeltaSeconds(G.world) end)
    return ok and t or -1.0
end

function G.apply_dilation()
    if G.dilation == nil or G.world == nil then return end
    pcall(function() G.gs:SetGlobalTimeDilation(G.world, G.dilation) end)
end

-- Frames entre dos comprobaciones de la dilatacion de tiempo.
G.DILATION_CHECK_FRAMES = 30

--- Vuelve a aplicar la dilatacion pedida si algo la cambio. La secuencia de muerte de algunos
--- jefes la deja en 1.0 (medido con el Chamuco) y el entrenamiento depende de ella.
function G.keep_dilation()
    if G.dilation == nil or G.frame % G.DILATION_CHECK_FRAMES ~= 0 then return end
    pcall(function()
        local now = G.gs:GetGlobalTimeDilation(G.world)
        if math.abs(now - G.dilation) > 1e-3 then G.gs:SetGlobalTimeDilation(G.world, G.dilation) end
    end)
end

--- Abre un mapa. La carga ocurre en el siguiente tick del motor.
function G.open_level(name)
    G.gs:OpenLevel(G.world, FName(name), true, "")
end

--- Ejecuta un comando de consola del motor en el mundo actual.
function G.console(cmd)
    G.ksl:ExecuteConsoleCommand(G.world, cmd, G.player_controller())
end

--- Cierra el juego de forma ordenada.
function G.quit()
    local ok = pcall(function() G.ksl:QuitGame(G.world, G.player_controller(), 0, false) end)
    if not ok then pcall(function() G.console("quit") end) end
end

return G
