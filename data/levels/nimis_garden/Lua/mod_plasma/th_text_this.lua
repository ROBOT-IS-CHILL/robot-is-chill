this_mod_globals = {}
local function reset_this_mod_globals()
    this_mod_globals = {
        active_this_property_text = {}, -- keep track of texts 
        undoed_after_called = false, -- flag for providing a specific hook of when we call code() after an undo
    }
end   
reset_this_mod_globals()

local utils = PlasmaModules.load_module("general/utils")
local UndoAnalyzer = PlasmaModules.load_module("general/undo_analyzer") 
local RaycastTrace = PlasmaModules.load_module("this/pnoun_raycast_trace")
local RaycastBank = PlasmaModules.load_module("this/raycast_bank")
local Pnoun = PlasmaModules.load_module("this/pnoun_group_defs")

local raycast_trace_tracker = RaycastTrace:new()
local raycast_analyzer = UndoAnalyzer.analyzers.raycast_analyzer

local blocked_tiles = {} -- all positions where "X is block" is active
local explicit_passed_tiles = {} -- all positions pointed by a "this is pass" rule. Used for cursor display 
local explicit_relayed_tiles = {} -- all positions pointed by a "this is relay" rule. Used for cursor display 
local on_level_start = false
local THIS_LOGGING = false

local indicator_layer_timer = 0 -- Used mainly for cycling through indicators if they are stacked
local TIMER_PERIOD = 180
local TIMER_CYCLE_PERIOD = TIMER_PERIOD/2

local checking_updatecode_status_flag = false
local checking_updatecode_curr_pnoun_ref = {}

local run_extract_bpr_subrules = false
local extracting_bpr_subrules = false

local function set_blocked_tile(tileid)
    if tileid then
        blocked_tiles[tileid] = true
    end
end
local function set_relay_tile(tileid)
    if tileid then
        explicit_relayed_tiles[tileid] = true
    end
end
local function set_passed_tile(tileid)
    if tileid then
        explicit_passed_tiles[tileid] = true
    end
end

local Pnoun_Op_To_Explicit_Tile_Func = {
    block = set_blocked_tile,
    relay = set_relay_tile,
    pass = set_passed_tile,
}

--[[ 
    local registered_pnoun_rules = {
        <Pnoun_Group> = {
            pnoun_features = [<feature>, <feature>],
            pnoun_units = (<pnoun unitid>, <pnoun unitid>),
        }
    }
]]
local registered_pnoun_rules = {}

--[[ 
    local pnoun_subrule_data = {
        pnoun_to_groups = {
            <pnoun unitid> = <Pnoun_Group>
            ...    
        }
        active_pnouns = (<pnoun unitid>, <pnoun unitid> ...),
        process_order = {
            <pnoun unitid> = <int>
        },
        pnouns_in_conds = (<pnoun unitid>, <pnoun unitid>, ...),
        pnoun_feature_extradata = {
            <feature> = {
                visible = <bool>
            }
        }
    }    
]]
local pnoun_subrule_data = {}

--[[ 
    local raycast_data = {
        <unit id of THIS text> = {
            -- list of all objects that were hit by the raycast
            raycast_objects = (<object>, <object>),
            raycast_object_count = <int>,

            -- Mapping of raycast positions to the objects that are in those objects. Note that these
            -- object lists aren't currently used since we don't need position specific logic. But maybe later
            -- if we come up with another word that needs that data.
            raycast_positions = { 
                <tileid> = [<object>, <object>], 
                <tileid> = [<object>, <object>], 
                ...
            },


            -- List of all extra spawned from relay or other raycast splitting
            cursors = {
                <tileid> = <unitid of cursor>,
                ...
            },
            
            pnoun_group = <Pnoun_Group>,
        }
    }
 ]]
local raycast_data = {}

--[[ 
    local relay_indicators = {
        <tileid + dir> = <unitid of indicator>,
        <tileid + dir> = <unitid of indicator>,
        ...
    }    
]]
local relay_indicators = {}

local PointerNouns = {
    this = true,
    that = true,
    these = true,
    those = true,
}

local playref = editor_objlist_reference["text_play"]
local feelingref = editor_objlist_reference["text_feeling"]
if editor_objlist[playref].argextra == nil then
    editor_objlist[playref].argextra = {}
end
if editor_objlist[feelingref].argextra == nil then
    editor_objlist[feelingref].argextra = {}
end
for pnoun_name, _ in pairs(PointerNouns) do
    table.insert(editor_objlist[playref].argextra, pnoun_name)
    table.insert(editor_objlist[feelingref].argextra, pnoun_name)
end


local function reset_this_mod_locals()
    blocked_tiles = {}
    explicit_passed_tiles = {}
    explicit_relayed_tiles = {}
    raycast_data = {}
    relay_indicators = {}
    registered_pnoun_rules = {}
    pnoun_subrule_data = {}
    indicator_layer_timer = 0
    run_extract_bpr_subrules = false
    extracting_bpr_subrules = false

    raycast_trace_tracker:clear()
    raycast_analyzer:reset()
end

local make_cursor, update_all_cursors, make_relay_indicator

table.insert(mod_hook_functions["rule_baserules"],
    function()
        addbaserule("empty", "is", "pass")
    end
)

-- Note: changed from "effect_always" to "always" since effect_always only activates when disable particle effects is off 
table.insert(mod_hook_functions["always"],
    function()
        if (generaldata.values[MODE] == 0) then
            utils.try_call(update_all_cursors, indicator_layer_timer)
            indicator_layer_timer = indicator_layer_timer + 1
            if indicator_layer_timer >= TIMER_PERIOD then
                indicator_layer_timer = 0
            end
        end
    end
)

table.insert(mod_hook_functions["level_start"], 
    function()
        on_level_start = true
        objectlist["text"] = 1 -- this fixes the "this(text) mimic x" + "X is this(Y)".
    end
)

table.insert( mod_hook_functions["undoed_after"],
    function()
        blocked_tiles = {}
        this_mod_globals.undoed_after_called = true
    end
)

table.insert(mod_hook_functions["rule_update"],
    function(is_this_a_repeated_update)
        this_mod_globals.active_this_property_text = {}
        blocked_tiles = {}
        explicit_passed_tiles = {}
        explicit_relayed_tiles = {}
        raycast_trace_tracker:clear()
        pnoun_subrule_data = {
            pnoun_to_groups = {},
            active_pnouns = {},
            process_order = {},
            pnouns_in_conds = {},
            pnoun_feature_extradata = {},
        }
        registered_pnoun_rules = {}
        for pnoun_group, value in pairs(Pnoun.Groups) do
            registered_pnoun_rules[value] = {
                pnoun_features = {},
                pnoun_units = {},
            }
        end
        run_extract_bpr_subrules = false

        if THIS_LOGGING then
            print(">>>>>>>>>>>>>>> rule_update start")
        end
    end
)
table.insert(mod_hook_functions["rule_update_after"],
    function()
        if on_level_start then
            on_level_start = false
        end
        if this_mod_globals.undoed_after_called then
            this_mod_globals.undoed_after_called = false
        end

        raycast_analyzer:reset()
        indicator_layer_timer = 0 -- Used for immediate feedback when making "THIS is pass/block/relay"

        if THIS_LOGGING then
            print("<<<<<<<<<<<<<< rule_update end")
        end
    end
)

table.insert( mod_hook_functions["command_given"],
    function()
        raycast_analyzer:reset()
    end
)
table.insert( mod_hook_functions["turn_end"],
    function()
        raycast_analyzer:reset()
    end
)

-- This actually returns the pointer name if valid. It should be named "get_pointer_noun_from_name()" But can't rename it because the BASED mod uses "is_name_text_this"
function is_name_text_this(name, check_not_)
    local check_not = check_not_ or false

    local isnot = false
    if string.sub(name, 1, 4) == "not " then
        isnot = true
        name = string.sub(name, 5)
    end

    if check_not and not isnot then
        return false
    end

    for noun, _ in pairs(PointerNouns) do
        if string.sub(name, 1, #noun) == noun then
            return noun
        end
    end
    return nil
end

local function dir_vec_to_dir_value(dir_vec)
    if dir_vec[1] > 0 and dir_vec[2] == 0 then
        return 0
    elseif dir_vec[2] < 0 and dir_vec[1] == 0 then
        return 1
    elseif dir_vec[1] < 0 and dir_vec[2] == 0 then
        return 2
    elseif dir_vec[2] > 0 and dir_vec[1] == 0 then
        return 3
    else
        return 4
    end
end

-- Determine the raycast velocity vectors, given a name of a pointer noun
local function get_rays_from_pointer_noun(name, x, y, dir, pnoun_unitid)
    local pointer_noun = is_name_text_this(name)
    local out_rays = {}

    if pointer_noun then
        local dir_vec = {dirs[dir+1][1], dirs[dir+1][2] * -1}

        if pointer_noun == "this" then
            table.insert(out_rays, {
                pos = {x, y},
                dir = dir_vec,
            })
        elseif pointer_noun == "that" then
            local cast_start_x = x
            local cast_start_y = y

            if dir == 0 then
                cast_start_x = roomsizex - 1
            elseif dir == 1 then
                cast_start_y = 0
            elseif dir == 2 then
                cast_start_x = 0
            elseif dir == 3 then
                cast_start_y = roomsizey - 1
            end
            table.insert(out_rays, {
                pos = {cast_start_x, cast_start_y},
                dir = {dir_vec[1] * -1, dir_vec[2] * -1},
            })
        elseif pointer_noun == "these" then
            table.insert(out_rays, {
                pos = {x, y},
                dir = dir_vec,
            })
        elseif pointer_noun == "those" then
            table.insert(out_rays, {
                pos = {x, y},
                dir = dir_vec,
            })
        end
    end

    return out_rays
end

-- Really useless function whose only purpose is to gatekeep calling update_raycast_units() in code() before checking updatecode.
function this_mod_has_this_text()
    for _,_ in pairs(raycast_data) do
        return true
    end
    return false
end

-- This is used in {{mod_injections}}
function reset_this_mod()
    for this_unitid, v in pairs(raycast_data) do
        for _, cursor in pairs(v.cursors) do
            MF_cleanremove(cursor)
        end
    end
    for tileid, relay_indicator_unitid in pairs(relay_indicators) do
        MF_cleanremove(relay_indicator_unitid)
    end
    reset_this_mod_globals()
    reset_this_mod_locals()
end

function on_add_this_text(this_unitid)
    if not raycast_data[this_unitid] then
        local unit = mmf.newObject(this_unitid)
        raycast_data[this_unitid] = {
            raycast_objects = {},
            raycast_object_count = 0,
            raycast_positions = {},
            cursors = {},
        }
    end
end

function on_delele_this_text(this_unitid)
    if raycast_data[this_unitid] then
        for _, cursor in pairs(raycast_data[this_unitid].cursors) do
            delunit(cursor)
            MF_cleanremove(cursor)
        end
        raycast_data[this_unitid] = nil
    end
end

--[[ 
    Every time addoption() gets called with a rule to submit, call this function to do a few things:
    - if the rule submitted has a pnoun as the target or effect, register the rule to be processed in do_subrule_pnouns()
    - if the rule submitted signifies potential enough reason to run extract_bpr_subrules(), set run_extract_bpr_subrules = true (see extract_bpr_subrules() for why)
    - if extracting_bpr_subrules = true, prevent any non-bpr rules from being submitted to the featureindex
]]
function scan_added_feature_for_pnoun_rule(rule, visible)
    local baserule = rule[1]
    local target = baserule[1]
    local verb = baserule[2]
    local property = baserule[3]
    local target_is_pnoun = is_name_text_this(target) or is_name_text_this(target, true)
    local property_is_pnoun = is_name_text_this(property) or is_name_text_this(property, true)
    local is_pnoun_rule = target_is_pnoun or property_is_pnoun

    local allow_add_to_featureindex = true

    local is_bpr_rule = property == "block" or property == "pass" or property == "relay"
    if extracting_bpr_subrules then
        if not is_bpr_rule then
            allow_add_to_featureindex = false
        elseif target_is_pnoun then
            allow_add_to_featureindex = false
        end
    else
        if verb == "mimic" then
            --[[ 
                Note: for "mimic", we cannot refine the above condition further. If for instance "baba mimic X" creates
                baba is pass" as a subrule, it is because the featureindex would've contained "X is pass" *after docode() is finished*.
                scan_added_feature_for_pnoun_rule() is called in addoption(), which in turn gets called *while* docode() is running.
                Therefore, accessing featureindex for lookup purposes in this function is unreliable, since the game is in the middle of
                repopulating featureindex.
             ]]
            run_extract_bpr_subrules = true
        elseif is_bpr_rule and (target == "all" or string.sub(target, 1, 5) == "group") then
            run_extract_bpr_subrules = true
        end
    end

    if is_pnoun_rule then
        local pnoun_group = nil
        if target_is_pnoun and not property_is_pnoun then
            if property == "block" then
                pnoun_group = Pnoun.Groups.THIS_IS_BLOCK
            elseif property == "relay" then
                pnoun_group = Pnoun.Groups.THIS_IS_RELAY
            elseif property == "pass" then
                pnoun_group = Pnoun.Groups.THIS_IS_PASS
            end
        end
        
        if pnoun_group == nil then
            pnoun_group = Pnoun.Groups.VARIABLE
        end
        
        -- A pnoun feature can only be in one pnoun group. There is no need to check for priority since
        -- each pnoun group is meant to be mutually exclusive in terms of features.
        table.insert(registered_pnoun_rules[pnoun_group].pnoun_features, rule)
        pnoun_subrule_data.pnoun_feature_extradata[rule] = {
            visible = visible
        }

        local pnouns_to_add = {}

        if target_is_pnoun then
            local target_this_unitid = get_target_unitid_from_rule(rule)
            if target_this_unitid ~= nil then
                table.insert(pnouns_to_add, target_this_unitid)
            end
        end
        if property_is_pnoun then
            local property_this_unitid = get_property_unitid_from_rule(rule)
            if property_this_unitid ~= nil then
                table.insert(pnouns_to_add, property_this_unitid)
            end
        end

        -- A pnoun unit can only belong to one pnoun group. If a pnoun can be categorized into two
        -- groups, only go for the group with the higher priority.
        for _, pnoun in ipairs(pnouns_to_add) do
            local prev_pnoun_group = pnoun_subrule_data.pnoun_to_groups[pnoun]

            if prev_pnoun_group ~= nil and pnoun_group < prev_pnoun_group then
                -- Replace with the pnoun group with the higher priority
                registered_pnoun_rules[prev_pnoun_group].pnoun_units[pnoun] = nil
                registered_pnoun_rules[pnoun_group].pnoun_units[pnoun] = true
                pnoun_subrule_data.pnoun_to_groups[pnoun] = pnoun_group
                pnoun_subrule_data.active_pnouns[pnoun] = true
            elseif prev_pnoun_group == nil then
                -- Assign the pnoun group to the pnoun unit
                registered_pnoun_rules[pnoun_group].pnoun_units[pnoun] = true
                pnoun_subrule_data.pnoun_to_groups[pnoun] = pnoun_group
                pnoun_subrule_data.active_pnouns[pnoun] = true
            end
        end
    end

    return allow_add_to_featureindex, target_is_pnoun, property_is_pnoun, is_pnoun_rule
end

function register_pnoun_in_cond(pnoun_unitid, condtype)
    local real_condtype = utils.real_condtype(condtype)
    pnoun_subrule_data.pnouns_in_conds[pnoun_unitid] = {
        condtype = real_condtype
    }
end

-- local
function update_all_cursors(timer)
    local order_explicit_indicators_on_top = timer <= TIMER_CYCLE_PERIOD
    for this_unitid, v in pairs(raycast_data) do
        local wordunit = mmf.newObject(this_unitid)
        for tileid, cursor_unitid in pairs(v.cursors) do
            local cursorunit = mmf.newObject(cursor_unitid)

            local x = wordunit.values[XPOS]
            local y = wordunit.values[YPOS]

            local nx = math.floor(tileid % roomsizex)
            local ny = math.floor(tileid / roomsizex)
            local cursor_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
            cursorunit.values[XPOS] = nx * cursor_tilesize + Xoffset + (cursor_tilesize / 2)
            cursorunit.values[YPOS] = ny * cursor_tilesize + Yoffset + (cursor_tilesize / 2)

            local c1 = 0
            local c2 = 0
            cursorunit.layer = 2
            if blocked_tiles[tileid] then
                if order_explicit_indicators_on_top then
                    cursorunit.values[ZLAYER] = 30
                else
                    cursorunit.values[ZLAYER] = 25
                end
                cursorunit.direction = 30
                MF_loadsprite(cursorunit.fixed,"this_cursor_blocked_0",30,true)
                c1,c2 = getuicolour("blocked")
            elseif explicit_relayed_tiles[tileid] then
                if order_explicit_indicators_on_top then
                    cursorunit.values[ZLAYER] = 29
                else
                    cursorunit.values[ZLAYER] = 24
                end
                cursorunit.direction = 29
                MF_loadsprite(cursorunit.fixed,"this_cursor_relay_0",29,true)
                c1,c2 = 5, 4
            elseif explicit_passed_tiles[tileid] then
                if order_explicit_indicators_on_top then
                    cursorunit.values[ZLAYER] = 28
                else
                    cursorunit.values[ZLAYER] = 23
                end
                cursorunit.direction = 31
                MF_loadsprite(cursorunit.fixed,"this_cursor_pass_0",31,true)
                c1,c2 = 4, 4
            else
                if ruleids[wordunit.fixed] then
                    cursorunit.values[ZLAYER] = 27 -- Note: the game only actually processes Zlayers between 0-30. We don't know what it does with layers outside of this range, but it seems
                else
                    cursorunit.values[ZLAYER] = 26
                end
                cursorunit.direction = 28
                MF_loadsprite(cursorunit.fixed,"this_cursor_0",28,true)
                -- MF_loadsprite(cursorunit.fixed,"stable_indicator_0",28,true)
                c1,c2 = wordunit.colour[1],wordunit.colour[2]
            end
        
            MF_setcolour(cursorunit.fixed,c1,c2)
            cursorunit.scaleX = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
            cursorunit.scaleY = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
            
            if (generaldata.values[DISABLEPARTICLES] ~= 0 or generaldata5.values[LEVEL_DISABLEPARTICLES] ~= 0) then
                cursorunit.visible = false
            else
                cursorunit.visible = true
            end
        end
    end

    for indicator_key, indicator_id in pairs(relay_indicators) do
        local relay_indicator = mmf.newObject(indicator_id)
        
        local dir = relay_indicator.values[DIR]
        local tileid = indicator_key - (dir * roomsizex * roomsizey)
        local x = math.floor(tileid % roomsizex)
        local y = math.floor(tileid / roomsizex)
        
        local cursor_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
        relay_indicator.values[XPOS] = x * cursor_tilesize + Xoffset + (cursor_tilesize / 2)
        relay_indicator.values[YPOS] = y * cursor_tilesize + Yoffset + (cursor_tilesize / 2)
        
        if (generaldata.values[DISABLEPARTICLES] ~= 0 or generaldata5.values[LEVEL_DISABLEPARTICLES] ~= 0) then
            -- Just to hide it
            relay_indicator.visible = false
        else
            relay_indicator.visible = true
        end

        relay_indicator.scaleX = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
        relay_indicator.scaleY = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
    end
end

-- local
function make_cursor()
    local unitid2 = MF_create("customsprite")
    local unit2 = mmf.newObject(unitid2)
    
    unit2.values[ONLINE] = 1
    
    unit2.layer = 2
    unit2.direction = 28
    MF_loadsprite(unitid2,"this_cursor_0",28,true)
    
    return unitid2
end

-- local
function make_relay_indicator(x, y, dir)
    local unitid = MF_create("customsprite")
    local unit = mmf.newObject(unitid)
    
    unit.values[ONLINE] = 1
    
    unit.layer = 2
    unit.direction = 27
    MF_loadsprite(unitid,"relay_indicator_0",27,true)

    unit.values[DIR] = dir
    if dir == 0 then
        unit.angle = 0
    elseif dir == 1 then
        unit.angle = 90
    elseif dir == 2 then
        unit.angle = 180
    elseif dir == 3 then
        unit.angle = 270
    end

    local cursor_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
    unit.values[XPOS] = x * cursor_tilesize + Xoffset + (cursor_tilesize / 2)
    unit.values[YPOS] = y * cursor_tilesize + Yoffset + (cursor_tilesize / 2)
    unit.scaleX = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
    unit.scaleY = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
    unit.values[ZLAYER] = 29

    MF_setcolour(unitid,5,4)
    
    return unitid
end

local function this_raycast(ray, checkemptyblock, raycast_trace, curr_cast_extradata)
    -- return values: ray_pos, is_emptyblock, select_empty, emptyrelay_dir
    local ox = ray.pos[1] + ray.dir[1]
    local oy = ray.pos[2] + ray.dir[2]
    while inbounds(ox,oy,1) do
        local tileid = ox + oy * roomsizex
        raycast_trace:add_tileid(tileid)
        
        if unitmap[tileid] == nil or #unitmap[tileid] == 0 then
            local pnoun_unitid = curr_cast_extradata.pnoun_unitid
            local empty_dir = emptydir(ox, oy)

            if checkemptyblock and raycast_trace:evaluate_raycast_property(pnoun_unitid, "empty", "block", 2, ox, oy) then
                return {ox, oy}, true, false, nil
            elseif raycast_trace:evaluate_raycast_property(pnoun_unitid, "empty", "relay", 2, ox, oy) and empty_dir ~= 4 then
                return {ox, oy}, false, false, empty_dir
            elseif not raycast_trace:evaluate_raycast_property(pnoun_unitid, "empty", "pass", 2, ox, oy) then
                return {ox, oy}, false, true, nil
            end
        elseif unitmap[tileid] ~= nil and #unitmap[tileid] > 0 then
            return {ox, oy}, false, false, nil
        end

        if curr_cast_extradata.pointer_noun == "those" then
            break
        else
            ox = ox + ray.dir[1]
            oy = oy + ray.dir[2]
        end
    end

    return nil
end

local function make_relay_indicator_key(tileid, dir)
    return tileid + dir * roomsizex * roomsizey
end

--[[ 
    Given a pnoun text, simulate a raycast with it without actually effecting anything.
    Return data about the results of the raycast/
]]
local function simulate_raycast_with_pnoun(pnoun_unitid, raycast_settings)
    --[[ 
        return value: {
            <tileid> = [<object>, <object>]
            ...
        }
     ]]
    local pointer_unit = mmf.newObject(pnoun_unitid)
    local pointer_noun = is_name_text_this(pointer_unit.strings[NAME])
    local rays = get_rays_from_pointer_noun(pointer_unit.strings[NAME], pointer_unit.values[XPOS], pointer_unit.values[YPOS], pointer_unit.values[DIR], pnoun_unitid)
    local ray_objects_by_tileid = {}
    local found_relay_indicators = {} -- indicator ids -> true
    local found_blocked_tiles = {}
    local found_passed_tiles = {} -- Currently only used for THOSE
    local found_ending_these_texts = {}
    local raycast_trace = RaycastTrace:new()

    for i, ray in ipairs(rays) do
        local stack = {
            {
                ray = ray, 
                extradata = {
                    pnoun_unitid = pnoun_unitid,
                    pointer_noun = pointer_noun,
                    these_ray_objects_by_tileid = {},
                    original_cast_pos = {pointer_unit.values[XPOS], pointer_unit.values[YPOS]}
                }
            } 
        }
        local visited_tileids = {}

        while #stack > 0 do
            local curr_cast_data = table.remove(stack)
            
            local ray_pos, is_emptyblock, select_empty, emptyrelay_dir = this_raycast(curr_cast_data.ray, raycast_settings.checkblocked, raycast_trace, curr_cast_data.extradata)
            if not ray_pos then
                -- Do nothing for now
            elseif (pointer_noun == "that" or pointer_noun == "those") and ray_pos[1] == curr_cast_data.extradata.original_cast_pos[1] and ray_pos[2] == curr_cast_data.extradata.original_cast_pos[2] then
                -- Do nothing. THAT and THOSE cannot refer to itself, except when it is relayed
            else
                local blocked = false
                local new_relay_indicators = {}
                local new_stack_entries = {}
                local ray_objects = {}
                local tileid = ray_pos[1] + ray_pos[2] * roomsizex
                local found_ending_these = false
                local found_valid_ending_these = false

                if pointer_noun == "these" and not gettilenegated(ray_pos[1], ray_pos[2]) then
                    -- If we found another THESE pointing in the opposite direction, terminate early
                    if unitmap[tileid] ~= nil then
                        for _, ray_unitid in ipairs(unitmap[tileid]) do
                            local ray_unit = mmf.newObject(ray_unitid)
                            if ray_unit.strings[NAME] == "these" then
                                found_ending_these = true

                                if ray_unitid ~= pnoun_unitid then
                                    local ray_dir_value = dir_vec_to_dir_value(curr_cast_data.ray.dir)
                                    if rotate(ray_dir_value) == ray_unit.values[DIR] then
                                        found_ending_these_texts[ray_unitid] = true
                                        found_valid_ending_these = true
                                    end
                                end
                            end
                        end
                    end
                end

                -- If a tile is marked as visited and we did not have this check, we know that any raycasts that 
                -- stop at the visited tile would get processed the same way everytime. The below check prevents
                -- this, removing repeated processing and infinite loops .
                if not visited_tileids[tileid] and not found_ending_these then
                    visited_tileids[tileid] = true
                    
                    if raycast_settings.checkblocked and is_emptyblock then
                        blocked = true
                    elseif emptyrelay_dir then
                        local indicator_key = make_relay_indicator_key(tileid, emptyrelay_dir)
                        new_relay_indicators[indicator_key] = {
                            x = ray_pos[1],
                            y = ray_pos[2],
                            dir = emptyrelay_dir
                        }
                        
                        for _, ray in ipairs(get_rays_from_pointer_noun(pointer_noun, ray_pos[1], ray_pos[2], emptyrelay_dir)) do
                            table.insert(new_stack_entries, {ray = ray, extradata = curr_cast_data.extradata})
                        end
                    elseif select_empty then
                        local add_to_rayunits = true
                        if add_to_rayunits then
                            local object = utils.make_object(2, ray_pos[1], ray_pos[2])
                            table.insert(ray_objects, object)
                        end
                    else
                        local total_pass_unit_count = 0
                        local found_relay = false
                        local relay_dirs = {}

                        -- Check through every unit in the specific space
                        for _, ray_unitid in ipairs(unitmap[tileid]) do
                            local ray_unit = mmf.newObject(ray_unitid)
                            local ray_unit_name = getname(ray_unit) -- If the unit is a text block, we want the name to be "text"
                            local add_to_rayunits = true

                            -- block logic
                            if raycast_settings.checkblocked then
                                if raycast_trace:evaluate_raycast_property(pnoun_unitid, ray_unit_name, "block", ray_unitid) then
                                    blocked = true
                                end

                                if blocked then
                                    break
                                end
                            end

                            -- relay logic
                            if raycast_settings.checkrelay and not blocked then
                                if raycast_trace:evaluate_raycast_property(pnoun_unitid, ray_unit_name, "relay", ray_unitid) then
                                    found_relay = true
                                    add_to_rayunits = false
                                    relay_dirs[ray_unit.values[DIR]] = true
                                    
                                    local indicator_key = make_relay_indicator_key(tileid, ray_unit.values[DIR])
                                    new_relay_indicators[indicator_key] = {
                                        x = ray_pos[1],
                                        y = ray_pos[2],
                                        dir = ray_unit.values[DIR]
                                    }
                                end
                            end

                            -- pass logic
                            if raycast_settings.checkpass and not blocked then
                                if raycast_trace:evaluate_raycast_property(pnoun_unitid, ray_unit_name, "pass", ray_unitid) then
                                    total_pass_unit_count = total_pass_unit_count + 1
                                    add_to_rayunits = false
                                end
                            end

                            if add_to_rayunits then
                                local object = utils.make_object(ray_unitid, ray_pos[1], ray_pos[2])
                                table.insert(ray_objects, object)
                            end
                        end

                        -- Consolidate findings from scanning all units in a single position.
                        if not blocked then
                            if found_relay then
                                curr_cast_data.extradata.original_cast_pos = {ray_pos[1], ray_pos[2]}
                                for dir, _ in pairs(relay_dirs) do
                                    for _, ray in ipairs(get_rays_from_pointer_noun(pointer_noun, ray_pos[1], ray_pos[2], dir)) do
                                        table.insert(new_stack_entries, {ray = ray, extradata = curr_cast_data.extradata})
                                    end
                                end
                            elseif raycast_settings.checkpass and total_pass_unit_count >= #unitmap[tileid] then
                                if pointer_noun == "those" then
                                    -- When processing THOSE, if all units are pass, then stop the re-raycasting.
                                    -- Note that this tileid is pass to indicate at which places stopped a THOSE raycast from
                                    -- going further.
                                    found_passed_tiles[tileid] = true
                                    ray_objects_by_tileid[tileid] = {}
                                else
                                    -- At this point, we know that all objects at this location are pass. The effect is to re-raycast
                                    -- in the same direction of the original raycast.
                                    local new_ray = {pos = ray_pos, dir = curr_cast_data.ray.dir}
                                    table.insert(new_stack_entries, {ray = new_ray, extradata = curr_cast_data.extradata})

                                    -- Since the direction of the re-raycast can be different depending on the original raycast, we
                                    -- don't mark this location as visited.
                                    visited_tileids[tileid] = false
                                end
                            end
                        end
                    end
                end

                if found_ending_these then
                    if found_valid_ending_these then
                        for tileid, ray_objects in pairs(curr_cast_data.extradata.these_ray_objects_by_tileid) do
                            if ray_objects_by_tileid[tileid] == nil then
                                ray_objects_by_tileid[tileid] = ray_objects
                            end
                        end
                    end
                elseif blocked then
                    -- If we find that the current tileid has a blocked unit, don't submit anything
                    found_blocked_tiles[tileid] = true
                    ray_objects_by_tileid[tileid] = {}
                elseif #new_stack_entries > 0 then
                    -- If we inserted into the stack, we intend to re-raycast. Don't submit the found ray objects.
                    for _, stack_entry in ipairs(new_stack_entries) do
                        table.insert(stack, stack_entry)
                    end
    
                    -- Do submit any relay indicators if we found any.
                    for indicator_key, data in pairs(new_relay_indicators) do
                        found_relay_indicators[indicator_key] = data
                    end
                else
                    -- At this point, we found a stopping point with valid ray objects.
                    if pointer_noun == "these" then
                        local new_extradata = curr_cast_data.extradata
                        if new_extradata.these_ray_objects_by_tileid[tileid] == nil then
                            new_extradata.these_ray_objects_by_tileid[tileid] = ray_objects
                        end

                        local new_ray = {pos = ray_pos, dir = curr_cast_data.ray.dir}
                        table.insert(stack, {ray = new_ray, extradata = new_extradata})
                    else
                        if pointer_noun == "those" and #ray_objects == 0 then
                            -- On THOSE, if there are no selectable ray objects, do not submit an empty list
                        else
                            if ray_objects_by_tileid[tileid] == nil then
                                -- @NOTE: for now we are assuming one cursor per cast (excluding relays). If there's a need
                                -- to distinguish between two cursors, and they both land on the same tileid, then we would
                                -- need to store this set of ray objects multiple times
                                ray_objects_by_tileid[tileid] = ray_objects
                            end
                        end
                    end
                    for indicator_key in pairs(new_relay_indicators) do
                        found_relay_indicators[indicator_key] = data
                    end

                    if pointer_noun == "those" and #ray_objects > 0 then
                        for i = 0,3 do
                            local new_dir = dirs[i+1]

                            -- Ensure that we are not raycasting in the direction that we just came
                            if (-new_dir[1] ~= curr_cast_data.ray.dir[1]) or (-new_dir[2] ~= curr_cast_data.ray.dir[2]) then
                                local new_ray = {pos = ray_pos, dir = new_dir}

                                local new_extradata = utils.deep_copy_table(curr_cast_data.extradata)
                                table.insert(stack, {ray = new_ray, extradata = new_extradata})
                            end
                        end
                    end
                end
            end
        end
    end

    -- For now, we are extending what BLOCK means by saying that if the raycast finds ANY block object, no matter where from,
    -- the pnoun will not refer to any object at all. This is to balance pnouns that can select multiple objects, like THESE
    -- and THOSE
    local found_blocked = false
    for _,_ in pairs(found_blocked_tiles) do
        found_blocked = true
        break
    end

    if found_blocked then
        for tileid, _ in pairs(ray_objects_by_tileid) do
            if not found_blocked_tiles[tileid] then
                ray_objects_by_tileid[tileid] = nil
            end
        end
    else
        for tileid, _ in pairs(ray_objects_by_tileid) do
            local x, y = plasma_utils.coords_from_tileid(tileid)
            if gettilenegated(x, y) then
                ray_objects_by_tileid[tileid] = {}
            end
        end
    end

    local extra_raycast_data = {
        found_relay_indicators = found_relay_indicators, 
        found_blocked_tiles = found_blocked_tiles,
        found_passed_tiles = found_passed_tiles,
        found_ending_these_texts = found_ending_these_texts,
    }

    return ray_objects_by_tileid, extra_raycast_data, raycast_trace
end

function check_updatecode_status_from_raycasting()
    for tileid in pairs(raycast_analyzer.tileids_updated) do
        if raycast_trace_tracker:is_tileid_recorded(tileid) then
            return true
        end
    end

    --[[ 
        This is part of a hack to prevent any rules involving THIS + block/pass/relay from constantly setting updatecode = 1
        when checking data from the raycast tracer. Since retest_features_for_testcond_change() can call testcond(),  
        setting checking_updatecode_status_flag == true signals to the condition function for "this" to run special logic.
        (see "condlist["this"]" for more info).

        Note: JAAANKY way of setting parameters of testcond without actually overriding testcond or hasfeature.
        Literally counting on the raycast tracer to update checking_updatecode_curr_pnoun_ref every time it checks a set of conditions
    ]]
    checking_updatecode_status_flag = true
    local result = raycast_trace_tracker:retest_features_for_testcond_change(checking_updatecode_curr_pnoun_ref)
    checking_updatecode_status_flag = false

    return result
end

function get_raycast_objects(this_text_unitid, exclude_bpr)
    if RaycastBank:is_valid_ray_id(this_text_unitid) then
        return RaycastBank:get_raycast_objects(this_text_unitid)
    end

    if raycast_data[this_text_unitid] == nil then
        return {}, 0
    else
        if exclude_bpr then
            local out_ray_objects = {}
            local count = 0
            for tileid, ray_objects in pairs(raycast_data[this_text_unitid].raycast_positions) do
                if not (blocked_tiles[tileid] or explicit_relayed_tiles[tileid] or explicit_passed_tiles[tileid]) then
                    for _, ray_object in ipairs(ray_objects) do
                        out_ray_objects[ray_object] = true
                        count = count + 1
                    end
                end
            end

            return out_ray_objects, count
        else
            return raycast_data[this_text_unitid].raycast_objects, raycast_data[this_text_unitid].raycast_object_count
        end
    end
end

function get_raycast_tileid(this_text_unitid)
    if RaycastBank:is_valid_ray_id(this_text_unitid) then
        return RaycastBank:get_raycast_tileids(this_text_unitid)
    end

    return raycast_data[this_text_unitid].raycast_positions
end

condlist["this"] = function(params,checkedconds,checkedconds_,cdata)
    if #params == 1 then
        valid = true
        local unitid, x, y = cdata.unitid, cdata.x, cdata.y
        local this_text_unitid = parse_this_unit_from_param_id(params[1])

        if checking_updatecode_status_flag then
            --[[ 
                When checking updatecode status with the raycast tracer, we want to emulate the same enviornment that
                was present when simulate_raycast_with_pnoun() was called with ref_pnoun below. This means the order that
                ref_pnoun was processed matters, since each process_round adds new features to the featureindex on the fly.
                To emulate this enviornment, we automatically return false for all "this" conditions that were generated by
                a pnoun processed later than or at the same time as ref_pnoun. This is because in the perspective of
                ref_pnoun, the only features that are relevant are the onces added before ref_pnoun was processed.
            ]]
            local ref_pnoun = checking_updatecode_curr_pnoun_ref[0]
            if pnoun_subrule_data.process_order[this_text_unitid] >= pnoun_subrule_data.process_order[ref_pnoun] then
                return false, checkedconds
            end
        end
        
        local object = utils.make_object(unitid, x, y)
        local raycast_objects = get_raycast_objects(this_text_unitid)
        if raycast_objects[object] then
            return true, checkedconds
        end
    end
    return false, checkedconds
end

-- Given an object/text pointed in "Baba <verb> THIS(X)", return whether or not X is valid
local function is_unit_valid_this_property(name, unittype, texttype, verb)
    if is_name_text_this(name) then
        return false
    end

    if verb == "is" then
        if unittype == "text" and (texttype == 2 or texttype == 0) then
            return true
        elseif unittype == "object" then
            return true
        end
    elseif verb == "write" then
        if unittype == "text" and (texttype == 0 or texttype == 2) then
            return true
        elseif unittype == "object" then
            return true
        end
    elseif verb == "play" then
        local play_realname = unitreference["text_play"]
        if (changes[play_realname] ~= nil) then
            local wchanges = changes[play_realname]

            local found = false
            for _, argtype in ipairs(wchanges.argtype) do
                if argtype == 8 then
                    found = true
                    break
                end
            end
            if (found) then
                for _, customobject in ipairs(wchanges.customobjects) do
                    if name == customobject then
                        return true
                    end
                end
            end
        end
    else
        if texttype == 0 then
            return true
        elseif unittype == "object" then
            return true
        end
    end

    return false
end

-- Given an object/text pointed in "Baba <infix_conf> THIS(X) is Y", return whether or not X is valid
function is_unit_valid_this_infix_param(name, unittype, texttype, infix_cond)
    if is_name_text_this(name) then
        return false
    end

    if infix_cond == "feeling" then
        if unittype == "text" and texttype == 2 then
            return true
        end
    else
        if unittype == "text" and texttype == 0 then
            return true
        elseif unittype == "object" then
            return true
        end
    end

    return false
end

-- Given a tileid that represents the location of a lettertext pointed by "baba is THIS(X)", get all valid words spelled out by letters
local function get_valid_letterwords(tileid, reason, reason_type)
    local found_letterwords = {}
    if (letterunits_map[tileid] ~= nil) then
        local single_letter_words = {}
        for i,v in ipairs(letterunits_map[tileid]) do
            local word = v[1]
            local wtype = v[2]
            local x = v[3]
            local y = v[4]
            local dir = v[5]
            local width = v[6]
            local unitids = v[7]

            local letter_tileid = utils.tileid_from_coords(x, y)
            if letter_tileid == tileid then
                if (string.len(word) > 5) and (string.sub(word, 1, 5) == "text_") then
                    word = string.sub(v[1], 6)
                end

                local is_valid = false
                if reason_type == "infix" then
                    is_valid = is_unit_valid_this_infix_param(word, "text", wtype, reason)
                else
                    is_valid = is_unit_valid_this_property(word, "text", wtype, reason)
                end
                
                if is_valid then
                    if width == 1 then
                        single_letter_words[word] = v
                    else
                        table.insert(found_letterwords, v)
                    end
                end
            end
        end

        if #found_letterwords == 0 then
            for word, letterword in pairs(single_letter_words) do
                table.insert(found_letterwords, letterword)
            end
        end
    end

    return found_letterwords
end

function get_raycast_infix_units(this_text_unitid, infix)
    local out_raycast_units = {}
    local found_letterwords = {}
    for ray_tileid, raycast_objects in pairs(raycast_data[this_text_unitid].raycast_positions) do
        for _, ray_object in ipairs(raycast_objects) do
            local ray_unit_details = utils.parse_object_full(ray_object)
            if ray_unit_details.texttype ~= 5 then
                if is_unit_valid_this_infix_param(ray_unit_details.name, ray_unit_details.unittype, ray_unit_details.texttype, infix) then
                    table.insert(out_raycast_units, ray_object)
                end
            end
        end

        for _, letterword in ipairs(get_valid_letterwords(ray_tileid, infix, "infix")) do
            table.insert(found_letterwords, letterword)
        end
    end

    return out_raycast_units, found_letterwords
end

-- Like get_raycast_objects, but factors in this redirection
local function get_raycast_property_units(this_text_unitid, verb)
    local out_raycast_units = {}
    local redirected_pnouns = {}
    local found_letterwords = {}

    if raycast_data[this_text_unitid] then
        local this_text_unit = mmf.newObject(this_text_unitid)
        local init_tileid = this_text_unit.values[XPOS] + this_text_unit.values[YPOS] * roomsizex
        
        local visited_tileids = {}
        visited_tileids[init_tileid] = true
        local raycast_this_texts = { this_text_unitid } -- This will be treated as a stack, meaning we are doing DFS instead of BFS
    
        while #raycast_this_texts > 0 do
            local curr_this_unitid = table.remove(raycast_this_texts) -- Pop the stack 
            
            for curr_raycast_tileid, raycast_objects in pairs(raycast_data[curr_this_unitid].raycast_positions) do
                if blocked_tiles[curr_raycast_tileid] then
                    -- do nothing if blocked
                elseif explicit_passed_tiles[curr_raycast_tileid] then
                    -- do nothing if the tile is explicitly passed
                elseif explicit_relayed_tiles[curr_raycast_tileid] then
                    -- do nothing if the tile is explicitly relayed
                elseif visited_tileids[curr_raycast_tileid] then
                    -- do nothing if we visited this current tile
                elseif curr_raycast_tileid then
                    visited_tileids[curr_raycast_tileid] = true
    
                    for _, ray_object in ipairs(raycast_objects) do
                        local ray_unit_details = utils.parse_object_full(ray_object)
                        if is_unit_valid_this_property(ray_unit_details.name, ray_unit_details.unittype, ray_unit_details.texttype, verb) then
                            table.insert(out_raycast_units, ray_object)
                        else
                            if is_name_text_this(ray_unit_details.name) then
                                table.insert(raycast_this_texts, ray_unit_details.unitid)
                                table.insert(redirected_pnouns, ray_unit_details.unitid)
                            end
                        end
                    end
    
                    for _, letterword in ipairs(get_valid_letterwords(curr_raycast_tileid, verb, "verb")) do
                        table.insert(found_letterwords, letterword)
                    end
                end
            end
        end
    end

    return out_raycast_units, redirected_pnouns, found_letterwords
end

local function populate_inactive_pnouns()
    local active_pnouns = {}
    for pnoun_group, data in pairs(registered_pnoun_rules) do
        for pnoun_unitid in pairs(data.pnoun_units) do
            active_pnouns[pnoun_unitid] = true
        end
    end

    local inactive_pnoun_group = registered_pnoun_rules[Pnoun.Groups.VARIABLE]
    for pnoun_unitid, _ in pairs(raycast_data) do
        if not active_pnouns[pnoun_unitid] then
            inactive_pnoun_group.pnoun_units[pnoun_unitid] = true
        end
    end
end

--[[ 
    Performes a "controlled call" on subrules() and grouprules() to extract any block/pass/relay rules created
    from group/mimic/all.

    For context, group/mimic/all are special words that can create what the game calls a "subrule". This is a normal baba is you rule
    but it is created *because* another rule exists. For instance, having "all is you" will generate "baba is you", "keke is you", "tree is you",
    and so on as subrules. Another example: "bird is group" + "group is word" will generate "bird is word" as a subrule.

    These subrules are generated when both subrules() and grouprules() are called. In vanilla, they are called right after docode() in rules.lua.

    Pnouns are also implemented in this way. They generate subrules after figuring out what each pnoun text points to. This all happens in
    do_subrule_pnouns(), which is called *before* subrules() and grouprules(), but *after* docode(). This order ensures that all pnoun texts
    get evaluated before passing it off to subrules() and grouprules() to do their own thing in evaluating group/mimic/all.

    But there's one flaw in this. do_subrule_pnouns() does all of the raycast logic for every pnoun text.
    The raycast logic depends on block/pass/relay rules (which I dub "bpr"). And bpr rules have to get added to the featureindex in order to be
    recognized.

    Let's look at trying to evaluate "all is block" + "this(baba) is you", keeping in mind of the call order of docode(), do_subrule_pnouns(), subrules() and grouprules() in rules.lua:
        - First, docode() extracts "all is block" and "this is you" (notice "this" instead of "this(baba)" because we haven't evaluated "this" yet)
        - Next, do_subrule_pnouns() determines that "this" is pointing to baba, and generates "baba is you"
        - Lastly, subrules()/grouprules() evaluates "all is block" to generate "baba is block", "keke is block", etc
    The problem is the "baba is block" subrule did not have a chance to be used in do_subrule_pnouns(). do_subrule_pnouns() is called *before* subrules().

    The solution I ultimately found (after several ideas leading to rabbit holes) is to do a *seperate* call to subrules()/grouprules() *before*
    evaluating all pnoun texts. However, this call ensures that subrules with *only* block/pass/relay are added to the featureindex, throwing out
    other subrules to avoid duplicates.

    There are still a few cases with pnouns + subrules that are broken:
    - X is this(group) + group is bpr
    - X mimic this(tree) + tree is bpr
    - X mimic this(group) + group is bpr
    However, these cases are so specific that I won't be fixing them (for now). I suspect though the fix would involve alternating calling
    between do_subrule_pnouns() and subrules()/grouprules() in some insanely restricted way. Or even worse, allow an infinite loop to happen.
]]
local function extract_bpr_subrules()
    extracting_bpr_subrules = true
    subrules()
    grouprules()
    extracting_bpr_subrules = false
end

-- The main function for generating pnoun subrules. This function turns "THIS(baba) is you" into "baba is you", for example
local function process_pnoun_features(pnoun_features, feature_extradata, filter_property_func, curr_pnoun_op, pnoun_group)
    local final_options = {}
    local processed_pnouns = {}
    local all_redirected_pnouns = {}
    
    for i=#pnoun_features,1,-1 do
        rules = pnoun_features[i]
        local rule, conds, ids, tags = rules[1], rules[2], rules[3], rules[4]
        local visible = feature_extradata[rules].visible
        local target, verb, property = rule[1], rule[2], rule[3]
        local redirected_pnouns_in_feature = {}
        local found_pnouns = {}

        local target_isnot = string.sub(target, 1, 4) == "not "
        if target_isnot then
            target = string.sub(target, 5)
        end
        local prop_isnot = string.sub(property, 1, 4) == "not "
        if prop_isnot then
            property = string.sub(property, 5)
        end

        -- Process properties first
        local property_options = {}
        if not is_name_text_this(property) then
            if filter_property_func(property) then
                table.insert(property_options, {rule = rule, conds = conds})    
            end
        else
            local this_text_unitid = get_property_unitid_from_rule(rules)
            if this_text_unitid then
                table.insert(found_pnouns, this_text_unitid)
                local raycast_objects, redirected_pnouns, found_letterwords = get_raycast_property_units(this_text_unitid, verb)
                redirected_pnouns_in_feature = redirected_pnouns
                for _, object in ipairs(raycast_objects) do
                    local unit_details = utils.parse_object_full(object)

                    local rulename = unit_details.name
                    if is_turning_text(rulename) then
                        rulename = get_turning_text_interpretation(unit_details.unitid)
                    end

                    if filter_property_func(rulename) and unit_details.texttype ~= 5 then
                        if unit_details.unittype == "text" then
                            this_mod_globals.active_this_property_text[unit_details.unitid] = true
                        end

                        if prop_isnot then
                            rulename = "not "..rulename
                        end
                        
                        local newrule = {rule[1],rule[2],rulename}
                        local newconds = {}
                        for a,b in ipairs(conds) do
                            table.insert(newconds, b)
                        end
                        table.insert(property_options, {rule = newrule, conds = newconds, newrule = nil, showrule = visible})
                    end
                end

                for _, letterword in ipairs(found_letterwords) do
                    local rulename = letterword[1]
                    local unittype = "text"

                    if (string.len(rulename) > 5) and (string.sub(rulename, 1, 5) == "text_") then
                        rulename = string.sub(letterword[1], 6)
                    end

                    if filter_property_func(rulename) then
                        for _, letterunitid in ipairs(letterword[7]) do
                            this_mod_globals.active_this_property_text[letterunitid] = true
                        end

                        if prop_isnot then
                            rulename = "not "..rulename
                        end
                        
                        local newrule = {rule[1],rule[2],rulename}
                        local newconds = {}
                        for a,b in ipairs(conds) do
                            table.insert(newconds, b)
                        end
                        table.insert(property_options, {rule = newrule, conds = newconds, newrule = nil, showrule = visible})
                    end
                end
            end
        end

        -- Process target next
        local target_options = {}
        if not is_name_text_this(target) then
            target_options = property_options
        elseif #property_options > 0 then
            local this_text_unitid = get_target_unitid_from_rule(rules)
            if this_text_unitid then
                table.insert(found_pnouns, this_text_unitid)
                local this_unit_as_param_id = convert_this_unit_to_param_id(this_text_unitid)
                if target_isnot then
                    for i,mat in pairs(objectlist) do
                        if (findnoun(i) == false) then
                            for _, option in ipairs(property_options) do
                                local newrule = {i, option.rule[2], option.rule[3]}
                                local newconds = {}
                                table.insert(newconds, {"not this", {this_unit_as_param_id} })
                                for a,b in ipairs(option.conds) do
                                    table.insert(newconds, b)
                                end
                                table.insert(target_options, {rule = newrule, conds = newconds, notrule = true, showrule = false})
                            end
                        end
                    end
                    
                    -- Rule display in pause menu
                    if #target_options > 0 and filter_property_func(property) then
                        local ray_names = {}
                        for ray_object in pairs(get_raycast_objects(this_text_unitid), true) do
                            local ray_unitid, _, _, ray_tileid = utils.parse_object(ray_object)
                            local ray_name = ""
                            if ray_unitid == 2 then
                                ray_name = "empty"
                            else
                                local ray_unit = mmf.newObject(ray_unitid)
                                ray_name = ray_unit.strings[NAME]
                                if ray_unit.strings[UNITTYPE] == "text" then
                                    ray_name = "text"
                                end
                            end

                            ray_names[ray_name] = true
                        end
                        for ray_name in pairs(ray_names) do
                            local newrule = {ray_name, rule[2], rule[3]}
                            local newconds = {}
                            table.insert(newconds, {"not this", {this_unit_as_param_id} })
                            for a,b in ipairs(conds) do
                                table.insert(newconds, b)
                            end
                            table.insert(visualfeatures, {newrule, newconds, ids, tags})
                        end 
                    end
                else
                    local ray_names = {}
                    for ray_object in pairs(get_raycast_objects(this_text_unitid, true)) do
                        local ray_unitid, _, _, ray_tileid = utils.parse_object(ray_object)
                        local ray_name = ""
                        if ray_unitid == 2 then
                            ray_name = "empty"
                        else
                            local ray_unit = mmf.newObject(ray_unitid)
                            ray_name = ray_unit.strings[NAME]
                            if ray_unit.strings[UNITTYPE] == "text" then
                                ray_name = "text"
                            end
                        end

                        ray_names[ray_name] = true
                    end
                    for ray_name in pairs(ray_names) do
                        for _, option in ipairs(property_options) do
                            local newrule = {ray_name, option.rule[2], option.rule[3]}
                            local newconds = {}
                            table.insert(newconds, {"this", {this_unit_as_param_id} })
                            for a,b in ipairs(option.conds) do
                                table.insert(newconds, b)
                            end

                            table.insert(target_options, {rule = newrule, conds = newconds, notrule = false, showrule = true})
                        end
                    end
                end
            end
        end

        -- If the current pnoun group doesn't have a place to redirect
        local mark_all_processed = Pnoun.Pnoun_Group_Lookup[pnoun_group].redirect_pnoun_group == nil
        
        if #target_options > 0 then
            mark_all_processed = true

            for _, option in ipairs(target_options) do
                table.insert(final_options, {rule = option.rule, conds=option.conds, ids=ids, tags=tags, notrule = option.notrule, showrule = option.showrule})
            end
            for _, unitid in ipairs(redirected_pnouns_in_feature) do
                all_redirected_pnouns[unitid] = true
            end
        else
            -- @ Note: this is meant to trick postrules to display the active particles even
            -- though we don't actually call addoption
            table.insert(features, {{"this","is","test"}, conds, ids, tags})
        end

        if mark_all_processed then
            -- For all "this" text in each option, mark it as processed so that future update_raycast_units() calls
            -- don't change the raycast units for each "this" text
            for _, pnoun_unitid in ipairs(found_pnouns) do
                processed_pnouns[pnoun_unitid] = true
            end
            table.remove(pnoun_features, i)
        end
    end

    for _, option in ipairs(final_options) do
        addoption(option.rule,option.conds,option.ids,option.showrule,nil,option.tags)
    end

    return {processed_pnouns, pnoun_features, all_redirected_pnouns}
end

--[[ 
    Calling this function locks in the raycast data generated from simulate_raycast_with_pnoun() for a single pnoun.
    It also applies the needed global state changes from doing a raycast. Once a pnoun's raycast data is committed,
    the pnoun cannot update it's raycast data until the featureindex gets refreshed again from calling code() with updatecode = 1.

    This function implies that raycast data for a pnoun *can* change while running do_subrule_pnouns(). See that function for how this is done.
]]
local function commit_raycast_data(pnoun_unitid, raycast_simulation_data, pnoun_group, op)
    local curr_raycast_data = raycast_data[pnoun_unitid]
    local raycast_objects_by_tileid = raycast_simulation_data.raycast_objects_by_tileid
    local extradata = raycast_simulation_data.extradata
    local raycast_trace = raycast_simulation_data.raycast_trace

    for tileid in pairs(extradata.found_blocked_tiles) do
        set_blocked_tile(tileid)
    end

    for tileid in pairs(extradata.found_passed_tiles) do
        set_passed_tile(tileid)
    end

    raycast_trace_tracker:add_traces(raycast_trace)

    if pnoun_subrule_data.active_pnouns[pnoun_unitid] or pnoun_subrule_data.pnouns_in_conds[pnoun_unitid] then
        for these_unitid in pairs(extradata.found_ending_these_texts) do
            this_mod_globals.active_this_property_text[these_unitid] = true
        end

        if pnoun_subrule_data.pnouns_in_conds[pnoun_unitid] then
            local pnoun_cond_data = pnoun_subrule_data.pnouns_in_conds[pnoun_unitid]
            if pnoun_cond_data.condtype == "feeling" then
                local raycast_objects, found_letterwords = get_raycast_infix_units(pnoun_unitid, pnoun_cond_data.condtype)
                for _, raycast_object in ipairs(raycast_objects) do
                    local ray_unitid = utils.parse_object(raycast_object)
                    if ray_unitid > 2 then
                        this_mod_globals.active_this_property_text[ray_unitid] = true
                    end
                end
                for _, letterword in ipairs(found_letterwords) do
                    for _, letterunitid in ipairs(letterword[7]) do
                        this_mod_globals.active_this_property_text[letterunitid] = true
                    end
                end
            end
        end
    end

    local new_positions = {}

    -- Mark explicit block/pass/relay tiles
    local explicit_tile_func = Pnoun_Op_To_Explicit_Tile_Func[op]
    for tileid, ray_objects in pairs(curr_raycast_data.raycast_positions) do
        if explicit_tile_func ~= nil then
            local x, y = utils.coords_from_tileid(tileid)
            for _, ray_object in ipairs(ray_objects) do
                local ray_unitid = utils.parse_object(ray_object)
                
                local has_prop = false
                local has_not_prop = false
                if ray_unitid == 2 then
                    has_prop = hasfeature("empty", "is", op, 2, x, y)
                    has_not_prop = hasfeature("empty", "is", "not "..op, 2, x, y)
                else
                    local ray_unit = mmf.newObject(ray_unitid)
                    local ray_unit_name = ray_unit.strings[NAME]
                    if ray_unit.strings[UNITTYPE] == "text" then
                        ray_unit_name = "text"
                    end
                    has_prop = hasfeature(ray_unit_name, "is", op, ray_unitid)
                    has_not_prop = hasfeature(ray_unit_name, "is", "not "..op, ray_unitid)
                end
                
                if has_prop and not has_not_prop then
                    explicit_tile_func(tileid)
                    break
                end    
            end

        end
        -- Add/Update/Remove cursors based on how many raycast positions we found
        if not curr_raycast_data.cursors[tileid] then
            table.insert(new_positions, tileid)
        end
    end
    
    local tileids_to_delete = {}
    for tileid, cursor_unitid in pairs(curr_raycast_data.cursors) do
        if not curr_raycast_data.raycast_positions[tileid] then
            table.insert(tileids_to_delete, tileid)
            MF_cleanremove(cursor_unitid)
            -- @Note: apparently we have to delete then remake all cursors to avoid a visual glitch with multiple cursors.
            -- Reassigning cursor positions without deleting causes the visual glitch for which I have no idea why it happens
            -- It isn't a race condition with the "always" modhook. But ehh. Reinvestigate if we need to optimize.
        end
    end
    for _, tileid in ipairs(tileids_to_delete) do
        curr_raycast_data.cursors[tileid] = nil
    end
    for _, tileid in ipairs(new_positions) do
        local cursor_unitid = make_cursor()
        curr_raycast_data.cursors[tileid] = cursor_unitid
    end
end

-- Starting point where all pnoun processing is. This is like what grouprules() is to all group processing.
function do_subrule_pnouns()
    if run_extract_bpr_subrules then
        if THIS_LOGGING then
            print("running extract_bpr_subrules")
        end
        extract_bpr_subrules()
    end
    populate_inactive_pnouns()

    local raycast_settings = {
        checkblocked = true,
        checkrelay = true,
        checkpass = true,
    }
    local all_found_relay_indicators = {}
    local new_relay_indicators = {}
    local process_round = 0

    --[[ 
        When adding pnoun rules to be processed in scan_added_feature_for_pnoun_rule(), we group them into "pnoun groups". These groups
        represent different priorities in processing order of the pnoun rules, mainly based around block/pass/relay.
        For instance, "THIS is block" gets processed before "THIS is relay".

        We process each pnoun group in a set order, submitting any subrules into the featureindex while doing so. Then the next pnoun group
        will use the previously submitted rules when it gets processed. (See comments on each action of each step of the process)
    ]]
    for pnoun_group, data in ipairs(registered_pnoun_rules) do
        if THIS_LOGGING then
            print("------ Processing Pnoun Group "..pnoun_group.." ------")
        end
        local recorded_raycast_simulations = {}
        local added_features_in_this_group = true
        local added_features_during_last_op = true
        local op_round = 0

        while added_features_in_this_group do
            added_features_in_this_group = false
            op_round = op_round + 1

            if op_round >= 1000 then
                -- Wow. First time I found a reason to add a too complex case.
                destroylevel("toocomplex")
                return
            end

            for _, op in ipairs(Pnoun.Pnoun_Group_Lookup[pnoun_group].ops) do
                if THIS_LOGGING then
                    print("------ Doing operation "..op.." Round "..op_round.." ------")
                end

                process_round = process_round + 1

                -- Main action 1: Simulate a raycasting for every pnoun in the current group. Submit the raycast objects to be used in
                -- process_pnoun_features(), but DON'T commit the results yet.
                if added_features_during_last_op then
                    recorded_raycast_simulations = {}
                    for pnoun_unitid in pairs(data.pnoun_units) do
                        if THIS_LOGGING then
                            print("-> Updating raycast units of "..utils.unitstring(pnoun_unitid))
                        end

                        local curr_raycast_data = raycast_data[pnoun_unitid]

                        -- Added an assert here since there are many cases where this happens
                        plasma_utils.debug_assert(curr_raycast_data, "Getting raycast data failed for unitid: "..pnoun_unitid.." | unitstring: "..utils.unitstring(pnoun_unitid))

                        local raycast_objects_by_tileid, extradata, raycast_trace = simulate_raycast_with_pnoun(pnoun_unitid, raycast_settings)
                        recorded_raycast_simulations[pnoun_unitid] = {
                            raycast_objects_by_tileid = raycast_objects_by_tileid,
                            extradata = extradata,
                            raycast_trace = raycast_trace,
                        }

                        local raycast_objects = {}
                        local count = 0
                        for tileid, ray_objects in pairs(raycast_objects_by_tileid) do
                            for _, ray_object in ipairs(ray_objects) do
                                if not raycast_objects[ray_object] then
                                    raycast_objects[ray_object] = true
                                    count = count + 1
                                end
                            end
                        end

                        curr_raycast_data.raycast_objects = raycast_objects
                        curr_raycast_data.raycast_object_count = count
                        curr_raycast_data.raycast_positions = raycast_objects_by_tileid
                    end
                end

                if THIS_LOGGING then
                    print("-> Processing pnoun features: ")
                    for _, feature in ipairs(data.pnoun_features) do
                        print("- "..utils.serialize_feature(feature))
                    end
                    print("________________")
                end

                -- Main action 2: Evaluate and submit all pnoun features under this current pnoun group. Return a list of all pnouns that were
                -- processed as part of a sentence, and the remaining pnouns and features to process
                local prev_pnoun_feature_count = #data.pnoun_features
                local processed_pnoun_units, remaining_pnoun_features, redirected_pnouns = nil, nil, nil
                if pnoun_group ~= Pnoun.Groups.OTHER_INACTIVE then
                    local process_result = process_pnoun_features(data.pnoun_features, pnoun_subrule_data.pnoun_feature_extradata, Pnoun.Ops[op].filter_func, op, pnoun_group)
                    processed_pnoun_units = process_result[1]
                    remaining_pnoun_features = process_result[2]
                    redirected_pnouns = process_result[3]
                else
                    processed_pnoun_units = data.pnoun_units
                    remaining_pnoun_units = {}
                    remaining_pnoun_features = {}
                    remaining_pnoun_features = {}
                    redirected_pnouns = {}
                end

                -- Main action 2.5
                for pnoun_unit in pairs(redirected_pnouns) do
                    if data.pnoun_units[pnoun_unit] then
                        processed_pnoun_units[pnoun_unit] = true
                        pnoun_subrule_data.active_pnouns[pnoun_unit] = true
                        this_mod_globals.active_this_property_text[pnoun_unit] = true
                    end
                end

                -- Main action 3: Of the processed pnouns, commit their raycast simulation data to the system.
                -- The other pnouns that weren't processed will go to the next action
                for pnoun_unitid in pairs(processed_pnoun_units) do
                    if THIS_LOGGING then
                        print("-> Committing pnoun: "..utils.unitstring(pnoun_unitid))
                    end
                    local simulation_data = recorded_raycast_simulations[pnoun_unitid]
                    if simulation_data then
                        commit_raycast_data(pnoun_unitid, simulation_data, pnoun_group, op)
                        
                        for indicator_key, data in pairs(simulation_data.extradata.found_relay_indicators) do
                            all_found_relay_indicators[indicator_key] = true
                            if relay_indicators[indicator_key] == nil and new_relay_indicators[indicator_key] == nil then
                                new_relay_indicators[indicator_key] = make_relay_indicator(data.x, data.y, data.dir)
                            end
                        end
                    end
                    
                    pnoun_subrule_data.process_order[pnoun_unitid] = process_round
                end

                -- Main action 4: Of the non-processed pnouns, update the current group's set of pnoun units and features. These 
                for pnoun in pairs(processed_pnoun_units) do
                    data.pnoun_units[pnoun] = nil
                end
                data.pnoun_features = remaining_pnoun_features

                added_features_during_last_op = #remaining_pnoun_features < prev_pnoun_feature_count
                if added_features_during_last_op then
                    added_features_in_this_group = true
                end

                if THIS_LOGGING then
                    print("-> Processed pnoun units: ")
                    for pnoun_unit in pairs(processed_pnoun_units) do
                        print("- "..utils.unitstring(pnoun_unit))
                    end
                    print("________________")
                    print("-> Redirected pnoun units: ")
                    for pnoun_unit in pairs(redirected_pnouns) do
                        print("- "..utils.unitstring(pnoun_unit))
                    end
                    print("________________")
                    print("-> Remaining pnoun units: ")
                    for pnoun_unit in pairs(data.pnoun_units) do
                        print("- "..utils.unitstring(pnoun_unit))
                    end
                    print("________________")
                    print("-> Remaining pnoun features: ")
                    for _, feature in ipairs(remaining_pnoun_features) do
                        print("- "..utils.serialize_feature(feature))
                    end
                    print("________________")
                end
            end

            if not Pnoun.Pnoun_Group_Lookup[pnoun_group].repeat_until_no_more_processing then
                break
            end
        end

        -- If there are still features to process and pnoun units to update, add both of those to the redirected pnoun group (if defined)
        -- Otherwise, throw them away
        local redirected_pnoun_group = Pnoun.Pnoun_Group_Lookup[pnoun_group].redirect_pnoun_group
        if redirected_pnoun_group ~= nil then
            for _, pnoun_feature in ipairs(data.pnoun_features) do
                table.insert(registered_pnoun_rules[redirected_pnoun_group].pnoun_features, pnoun_feature)
            end
            for pnoun_unit in pairs(data.pnoun_units) do
                registered_pnoun_rules[redirected_pnoun_group].pnoun_units[pnoun_unit] = true
            end
        else
            if THIS_LOGGING then
                -- Purely for error checking purposes
                if #registered_pnoun_rules[pnoun_group].pnoun_features ~= 0 then
                    local err_str = "Reached end of processsing Pnoun Group "..tostring(pnoun_group).." but there are still features left that we are throwing out!\nList of features: "

                    local feature_list = {}
                    for _, feature in ipairs(registered_pnoun_rules[pnoun_group].pnoun_features) do
                        feature_list[#feature_list + 1] = utils.serialize_feature(feature)
                    end

                    print(err_str..table.concat(feature_list))
                end

                local discarded_pnoun_units = {}
                local err_str = "Reached end of processsing Pnoun Group "..tostring(pnoun_group).." but there are still pnoun units left that we are throwing out!\nList of pnoun units: "
                for pnoun_unit in pairs(registered_pnoun_rules[pnoun_group].pnoun_units) do
                    found_pnoun = true
                    for pnoun_unit in pairs(data.pnoun_units) do
                        discarded_pnoun_units[#discarded_pnoun_units + 1] = utils.unitstring(pnoun_unit)
                    end 
                end
                if #discarded_pnoun_units > 0 then
                    print(err_str..table.concat(discarded_pnoun_units))
                end
            end
        end
    end


    -- Updating the set of relay indicators
    for indicator_key, indicator in pairs(relay_indicators) do
        if not all_found_relay_indicators[indicator_key] then
            MF_cleanremove(indicator)
            relay_indicators[indicator_key] = nil
        end
    end
    for indicator_key, indicator in pairs(new_relay_indicators) do 
        relay_indicators[indicator_key] = indicator
    end
end