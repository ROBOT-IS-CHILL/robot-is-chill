table.insert(editor_objlist_order, "text_guard")

editor_objlist["text_guard"] = 
{
	name = "text_guard",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = {2, 1},
    colour_active = {2, 2},
}

formatobjlist()

local COND_CATEGORIES = {
    existence = { -- if these conditions exist, check when the appropriate unit exists (addundo({"create"}) or addundo({"delete"}))
        without= {check = "condparam"},
    },
    features = { -- if you have these conditions, check when featureindex gets updated ("rule_update_after" modhook)
        powered=true, -- note: special case for powered since there are now multiple powers and the condtype is "powered<some string>". So we need a special case to handle that
        feeling=true
    },
    update = { -- if these conditions exist, check on addundo({"update"})
        lonely= {check= "target"},
        this=true,
        stable=true   
    },
    onturnstart = { -- if these conditions exist, always recalculate guard units at command_given
        seldom=true,
        often=true,
        idle=true
    },
    ignore = { -- exclude these conditions when considering to recalculate guard units 
        never=true
    },

    --[[ 
        - If #conds == 0, then check based on addundo({"create"}) or addundo({"delete"})
        - If #conds > 0, but it only consists of conds in the "ignore" category, do not recalculate guards at all
        - If a condtype is not found here, but has parameters, check based on addundo({"update"})
        - Otherwise, always recalculate guards on every guard checkpoint

        - if at any point featureindex gets updated, then recalculate guards. (This will make the "features" category seem obsolete, but it still counts as a nontrivial condition)
    ]]
}

GLOBAL_disable_guard_checking = false

local guard_relation_map = {} -- <guardee name> -> list of unitids to destroy if a unit named <guardee name> is about to be destroyed
local processed_destroyed_units = {} -- list of objects which we already handled delete()-ing of, whether normally or through guards
local units_to_guard_destroy = {} -- list of objects that we marked for guard destroys on handle_guard_dels()
local units_to_save = {} -- list of objects that we marked to set deleted[unitid] = nil on between guard checkpoints
local update_guards = false -- when set to true during a turn, guard_checkpoint() calls recalculate_guards().
local too_complex_guard = false
local guard_update_criteria = {}

-- List of objects that were saved by a guard unit during a turn. Saved units cannot be normal destroyed until the end of the turn.
-- This implements what I call the "pin cushion effect", where a guard unit would take the blow for all direct hits.
-- Note: Saved units can still be destroyed from guarding other units
local all_saved_units = {}

local utils = PlasmaModules.load_module("general/utils")
local PlasmaSettings = PlasmaModules.load_module("general/gui")

local enable_guard_chaining = not PlasmaSettings.get_toggle_setting("disable_guard_chain") 

local GUARD_LOGGING = false
local GUARD_ALG_LOGGING = false
local GUARD_CHECK_LOGGING = false

function clear_guard_mod()
    guard_relation_map = {}
    processed_destroyed_units = {}
    units_to_guard_destroy = {}
    units_to_save = {}
    guard_update_criteria = {}
    update_guards = false
    too_complex_guard = false
    GLOBAL_disable_guard_checking = false
end

function is_unit_guarded(unitid)
    local unitname = nil
    if unitid == 1 then
        unitname = "level"
    elseif unitid == 2 then
        unitname = "empty"
    else
        local unit = mmf.newObject(unitid)
        unitname = getname(unit)
    end
    return is_name_guarded(unitname)
end

function is_unit_saved_by_guard(unitid, x, y)
    local object = utils.make_object(unitid, x, y)
    return units_to_save[object] ~= nil
end

function is_name_guarded(name)
    return guard_relation_map[name] ~= nil
end

local function get_guard_units(name)
    return guard_relation_map[name]
end

table.insert(mod_hook_functions["level_start"],
    function()
        clear_guard_mod()
        enable_guard_chaining = not PlasmaSettings.get_toggle_setting("disable_guard_chain")

        update_guards = true -- On start, set up guard_relation_map
        guard_checkpoint("level_start")
    end
)

table.insert(mod_hook_functions["command_given"],
    function()
        if GUARD_LOGGING then
            print("========GUARD START=========")
        end
        -- clear_guard_mod()
        guard_checkpoint("command_given")
        all_saved_units = {}
    end
)

table.insert(mod_hook_functions["turn_end"],
    function()
        guard_checkpoint("turn_end")
        if GUARD_LOGGING then
            print("========GUARD END=========")
        end
    end
)

table.insert(mod_hook_functions["rule_update_after"],
    function()
        update_guards = true
        if GUARD_CHECK_LOGGING then
            print(">>> setting update_guards to true from rule_update_after")
        end
    end
)
table.insert(mod_hook_functions["undoed_after"],
    function()
        update_guards = true
        if GUARD_CHECK_LOGGING then
            print(">>> setting update_guards to true from undo")
        end
    end
)

-- A special list of functions to forcibly ignore guard logic, due to special cases
local funcs_to_ignore_guard_units = {
    [destroylevel_do] = true,
    [createall] = true,
    [ending] = true
}
-- Called on delete(). Returns true if the about-to-be-deleted unit is guarded. 
-- Used for proceeding with the regular delete() logic if the unit isn't guarded.
function handle_guard_delete_call(unitid, x, y, caller_func)
    local object = utils.make_object(unitid, x, y)
    -- Neat trick to figure out calling function. Can't use debug.getinfo(2).name since it returns nil.
    if funcs_to_ignore_guard_units[caller_func] then
        processed_destroyed_units[object] = true
        return false
    end
    
    local is_guarded = ack_endangered_unit(object)
    if is_guarded then
        if GUARD_LOGGING then
            print("Endangered unit is guarded: "..utils.objectstring(object))
        end
        return true
    elseif processed_destroyed_units[object] then
        if GUARD_LOGGING then
            print("handle_guard_delete_call: Already destroyed "..utils.objectstring(object))
        end
        return true
    else
        if GUARD_LOGGING then
            print("Normal destroy "..utils.objectstring(object))
        end
        processed_destroyed_units[object] = true
        return false
    end
end

function ack_endangered_unit(object)
    if all_saved_units[object] then
        if GUARD_LOGGING then
            print("Endangered unit is already saved: ", utils.objectstring(object))
        end 
        return true
    end
    local unitid, x, y = utils.parse_object(object)
    local unitname = nil
    if unitid == 1 then
        unitname = "level"
    elseif unitid == 2 then
        unitname = "empty"
    else
        local unit = mmf.newObject(unitid)
        unitname = getname(unit)
    end
    if is_name_guarded(unitname) then
        for unitid, _ in pairs(get_guard_units(unitname)) do
            units_to_guard_destroy[unitid] = true
            if GUARD_LOGGING then
                print("Marking guard unit to destroy: ", utils.unitstring(unitid))
            end
        end
        units_to_save[object] = true
        if GUARD_LOGGING then
            print("Marking guard unit to save: ", utils.objectstring(object))
        end
        return true
    else
        return false
    end
end

-- Destroys all marked objects from units_to_guard_destroy, if not already deleted
local function handle_guard_dels()
    for saved_object, _ in pairs(units_to_save) do
        local unitid, x, y = utils.parse_object(saved_object)
        if unitid ~= 1 then
            local deleted_unitid = utils.get_deleted_unitid_key(saved_object)
            deleted[deleted_unitid] = nil
        end

        all_saved_units[saved_object] = true
    end

    GLOBAL_disable_guard_checking = true
    for guard, _ in pairs(units_to_guard_destroy) do
        if not processed_destroyed_units[guard] then
            local unitid, x, y = utils.parse_object(guard)
            if unitid == 1 then
                if not issafe(unitid) then
                    destroylevel()
                end
            else
                if GUARD_LOGGING then
                    print("- Destroying unit: ", utils.objectstring(guard))
                end

                local unit = mmf.newObject(unitid)
        
                if not units_to_save[guard] then
                    local pmult,sound = checkeffecthistory("defeat")
                    MF_particles("destroy", x, y, 5 * pmult, 0, 3, 1, 1)
                    setsoundname("removal", 1, sound)
                end
        
                if not issafe(unitid) then
                    local deleted_unitid = utils.get_deleted_unitid_key(guard)
                    deleted[deleted_unitid] = nil
                    delete(unitid, x, y, nil, nil, true)
                end
            end

            processed_destroyed_units[guard] = true
        else
            if GUARD_LOGGING then
                print("- Already destroyed unit: ", guard)
            end
        end
    end
    GLOBAL_disable_guard_checking = false
    units_to_save = {}
    processed_destroyed_units = {}
    units_to_guard_destroy = {}
end

local function get_table_value(table, key)
    if not table[key] then
        table[key] = {}
    end
    return table[key]
end

local function make_typedata(feature)
    return {feature[1][1], feature[2]}
end

local function found_units_for_feature(feature)
    local name = feature[1][1]
    local conds = feature[2]
    local typedata = {name, conds}
    
    local found_units = #findall(typedata, false, true) > 0
    if not found_units then
        if name == "level" then
            found_units = testcond(conds, 1)
        elseif name == "empty" then
            found_units = #findempty(conds, true) > 0
        end
    end

    return found_units
end

local function found_units_for_typedata(typedata)
    local guard_name = typedata[1]
    local conds = typedata[2]
    
    GLOBAL_checking_stable = true
    local found_units = false
    if guard_name == "empty" then
        found_units = #findempty(conds, true) > 0
    else
        found_units = #findall(typedata, false, true) > 0
        if not found_units then
            if guard_name == "level" then
                found_units = testcond(conds, 1)
            end
        end
    end
    GLOBAL_checking_stable = false

    return found_units
end

local function print_scc(scc)
    local v = {}
    for _, vertex in ipairs(scc) do
        table.insert(v, vertex.typedata_hash)
    end

    return string.format("{%s}", table.concat(v, " | "))
end

local function evaluate_typedata_for_update_criteria(typedata)
    local name = typedata[1]
    local conds = typedata[2]

    local s = get_table_value(guard_update_criteria, "create")
    s[name] = true
    s = get_table_value(guard_update_criteria, "remove")
    s[name] = true

    local has_valid_conds = true
    for _, cond in ipairs(conds) do
        local condtype = utils.real_condtype(cond[1])
        local params = cond[2]

        if not COND_CATEGORIES.ignore[condtype] then
            has_valid_conds = true

            if COND_CATEGORIES.onturnstart[condtype] then
                guard_update_criteria["onturnstart"] = true
            elseif COND_CATEGORIES.features[condtype] then
                guard_update_criteria["features"] = true
            elseif COND_CATEGORIES.existence[condtype] then
                local data = COND_CATEGORIES.existence[condtype]
                if type(data) == "table" and data.check == "condparam" then
                    for _, check_name in ipairs(params) do
                        local s = get_table_value(guard_update_criteria, "create")
                        s[check_name] = true
                        s = get_table_value(guard_update_criteria, "remove")
                        s[check_name] = true
                    end
                end
            elseif COND_CATEGORIES.update[condtype] then
                local data = COND_CATEGORIES.existence[condtype]
                if type(data) == "table" and data.check == "target" then
                    local s = get_table_value(guard_update_criteria, "update")
                    s[name] = true
                end
            elseif #params > 0 then
                local create_criteria = get_table_value(guard_update_criteria, "create")
                local remove_criteria = get_table_value(guard_update_criteria, "remove")
                local update_criteria = get_table_value(guard_update_criteria, "update")

                -- create_criteria[name] = true
                -- remove_criteria[name] = true
                update_criteria[name] = true

                for _, param in ipairs(params) do
                    create_criteria[param] = true
                    remove_criteria[param] = true
                    update_criteria[param] = true
                end
            else
                guard_update_criteria["always"] = true
            end
        end
    end
end

function check_undo_data_for_updating_guards(line)
    if update_guards then
        return
    end
    
    local style = line[1]
    if style == "create" or style == "remove" or style == "update" then
        local changed_name = line[2]
        if guard_update_criteria[style] and guard_update_criteria[style][changed_name] then
            update_guards = true
        end
    end

    if GUARD_CHECK_LOGGING and update_guards then
        print(">>> setting update_guards to true from "..style)
    end
end


--[[ 
    vertex = represents each unique typedata
    scc = list of vertices
]]

-- Following Tarjan's algorithm for generating SCC https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
-- Note that the algorithm doesn't do anything to link each scc together in a super graph. This is what the "predecessor" field is for.
local function strongconnect(vertex, data)
    vertex.index = data.index
    vertex.lowlink = data.index
    data.index = data.index + 1

    table.insert(data.stack, vertex)
    vertex.on_stack = true

    local curr_guard_name = vertex.typedata[1]

    if GUARD_ALG_LOGGING then
        print("current guard "..curr_guard_name)
    end

    local guardee_vertices = data.guard_graph[curr_guard_name] or {}
    for typedata_hash, next_vertex in pairs(guardee_vertices) do
        utils.debug_assert(next_vertex, curr_guard_name)

        if next_vertex.index == nil then
            next_vertex.predecessor = vertex
            strongconnect(next_vertex, data)
            vertex.lowlink = math.min(vertex.lowlink, next_vertex.lowlink)
        elseif next_vertex.on_stack then
            vertex.lowlink = math.min(vertex.lowlink, next_vertex.index)
        elseif next_vertex.containing_scc ~= nil then
            -- If we already created the scc that the next vertex belongs to, note the vertex -> scc edge in scc_graph
            local scc_list = get_table_value(data.scc_graph, vertex)

            if not scc_list[next_vertex.containing_scc] then
                scc_list[next_vertex.containing_scc] = true

                if GUARD_LOGGING then
                    print(string.format("%s -> %s (Copy)", vertex.typedata_hash, print_scc(next_vertex.containing_scc)))
                end
            end
        end
    end

    if (vertex.lowlink == vertex.index) then
        local predecessor = vertex.predecessor or data.start_vertex
        local pred_scc_list = get_table_value(data.scc_graph, predecessor)

        -- Make a new scc
        local new_scc = {}
        local stack_vertex = vertex
        repeat
            stack_vertex = table.remove(data.stack)
            stack_vertex.on_stack = false
            table.insert(new_scc, stack_vertex)

            stack_vertex.containing_scc = new_scc -- Make all vertices point to the scc that contains the vertices
        until stack_vertex == vertex

        -- Add the vertex -> scc link, aka the predecessor -> new scc
        pred_scc_list[new_scc] = true

        if GUARD_LOGGING then
            print(string.format("%s -> %s", predecessor.typedata_hash, print_scc(new_scc)))
        end
    end
end

local function recalculate_guards()
    if GUARD_LOGGING then
        print("- Recalculating guards")
    end
    guard_relation_map = {}
    guard_update_criteria = {}

    --[[ 
        Recalculating guards is divided into two phases:
        - Phase 1: 
            - build a guard graph of guard relations based on all guard features.
            - Then run Tarjan's algorithm to generate a super graph of Strongly Connected Components (SCC).
                - SCCs are subgraphs where all vertices can be accessed from any other vertex in the graph.
                - By building a super graph of SCCs, where each SCC can be thought as its own vertex, it simplifies the original guard graph to eliminate all loops
        - Phase 2:
            - Run a DFS traversal on the super graph to apply guard chaining logic and calculate guard units 
    ]]

    --[[ Start of Phase 1 ]]

    local guard_graph = {} -- guardee name -> list of vetices. An adjacency list that provides each traversal option in terms of the guardee name
    local guardee_vertices = {} -- guardee name -> vertex. The mapped vertex represents the starting point of iteration for each guardee name in the guard graph
    local all_vertices = {} -- typedata_hash -> vertex. This tracks each uniquely created vertex in the guard graph. Its mainly used to avoid duplications based on the typedata_hash

    -- Go through all guard features and populate the guard_graph and guardee_vertices
    if featureindex["guard"] ~= nil then
        for _, feature in ipairs(featureindex["guard"]) do
            if feature[1][3] ~= "all" and feature[1][1] ~= "all" then
                local guardee_name = feature[1][3]

                if string.sub(guardee_name, 1, 4) ~= "not " then

                    -- Check that a unit with the guardee name exists before checking the guard typedata
                    local check_guard_typedata = false
                    if guardee_vertices[guardee_name] then
                        check_guard_typedata = true
                    else
                        local guardee_typedata = {guardee_name, {}}
                        evaluate_typedata_for_update_criteria(guardee_typedata)

                        if found_units_for_typedata(guardee_typedata) then
                            check_guard_typedata = true
                            
                            -- Make a "guardee" vertex. This represents a starting point for getting all guard units of a guardee
                            local guardee_typedata_hash = "[start] "..utils.serialize_typedata(guardee_typedata)
                            local name_vertex = {
                                -- Tarjan's algorithm fields
                                index = nil,
                                lowlink = nil,
                                on_stack = false,
                
                                -- Other fields
                                typedata = guardee_typedata,
                                typedata_hash = guardee_typedata_hash,
                                predecessor = nil, -- Will point to the previous vertex used to go to the current vertex. Used for linking each scc in the scc_graph
                                containing_scc = nil, -- Will point to the scc that contains this vertex.
                            }
                            guardee_vertices[guardee_name] = name_vertex
                        end
                    end

                    if check_guard_typedata then
                        local typedata = make_typedata(feature)
                        local typedata_hash = utils.serialize_typedata(typedata)
                        evaluate_typedata_for_update_criteria(typedata)
                        
                        local found_units = found_units_for_typedata(typedata)
                        if found_units then
                            local vertexlist = get_table_value(guard_graph, guardee_name)

                            if all_vertices[typedata_hash] == nil then
                                local new_vertex = {
                                    -- Tarjan's algorithm fields
                                    index = nil,
                                    lowlink = nil,
                                    on_stack = false,

                                    -- Other fields
                                    typedata = typedata,
                                    typedata_hash = typedata_hash,
                                    predecessor = nil, -- Will point to the previous vertex used to go to the current vertex. Used for linking each scc in the scc_graph
                                    containing_scc = nil -- Will point to the scc that contains this vertex.
                                }
                                
                                vertexlist[typedata_hash] = new_vertex
                                all_vertices[typedata_hash] = new_vertex
        
                                if GUARD_ALG_LOGGING then
                                    print("Adding vertex with typedata: "..typedata_hash.." for guardee "..guardee_name)
                                end
                            else
                                if GUARD_ALG_LOGGING then
                                    print("Copying vertex with typedata: "..typedata_hash.." for guardee "..guardee_name)
                                end
                                vertexlist[typedata_hash] = all_vertices[typedata_hash]
                            end    
                        end
                    end
                end
            end
        end
    end

    -- Persistent data used for running Tarjan's algorithm
    local scc_data = {
        -- Tarjan's algorithm fields
        stack = {},
        index = 0,
        
        -- Other fields
        guard_graph = guard_graph, -- Save reference to the guard graph

        -- vertex -> scc. This gets populated with all edges from a vertex (V) to an scc (S) such that V is not in S.
        -- Used for traversing through a super graph that only consists of sccs from guard_graph.
        scc_graph = {},

        -- A dummy vertex used when the current vertex does not have a predecessor. Represents the singular starting point for traversal
        start_vertex = {
            typedata = {"start_vertex", {}},
            typedata_hash = "start_vertex",
            
            index = nil,
            lowlink = nil,
            on_stack = false,
            predecessor = nil,
        },
    }

    -- Run Tarjan's algorithm
    for _, guardee_vertex in pairs(guardee_vertices) do
        strongconnect(guardee_vertex, scc_data)
    end

    if GUARD_LOGGING or GUARD_ALG_LOGGING then
        print("-----------------")
    end

    --[[ Start of Phase 2 ]]
    --[[ 
        The next section is several DFS traversals on the scc_graph, one traversal for each guardee name to resolve.
        This is where we calculate the final set of guard units for each guardee.
     ]]
    
    -- typedata_hash -> list of units. Optimization: this saves the list of guard units calculated from a given typedata.
    -- Used for avoiding redundant calculations when processing duplicate typedata
    local calculated_guard_units = {}
    
    for guardee_name, _ in pairs(guardee_vertices) do
        if GUARD_ALG_LOGGING then
            print("resolving guardee name "..guardee_name)
        end
        local starting_vertices = guard_graph[guardee_name] or {}
        
        local stack = {}
        local guardee_sccs = {}
        -- Initially populate the stack with all unique sccs that contain the guardee
        for typedata_hash, vertex in pairs(starting_vertices) do
            if not guardee_sccs[vertex.containing_scc] then
                guardee_sccs[vertex.containing_scc] = true
                table.insert(stack, vertex.containing_scc)
            end
        end

        local final_guard_units = {}
        local found_guard = false
        while #stack > 0 do
            local curr_scc = table.remove(stack)

            if GUARD_ALG_LOGGING then
                print("Processing scc with verticies: "..print_scc(curr_scc))
            end
            
            -- Find any sccs branching off from this current scc and add to the stack.
            local get_guard_units = true

            if enable_guard_chaining then
                local sccs_to_insert = {}
                for _, vertex in ipairs(curr_scc) do
                    local successor_sccs = scc_data.scc_graph[vertex] or {}
                    for next_scc, _ in pairs(successor_sccs) do
                        if next_scc ~= curr_scc then
                            sccs_to_insert[next_scc] = true
                            get_guard_units = false
                        end
                    end
                end
                for scc, _ in pairs(sccs_to_insert) do
                    table.insert(stack, scc)

                    if GUARD_ALG_LOGGING then
                        print("Adding to stack scc with verticies: "..print_scc(curr_scc))
                    end
                end
            end
            

            if get_guard_units then
                -- At this point, we've determined that the current scc has no other sccs to branch to. So add the guard units of the scc to the final guard list
                for _, curr_vertex in ipairs(curr_scc) do
                    local found_guards = {}
                    if calculated_guard_units[curr_vertex.typedata_hash] then
                        found_guards = calculated_guard_units[curr_vertex.typedata_hash]

                        if GUARD_ALG_LOGGING then
                            print("using calculated guards for "..curr_vertex.typedata_hash)
                        end
                    else
                        if GUARD_ALG_LOGGING then
                            print("calculating guards for "..curr_vertex.typedata_hash)
                        end
                        for _, unitid in ipairs(findall(curr_vertex.typedata, false, false)) do
                            found_guard = true
                            local object = utils.make_object(unitid)
                            found_guards[object] = true
                        end

                        local curr_guard_name = curr_vertex.typedata[1]
                        local conds = curr_vertex.typedata[2]
                        if curr_guard_name == "empty" then
                            for _, tileid in ipairs(findempty(conds, false)) do
                                found_guard = true

                                local x = tileid % roomsizex
                                local y = math.floor(tileid / roomsizex)
                                local empty_object = utils.make_object(2, x, y)
                                found_guards[empty_object] = true
                            end
                        elseif curr_guard_name == "level" then
                            if testcond(conds, 1) then
                                found_guard = true

                                local level_object = utils.make_object(1)
                                found_guards[level_object] = true
                            end
                        end

                        calculated_guard_units[curr_vertex.typedata_hash] = found_guards
                    end
                    for guard_unit, _ in pairs(found_guards) do
                        if GUARD_LOGGING and not final_guard_units[guard_unit] then
                            print(string.format("%s => %s", guardee_name, utils.objectstring(guard_unit)))
                        end

                        final_guard_units[guard_unit] = true 
                        found_guard = true
                    end
                end
            end
        end

        if found_guard then
            guard_relation_map[guardee_name] = final_guard_units
        end
    end
end

-- The main entrypoint for guard logic. This gets called from: start and end of turn, after code(), after handledels(), and after levelblock()
function guard_checkpoint(calling_func)
    if not too_complex_guard then
        if GUARD_CHECK_LOGGING then
            print(string.format("> guard_checkpoint from %s", calling_func))
        end
        handle_guard_dels()

        if guard_update_criteria["always"] then
            if GUARD_CHECK_LOGGING then
                print(">>> setting update_guards to true from always")
            end
            update_guards = true
        elseif (calling_func == "command_given" and guard_update_criteria["onturnstart"]) then
            if GUARD_CHECK_LOGGING then
                print(">>> setting update_guards to true from onturnstart")
            end
            update_guards = true
        end
        
        if update_guards then
            recalculate_guards()
            update_guards = false
        end
    end
end