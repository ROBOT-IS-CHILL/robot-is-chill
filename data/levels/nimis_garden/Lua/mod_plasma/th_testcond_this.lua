local plasma_utils = PlasmaModules.load_module("general/utils")
local RaycastBank = PlasmaModules.load_module("this/raycast_bank")

function get_this_parms_in_conds(conds, ids)
    local id_index = 4 -- start at 4 since 1-3 ids is target, verb, property
    local conds_with_this_as_param = {} -- cond object -> {index -> unitid}

    if #conds > 0 then
        -- skip through all extraids (aka ands and nots and filler texts)
        while id_index <= #ids do
            local unit = mmf.newObject(ids[id_index][1])
            local type = unit.values[TYPE]

            if type ~= 4 and type ~= 6 and type ~= 11 then
                break
            end
            id_index = id_index + 1
        end

        for i, cond in ipairs(conds) do
            local condtype = plasma_utils.real_condtype(cond[1])
            local params = cond[2]
            
            if condtype == "never" then
                -- skip the special and technically-not-a-condition "never"
            elseif condtype == "this" or condtype == "not this" or condtype == "stable" then
                -- skip params if the condtype is "this", since the params are actually unitids
            else
                id_index = id_index + 1 -- consume the condition
                for i, param in ipairs(params) do
                    if string.sub(param, 1, 4) == "not " then
                        param = string.sub(param, 5)
                    end
                    local this_param_name, this_param_id = parse_this_param(param)
                    if this_param_name and not RaycastBank:is_valid_ray_id(this_param_id) then
                        local this_unitid = ids[id_index][1]
                        if not conds_with_this_as_param[cond] then
                            conds_with_this_as_param[cond] = {}
                        end
                        conds_with_this_as_param[cond][i] = this_unitid
                    end
                    id_index = id_index + 1
                end

                -- Special case when group is involved. The list of conditions is formed by concatinating the conds from "X is group" and "group is Y". However, the list of ids are formed by concatenating ids from the same sentences together
                -- Say we have "baba on rock is group" and "lonely group is push", the list of ids would look like {baba, is, group, on, rock, group, is, push, lonely }.
                -- In this case we skip over "group is push" to get to "lonely"
                if i < #conds and id_index <= #ids then
                    local u = mmf.newObject(ids[id_index][1])
                    if u and u.strings[NAME] == "group" then
                        id_index = id_index + 3 -- Consume the "group is X"
                        -- skip through all extraids (aka ands and nots and filler texts)
                        while id_index <= #ids do
                            local unit = mmf.newObject(ids[id_index][1])
                            local type = unit.values[TYPE]
    
                            if type ~= 4 and type ~= 6 and type ~= 11 then
                                break
                            end
                            id_index = id_index + 1
                        end
                    end
                end
            end
        end    
    end

    return conds_with_this_as_param
end

function parse_this_param(this_param)
    local this_param_name = ""
    if string.sub(this_param, 1, 4) == "not " then 
        this_param_name = "not "
        this_param = string.sub(this_param, 5, #this_param)
    end
    if not is_name_text_this(this_param) then 
        return nil, nil
    end
    local end_index = string.find(this_param, " ", 5)
    if not end_index then
        end_index = #this_param
        this_param_name = this_param_name..string.sub(this_param, 1, end_index) 
    else
        this_param_name = this_param_name..string.sub(this_param, 1, end_index-1) 
    end
    local param_id = string.sub(this_param, end_index + 1)

    return this_param_name, param_id
end


function parse_this_param_and_get_raycast_units(this_param)
    local this_param_name, param_id = parse_this_param(this_param)
    local this_unitid = parse_this_unit_from_param_id(param_id)
    if not this_unitid then
        return false, nil, nil, 0, nil
    end
    
    local raycast_objects, count = get_raycast_objects(this_unitid)            
    local tileids = get_raycast_tileid(this_unitid)
    local out = {}
    for ray_object in pairs(raycast_objects) do
        local ray_unit = plasma_utils.parse_object(ray_object)
        out[ray_unit] = true
    end

    return this_param_name, out, tileids, count, this_unitid
end

function parse_this_param_and_get_raycast_infix_units(this_param, infix)
    local this_param_name, param_id = parse_this_param(this_param)
    local this_unitid = parse_this_unit_from_param_id(param_id)
    if not this_unitid then
        return {}, {}
    end
    
    local raycast_objects, found_letterwords = get_raycast_infix_units(this_unitid, infix)
    return raycast_objects, found_letterwords
end

--[[ 
    This gets the unitid of the target/noun text that is stored in the rule.
    If the target word is formed by letters:
        - if include_letters == true then return a table of letter unitids
        - otherwise, return nil
]]
function get_target_unitid_from_rule(rule, include_letters)
    local tags = rule[4]
    if has_stable_tag(tags) then --@mods(stable)
        return nil
    end
    local rule_metadata = pf_rule_metadata_index:get_rule_metadata(rule[1])
    if rule_metadata == nil then return nil end
    return rule_metadata.target_unitid[1]
end

--[[ 
    This gets the unitid of the property text that is stored in the rule.
    If the property word is formed by letters:
        - if include_letters == true then return a table of letter unitids
        - otherwise, return nil
]]
function get_property_unitid_from_rule(rule, include_letters)
    local tags = rule[4]
    if has_stable_tag(tags) then
        return nil
    end

    local rule_metadata = pf_rule_metadata_index:get_rule_metadata(rule[1])
    if rule_metadata == nil then return nil end
    return rule_metadata.property_unitid[1]
end


--@TODO: might delete or refactor this later when we make THIS mod use values[ID] to represent the specific THIS text instead of unitids
function make_this_param(param_name, param_id)
    return param_name.." "..param_id
end

--[[ 
    Return a string representing the THIS text that can be used in parameters for rule conditions. Throws an error if the provided unitid isn't a THIS text
 ]]
function convert_this_unit_to_param_id(this_unitid)
    local this_unit = mmf.newObject(this_unitid)
    if not this_unit or not is_name_text_this(this_unit.strings[NAME]) then
        if this_unit then
            error("Provided unit id that points to invalid THIS text. unit name: "..this_unit.strings[NAME]..". Stack trace: "..debug.traceback())
        else
            error("Provided unit id that points to invalid THIS text. unit id: "..tostring(this_unitid)..". Stack trace: "..debug.traceback())
        end
    end
    return tostring(this_unit.values[ID])
end

--[[ 
    Return the unitid of a THIS text from the output of convert_this_unit_to_param_id(). Returns nil if this_param_id isn't a number. 
    If this_param_id is a stable_this_id, it returns the stable_this_id directly.
 ]]
function parse_this_unit_from_param_id(this_param_id)
    local this_unitid = tonumber(this_param_id)
    if not this_unitid then
        return nil
    end

    if not RaycastBank:is_valid_ray_id(this_unitid) then
        this_unitid = MF_getfixed(this_unitid)
    end

    return this_unitid
end