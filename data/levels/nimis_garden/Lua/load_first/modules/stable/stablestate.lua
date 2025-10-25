local utils = PlasmaModules.load_module("general/utils")
local RaycastBank = PlasmaModules.load_module("this/raycast_bank")

local StableUnit = {}
StableUnit.__index = StableUnit

local get_stablerule_display

function StableUnit:new(object, ruleids)
    local new_stableunit = {
        object = object,
        ruleids = ruleids,
    }
    setmetatable(new_stableunit, self)

    -- Making the StableUnit read-only
    local proxy = {}
    setmetatable(proxy, {
        __index = new_stableunit,
        __newindex = function()
            error("attempt to update a stableunit, which is read-only", 2)
        end
    })

    return proxy
end

--[[ 
    Some semantics/definitions:
    - ruleid - a string identifying a sentence in string form, regardless of what text unitids form it (i.e. "baba on keke is you", "hedge is stop", etc)
]]

local StableState = {
    objects = {},
    --[[ 
        object -> {
            ruleids: {
                ruleid: string -> {
                    stack_count: int
                }
            }
        }
     ]]

    rules = {},
    --[[ 
        ruleids: string -> {
            feature : featureindex item
            objects : (Set of objects)
            unit_count : int,
            display : string,
            stable_this_ids : [List of stable_this_ids],
            max_stack_count : int,
        }
     ]]

    feature_cache = {},
    --[[ 
        ruleid -> prototype feature

        This stores prototype features to make stablerules from when associating an object with a set of rules. The contents of the
        cache persist even after undo and are only reset when the level is restarted/exited/entered.
     ]]
}
StableState.__index = StableState

function StableState:new(logging)
    local o = {}
    setmetatable(o, self) 
    o:reset()
    o.logging = logging or false
    return o
end

function StableState:reset()
    self.objects = {}
    self.rules = {}
    self.feature_cache = {}
end

function StableState:add_object(object, features)
    utils.debug_assert(object)
    utils.debug_assert(features)
    
    if not self.objects[object] then
        local ruleids = {}
        local ruleid_to_feature = {}
        for _, feature in ipairs(features) do
            local dup_feature = utils.deep_copy_table(feature)
            local newconds, stable_this_ids = RaycastBank:evaluate_and_store_pnouns_in_conds(dup_feature[2])
            dup_feature[2] = newconds
            dup_feature.stable_this_ids = stable_this_ids

            local ruleid = utils.serialize_feature(dup_feature)

            if not ruleids[ruleid] then
                ruleids[ruleid] = {
                    stack_count = 1
                }
                ruleid_to_feature[ruleid] = dup_feature
            else
                ruleids[ruleid].stack_count = ruleids[ruleid].stack_count + 1
            end
        end

        local stableunit = StableUnit:new(object, ruleids)

        for ruleid, rule_data in pairs(stableunit.ruleids) do
            if not self.rules[ruleid] then
                -- Make a new stablerule to associate the object with.
                local prototype_feature = ruleid_to_feature[ruleid]
                if self.feature_cache[ruleid] == nil then
                    self.feature_cache[ruleid] = prototype_feature

                    if self.logging then
                        print("[StableState] adding prototype feature to cache: ", ruleid)
                    end
                else
                    prototype_feature = self.feature_cache[ruleid]
                    if self.logging then
                        print("[StableState] using prototype feature from cache: ", ruleid)
                    end
                end

                self:add_rule_with_object(ruleid, prototype_feature, object, rule_data)
            else
                -- There's an existing stablerule. Associate the object with the stablerule.
                self:link_rule_with_object(ruleid, object, rule_data)
            end
        end

        self.objects[object] = stableunit

        if self.logging then
            print("[StableState] Added new stableunit: ", object)
        end
        return true
    end
    return false
end

function StableState:remove_object(object, onundo_)
    local stableunit = self.objects[object]
    if stableunit then
        for ruleid, rule_data in pairs(stableunit.ruleids) do
            self:detach_rule_from_object(ruleid, object, onundo_)
        end

        self.objects[object] = nil

        if self.logging then
            print("[StableState] Removed stableunit: ", object)
        end
        return true
    end

    return false
end

function StableState:restore_stableunit(stableunit)
    utils.debug_assert(stableunit.object)
    utils.debug_assert(not self.objects[stableunit.object])
    utils.debug_assert(stableunit)

    if not self.objects[stableunit.object] then
        self.objects[stableunit.object] = stableunit

        for ruleid, rule_data in pairs(stableunit.ruleids) do
            if not self.rules[ruleid] then
                local prototype_feature = self.feature_cache[ruleid]
                utils.debug_assert(prototype_feature, "Prototype feature not found! "..ruleid)

                if self.logging then
                    print("[StableState] Restoring ruleid: ", ruleid)
                end

                self:add_rule_with_object(ruleid, prototype_feature, stableunit.object, rule_data)
            else
                -- There's an existing stablerule. Associate the object with the stablerule.
                self:link_rule_with_object(ruleid, stableunit.object, rule_data)
            end
        end

        return true
    end

    return false
end

function StableState:get_stableunit(object)
    if self.objects[object] then
        return self.objects[object]
    else
        return nil
    end
end

function StableState:print_stable_state(on_stable_undo)
    print("--------Stable State---------")
    print("===objects===")
    for object, v in pairs(self.objects) do
        local unitstring = ""
        if not on_stable_undo then
            unitstring = utils.objectstring(object)
        else
            unitstring = object
        end
        print("Object: "..unitstring)
        for ruleid, ruleid_data in pairs(v.ruleids) do
            print("\t"..ruleid, ", Stack Count: "..ruleid_data.stack_count)
        end
    end
    print("===stablerules===")
    for ruleid, v in pairs(self.rules) do
        print("{")
        print("ruleid = "..ruleid)
        print("unit_count = "..v.unit_count)
        print("feature: "..utils.serialize_feature(v.feature))
        -- print("max_stack_count: "..v.max_stack_count)
        print("}")
    end
    
    print("------------------------")
end

--[[ The below functions should not be called outside of StableState ]]

function StableState:add_rule_with_object(ruleid, feature, object, rule_data)
    local dup_feature = utils.deep_copy_table(feature)

    table.insert(dup_feature[2], {"stable", { ruleid }})

    dup_feature[3] = {}
    table.insert(dup_feature[4], "stable")

    self.rules[ruleid] = {
        feature = dup_feature,
        objects = {[object] = true},
        unit_count = 1,
        display = get_stablerule_display(feature),
        stable_this_ids = dup_feature.stable_this_ids,
        max_stack_count = rule_data.stack_count,
    }

    if self.logging then
        print("[StableState] Added new stablerule: "..ruleid.."\n\t\t...and linked it with object: "..object)
    end
end

function StableState:link_rule_with_object(ruleid, object, rule_data)
    utils.debug_assert(self.rules[ruleid], ruleid)
    utils.debug_assert(object, object)

    self.rules[ruleid].objects[object] = true
    self.rules[ruleid].unit_count = self.rules[ruleid].unit_count + 1
    self.rules[ruleid].max_stack_count = math.max(self.rules[ruleid].max_stack_count, rule_data.stack_count)

    if self.logging then
        print("[StableState] Linked stablerule: "..ruleid.."\n\t...with object: "..object)
    end
end

function StableState:detach_rule_from_object(ruleid, object, onundo_)
    local stablerule = self.rules[ruleid]
    local onundo = onundo_ or false
    utils.debug_assert(stablerule)
    utils.debug_assert(stablerule.objects[object])

    stablerule.unit_count = stablerule.unit_count - 1
    if stablerule.unit_count == 0 then
        if onundo then
            for _, stable_this_id in ipairs(stablerule.stable_this_ids) do
                RaycastBank:revoke_ray_id(stable_this_id)
            end
        end
        self.rules[ruleid] = nil

        if self.logging then
            print("[StableState] Removed stablerule: "..ruleid.."\n\t...in the process of detaching it from object: "..object)
        end
    else
        stablerule.objects[object] = nil

        local new_max_stack = 0
        for object, _ in pairs(stablerule.objects) do
            local rule_data = self.objects[object].ruleids[ruleid]
            new_max_stack = math.max(new_max_stack, rule_data.stack_count)
        end

        stablerule.max_stack_count = new_max_stack

        if self.logging then
            print("[StableState] Detached stablerule: "..ruleid.."\n\t...from object: "..object)
        end
    end
end

function StableState:register_this_text_in_stablerule(this_unitid)
    local raycast_objects, ray_count = get_raycast_objects(this_unitid)
    local raycast_tileids = get_raycast_tileid(this_unitid)

    local stable_this_id = RaycastBank:register_ray_objects(raycast_objects, raycast_tileids, ray_count)

    if self.logging then
        print("[StableState] Registering this text in stablerule: "..utils.unitstring(this_unitid).."\n\t...With stable this ID: ", stable_this_id)
    end

    return stable_this_id
end

-- local
function get_stablerule_display(feature)
    local custom = MF_read("level","general","customruleword")

    local text = ""
   
    local rule = feature[1]
    if (#custom == 0) then
        text = text .. rule[1] .. " "
    else
        text = text .. custom .. " "
    end

    local conds = feature[2]
    local ids = feature[3]
    local tags = feature[4]

    local has_base_tag = false
    local has_mimic_tag = false
    for _,b in ipairs(tags) do
        if (b == "mimic") then
            has_mimic_tag = true
        elseif (b == "base") then
            has_base_tag = true
        end
    end

    if #ids == 0 and not has_base_tag then
        if (#custom == 0) then
            return rule[1].." "..rule[2].." "..rule[3]
        else
            return custom.." "..custom.." "..custom 
        end
    end

    if (#conds > 0) then
        local handling_or = false -- Adding logic to ignore all conditions that are in an "or" clause (aka, between parenthesis)
        for a,cond in ipairs(conds) do
            local condtype = cond[1]

            -- invertconds() can directly modify the feature's conds. But it tends to add new conds with
            -- parenthesis around the condtype (EX: "(not near)", "(on)" ). So to show the original rule,
            -- ignore the conds with parenthesis
            if string.find(condtype, "%(") then
                handling_or = true
            end

            if not (condtype == "this" or condtype == "not this" or condtype == "never" or handling_or) then
                local middlecond = true
                
                if (cond[2] == nil) or ((cond[2] ~= nil) and (#cond[2] == 0)) then
                    middlecond = false
                end
                
                if middlecond then
                    if (#custom == 0) then
                        text = text .. condtype .. " "
                    else
                        text = text .. custom .. " "
                    end
                    
                    if (cond[2] ~= nil) then
                        if (#cond[2] > 0) then
                            for c,d in ipairs(cond[2]) do
                                local this_param_name,_,_,_,this_unitid = parse_this_param_and_get_raycast_units(d)
                                if this_param_name then
                                    text = text .. this_param_name.." "

                                    local names = {}
                                    local raycast_objects, raycast_count = get_raycast_objects(this_unitid)
                                    for ray_object in pairs(raycast_objects) do
                                        local ray_unitid = utils.parse_object(ray_object)
                                        if ray_unitid == 2 then
                                            names["empty"] = true
                                        else
                                            local ray_unit = mmf.newObject(ray_unitid)
                                            names[ray_unit.strings[NAME]] = true
                                        end
                                    end

                                    if raycast_count > 0 then
                                        text = text.."("
                                        local first = true
                                        for name, _ in pairs(names) do
                                            if not first then
                                                text = text.." or "
                                            end
                                            first = false
                                            text = text..name
                                        end
                                        text = text..") "
                                    end
                                else
                                    if (#custom == 0) then
                                        text = text .. d .. " "
                                    else
                                        text = text .. custom .. " "
                                    end
                                end
                                    
                                if (#cond[2] > 1) and (c ~= #cond[2]) then
                                    text = text .. "& "
                                end
                            end
                        end
                    end
                    
                    if (a < #conds) then
                        text = text .. "& "
                    end
                else
                    if (#custom == 0) then
                        text = condtype .. " " .. text
                    else
                        text = custom .. " " .. text
                    end
                end
            end

            if string.find(condtype, "%)") then
                handling_or = false
            end
        end
    end
    
    local target = rule[3]
    local isnot = string.sub(target, 1, 4)
    local target_ = target
    
    if (isnot == "not ") then
        target_ = string.sub(target, 5)
    else
        isnot = ""
    end
    
    if (word_names[target_] ~= nil) then
        target = isnot .. word_names[target_]
    end
    
    if (#custom == 0) then
        text = text .. rule[2] .. " " .. target
    else
        text = text .. custom .. " " .. custom
    end
    
    if has_mimic_tag then
        text = text .. " (mimic)"
    end

    return text
end

return StableState