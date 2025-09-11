table.insert(editor_objlist_order, "text_stable")

editor_objlist["text_stable"] = 
{
	name = "text_stable",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {3, 2},
    colour_active = {3, 3},
}

formatobjlist()

-- When a stableunit gets deleted, queue its su_key to be deleted in the next call to update_stable_state
-- @TODO: would it be better if the su_key was deleted instantly?
local queued_deleted_su_keys = {}
    
local on_stable_undo = false

local STABLE_LOGGING = false

local utils = PlasmaModules.load_module("general/utils")
local PlasmaSettings = PlasmaModules.load_module("general/gui")
local StableState = PlasmaModules.load_module("stable/stablestate")
local StableDisplay = PlasmaModules.load_module("stable/stabledisplay")
local enable_stablerule_display_setting = not PlasmaSettings.get_toggle_setting("disable_stable_display")

local stablestate = StableState:new(STABLE_LOGGING)
local stabledisplay = StableDisplay:new(stablestate)

GLOBAL_disable_stablerule_update = false
GLOBAL_checking_stable = false -- set to true whenever we are finding "X is stable" rules. This is to indirectly tell testcond()

function clear_stable_mod()
    MF_letterclear("stablerules")

    stablestate:reset()
    stabledisplay:reset()

    queued_deleted_su_keys = {}
end

function on_add_stableunit(unitid)
    local object = utils.make_object(unitid)
    if stablestate.objects[object] then
        stabledisplay:add_stableunit(object)
    end
end

function on_delete_stableunit(unitid)
    local object = utils.make_object(unitid)

    if not on_stable_undo and stablestate.objects[object] then
        table.insert(queued_deleted_su_keys, object)
    end
end

--@mods(stable x persist)
function get_persist_stablestate_info(unitid)
    local object = utils.make_object(unitid)
    local stableunit = stablestate:get_stableunit(object)

    if not stableunit then
        return nil
    end
    
    local prototype_features = {}
    for ruleid, rule_data in pairs(stableunit.ruleids) do
        for i = 1, rule_data.stack_count do
            table.insert(prototype_features, stablestate.feature_cache[ruleid])
        end
    end

    return {
        stableunit = stableunit,
        prototype_features = prototype_features
    }
end

function apply_persist_stablestate_info(unitid, persist_stable_info)
    print("apply_persist_stablestate_info")
    local object = utils.make_object(unitid)
    local stableunit = persist_stable_info.stableunit
    local prototype_features = persist_stable_info.prototype_features

    stablestate:remove_object(object)
    local added = stablestate:add_object(object, prototype_features)
    if added then
        print("Added persist stablerule to object: ", utils.objectstring(object))
        stabledisplay:add_stableunit(object)
    end
    -- utils.debug_assert(added, "object was not added!"..utils.objectstring(object))
end

function has_stable_tag(tags)
    for i, tag in ipairs(tags) do
        if tag == "stable" then
            return true
        end
    end
    return false
end

--[[ 
    Given a name of a unit, go through the featureindex to determine which features are qualified to
    be stablerules. All features that target the given name are qualified unless it falls under one these rules:
        - The feature has the "stable" tag, meaning it is a stablerule
        - The feature is in the form of "X is stable"
        - The feature is in the form of "X is THIS"
]]
local function get_stablerules_from_name(name)
    local stable_features = {}
    if featureindex[name] ~= nil then
        for _, feature in ipairs(featureindex[name]) do
            local rule = feature[1]
            
            if rule[1] == name and rule[3] ~= "stable" and not is_name_text_this(rule[3]) then
                local copy_this_rule = true
                
                local tags = feature[4]
                for _, tag in ipairs(tags) do
                    if tag == "stable" or tag == "mimic" then
                        copy_this_rule = false
                        break
                    end
                end
                
                --[[ 
                    Note: one "special" rule that we haven't covered is "X is crash". But excluding this property from stablerules seems to prevent the infinite loop screen from happening when "X is stable" + "X feeling stable is not stable"
                ]]
                if copy_this_rule and rule[1] == name and rule[3] ~= "stable" then
                    for i, cond in ipairs(feature[2]) do
                        local condtype = cond[1]
                        local real_condtype = utils.real_condtype(condtype)
                        if real_condtype == "stable" or real_condtype == "not stable" then
                            -- Don't copy this rule if the rule has stable condition. 
                            -- @Note: This is a fix to an issue with group where in some cases in grouprules(), tags aren't preserved. (Look for "local newtags = concatenate(tags)") - 3/6/22
                            copy_this_rule = false
                        end
                    end

                end

                if copy_this_rule then
                    table.insert(stable_features, feature)
                end
            end
        end
    end

    return stable_features
end

--[[ 
    Given a set of stablerules determined by get_stablerules_from_name(), filter the stablerules based on
    specific condtypes, with an object to test it on. Right now, the only special condtype to check is 
    "this" since it only applies to units that are pointed by a pnoun.
]]
local function filter_stablerules_by_conds(stablerules, object)
    local new_stablerules = {}
    local unitid, x, y = utils.parse_object(object)
    for _, feature in ipairs(stablerules) do
        --@TODO - optimization since most rules will not have "this" condtype
        local conds_to_test = {}  -- These are special conditions that we have to test before adding the stablerule
        for _, cond in ipairs(feature[2]) do
            local condtype = utils.real_condtype(cond[1])
            if condtype == "this" or condtype == "not this" then
                table.insert(conds_to_test, cond)
            end
        end

        local add_ruleid = true
        if #conds_to_test > 0 then
            add_ruleid = testcond(conds_to_test, unitid, x, y)
        end

        if add_ruleid then
            table.insert(new_stablerules, feature)
        end
    end

    return new_stablerules
end

table.insert(mod_hook_functions["level_start"],
    function()
        update_stable_state() -- Only reason we update stable state here is because of a bug where the stable cursor doesn't show at level startup

        enable_stablerule_display_setting = not PlasmaSettings.get_toggle_setting("disable_stable_display")
    end
)

function is_stableunit(unitid, x, y)
    local object = utils.make_object(unitid, x, y, true)
    return stablestate.objects[object] ~= nil
end

--[[ Core logic ]]
function update_stable_state(alreadyrun)
    if on_stable_undo then
        if STABLE_LOGGING then
            print("Skipped update_stable_state() because on_stable_undo = true")
        end
        return
    end
    if GLOBAL_disable_stablerule_update then
        if STABLE_LOGGING then
            print("skipped update_stable_state() because GLOBAL_disable_stablerule_update = true")
        end
        return
    end

    if STABLE_LOGGING then
        print("Calling update_stable_state() with alreadyrun = "..tostring(alreadyrun))
    end

    -- Deleting items from stablestate.units and rules
    local deleted_su_key_count = 0
    deleted_su_key_count = #queued_deleted_su_keys
    for _, object in ipairs(queued_deleted_su_keys) do

        local stableunit = stablestate:get_stableunit(object)
        local removed = stablestate:remove_object(object)
        if removed then
            stabledisplay:remove_stableunit(object)

            utils.debug_assert(stableunit)
            addundo({"stable","remove", object, stableunit})
            deleted_su_key_count = deleted_su_key_count + 1
        end
    end
    queued_deleted_su_keys = {}


    GLOBAL_checking_stable = true
    local code_stableunits, code_stableempties = findallfeature(nil, "is", "stable")
    if hasfeature("level", "is", "stable", 1) then
        table.insert(code_stableunits, 1)
    end
    GLOBAL_checking_stable = false
    
    local new_stableunit_count = 0

    local cached_stablerules_by_name = {}
    local found_stableunits = {}
    for _, unitid in ipairs(code_stableunits) do
        if unitid ~= 2 then
            local object = utils.make_object(unitid)
            found_stableunits[object] = true

            local name = ""
            if unitid == 1 then
                name = "level"
            else
                local unit = mmf.newObject(unitid)
                name = getname(unit)
            end

            local stablerules = nil
            if cached_stablerules_by_name[name] then
                stablerules = cached_stablerules_by_name[name]
            else
                stablerules = get_stablerules_from_name(name)
                cached_stablerules_by_name[name] = stablerules
            end
            stablerules = filter_stablerules_by_conds(stablerules, object)

            local added = stablestate:add_object(object, stablerules)
            if added then
                stabledisplay:add_stableunit(object)
                addundo({"stable","add", object, stablestate:get_stableunit(object)})
                new_stableunit_count = new_stableunit_count + 1
            end 
        end
    end
    for _, group in ipairs(code_stableempties) do
        for tileid, _ in pairs(group) do
            local x, y = utils.coords_from_tileid(tileid)
            local object = utils.make_object(2, x, y)
            found_stableunits[object] = true

            if cached_stablerules_by_name["empty"] then
                stablerules = cached_stablerules_by_name["empty"]
            else
                stablerules = get_stablerules_from_name("empty")
            end
            stablerules = filter_stablerules_by_conds(stablerules, object)

            local added = stablestate:add_object(object, stablerules)
            if added then
                stabledisplay:add_stableunit(object)
                addundo({"stable","add", object, stablestate:get_stableunit(object)})
                new_stableunit_count = new_stableunit_count + 1
            end
        end
    end
    for object, _ in pairs(stablestate.objects) do
        if not found_stableunits[object] then
            local stableunit = stablestate:get_stableunit(object)

            local removed = stablestate:remove_object(object)
            if removed then
                stabledisplay:remove_stableunit(object)

                utils.debug_assert(stableunit)
                addundo({"stable","remove", object, stableunit})
                deleted_su_key_count = deleted_su_key_count + 1
            end
        end
    end

    if new_stableunit_count > 0 or deleted_su_key_count > 0 then
        updatecode = 1
        return true
    end
    return false
end

--[[ 
    Called in rule_baserules modhook, this function adds all stablerules to the featureindex. For each 
    stablerule, it adds enough duplicates to account for stableunits with stacked rules.
]]
local function add_stable_rules()
    -- adding all stablestate.rules into the featureindex
    for _, v in pairs(stablestate.rules) do
        for s = 1, v.max_stack_count do
            local feature = utils.deep_copy_table(v.feature)

            for _, cond in ipairs(feature[2]) do
                local condtype = cond[1]
                if condtype == "stable" then
                    local params = cond[2]
                    params[2] = s

                    break -- Assuming that each of the rules from stablestate will only have one stable cond
                end
            end

            addoption(feature[1], feature[2], feature[3], false, nil, feature[4])
            
            if STABLE_LOGGING then
                print("Inserted stablerule into featureindex: "..utils.serialize_feature(feature), "Stack count: "..s)
            end
        end
    end
end

local function stableunit_has_ruleid(unitid, ruleid, x, y, rule_stack_count)
    local object = utils.make_object(unitid, x, y)
    if not stablestate.objects[object] then
        return false
    end
    local ruleid_list = stablestate.objects[object].ruleids

    -- Second clause handles stacked stable properties per stableunit
    return (ruleid_list[ruleid] ~= nil) and (ruleid_list[ruleid].stack_count >= rule_stack_count)
end

condlist["stable"] = function(params,checkedconds,checkedconds_,cdata)
    utils.debug_assert(#params == 2)
    if #params == 2 then
        valid = true
        local unitid, x, y = cdata.unitid, cdata.x, cdata.y

        --[[ 
            JANK WARNING: the parameters are actually passed in reverse order due to testcond.
            For reference, look for this piece of code in testcond:

                if (string.sub(b, 1, 4) == "not ") then
                    table.insert(params, b)
                else
                    table.insert(params, 1, b)
                end
         ]]
        local cond_ruleid = params[2]
        local stack_count = params[1]

        local result = stableunit_has_ruleid(unitid, cond_ruleid, x, y, stack_count)

        return result, checkedconds
    end
    return false, checkedconds
end

table.insert(mod_hook_functions["rule_baserules"],
    function()
        add_stable_rules()
    end
)

table.insert(mod_hook_functions["rule_update_after"],
    function()
        if on_stable_undo then
            on_stable_undo = false

            if STABLE_LOGGING then
                stablestate:print_stable_state(on_stable_undo)
            end
        end
    end
)

--[[ UNDO Management ]]

function handle_stable_undo(line)
    local action = line[2]
    local object = line[3]
    local stableunit = line[4]
    if action == "add" then
        if STABLE_LOGGING then
            print("Removing stableunit on undo "..object)
        end

        local removed = stablestate:remove_object(object, true)
        if removed then
            stabledisplay:remove_stableunit(object)
            updatecode = 1
        end
    elseif action == "remove" then
        if STABLE_LOGGING then
            print("Restored stableunit on undo "..object)
        end

        local restored = stablestate:restore_stableunit(stableunit)
        if restored then
            stabledisplay:add_stableunit(object)
            updatecode = 1
        end
    end
end

-- Note: we use "undoed" instead of "undoed_after" since the former fires if the game's undo stack has an entry to pop.
-- Ideally: we won't need to rely on the undo entry being applied *before* calling this function 
table.insert(mod_hook_functions["undoed"],
    function()
        on_stable_undo = true
    end
)

table.insert(mod_hook_functions["command_given"],
    function()
        if STABLE_LOGGING then
            print("--------turn start--------")
        end
        on_stable_undo = false
    end
)

table.insert(mod_hook_functions["turn_end"],
    function()
        if STABLE_LOGGING then
            stablestate:print_stable_state(on_stable_undo)
            print("--------turn end--------")
        end
    end
)

-- Note: changed from "effect_always" to "always" since effect_always only activates when disable particle effects is off 
table.insert(mod_hook_functions["always"],
    function()
        if generaldata.values[MODE] == 0 then
            if utils.try_call(stabledisplay.update_stable_indicators, stabledisplay) then
            if enable_stablerule_display_setting then
                    utils.try_call(stabledisplay.show_stablerule_display, stabledisplay)
                end
            end
        end
    end
)