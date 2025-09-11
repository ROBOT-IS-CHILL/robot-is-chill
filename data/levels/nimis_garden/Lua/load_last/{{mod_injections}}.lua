--[[ 
    Which functions to inject:
    Definetly - clearunits(), findnoun(), delunit()
    Maybe - all functions in features.lua, getunitswitheffect("select",true), addunit()
    Partially - code()

 ]]


-- @mods(this), @mods(stable) - Injection reason: provide hook for clearing mod globals/locals
local utils = PlasmaModules.load_module("general/utils")

local old_clearunits = clearunits

function clearunits(...)
    local ret = old_clearunits(...)
    reset_this_mod()
	clear_stable_mod()
    clear_turning_text_mod()
    reset_arrow_properties()
    clear_guard_mod()

    return ret
end


local old_addunit = addunit
function addunit(id, ...)
    local ret = old_addunit(id, ...)

    local unit = mmf.newObject(id)
    local name = getname(unit)
	local name_ = unit.strings[NAME]

	if is_name_text_this(name_) then
		on_add_this_text(unit.fixed)
	end

	on_add_stableunit(unit.fixed)
end

-- @mods(stable), @mods(this) - Injection reason: provide hook for when a unit gets deleted. This is to clear that unit from each mod's internal tables
local old_delunit = delunit
function delunit(unitid)
    local ret = old_delunit(unitid)
    on_delete_stableunit(unitid)
    on_delele_this_text(unitid)

    return ret
end

--[[ 
    @mods(this) - Injection reason: in the many cases where the game iterates through objectlist, it uses this function to exclude special nouns from "all". 
    Since we want THIS and all of its variations to be excluded, override this function, not just nlist.full 
]]
local old_findnoun = findnoun
function findnoun(noun, ...)
    if is_name_text_this(noun) then
        return true
	else
        return old_findnoun(noun, ...)
    end
end

--[[ 
    @mods(guard) - Injection reason: provide a guard checkpoint after every handledels call
]]
local old_handledels = handledels
function handledels(delthese, ...)
    local ret = table.pack(old_handledels(delthese, ...))
    guard_checkpoint("handledels")
    return table.unpack(ret)
end

local old_delete = delete
function delete(unitid, x_, y_, ...)
	if not GLOBAL_disable_guard_checking then
		local caller_func = debug.getinfo(2).func
		local is_guarded = handle_guard_delete_call(unitid, x_, y_, caller_func)
		if is_guarded then
			return
		end
	end

    return old_delete(unitid, x_, y_, ...)
end

--[[ 
    @mods(guard) - Injection reason: prevent triggering "has" if the unit is guarded
]]
local old_inside = inside
function inside(name,...)
    if not GLOBAL_disable_guard_checking and is_name_guarded(name) then
		return
	end
    return old_inside(name, ...)
end

--[[ 
    @mods(guard) - Injection reason: detect all changes to units on the level in order to determine whether or not to recalculate guards
]]
local old_addundo = addundo
local UndoAnalyzer = PlasmaModules.load_module("general/undo_analyzer")
function addundo(line,...)
    local ret = table.pack(old_addundo(line, ...))

    UndoAnalyzer.analyze_undo_line(line)
    check_undo_data_for_updating_guards(line)
    return table.unpack(ret)
end

--[[ 
    @mods(stable) - Injection reason: if the game calls destroylevel() for any reason (e.g infinite loop, too complex, level is weak),
        all units in the level will be destroyed. If there were any stableunits before, we need to update the stable state to account
        for this suddden deletion of all stableunits. This would automatically delete any stable indicators to prevent nil errors from
        stabledisplay trying to access a stableunit to attach the stable indicator to, and only getting nil.
]]
local old_destroylevel_do = destroylevel_do
function destroylevel_do(...)
    local do_stablestate_update = (generaldata.values[MODE] ~= 5) and destroylevel_check

    local ret = table.pack(old_destroylevel_do(...))

    if do_stablestate_update then
        update_stable_state(false)
    end

    return table.unpack(ret)
end

--[[ 
    @mods(stable) - Injection reason:
        - (see giant block comment below)
        - treat "not stable" condtype as "stable".
            - invertconds() can change condtype "stable" to "not stable". "stable" as a condtype technically doesn't make sense. It's more of a hacky way to prevent other non-stable rules from affecting stable objects.
              So if we somehow end up with a "not stable" condition, treat it as a normal "stable" condtype.
]]
local old_testcond = testcond
function testcond(conds, unitid, x_, y_, ...)
    local x,y = 0,0
    if (unitid ~= 0) and (unitid ~= 1) and (unitid ~= 2) and (unitid ~= nil) then
        local unit = mmf.newObject(unitid)
		x = unit.values[XPOS]
		y = unit.values[YPOS]
    elseif (unitid == 2) then
		x = x_
		y = y_
    end

    --[[ 
		@mods(stable) - if a stableunit is being checked, the set of conditions must have the "stable" cond in order
		to make sure that the stableunit only has stablerules applied.

		EXCEPTIONS:
		- When GLOBAL_checking_stable == true. This is a global that tells is set to true whenever we intend to do a testcond() on a "X is stable" rule.
			Rules in the form of "X is stable" should never be a stablerule (aka, appear in the list of rules when hovering a stableunit with mouse).
			So it makes sense that we should not perform this check when testing "X is stable".
		- When conds == nil. NOT WHEN conds == {}. In the rare cases where testcond() gets passed in nil instead of an empty table
			for conds, the code just wants all units regardless of conditions. (At least, thats what I gathered from
			handling the special case of "teeth eat baba"; It calls findtype() while passing in nil conds). I'm guessing that
			other cases where the game passes an empty table means that the game wants to consider conditions.
			WARNING: this is a pretty unfounded assumption that can collapse easily.
	]]
    local found_stablecond = false

    if conds ~= nil then
    for _,cond in ipairs(conds) do
        local condtype = utils.real_condtype(cond[1])
        if condtype == "stable" or condtype == "not stable" then
            found_stablecond = true
            break
            end
        end
    end
    if not found_stablecond and (not GLOBAL_checking_stable and conds ~= nil and is_stableunit(unitid, x, y)) then
        return false
    else
        return old_testcond(conds, unitid, x_, y_, ...)
    end
    --[[ 
        @NOTE: There was a bug that was originally handled in commit #582f9fd8cef2585c37cabc85880de90a7d66a6cf. However, with the new condition system, I basically removed that fix.
        But for *some* reason, the issue doesn't *seem* to happen even without the fix. And I don't know why lol. For now, I'll trust that the issue won't happen.
        But this will bug me to no end.

        To explain what I found so far, if you have "text on baba is stable", "text is push", "text is not push", and trigger the stable, the issue would make non-stable texts be pushable.
        But in the new system, the non-stablerule "text is push" gets modified by both stablerule and non-stablerule versions of "text is not push". This will end up with a final rule of:
            text is push | (not stable)[(text is not push)] && never[] && stable[(text is push | (not stable)[(text is not push)] && never[])]

        So the "never" effectively cancels out the non-stablerule "text is push". *sigh* Buuttt, there's a whole alot of implications that I don't want to dive into right now.

        Revisit this later on.
     ]]
    -- elseif found_stablecond then
    --     local newconds = {}

    --     --[[ 
    --         @NOTE: the original intention here is to replace "not stable" conds with "stable" conds. However, I would have to go through the trouble of parsing through
    --             the weird syntax of parenthesis generated from invertconds. 
    --             Yet somehow even without my interference, I don't run into the issue outlined in commit #582f9fd8cef2585c37cabc85880de90a7d66a6cf
    --      ]]

    --     for _,cond in ipairs(conds) do
    --         local condtype = cond[1]
    --         if condtype == "not stable" then
    --             print("testklsndflsnkfd")
    --         end
    --         if condtype == "stable" or condtype == "not stable" then
    --             condtype = "stable"
    --         end
    --         table.insert(newconds, {condtype, cond[2]})
    --     end

    --     return old_testcond(newconds, unitid, x_, y_, ...)
    -- else
    --     return old_testcond(conds, unitid, x_, y_, ...)
    -- end
end