table.insert(editor_objlist_order, "text_nuhuhright")
table.insert(editor_objlist_order, "text_nuhuhup")
table.insert(editor_objlist_order, "text_nuhuhleft")
table.insert(editor_objlist_order, "text_nuhuhdown")

editor_objlist["text_nuhuhright"] = 
{
	name = "text_nuhuhright",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods"},
	tiling = -1,
	type = 8,
	layer = 20,
	colour = {2, 2},
	colour_active = {2, 2},
}
editor_objlist["text_nuhuhup"] = 
{
	name = "text_nuhuhup",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods"},
	tiling = -1,
	type = 8,
	layer = 20,
	colour = {2, 2},
	colour_active = {2, 2},
}
editor_objlist["text_nuhuhleft"] = 
{
	name = "text_nuhuhleft",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods"},
	tiling = -1,
	type = 8,
	layer = 20,
	colour = {2, 2},
	colour_active = {2, 2},
}
editor_objlist["text_nuhuhdown"] = 
{
	name = "text_nuhuhdown",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods"},
	tiling = -1,
	type = 8,
	layer = 20,
	colour = {2, 2},
	colour_active = {2, 2},
}

--@Merge(nuhuh x plasma) add support for turning nuhuh
table.insert(editor_objlist_order, "text_turning_nuhuh")
editor_objlist["text_turning_nuhuh"] = 
{
	name = "text_turning_nuhuh",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods", "turning text"},
	tiling = 0,
	type = 8,
	layer = 20,
	colour = {2, 2},
	colour_active = {2, 2},
}

formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Btd456creeper", {0,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
            name = "nuhuh",
			thumbnail_obj = "text_nuhuhright",
			author = "Btd456creeper",
			display_sprites = {"text_nuhuhright", "text_nuhuhup", "text_nuhuhleft", "text_nuhuhdown", "text_turning_nuhuh"},
			description = 
[[Disables texts from being used in a rule.

- The texts that nuhuh targets are 1 tile in the direction of the arrow, relative to the position of nuhuh.

- Not used in normal parsing.

- Includes turning text variant]],
        }
    })
end

--New function to check if a Nuh Uh! text is negating a given space on the grid
function gettilenegated(x,y,infloopcount_)
    infloopcount = infloopcount_ or 0
    if (infloopcount > 600) then --600 is slightly more than the number of tiles in a full-size level (594)
        HACK_INFINITY = 200 --???
        destroylevel("infinity")
        return false
    end
    infloopcount = infloopcount + 1

	local result = false

    local dirnames = {"right","up","left","down"}
    for i=1,4 do
        local dir = ndirs[i]
        local dirx = dir[1] * -1
        local diry = dir[2] * -1

        local checkx = x + dirx
        local checky = y + diry

        local potentialnegates = findallhere(checkx,checky)
        for j,v in ipairs(potentialnegates) do
			local unit = mmf.newObject(v)
            if (unit.strings[UNITNAME] == "text_nuhuh" .. dirnames[i] --@Merge: Add support for turning nuhuh
			  or (unit.strings[UNITNAME] == "text_turning_nuhuh" and unit.values[DIR] == i-1)) and (gettilenegated(checkx,checky,infloopcount) == false) then
                result = true
            end
        end
    end

    return result
end


--[[ @Merge: codecheck() was merged ]]



--[[ @Merge: formlettermap() was merged ]]


--DEBUG: Spawn particles on all tiles affected by "Nuh Uh!" texts.
--Causes bugs when an infinite loop happens.
--[[
table.insert(mod_hook_functions["effect_always"],
	function()
		for x=1,roomsizex do
            for y=1,roomsizey do
                if gettilenegated(x,y) then
                    MF_particle("error",x,y,2,2,30)
                end
            end
        end
	end
)
]]