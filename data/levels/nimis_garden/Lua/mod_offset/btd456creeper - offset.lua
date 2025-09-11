--First things first: global constants!
--These control where offsets are stored in undo data for the "remove" undo type.
--Changing these may help resolve some conflicts with other mods.
--These two values should be different, and both greater than or equal to 20.
--@Merge: originally these two were 20 and 21. But patashu mods also uses 20 and 21 in its undo data. And KARMA uses 22. So changing these globals to different values
XOFFSETUNDOLINE = 23
YOFFSETUNDOLINE = 24

--@Merge: local forward declarations to avoid polluting the namespace
local getoffsetrules
local gettrueoffset
local getunitswithoffset
local updateleveloffset

--If true, the game will print detailed information to the console whenever an offset happens.
--This uses the print() function, so the game has to be set up to output those.
local offsetdebuglog = true -- @nocommit


--New in 1.1: Global variables to store the outerlevel's offset.
offset_levelxoffset = 0
offset_levelyoffset = 0

table.insert(mod_hook_functions["level_start"],
	function()
		offset_levelxoffset = 0
		offset_levelyoffset = 0

		-- @Merge: Offset doesn't really account for cases where code() can get called during "level_start".
		-- In mega modpack, this currently happens in persist mod's modsupport.lua.
		-- As an extra guard, when the above global variables are set to 0 during a level_start, reset the level's offset.
		-- This is in response to a bug where LEVEL IS OFFSET shifts the level twice on level startup.
		local xoffset = (screenw - roomsizex * tilesize * spritedata.values[TILEMULT]) * 0.5
		local yoffset = (screenh - roomsizey * tilesize * spritedata.values[TILEMULT]) * 0.5
		MF_setroomoffset(xoffset, yoffset)
	end
)

--Also new in 1.1: Function to handle the outerlevel's offset.
function updateleveloffset()
	local levelxoffsetrules = hasfeature_count("level", "is", "offsetright", 1) - hasfeature_count("level", "is", "offsetleft", 1)
	local levelyoffsetrules = hasfeature_count("level", "is", "offsetdown", 1) - hasfeature_count("level", "is", "offsetup", 1)

	local xchange = levelxoffsetrules - offset_levelxoffset
	local ychange = levelyoffsetrules - offset_levelyoffset

	if ((xchange > 0) and cantmove("level",1,0)) --lockedright
	or ((xchange < 0) and cantmove("level",1,2)) --lockedleft
	then
		xchange = 0
	end

	if ((ychange < 0) and cantmove("level",1,1)) --lockedup
	or ((ychange > 0) and cantmove("level",1,3)) --lockeddown
	then
		ychange = 0
	end

	if ((xchange ~= 0) or (ychange ~= 0)) then
		--Note: Xoffset and Yoffset are global variables in the base game,
		--separate from the Offset mod! They track the outerlevel's position.
		addundo({"leveloffset",offset_levelxoffset,offset_levelyoffset,Xoffset,Yoffset})

		local oldxoffset = offset_levelxoffset
		local oldyoffset = offset_levelyoffset
		offset_levelxoffset = offset_levelxoffset + xchange
		offset_levelyoffset = offset_levelyoffset + ychange
		MF_scrollroom(xchange * tilesize,ychange * tilesize)

		if offsetdebuglog then
			print("The outerlevel moved via Offset! Old offset was (" .. oldxoffset .. "," .. oldyoffset .. "), new offset is (" .. offset_levelxoffset .. "," .. offset_levelyoffset .. ").")
			print(debug.traceback())
		end
	end
end

--The meat of the mod: Update objects' positions based on their offsets.
local function updateoffsets()
	--Offsetting must happen simultaneously, so calculate all offsets first, THEN apply them all at once.
	local offsetdebugtable = {}
	if offsetdebuglog then
		table.insert(offsetdebugtable,"OFFSET: (hack_infinity = " .. HACK_INFINITY .. ")")
	end

	local poschanges = {}
	local somethingchanged = false
	for i,unit in ipairs(units) do
		local truexoffset,trueyoffset = gettrueoffset(unit)
		local rulexoffset,ruleyoffset = getoffsetrules(unit)
		local xchange = rulexoffset - truexoffset
		local ychange = ruleyoffset - trueyoffset
		local unitname = getname(unit)
		local unitid = unit.fixed

		if ((xchange > 0) and cantmove(unitname,unitid,0)) --lockedright
		or ((xchange < 0) and cantmove(unitname,unitid,2)) --lockedleft
		then
			xchange = 0
		end

		if ((ychange < 0) and cantmove(unitname,unitid,1)) --lockedup
		or ((ychange > 0) and cantmove(unitname,unitid,3)) --lockeddown
		then
			ychange = 0
		end

		poschanges[unit] = {xchange, ychange}
	end

	for unit,change in pairs(poschanges) do
		local xchange = change[1]
		local ychange = change[2]
		local unitid = unit.fixed
		if ((xchange ~= 0) or (ychange ~= 0)) then
				somethingchanged = true
				addundo({"offset",unit.values[ID],unit.xoffset,unit.yoffset})
				unit.xoffset = unit.xoffset + xchange
				unit.yoffset = unit.yoffset + ychange
			local oldx = unit.values[XPOS]
			local oldy = unit.values[YPOS]
			local newx = oldx + xchange
			local newy = oldy + ychange
			update(unitid,newx,newy)

			if offsetdebuglog then
				table.insert(offsetdebugtable,unit.strings[NAME] .. " with ID " .. unitid .. " at (" .. oldx .. "," .. oldy .. ") moved via Offset to (" .. newx .. "," .. newy .. ")")
			end
		end
	end

	updateleveloffset()

	if somethingchanged then
		if offsetdebuglog and (#offsetdebugtable > 0) then
			for i,v in ipairs(offsetdebugtable) do
				print(v)
			end
		end

		updatecode = 1
		code(true)
	end
end

--Patch to code to update offsets whenever the rules are checked, unless an undo happened.
oldcode = code
-- @Merge(injection)
function code(alreadyrun_)
	oldcode(alreadyrun_)

	if (HACK_INFINITY < 200) and not undoing then
		offsetshappened = updateoffsets()
	end
end

--Check if an undo happened (see above) using a global variable.
table.insert(mod_hook_functions["undoed"],
	function()
		undoing = true
	end
)

--completely normal patch to diceblock created by a very sane person
--The undoed_after modhook doesn't work because code() is called AFTER it.
--But according to PlasmaFlare's lovely modding guide, diceblock, a deprecated function,
--is called at the very end of every undo, so it can be used to reset the "undoing" variable.
olddiceblock = diceblock
-- @Merge(injection)
function diceblock()
	olddiceblock()
	undoing = false
end

--Function to get what an object's offset "should be" based on its Offset rules.
--An object's actual offset might vary from this due to Still and Locked cancelling Offset.
function getoffsetrules(unit)
	local unitid = unit.fixed
	local name = getname(unit)
	local x = unit.values[XPOS]
	local y = unit.values[YPOS]

	local offsettotalx = hasfeature_count(name, "is", "offsetright", unitid, x, y) - hasfeature_count(name, "is", "offsetleft", unitid, x, y)
	local offsettotaly = hasfeature_count(name, "is", "offsetdown", unitid, x, y) - hasfeature_count(name, "is", "offsetup", unitid, x, y)

	return offsettotalx,offsettotaly
end

--Function to get an object's actual offset from its original location.
--This takes Still and Locked into account.
function gettrueoffset(unit)
	if not (unit.xoffset and unit.yoffset) then
		unit.xoffset = 0
		unit.yoffset = 0
	end

	return unit.xoffset,unit.yoffset
end

--Function to get every unit with an Offset rule.
--This makes coding easier because there are four directions and each one has to be checked individually.
--I might not actually use this oh no
function getunitswithoffset()
	local unitswithoffset = getunitswitheffect("offsetright")

	for i,v in ipairs(getunitswitheffect("offsetup")) do
		table.insert(unitswithoffset, v)
	end

	for i,v in ipairs(getunitswitheffect("offsetleft")) do
		table.insert(unitswithoffset, v)
	end
	
	for i,v in ipairs(getunitswitheffect("offsetdown")) do
		table.insert(unitswithoffset, v)
	end

	--Remove duplicates (can happen if an object has multiple Offset rules in different directions)
	local unitswithoffsetnodupes = {}
	local dupes = {}
	for i,v in ipairs(unitswithoffset) do
		if not dupes[v] then
			table.insert(unitswithoffsetnodupes, v)
			dupes[v] = true
		end
	end

	return unitswithoffsetnodupes
end

--Undo stuff! Since offsets are stored in object memory, that information has to be
--stored somewhere when the object is destroyed, so it can be brought back by undoing.

--Patch to addundo to add a destroyed object's offsets to the line of undo data, if applicable.
oldaddundo = addundo
-- @Merge(injection)
function addundo(line,uid_)
	local ename = line[1]
	if (ename == "remove") then
		-- NIMI MOD EDIT: make karma undo data always boolean (yes, THIS is what causes the offset undo bug)
		if (#line < 22) then
			line[22] = false
		end
		local unit = mmf.newObject(getunitid(line[6]))
		line[XOFFSETUNDOLINE] = unit.xoffset
		line[YOFFSETUNDOLINE] = unit.yoffset
	end
	oldaddundo(line,uid_)
end

--Override to undo to put the offset data back on an object when its destruction is undone,
--as well as to add a new undo data type to handle when an object's offset changes.

--[[ @Merge: undo() was merged ]]


--add the offset texts
table.insert(editor_objlist_order, "text_offsetdown")
editor_objlist["text_offsetdown"] = 
{
	name = "text_offsetdown",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods", "arrow properties"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_offsetright")
editor_objlist["text_offsetright"] = 
{
	name = "text_offsetright",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods", "arrow properties"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_offsetup")
editor_objlist["text_offsetup"] = 
{
	name = "text_offsetup",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods", "arrow properties"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_offsetleft")
editor_objlist["text_offsetleft"] = 
{
	name = "text_offsetleft",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods", "arrow properties"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

--@Merge (offset x plasma): add turning offset
table.insert(editor_objlist_order, "text_turning_offset")
editor_objlist["text_turning_offset"] = 
{
	name = "text_turning_offset",
	sprite_in_root = false,
	unittype = "text",
	tags = {"turning text", "text", "btd456creeper mods", "arrow properties"},
	tiling = 0,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

formatobjlist()

word_names["offsetright"] = "offset (right)"
word_names["offsetup"] = "offset (up)"
word_names["offsetleft"] = "offset (left)"
word_names["offsetdown"] = "offset (down)"

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Btd456creeper", {0,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
            name = "offset",
			thumbnail_obj = "text_offsetright",
			author = "Btd456creeper",
			display_sprites = {"text_offsetright", "text_offsetup", "text_offsetleft", "text_offsetdown", "text_turning_offset"},
			description = 
[[Adjusts an objects position by 1 tile in the indicated direction. When the object is no longer OFFSET, the object's position adjusts back, undoing the effects of offset.

- An object can stack offsets to get a larger object displacement.

- Includes turning text variant]],
        }
    })
end