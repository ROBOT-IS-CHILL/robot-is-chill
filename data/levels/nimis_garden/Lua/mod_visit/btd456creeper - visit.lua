--Visit Mod - by Btd456Creeper

--Full levelsurrounds format:
--Divided into multiple level entries. Also, every term is followed by a comma.
--Each entry starts with "levelseparator", then the ID of the level tile on the map,
--then that tile's target level file, level number and visual type (number, letter, or dots), and direction.
--Then for each direction (including diagonals and no direction): The direction name (in one or two letters),
--followed by every object in that direction (or a hyphen if there's nothing there).
--Special case: if an object has level data, put its tile ID right before its object name.

--As of v1.2.1, at the start of the levelsurrounds is the parent level.
--This is used to prevent a bug caused by the fact that entering the level you're currently in doesn't update the leveltree.

--Example: 1level,levelseparator,2.59410404,2level,1,0,0,r,baba,u,skull,house,l,-,d,2.74583175,level,dr,-,ur,-,ul,-,dl,-,o,-,levelseparator,2.74583175,3level,2,1,0,r,-,u,2.59410404,level,l,-,d,-,dr,-,ur,baba,ul,-,dl,-,o,-,


--A new global string is needed to store the tile ID of the current level on the map.
--visit_innerlevelid is used to track where to visit from.
visit_innerlevelid = ""

--Another global string is needed to store the full levelsurrounds (separate from the surrounds for a single level).
--visit_fullsurrounds is used for this.
visit_fullsurrounds = ""

--editor stuff
table.insert(editor_objlist_order, "text_visit")

editor_objlist["text_visit"] = 
{
	name = "text_visit",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "btd456creeper mods"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Btd456creeper", {0,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
            name = "visit",
			author = "Btd456creeper",
			description =
[[Changes the current level to the level directly adjacent to it in the parent level of the current level.

- This effect applies when a you object moves onto a visit object.

- The direction of the visit object determines which direction to look for an adjacent level. If there isn't an adjacent level, the current level gets destroyed instead.]],
        }
    })
end

--The code that actually performs a visit, based on levelsurrounds and the direction of the Visit object.
function dovisit(visitdir)
	if visitdir == 4 then
		visitdir = math.random(0,3)
	end
	if visitdir == 0 then
		visitdir = "r"
	end
	if visitdir == 1 then
		visitdir = "u"
	end
	if visitdir == 2 then
		visitdir = "l"
	end
	if visitdir == 3 then
		visitdir = "d"
	end
	
	--Get the level to go to from fullsurrounds (janky code warning).
	--A table can't be made from fullsurrounds as it would get too big for the game to handle (apparently).
	local stage = 0
	local levelfound
	local v = ""
	
	for i = 1,#visit_fullsurrounds,1 do
		local char = string.sub(visit_fullsurrounds,i,i)
		if char == "," then
			--Find "levelseparator"...
			if v == "levelseparator" then
				stage = 1
				--...and make sure it's followed by the current level's tile ID on the map...
			elseif stage == 1 then
				if v == visit_innerlevelid then
					stage = 2
				else
					stage = 0
				end
				--...then skip ahead two spots...
			elseif stage == 2 then
				stage = 3
			elseif stage == 3 then
				stage = 4
				--...then wait to get to the direction in the surrounds that matches the visit direction...
			elseif stage == 4 and v == visitdir then
				stage = 5
				--...and finally, keep going until a number (which should be a level ID) is found. Or until the code hits a
				--different direction in the surrounds, meaning everything has been checked and no level was found.
			elseif stage == 5 then
				if v == "u" or v == "l" or v == "d" or v == "dr" then
					break
				elseif tonumber(v) ~= nil then
					levelfound = v
					break
				end
			end
			v = ""
		else
			v = v .. char
		end
		i = i + 1
	end
	
	--Visiting is disabled in the editor because making it work in the editor is hard.
	if levelfound and editor.values[E_INEDITOR] == 0 then
		--Now just find the level to go to in fullsurrounds to display the correct information in the transition screen.
		stage = 0
		local v = ""
		local levelfile
		local levelnum
		local leveltype
		local notfirst = false
		local parentlevel
		
		for i = 1,#visit_fullsurrounds,1 do
			local char = string.sub(visit_fullsurrounds,i,i)
			if char == "," then
				if notfirst == false then
					parentlevel = v
					notfirst = true
				elseif v == "levelseparator" then
					stage = 1
				elseif stage == 1 then
					if v == levelfound then
						stage = 2
					else
						stage = 0
					end
				elseif stage == 2 then
					levelfile = v
					stage = 3
				elseif stage == 3 then
					levelnum = v
					stage = 4
				elseif stage == 4 then
					leveltype = v
					
					findpersists() -- @Merge(Visit x Persist): support for persisting objects via visit
					-- @nocommit - does it matter exactly when to call findpersists() in relation to calling uplevel() and sublevel()?

					if parentlevel ~= generaldata.strings[CURRLEVEL] then
						uplevelkeepsurrounds()
					end
					sublevel(levelfile,tonumber(levelnum),tonumber(leveltype))
					
					--These next six lines are just to switch the level while displaying the proper effects.
					--I don't know what most of these lines do, they're taken from an example Hempuli posted once.
					generaldata.values[TRANSITIONREASON] = 9
					generaldata.values[IGNORE] = 1
					generaldata3.values[STOPTRANSITION] = 1
					generaldata2.values[UNLOCK] = 0
					generaldata2.values[UNLOCKTIMER] = 0
					MF_loop("transition",1)
					
					visit_innerlevelid = levelfound
					
					break
				end
				v = ""
			else
				v = v .. char
			end
		end
	else
		--If there's nowhere to visit to, destroy the current level instead (like Level Is Weak).
		destroylevel()
		destroylevel_do()
	end
end

--getlevelsurrounds function override (called when entering a level on the map)
--Takes the tile ID of the level tile you're entering.

--[[ @Merge: getlevelsurrounds() was merged ]]


--Whenever a level is started, turn the full surrounds into regular surrounds that parsesurrounds can handle.
table.insert(mod_hook_functions["level_start"],
	function()
		local result = ""
	local stage = 0
	local v = ""

	--Can't use parsestring as the resulting table might end up too big.
	for i = 1,#visit_fullsurrounds,1 do
		local char = string.sub(visit_fullsurrounds,i,i)
		if char == "," then
			if v == "levelseparator" then
				stage = 1
			elseif stage == 1 and v == visit_innerlevelid then
				stage = 2
			elseif stage == 1 then
				stage = 0
			elseif stage == 2 then
				stage = 3
			elseif stage == 3 then
				stage = 4
			elseif stage == 4 then
				stage = 5
			elseif stage == 5 then
				if v == "levelseparator" then
					break
				else
					result = result .. v .. ","
				end
			end
			v = ""
		else
			v = v .. char
		end

		i = i + 1
	end

	generaldata2.strings[LEVELSURROUNDS] = result
	end
)

--Override for uplevel to clear fullsurrounds (but not normal surrounds, as is base game behavior) on level exit.
--(There's no modhook for this.)

--[[ @Merge: uplevel() was merged ]]


--New function that is literally just a copy of the normal uplevel() function.
--This version doesn't clear visit_fullsurrounds.
function uplevelkeepsurrounds()
	local id = #leveltree
	local parentid = #leveltree - 1
	
	local oldlevel = generaldata.strings[CURRLEVEL]
	generaldata2.strings[PREVIOUSLEVEL] = oldlevel
	MF_store("save",generaldata.strings[WORLD],"Previous",oldlevel)
	latestleveldetails = {lnum = -1, ltype = -1}
	
	if (id == 0) then
		MF_alert("Already at map root")
		
		if (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) and (editor.values[INEDITOR] == 0) then
			MF_end_single()
			MF_credits(1)
		end
	end
	
	if (parentid > 1) then
		generaldata.strings[PARENT] = leveltree[parentid - 1]
	else
		generaldata.strings[PARENT] = ""
	end
	
	if (id > 1) then
		generaldata.strings[CURRLEVEL] = leveltree[parentid]
	else
		generaldata.strings[CURRLEVEL] = ""
	end
	
	table.remove(leveltree, id)
	table.remove(leveltree_id, id)

	return oldlevel
end

--Override to changemenu to clear some strings when returning to the main menu. 

--[[ @Merge: changemenu() was merged ]]


--Override for the block function to add Visit.

--[[ @Merge: block() was merged ]]


--Override to levelblock to handle how Visit works with empties and the outerlevel.
--No documentation here because I already forgot how this part works.

--[[ @Merge: levelblock() was merged ]]


--Override to effects to add the visit effect.

--[[ @Merge: effects() was merged ]]


--Override to doeffects to add a special rule for the visit effect (namely, basing it on the object's direction).

--[[ @Merge: doeffect() was merged ]]
