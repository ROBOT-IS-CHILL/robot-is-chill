--[[ 
	@Merge - did some heavy modification with this file to make it more portable and not levelpack specific.
	- removed the giant modhook table and replaced with table.insert(mod_hook_functions["whatever"])
	- removed references to background images used in the levelpack
	- removed references to the ending, and 42level
	- removed all sprites except for text sprites
	- added persist to be selectable in the editor (TODO: maybe I should do the same with baserules?)
 ]]


persists = {}
levelpersist = {}
persistreverts = {}
persistrevert = nil
persistbaserules = {}
persistbaserulestoadd = {}

prevpersists = {}
prevlevelpersist = {}

local enable_persist_in_editor = false
local persist_stablerules = {}
local utils = PlasmaModules.load_module("general/utils")

exitedlevel = false

local function clearpersists()
	persists = {}
	levelpersist = {}
	persistreverts = {}
	persistrevert = nil
	persistbaserules = {}
	persistbaserulestoadd = {}

	prevpersists = {}
	prevlevelpersist = {}

	persist_stablerules = {}
end
clearpersists()

function apply_persist_settings(settings_dict)
	for setting_name, value in pairs(settings_dict) do
		if setting_name == "allow_persist_in_editor" then
			enable_persist_in_editor = value
		end
	end
end

function getprevpersists()
	persists = {}
	if prevpersists[currentlevel] ~= nil then
		for i,v in pairs(prevpersists[currentlevel]) do
			persists[i] = {}
			for j,k in pairs(v) do
				if type(k) == "table" then
					persists[i][j] = utils.deep_copy_table(k)
				else
					persists[i][j] = k
				end
			end
		end
	end

	levelpersist = {}
	if prevlevelpersist[currentlevel] ~= nil then
		for i,j in ipairs(prevlevelpersist[currentlevel]) do
			levelpersist[i] = j
		end
	end
end

--for optimisation purposes, this remembers ONLY the oldest object and will not try subsequent ones if X IS NOT Y is present in future levels
--will fix this if I make a general release of this
function getreverts(unit)
	persistrevert = nil
	if persistreverts ~= nil then
		id = unit.values[ID]
		persistrevert = persistreverts[id]
	end
	
	local originalname = ""
	local oldest = 0
	local source = ""
	local baseid = -1
	local nametotest = getname(unit)
	local ingameid = unit.values[ID]
	local baseingameid = unit.values[ID]
	local unitid = unit.values[ID]
	
	for i=1,#undobuffer do
		local curr = undobuffer[i]
		
		for a,b in ipairs(curr) do
			if (b[1] == "create") and (b[3] == baseingameid) then
				oldest = i
				originalname = b[2]
				source = b[5]
				baseid = b[4]
				break
			end
		end
	end
	
	local oldestundo = undobuffer[oldest] or {}
	
	for i,v in ipairs(oldestundo) do
		if (v[1] == "remove") and ((v[6] == unit.values[ID]) or (v[7] == unit.values[ID]) or ((baseid == v[6]) and (baseid > -1))) then
			if (hasfeature(nametotest,"is","not " .. v[2],unitid,x,y) == nil) then
				originalname = v[2]
				break
			end
		end
	end
	if persistrevert ~= nil then
		originalname = persistrevert
	end
	if (string.len(originalname) > 0) then
		return originalname
	else
		return nil
	end
end

--permanently add rules made from text with the "baserule" property. for optimisation doesn't care about conditions. this is easy to fix though
function findpersistrules()
	for i,rules in ipairs(visualfeatures) do
		local conds = rules[2]
		local ids = rules[3]
		local tags = rules[4]
		
		local fullpersist = true
		for a,b in ipairs(ids) do
			local dunit = mmf.newObject(b[1])
			
			if not hasfeature(getname(dunit), "is", "baserule", dunit.fixed) then
				fullpersist = false
				break
			end
		end
		

		if (fullpersist == true) and (#conds==0) then
			--do not persist rules that are disabled
			if not hasfeature(rules[1][1],rules[1][2],"not "..rules[1][3]) and not (objectlist[rules[1][3]] ~= nil and hasfeature(rules[1][1],"is",rules[1][1])) then
				table.insert(persistbaserulestoadd,{rules[1][1],rules[1][2],rules[1][3]})
			end
		end
	end
	if #persistbaserulestoadd ~= 0 then
		if persistbaserules == nil then
			persistbaserules = {}
		end
		--if we already have rules for this level, don't add anymore; prevents duplicate entries due to hook for WIN getting called more than once with multiple YOU objects, or level being transformed into multiple things at once, etc
		persistbaserules[currentlevel] = persistbaserulestoadd
		persistbaserulestoadd = {}
	end
end

function findpersists(reason)
	if not enable_persist_in_editor and editor.values[INEDITOR] ~= 0 then
		-- A value of zero seems to indicate that we are actually playing the level in game, not in the editor, nor in a single level
		clearpersists()
		return
	end

	if reason ~= "levelentry" then
		prevpersists = {}
		prevlevelpersist = {}
	end
	findpersistrules()
	--update persistent object info
	persists = {}
	levelpersist = {}
	persist_stablerules = {}
	if hasfeature("level","is","persist",1) then
		levelpersist = {Xoffset-Xoffsetorig,Yoffset,mapdir,maprotation}
		persistxoffset = 0
		persistyoffset = 0
	else
		persistxoffset = (Xoffset-Xoffsetorig)/tilesize
		persistyoffset = (Yoffset-Yoffsetorig)/tilesize
	end
		
	ispersist = getunitswitheffect("persist",delthese)
	for id,unit in ipairs(ispersist) do
		x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
		name = getname(unit)
		leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
		persistobjectdata = {unit.strings[UNITNAME],x+(persistxoffset),y+(persistyoffset),dir,x+(persistxoffset),y+(persistyoffset),nil,nil,leveldata,{getreverts(unit),unit.karma}}

		local stableunit_entry = get_persist_stablestate_info(unit.fixed)
		if stableunit_entry then
			persistobjectdata[11] = stableunit_entry
		end
		--persistobjectdata = {unit.strings[UNITNAME],x,y,dir,unit.values[ID],y,nil,nil,leveldata,unit.followed,unit.back_init}
		table.insert(persists,(persistobjectdata))
	end
end


table.insert(mod_hook_functions["level_start"], 
	function()
		currentlevel = generaldata.strings[CURRLEVEL]

		if exitedlevel == true then
			getprevpersists()
		end
		exitedlevel = false
		
		--check for persistent rules
		for level,v in pairs(persistbaserules) do
			for j,rule in ipairs(v) do
				--need to be able to create objects that aren't already in the level
				if (unitreference[rule[3]] ~= nil or rule[3] == "empty" or rule[3] == "text") then
					objectlist[rule[3]] = 1
				end
			end
		end
		
	
		Xoffsetorig = Xoffset
		Yoffsetorig = Yoffset
		--create persistent objects from previous level
		prevpersists[currentlevel] = {}

		local stable_persists_to_apply = {}
		local created_new_persist_units = false
		if persists ~= nil then
			for i,v in pairs(persists) do
				--do not bring persistent objects if their persistence is disabled in the new level
				if hasfeature(v[1], "is","not persist") == nil and not (hasfeature("all", "is","not persist") and not (string.sub(v[1], 1, 5) == "text_")) and not ((string.sub(v[1], 1, 5) == "text_") and (hasfeature("text", "is","not persist"))) then
					local newunitid, id = create(v[1],v[2],v[3],v[4],v[5],v[6],nil,true,v[9],v[10])
					created_new_persist_units = true

					if v[11] then
						table.insert(stable_persists_to_apply, {newunitid, v[11]})
					end

					prevpersists[currentlevel][i] = {}
					--backup to prevpersists; if entry is a table (leveldata) make sure it copies the whole table
					for j,k in pairs(v) do
						if type(k) == "table" then
							prevpersists[currentlevel][i][j] = utils.deep_copy_table(k)
						else
							prevpersists[currentlevel][i][j] = k
						end
					end
				end
			end
		end

		-- @Merge: small fix where persisted objects don't update their sprites
		if created_new_persist_units then
			animate()
		end
		
		prevlevelpersist[currentlevel] = {}
		if levelpersist ~= nil then
			for i,j in ipairs(levelpersist) do
				prevlevelpersist[currentlevel][i] = j
			end
			if levelpersist[1] ~= nil and hasfeature("level", "is","not persist",1) == nil then
				MF_scrollroom(levelpersist[1],levelpersist[2])
				mapdir = levelpersist[3]
				maprotation = levelpersist[4]
				MF_levelrotation(maprotation)
			end
		end
		
		persistbaserulestoadd = {}
		updatecode = 1

		-- @mods(plasma x persist) - if we are going to apply persist stablerules, don't update the stablestate until
		-- the call to code() after. This is to account 
		if #stable_persists_to_apply > 0 then
			GLOBAL_disable_stablerule_update = true
		end

		code(alreadyrun_) --reparse any new rules formed by persisted text

		if #stable_persists_to_apply > 0 then
			GLOBAL_disable_stablerule_update = false
		end

		GLOBAL_checking_stable = true
		for _, stable_persist in ipairs(stable_persists_to_apply) do
			local newunitid = stable_persist[1]
			local stableunit_entry = stable_persist[2]
			local newunit = mmf.newObject(newunitid)
			
			if stableunit_entry and hasfeature(newunit.strings[NAME], "is", "stable", newunitid) then
				apply_persist_stablestate_info(newunitid, stableunit_entry)
				updatecode = 1
			end
		end
		GLOBAL_checking_stable = false

		if updatecode == 1 then
			code(alreadyrun_)
		end
	end
)


table.insert(mod_hook_functions["level_end"], 
	function()
		--if the player exits the level via the menu
		exitedlevel = true
	end
)

table.insert(mod_hook_functions["level_win"], findpersists)

table.insert(editor_objlist_order, "text_persist")
editor_objlist["text_persist"] = 
{
	name = "text_persist",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "abstract", "text_quality", "persist"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {0, 1},
	colour_active = {0, 3},
}

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Randomizer", {0,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
            name = "persist",
            author = "Randomizer",
            description =
[[When switching from one level to another, objects that are "PERSIST" will also move to the next level, retaining their position and state from the previous level.
        
- If an object PERSISTs onto a level that doesn't have the object in its palette, the "PERSIST" object shows as an error instead.]]
        }
    })
end