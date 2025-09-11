--[[ ========== OVERRIDDEN FUNCTIONS ==========

tools.lua
-- isgone(): add check for ALIVE
-- delete(): keep karma after undoing
-- delunit(): update ECHO
-- create(): update ECHO
-- update(): update ECHO
-- getlevelsurrounds(): keep track of any text at the level's position (including the level itself if it was converted) + keep track of level sinful status and alignment
-- writerules(): replace instances of "AMBIENT" with "AMBIENT (OBJECT)" in pause menu rules (currently unused because of the hacky ambient implementation), allow prefixes to have custom visual names

undo.lua
-- undo(): add "levelkarma" and "unitkarma" events, keep karma when undoing destruction/conversion. Add code checks when something related to ECHO is undone
-- newundo(): keep track of ECHO stuff

syntax.lua
-- command(): add check for VESSEL2
-- setunitmap(): keep karma after undoing
-- createall(): keep karma after undoing (?)

map.lua
-- displaysigntext(): add check for ALIVE
-- unlockeffect(): also get ENTER units
-- mapunlock(): also get ENTER units
-- gateindicatorcheck(): also get ENTER units

convert.lua
-- conversion(): also check for ECHO
-- convert(): also check for ENTER
-- doconvert(): keep karma after a conversion or when undoing, add checks for ECHO

effects.lua
-- effects(): add particles for ECHO
-- doeffects(): add a "reducedlvl" option, to spawn less particles when LEVEL IS ECHO is formed

load.lua
-- init(): start with empty ECHO lists

mapcursor.lua
-- mapcursor_load(): check for ENTER
-- mapcursor_move(): ENTER
-- mapcursor_move(): update ECHO stuff when moving a cursor, also check for VEHICLE
-- mapcursor_enter(): ENTER
-- mapcursor_hardset(): ENTER
-- mapcursor_levelstart(): ENTER
-- mapcursor_displayname(): ENTER
-- idleblockcheck(): ENTER
-- cursorcheck(): ENTER
-- mapcursor_tofront(): ENTER
-- hidecursor(): ENTER
-- mapcursor_setonlevel(): ENTER

movement.lua
-- movecommand(): add checks for VESSEL and VESSEL2, set karma when a weak object moves into an obstacle unless the obstacle is REPENT, implement HOP/HOPS for direct movements, implement BOOST/BOOSTS
-- move(): add karma for OPEN/SHUT and EAT unless the sinner is REPENT
-- trypush(): add HOP/HOPS for pushable objects
-- dopush(): add karma for when a WEAK object is pushed into an obstacle unless it's REPENT, add HOP/HOPS for pushable objects

blocks.lua
-- moveblock(): keep karma after undoing, ECHO/BACK interaction
-- block(): add check for ALIVE, set karma for destructions by overlap or BOOM, implement REPENT, add special interaction for LEVEL IS ENTER
-- levelblock(): add check for ALIVE, set level karma for destructions unless level is REPENT
-- findplayer(): add check for ALIVE and VESSEL

update.lua
-- doupdate(): keep track of ECHO units

rules.lua
-- code(): look for ECHO units?
-- codecheck(): add rules from ECHO objects (?)
-- addoption(): implement AMBIENT (hacky af)

clears.lua
-- clearunits(): clear ECHO stuff
-- clear(): also clear ECHO stuff

letterunits.lua
-- formlettermap(): check for ECHO objects that are echoing letters

-- ========== OVERRIDDEN FUNCTIONS ==========]]

-- TODO list:
---- improve handling of karma units (add a map to keep track of which unitids have karma?)
---- fix ECHO bugs (ECHOing letters sometimes disables parsing, cannot ECHO multiple words in the same spot at once)
---- make AMBIENT less jank

-- Change the name of ALIGNEDX and ALIGNEDY in the rule list
word_names["alignedx"] = "x-aligned"
word_names["alignedy"] = "y-aligned"

-- Set the initial karma value of the outer level
table.insert(mod_hook_functions["level_start"],
    function()
		if (editor.values[E_INEDITOR] == 0 and WS_KEEP_LEVEL_KARMA) then -- We probably don't want to keep the karma status when entering a level from the editor
			levelKarma = ws_wasLevelSinful 
		else
			levelKarma = false
		end
	end
)

-- Clear the "Was level sinful" value when ending a level (we can't do that in the level start hook, because it's also called when restarting!)
table.insert(mod_hook_functions["level_end"],
    function()
		ws_wasLevelSinful = false
		ws_levelAlignedRow=false
		ws_levelAlignedColumn=false
		ws_ambientObject = "level"
	end
)

-------- VESSEL/ALIVE FUNCTIONS --------

-- Checks if the given property is a "player" property
function ws_isPlayerProp(property)
	return property == "you" or property == "you2" or property == "3d" or property == "alive"
end


-- Checks if the level is a "player"
function ws_isLevelPlayer(i,j)
	if (i == nil) or (j == nil) then
		return (hasfeature("level","is","you",1) ~= nil) or (hasfeature("level","is","you2",1) ~= nil) or (hasfeature("level","is","3d",1) ~= nil) or (hasfeature("level","is","alive",1) ~= nil)
	else
		return (hasfeature("level","is","you",1,i,j) ~= nil) or (hasfeature("level","is","you2",1,i,j) ~= nil) or (hasfeature("level","is","3d",1,i,j) ~= nil) or (hasfeature("level","is","alive",1,i,j) ~= nil)
	end
end


-- Returns all the features that match a "player" property
function ws_findPlayers()
	local yous = findfeature(nil,"is","you") or {}
	local yous2 = findfeature(nil,"is","you2")
	local yous3 = findfeature(nil,"is","3d")
	local yous4 = findfeature(nil,"is","alive") -- Added check for ALIVE

	if (yous2 ~= nil) then
		for _,v in ipairs(yous2) do
			table.insert(yous, v)
		end
	end

	if (yous3 ~= nil) then
		for _,v in ipairs(yous3) do
			table.insert(yous, v)
		end
	end

	if (yous4 ~= nil) then -- Add ALIVE entities to "yous"
		for _,v in ipairs(yous4) do
			table.insert(yous, v)
		end
	end
	
	return yous
end


-- Checks if empty should be considered a player
function ws_areTherePlayerEmpties()
	return (#findallfeature("empty","is","you") > 0) or (#findallfeature("empty","is","you2") > 0) or (#findallfeature("empty","is","3d") > 0) or (#findallfeature("empty","is","alive") > 0)
end


-- Returns all the features of the type "level is you-id" or "level is vessel-id"
function ws_findLevelVessel(id_)
	local levelmove
	local id = id_ or ""
	
	levelmove = findfeature("level","is","you"..id) or {}
	local levelvessel = findfeature("level","is","vessel"..id)
	if (levelvessel ~= nil) then
		for i,v in ipairs(levelvessel) do
			table.insert(levelmove, v)
		end
	end
	
	return levelmove
end


-- Returns all the features of the type "x/empty is you-id" or "x/empty is vessel-id"
function ws_findVessels(id_)
	local players = {}
	local empty = {}
	local id = id_ or ""
	
	players,empty = findallfeature(nil,"is","you"..id)
	local vessels,emptyvessels = findallfeature(nil,"is","vessel"..id)
	if (vessels ~= nil) then
		for i,v in ipairs(vessels) do
			table.insert(players, v)
		end
		for i,v in ipairs(emptyvessels) do
			table.insert(empty, v)
		end
	end
	
	return players,empty
end


-- Returns all the features at a certain position that are "players"
function ws_findplayerfeatureat(x,y)
	local result = findfeatureat(nil,"is","you",x,y) or {}
	local you2 = findfeatureat(nil,"is","you2",x,y) or {}
	local you3 = findfeatureat(nil,"is","3d",x,y) or {}
	local you4 = findfeatureat(nil,"is","alive",x,y) or {}

	for _,v in ipairs(you2) do
		table.insert(result, v)
	end

	for _,v in ipairs(you3) do
		table.insert(result, v)
	end

	for _,v in ipairs(you4) do
		table.insert(result, v)
	end

	return result
end


-------- KARMA/SINFUL/REPENT FUNCTIONS --------

-- Sets the level karma to a new value (true by default) and adds it to the undo queue if it was changed
function ws_setLevelKarma(_newKarma)
	local newKarma = true
	if (_newKarma ~= nil) then 
		newKarma = _newKarma 
	end
	if (levelKarma ~= newKarma) then
		addundo({"levelkarma", levelKarma})
		levelKarma = newKarma
	end
end


-- Sets the karma of an object to a new value (true by default) and adds it to the undo queue if it was changed
function ws_setKarma(unitid, _newKarma)
	local unit = mmf.newObject(unitid)
	local newKarma = true
	if (_newKarma ~= nil) then 
		newKarma = _newKarma 
	end
	if (unit.karma ~= newKarma) then
		addundo({"unitkarma",unit.values[ID], unit.karma})
		unit.karma = newKarma
	end
end


--[[ Apply karma to destroyer tiles: 
PARAMS
	x, y: position of the victim
	reason: why the destroyer killed something (eg "melt", "weak", "bonus" etc)
	victimid: the unitid of the object that got destroyed
]]
-- Do we even need to pass/return delthese, removalshort and removalsound? Seems to work ¯\_(ツ)_/¯
function ws_karma(x, y, reason, victimid, delthese, removalshort, removalsound)
	if reason == "melt" then
		local hotfeatures = findfeatureat(nil,"is","hot",x,y)
		for _,hotid in ipairs(hotfeatures) do
			if floating(hotid,victimid,x,y) then -- Only the lavas on the same float level of the melted object are evil
				delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,hotid,delthese,removalshort,removalsound)
			end
		end
	elseif reason == "weak" then
		local others = findallhere(x,y) -- Get all the other overlapping items
		for _,otherid in ipairs(others) do
			if floating(otherid,victimid,x,y) and (otherid ~= victimid) then -- Cornercase, but don't karma self
				delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,otherid,delthese,removalshort,removalsound)
			end
		end
	elseif reason == "defeat" then
		local defeatfeatures = findfeatureat(nil,"is","defeat",x,y)
		for _,defeatid in ipairs(defeatfeatures) do
			if floating(defeatid,victimid,x,y) then -- Only the skulls on the same float level of the defeated object are evil
				delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,defeatid,delthese,removalshort,removalsound)
			end
		end
	elseif reason == "bonus" then
		local playerfeatures = ws_findplayerfeatureat(x,y) --(find all the features that are you, you2, 3d, alive)
		for _,playerid in ipairs(playerfeatures) do
			if floating(playerid,victimid,x,y) then
				delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,playerid,delthese,removalshort,removalsound)
			end
		end
	elseif reason == "sink" then -- Object got sinked by something that is SINK
		local sinkfeatures = findfeatureat(nil,"is","sink",x,y)
		for _,sinkid in ipairs(sinkfeatures) do
			if floating(sinkid,victimid,x,y) then -- Only the sinkers on the same float level of the defeated object are evil
				delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,sinkid,delthese,removalshort,removalsound)
			end
		end
	elseif reason == "sink2" then -- Object that is SINK got sinked by something else
		-- @mods(word salad x plasma) - rewrote this entire section since guard allows units to still survive while not being "safe".
		local others = findallhere(x,y) -- Get all the other overlapping items
		for _,otherid in ipairs(others) do
			if otherid ~= victimid and floating(otherid,victimid,x,y) and not ws_isrepent(otherid,x,y) then -- Only scan objects that aren't the victim with the same float level
				if is_unit_guarded(otherid) or issafe(otherid) then -- Objects that survive (either through SAFE or GUARD) after the sink action are evil
					ws_setKarma(otherid)
				end
			end
		end
		-- local sinkerfeatures = findfeatureat(nil,"is","safe",x,y) -- Only the survived items are to blame
		-- for _,sinkerid in ipairs(sinkerfeatures) do
		-- 	if floating(sinkerid,victimid,x,y) then -- Only the skulls on the same float level of the defeated object are evil
		-- 		ws_setKarma(sinkerid) -- The sinker is safe, we can directly set its karma
		-- 	end
		-- end
	end
	return delthese, removalshort, removalsound
end


--[[ Destroy tiles by karma, or sets the karma flag, unless the object is REPENT
-- x, y: pos of the tile
-- karmaid: id of the unit to destroy
-- delthese: list of unitids to destroy (passed by reference? do we need to return it?)
]]
function ws_setKarmaOrDestroy(x, y, karmaid, delthese, removalshort, removalsound)
	local karmaunit = mmf.newObject(karmaid)
	if not ws_isrepent(karmaid,x,y) then
		if hasfeature(getname(karmaunit), "is", "karma", karmaid) and not issafe(karmaid) then
		local pmult,sound = checkeffecthistory("karma")
		MF_particles("unlock",x,y,5 * pmult,2,2,1,1)
		removalshort = sound
		removalsound = 1
		generaldata.values[SHAKE] = 4
		table.insert(delthese, karmaid)

		--[[ 
			@mods(word_salad x plasma) - The below line wasn't added in the original word salad since at this point in the
			code, the karma unit is set to be destroyed. Why bother updating the karma status when the unit will be destroyed?
			Guard however can "rescue" the karma unit from actually being destroyed, meaning the karma unit is in
			delthese but the unit survives. So we still need to update the unit's karma status.
		]]
		if is_unit_guarded(karmaid) then
			ws_setKarma(karmaid)
		end
	else
		ws_setKarma(karmaid)
	end
	end
	return delthese, removalshort, removalsound
end

-- Function to check if something is repent (basically the same as issafe)
function ws_isrepent(unitid,x,y)
	name = ""
	
	if (unitid ~= 1) and (unitid ~= 2) then
		local unit = mmf.newObject(unitid)
		name = unit.strings[UNITNAME]
	elseif (unitid == 1) then
		name = "level"
	else
		name = "empty"
	end
	
	local safe = hasfeature(name,"is","repent",unitid,x,y)
	
	if (safe ~= nil) then
		return true
	end
	
	return false
end


-------- HOP/HOPS FUNCTIONS --------

-- Function to check if a unit *could* hop if it bumped into an obstacle: that is, if the unit is HOP, or if it HOPS any unit in the next tile
function ws_couldHop(unitid,x,y,ox,oy)
	local name = ""
	if (unitid ~= 2) then
		name = getname(mmf.newObject(unitid))
	else
		name = "empty"
	end
	-- First, check if the unit is HOP
	if hasfeature(name,"is","hop",unitid,x,y) then return true end
	-- Then, we check if it HOPS anything in the next tile
	if (featureindex["hops"] ~= nil) then
		local obs = findobstacle(x+ox,y+oy)
		if (#obs > 0) then -- The next tile isn't empty
			for i,obsid in ipairs(obs) do
				if (obsid == -1) then -- Bumping into the level edge, check if "x HOPS LEVEL"
					if hasfeature(name,"hops","level",unitid,x,y) then return true end
				else -- Bumping into something, check if the unit hops anything
					local obsunit = mmf.newObject(obsid)
					local obsname = getname(obsunit)
					-- timedmessage("Checking if "..name.." hops "..obsname,0,i)
					if hasfeature(name,"hops",obsname,unitid,x,y) then return true end
				end
			end
		else -- The next tile is empty, check if "x HOPS EMPTY"
			if hasfeature(name,"hops","empty",unitid,x,y) then return true end
		end
	end

	return false
end


-------- ECHO FUNCTIONS --------

-- Function to query the text data from the echo map, given the name of a echo unit
function ws_getTextDataFromEchoMap(unitname)
	local matching_texts = echomap[unitname] or {}
	-- If the object being echo is the outer level, we also add all the texts from the outer level. hooray for meta stuff
	if (unitname == "level") then
		local outer_text = echomap["_outerlevel"] or {}
		for _,outer_data in ipairs(outer_text) do
			table.insert(matching_texts, outer_data)
		end
	end

	return matching_texts
end

-- Function to check for changes to echo units (similar to checkwordchanges in utils.lua)
function ws_checkechochanges(unitid)
	if (#echounits > 0) then
		for i,v in ipairs(echounits) do
			if (v[1] == unitid) then
				updatecode = 1
				return
			end
		end
	end
	
	if (#echorelatedunits > 0) then
		for i,v in ipairs(echorelatedunits) do
			if (v[1] == unitid) then
				updatecode = 1
				return
			end
		end
	end
end
-- Find echo units
-- How to implement the echo map(?):
-- ICE -> [text_baba, text_keke]
-- BABA -> []
-- KEKE -> [text_keke] etc.
-- When checking for echo units, remove overlapping texts from list
-- 
function ws_findechounits()
	local result = {}
	local alreadydone = {}
	local checkrecursion = {}
	local related = {}
	
	local identifier = ""
	local fullid = {}
	
	if (featureindex["echo"] ~= nil) then -- Something is ECHO
		for i,v in ipairs(featureindex["echo"]) do -- For all rules that contain ECHO
									-- Example for KEKE ON BABA AND NOT NEAR ME IS ECHO
			local rule = v[1]		-- The basic rule: {"keke", "is", "echo"}
			local conds = v[2]		-- The conditions: {"on", {"baba"}}, {"not near", {"me"}}
			local ids = v[3]		-- Unit ids of the text pieces: {{unitid of "keke"}, {unitid of "is"}, {unitid of "echo"}...}
			
			local name = rule[1]	-- The object in the rule being ECHO ("keke", "not baba" etc.)
			local subid = ""

			if (rule[2] == "is") then	-- Only rules of the type "x IS echo" are valid
				if (objectlist[name] ~= nil) and (name ~= "text") and (alreadydone[name] == nil) then -- TEXT can't be ECHO
					local these = findall({name,{}}) -- Find all unit ids with the given name and no conditions
					alreadydone[name] = 1 -- This specific name is already checked (prevents checking for ECHO KEKEs twice for example)
					
					if (#these > 0) then -- If there are objects
						for _,b in ipairs(these) do
							local bunit = mmf.newObject(b)
							local valid = true
							
							if (featureindex["broken"] ~= nil) then -- Skip BROKEN objects
								if (hasfeature(getname(bunit),"is","broken",b,bunit.values[XPOS],bunit.values[YPOS]) ~= nil) then
									valid = false
								end
							end
							
							if valid then
								table.insert(result, {b, conds}) -- Apped the unit ids + conditions to the result table
								subid = subid .. name -- From "" to "name" (e.g. "" -> "keke")
								-- LISÄÄ TÄHÄN LISÄÄ DATAA (translated: Add more data here)
							end
						end
					end
				end
				
				if (#subid > 0) then -- If subid has lenght > 0 (there was at least one valid unit)
					for _,b in ipairs(conds) do		-- conds is {"on", {"baba"}}, {"not near", {"me"}}
						local condtype = b[1]		-- The operator (ON, NEAR, LONELY...)
						local params = b[2] or {}	-- The parameters (on BABA, near NOT WALL), empty for prefixes
						
						subid = subid .. condtype	-- from "namenamename" to "namenamenamecondtype"? Seems so
						
						if (#params > 0) then		-- If the condition has parameters
							for _,d in ipairs(params) do -- For all condition parameters
								subid = subid .. tostring(d) -- append them to subid
								
								related = findunits(d,related,conds) -- Related is initially empty
							end
						end
					end
				end
				
				table.insert(fullid, subid) -- Insert the subids inside the fullid table ("kekekekeonwall", "baba" etc.)
				
				--MF_alert("Going through " .. name)
				
				if (#ids > 0) then -- Unitids of the texts that make the "x is echo" rules
					if (#ids[1] == 1) then -- subject of rule has exactly one entry
						local firstunit = mmf.newObject(ids[1][1])

						local notname = name
						if (string.sub(name, 1, 4) == "not ") then
							notname = string.sub(name, 5)
						end
						
						-- If the subject of rule isn't "text_name" or "text_name" (not removed), check for recursion
						if (firstunit.strings[UNITNAME] ~= "text_" .. name) and (firstunit.strings[UNITNAME] ~= "text_" .. notname) then 
							--MF_alert("Checking recursion for " .. name)
							-- timedmessage("Found echo rule starting with non text object", 0, 1) -- DEBUG
							table.insert(checkrecursion, {name, i}) -- Add the object name to the list of stuff to check recursion for (pairs of {name, index of rule with echo})
						end
					end
				else
					MF_alert("No ids listed in Echo-related rule! rules.lua line 1302 - this needs fixing asap (related to grouprules line 1118)") -- um, this was clearly copied from the WORD code
				end
			end
		end
		
		table.sort(fullid) -- After checking all the rules with echo, sort the table of fullid
		for _,v in ipairs(fullid) do
			-- MF_alert("Adding " .. v .. " to id")
			identifier = identifier .. v -- Concatenate all the fullids "kekekekeonwallbabafrogfrog" etc.
		end
		
		--MF_alert("Identifier: " .. identifier)
		
		-- ISSUE IS PROBABLY HERE!!
		for _,checkname_ in ipairs(checkrecursion) do -- For all objects in the "check recursion table"
			local found = false
			
			local checkname = checkname_[1] -- The name of the object to check recursion for
			
			local b = checkname -- Actual name without the "not " at the beginning
			if (string.sub(b, 1, 4) == "not ") then
				b = string.sub(checkname, 5)
			end
			
			for _,v in ipairs(featureindex["echo"]) do -- Check all rules with "echo" again
				local rule = v[1]	-- The basic rule: {"keke", "is", "echo"}
				local ids = v[3]	-- Unit ids of the text pieces: {{unitid of "keke"}, {unitid of "is"}, {unitid of "echo"}...}
				local tags = v[4]	-- Rule tags such as "base" or "mimic"
				
				-- If the rule subject is the object, ALL, or not the object and the subject starts with not
				if (rule[1] == b) or (rule[1] == "all") or ((rule[1] ~= b) and (string.sub(rule[1], 1, 3) == "not")) then 
					for _,g in ipairs(ids) do
						for _,d in ipairs(g) do
							local idunit = mmf.newObject(d)	-- Single text piece making up part of a word of the rule
							
							-- Tässä pitäisi testata myös Group! (Translated: group should be tested here too)
							if (idunit.strings[UNITNAME] == "text_" .. rule[1]) or (rule[1] == "all") then
								--MF_alert("Matching objects - found")
								found = true
							elseif (string.sub(rule[1], 1, 5) == "group") then
								--MF_alert("Group - found")
								found = true
							elseif (rule[1] ~= checkname) and (string.sub(rule[1], 1, 3) == "not") then
								--MF_alert("Not Object - found")
								found = true
							-- TEST!!!!
							elseif (idunit.strings[UNITNAME] ~= b) then
								-- The unit in the rule isn't the unit that risks being recursive
								found = true
							end
							-- timedmessage("d name is "..idunit.strings[UNITNAME],0,13+www) -- DEBUG
						end
					end
					
					for _,g in ipairs(tags) do -- Check for mimiced rules
						if (g == "mimic") then
							found = true
						end
					end
				end
			end
			
			if (found == false) then -- If the rule wasn't caused by any text (?)
			
				-- timedmessage("Possible recursion for "..b, 0, 12) -- DEBUG
				--MF_alert("Wordunit status for " .. b .. " is unstable!")
				identifier = "null"
				echounits = {} -- Clear all word units???
				
				for _,v in pairs(featureindex["echo"]) do -- Check all the rules with echo once more (???)
					local rule = v[1]
					local ids = v[3]
					
					--MF_alert("Checking to disable: " .. rule[1] .. " " .. ", not " .. b)
					
					if (rule[1] == b) or (rule[1] == "not " .. b) then
						v[2] = {{"never",{}}} -- Disable rules 
					end
				end
				
				if (string.sub(checkname, 1, 4) == "not ") then
					local notrules_word = notfeatures["echo"]
					local notrules_id = checkname_[2]
					local disablethese = notrules_word[notrules_id]
					
					for _,v in ipairs(disablethese) do
						v[2] = {{"never",{}}}
					end
				end
			end
		end
	end
	
	--MF_alert("Current id (end): " .. identifier)
	
	--[[
	for i,name in ipairs(result) do
		timedmessage(name[1], 2, 2 + i) -- Print the id of all the objects that could be echo (ignoring conditions)
	end
	--]]
	
	-- Populate echomap:

	echomap = {} -- Clear the echomap
	-- local all_echo_unitids = findallfeature(nil, "is", "echo", true)
	local all_echo_units = getunitswitheffect("echo")
	-- For each echo object
	for _,unit in ipairs(all_echo_units) do
		local unit_name = getname(unit)
		-- Skip text objects
		if (unit_name ~= "text") then
			 -- We use the unit name as the key, but it might be the first time it has to be added to the echo map
			if (not echomap[unit_name]) then
				echomap[unit_name] = {}
			end
			-- We then get all the text objects at the same position
			local this_x = unit.values[XPOS]
			local this_y = unit.values[YPOS]

			if (gettilenegated(this_x, this_y) == false) then
				local text_ids = findtype({"text"}, this_x, this_y) -- We don't need to specify the id of this unit because it can't be text
			
				-- EXPERIMENT: support echoing word units
				if (WS_CAN_ECHO_WORD_UNITS) then
					for _,words in ipairs(wordunits) do -- The echomap is always filled after the word map, so this should work??
						if (words[1] ~= unit.fixed) then -- If a unit is both ECHO and WORD at the same time, it can't add itself to the echo map
							local word_unit = mmf.newObject(words[1])
							if (word_unit.values[XPOS] == this_x) and (word_unit.values[YPOS] == this_y) then
								table.insert(echomap[unit_name], {word_unit.strings[NAME], 0, this_x + this_y*roomsizex}) -- WORD objects are always nouns
							end
						end
					end
				end
			
				for _,textid in ipairs(text_ids) do
					local text_unit = mmf.newObject(textid)
					-- Pair of {name, type, position}, for example {"baba", 0, 6}, {"win", 2, 11} etc.
					-- The position is used to skip overlapping texts
					local text_data = {text_unit.strings[NAME], text_unit.values[TYPE], this_x + this_y*roomsizex} 
					text_data.echotext_unitid = textid --@Merge(Word Salad x Plasma) Store the text id of the text unit overlapping the ECHO unit. This is used in codecheck() to allow support for turning text and pointer nouns
					table.insert(echomap[unit_name], text_data)
				end
				-- Special case for LEVEL: aside from normal ECHO, we also repeat any text at the level's position on the map (including the level itself if it was converted)
				-- We need to check this only the first time we have LEVEL IS ECHO, since this can't be changed from within the level
				if (unit_name == "level") then
					if (not echomap["_outerlevel"]) then
						echomap["_outerlevel"] = {}
						if hasfeature("level","is","echo",1) then
							local overlapping = ws_overlapping_texts or {} -- Should  never be nil, but let's be safe
							for i,ovtextdata in ipairs(overlapping) do
								if (unitreference["text_"..ovtextdata[1]] ~= nil) then -- Make sure that the text being echoed is in the level palette
									table.insert(echomap["_outerlevel"], ovtextdata)
								else
									timedmessage("unitreference for " .. ovtextdata[1] .. " was nil!", 0, 2)
								end
							end
						end
					end
				end
			end
		end
	end
	--[[
		(hopefully) We should end up with something like
		echomap = {baba: {{"win", 2, 7}, {"make", 1, 3}}, keke: {{"green", 2, 7}}, ice: {}, flag: {{"baba", 0, 11}}}
	--]]
	
	return result,identifier,related
end


-------- VEHICLE/ENTER FUNCTIONS --------

-- Gets cursors that can enter levels (SELECT or ENTER)
function ws_getSelectOrEnterUnits(nolevels_,ignorethese_,checkedconds,ignorebroken_)
	local result = getunitswitheffect("select",nolevels_,ignorethese_,checkedconds,ignorebroken_)
	local enterCursors = getunitswitheffect("enter",nolevels_,ignorethese_,checkedconds,ignorebroken_)
	
	for _,v in ipairs(enterCursors) do
		table.insert(result, v)
	end
	
	return result
end

-- Gets cursors that can move on paths or opened levels (SELECT or VEHICLE)
function ws_getSelectOrVehicleUnits(nolevels_,ignorethese_,checkedconds,ignorebroken_)
	local result = getunitswitheffect("select",nolevels_,ignorethese_,checkedconds,ignorebroken_)
	local vehicleCursors = getunitswitheffect("vehicle",nolevels_,ignorethese_,checkedconds,ignorebroken_)
	
	for _,v in ipairs(vehicleCursors) do
		table.insert(result, v)
	end
	
	return result
end