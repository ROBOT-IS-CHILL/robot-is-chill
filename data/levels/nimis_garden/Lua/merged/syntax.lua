

function command(key,player_,keyid,keyid2)
	--[[ 
		@mods(turning text) - Override reason: fixes a bug where if "level is auto" "baba is you" "bird is you2 (down)", pressing "S" will make both baba and bird move down. (Only bird should move down)
		@mods(past) - Override reason: handles direct keyids, turns off during past turns, and add keys.
	 ]]
	if doingpast == false then
		redokeys = {}
	end
	amundoing = false
	if keyid == nil then
		keyid = -1
	if (keys[key] ~= nil) then
		keyid = keys[key]
	else
		print("no such key")
		return
	end
	end
	
	local player = 1
	if (player_ ~= nil) then
		player = player_
	end
	
	do_mod_hook("command_given", {key,player})
	
	if (keyid <= 4) then
		if (generaldata5.values[AUTO_ON] == 0) or doingpast then
			local drs = ndirs[keyid+1]
			local ox = drs[1]
			local oy = drs[2]
			local dir = keyid
			
			last_key = keyid
			
			if (auto_dir[player] == nil) then
				auto_dir[player] = 4
			end
			
			auto_dir[player] = keyid
			
			if (spritedata.values[VISION] == 1) and (dir == 3) then
				if (#units > 0) then
					changevisiontarget()
				end
				movecommand(ox,oy,dir,player,nil,true)
				MF_update()
        		past_addkey(dir,player,keyid2)
			else
				movecommand(ox,oy,dir,player,keyid2)
				MF_update()
        		past_addkey(dir,player,keyid2)
			end
		else
			if (auto_dir[player] == nil) then
				auto_dir[player] = 4
			end
			
			auto_dir[player] = keyid
			
			if (auto_dir[1] == nil) and (featureindex["you2"] == nil) 
				and featureindex["you2right"] == nil 
				and featureindex["you2left"] == nil 
				and featureindex["you2up"] == nil 
				and featureindex["you2down"] == nil 
				and featureindex["vessel2"] == nil -- Added check for VESSEL2 here
				then
				auto_dir[1] = keyid
			end
		end
		--@mods(past x patashu) - if we hit a reset object during a past replay, don't handle reset here. Handle it in
		--the past mod. Need to do so since it uses the "always" modhook
		if doreset and not doingpast then
			resetlevel()
			MF_update()
		else
			resetmoves = math.max(0, resetmoves - 1)
		end
	end
	
	if (hasfeature("level","is","noreset",1) == nil) then
		if (keyid == 5) then
			MF_restart(false)

			-- @Question: do we want to not call this here? For reference, this was commented out in Version 463 in an attempt 
			-- to fix the restart modhook firing even when you say no to the restart prompt
			-- do_mod_hook("level_restart", {})
		elseif (keyid == 8) then
			MF_restart(true)
			-- do_mod_hook("level_restart", {})
		end
	end
	
	dolog(key)

	-- @mods(past)
	if (generaldata5.values[AUTO_ON] <= 0) or doingpast then
		dopast()
	end
end

function setunitmap()
	unitmap = {}
	unittypeshere = {}
	local delthese = {}
	
	local limit = 6
		
	if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) and ((generaldata.strings[CURRLEVEL] == "89level") or (generaldata.strings[CURRLEVEL] == "33level")) then
		limit = 3
	end
	
	if (generaldata.strings[WORLD] == "baba_m") and ((generaldata.strings[CURRLEVEL] == "89level") or (generaldata.strings[CURRLEVEL] == "33level")) then
		limit = 2
	end
	
	for i,unit in ipairs(units) do
		local tileid = unit.values[XPOS] + unit.values[YPOS] * roomsizex
		local valid = true
		
		--print(tostring(unit.values[XPOS]) .. ", " .. tostring(unit.values[YPOS]) .. ", " .. unit.strings[UNITNAME])
		
		if (unitmap[tileid] == nil) then
			unitmap[tileid] = {}
			unittypeshere[tileid] = {}
		end
		
		local uth = unittypeshere[tileid]
		local name = unit.strings[UNITNAME]
		
		if (uth[name] == nil) then
			uth[name] = 0
		end
		
		if (uth[name] < limit) then
			uth[name] = uth[name] + 1
		elseif (string.len(unit.strings[U_LEVELFILE]) == 0) then
			table.insert(delthese, unit)
			valid = false
		end
		
		if valid then
			table.insert(unitmap[tileid], unit.fixed)
		end
	end
	
	for i,unit in ipairs(delthese) do
		local x,y,dir,unitname = unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.strings[UNITNAME]
		addundo({"remove",unitname,x,y,dir,unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init,unit.originalname,unit.strings[UNITSIGNTEXT],false,unit.fixed,unit.karma}) -- EDIT: keep karma after undoing
		delunit(unit.fixed)
		MF_remove(unit.fixed)
	end
end

function command_auto()
	-- @mods(past)
	if doingpast == false then
		redokeys = {}
	end
	amundoing = false

	local moving = false
	local firstp = -1
	local secondp = -1

	if (auto_dir[1] ~= nil) then
		firstp = auto_dir[1]
		moving = true
	else
		firstp = 4
		moving = true
	end

	if (auto_dir[2] ~= nil) then
		secondp = auto_dir[2]
		moving = true
	else
		secondp = 4
		moving = true
	end

	do_mod_hook("turn_auto", {firstp,secondp,moving})

	if moving and (generaldata5.values[AUTO_ON] > 0) and not doingpast then
		for i=1,generaldata5.values[AUTO_ON] do
			if (firstp ~= 4) then
				last_key = firstp
			elseif (secondp ~= 4) then
				last_key = secondp
			else
				last_key = 4
			end

			local drs = ndirs[firstp+1]
			local ox = drs[1]
			local oy = drs[2]
			local dir = firstp

			if (spritedata.values[VISION] == 1) and (dir == 3) then
				if (#units > 0) then
					changevisiontarget()
				end
				movecommand(ox,oy,dir,3,secondp,true)
			else
				movecommand(ox,oy,dir,3,secondp)
			end
		end

		MF_update()
    	past_addkey(dir,3,secondp)
	end

	auto_dir = {}
  	if (generaldata5.values[AUTO_ON] > 0) and not doingpast then
		dopast()
	end
end