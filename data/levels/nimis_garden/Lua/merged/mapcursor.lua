

function mapcursor_move(ox_,oy_,mdir)
	--[[ 
		@mods(turning text) - Override reason: mod hook for directional select
	 ]]
	local dir = mdir or 4
    
    arrow_prop_mod_globals.group_arrow_properties = false
    local cursors = ws_getSelectOrVehicleUnits(true)
    for i,dircursorunit in ipairs(do_directional_select(mdir)) do
        table.insert(cursors, dircursorunit)
    end
    arrow_prop_mod_globals.group_arrow_properties = true
	
	for i,unit in ipairs(cursors) do
		local x_,y_,dir_ = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
		local currlevel = unit.values[CURSOR_ONLEVEL]
		local ox,oy = ox_,oy_
		
		if (featureindex["reverse"] ~= nil) then
			dir,ox,oy = reversecheck(unit.fixed,dir,x_,y_,ox_,oy_)
		end
		
		local unitname = getname(unit)
		local still = cantmove(unitname,unit.fixed,dir) --hasfeature(unitname,"is","still",unit.fixed,x_,y_)
		
		local x = x_ + ox
		local y = y_ + oy
		
		unit.flags[CURSOR_LIMITER] = false
		local levelfound = false
		local moved = false
		
		if (still == false) then
			local targets = findallhere(x,y,unit.fixed,true)
			
			for a,b in ipairs(targets) do
				local lunit = mmf.newObject(b)
				
				if lunit.visible and (lunit.flags[DEAD] == false) and (lunit.values[COMPLETED] > 1) then
					editor.values[NAMEFLAG] = 0
					moved = true
					
					if (unit.flags[CURSOR_LIMITER] == false) and (string.len(lunit.strings[U_LEVELFILE]) == 0) then
						unit.values[CURSOR_ONLEVEL] = b
						break
					elseif (string.len(lunit.strings[U_LEVELFILE]) > 0) then
						unit.values[CURSOR_ONLEVEL] = b
						unit.flags[CURSOR_LIMITER] = true
						levelfound = true
						break
					end
				end
			end
		end
		
		if moved then
			unit.values[XPOS] = x
			unit.values[YPOS] = y
			unit.values[DIR] = dir
			unit.values[POSITIONING] = 0
			addundo({"mapcursor",currlevel,x_,y_,dir_,unit.values[CURSOR_ONLEVEL],x,y,dir,unit.values[ID]})
			updateundo = true
			
			checkwordchanges(unit.fixed,unit.strings[UNITNAME])
			ws_checkechochanges(unit.fixed) -- Added check for ECHO
			
			if (levelfound == false) then
				editor.values[NAMEFLAG] = 0
				unit.values[CURSOR_ONLEVEL] = 0
			end
			
			if (unit.strings[UNITTYPE] == "text") then
				updatecode = 1
			end
			
			if (generaldata5.values[AUTO_ON] == 1) and (generaldata2.strings[TURNSOUND] == "") and (dir == 4) then
				generaldata2.strings[TURNSOUND] = "silent"
			end
		end
	end
end

function mapcursor_enter(varsunitid)
	local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
	local varsunit = mmf.newObject(varsunitid)
	local entering = {}
	
	for i,unit in ipairs(cursors) do
		local targetfound = MF_findunit_fixed(unit.values[CURSOR_ONLEVEL])
		
		if targetfound then
			local allhere = findallhere(unit.values[XPOS],unit.values[YPOS],unit.fixed)
			
			for a,b in ipairs(allhere) do
				local lunit = mmf.newObject(b)
				
				if (string.len(lunit.strings[U_LEVELFILE]) > 0) and (string.len(lunit.strings[U_LEVELNAME]) > 0) and (generaldata.values[IGNORE] == 0) and (lunit.values[COMPLETED] > 1) then
					local valid = true
					
					for c,d in ipairs(cursors) do
						if (d.fixed == b) then
							valid = false
							break
						end
					end
					
					if valid then
						table.insert(entering, {b, lunit.strings[U_LEVELNAME], lunit.strings[U_LEVELFILE]})
					end
				end
			end
		end
	end
	
	if (#entering > 0) then
		dolog("end","event")
	end
	
	if (#entering == 1) then
		findpersists("levelentry")
		generaldata2.values[UNLOCK] = 0
		generaldata2.values[UNLOCKTIMER] = 0
		varsunit.values[1] = entering[1][1]
		MF_loop("enterlevel", 1)
	elseif (#entering > 0) then
		findpersists("levelentry")
		MF_menuselector_hack(1)
		submenu("enterlevel_multiple",entering)
		print("Trying to enter multiple levels!")
	end
end

function mapcursor_load()
	local ix = tonumber(MF_read("level","general","selectorX")) or 0
	local iy = tonumber(MF_read("level","general","selectorY")) or 0
	
	generaldata4.values[MAINCURSOR] = 0
	
	if (ix > 0) and (iy > 0) then
		--MF_alert("Default cursor position detected at " .. tostring(ix) .. ", " .. tostring(iy))
		local tileid = ix + iy * roomsizex
		local createnew = true
		
		if (unitmap[tileid] ~= nil) and (#unitmap[tileid] > 0) then
			for a,b in ipairs(unitmap[tileid]) do
				local lunit = mmf.newObject(b)
				
				if (lunit.strings[UNITNAME] == "cursor") then
					--MF_alert("Another cursor detected at default cursor position")
					createnew = false
					generaldata4.values[MAINCURSOR] = b
				end
			end
		end
		
		if createnew and (unitreference["cursor"] ~= nil) then
			--MF_alert("Creating a cursor at " .. tostring(ix) .. ", " .. tostring(iy))
			local maincursorid = create("cursor",ix,iy,0,ix,iy,0,true)
			generaldata4.values[MAINCURSOR] = maincursorid
		end
	end
	
	local cursors = ws_getSelectOrEnterUnits(true,nil,nil,true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		unit.values[POSITIONING] = 7
		editor.values[NAMEFLAG] = 0
		local something = false
		
		if (generaldata4.values[MAINCURSOR] == 0) then
			generaldata4.values[MAINCURSOR] = unit.fixed
		end
		
		if (generaldata4.values[MAINCURSOR] == unit.fixed) then
			local parentlevel = MF_findlevelunit(generaldata.strings[PARENT])
			
			local parent_ini = MF_read("save",generaldata.strings[WORLD],"Previous")
			local previous = generaldata2.strings[PREVIOUSLEVEL]
			local parentlevel_ini = {}
			local previouslevel = {}
			
			if (parent_ini ~= nil) then
				parentlevel_ini = MF_findlevelunit(parent_ini)
			end
			
			if (#parentlevel > 0) then
				for a,b in ipairs(parentlevel) do
					local lunit = mmf.newObject(b)
					
					if (lunit.values[COMPLETED] >= 2) and (lunit.flags[DEAD] == false) then
						unit.values[XPOS] = lunit.values[XPOS]
						unit.values[YPOS] = lunit.values[YPOS]
						unit.values[CURSOR_ONLEVEL] = b
						
						unit.x = lunit.x
						unit.y = lunit.y
						something = true
						break
					end
				end
			end
			
			if (#parentlevel_ini > 0) then
				for a,b in ipairs(parentlevel_ini) do
					local lunit = mmf.newObject(b)
					
					if (lunit.values[COMPLETED] >= 2) and (lunit.flags[DEAD] == false) then
						unit.values[XPOS] = lunit.values[XPOS]
						unit.values[YPOS] = lunit.values[YPOS]
						unit.values[CURSOR_ONLEVEL] = b
						
						unit.x = lunit.x
						unit.y = lunit.y
						something = true
						break
					end
				end
			end
			
			--MF_alert(tostring(something) .. ", " .. tostring(parentlevel) .. ", " .. tostring(parent_ini) .. ", " .. tostring(previous))
			
			if (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) and (something == false) and (#previous > 0) then
				previouslevel = MF_findlevelunit(previous)
				
				for a,b in ipairs(previouslevel) do
					local lunit = mmf.newObject(b)
					
					if (lunit.values[COMPLETED] >= 2) and (lunit.flags[DEAD] == false) then
						unit.values[XPOS] = lunit.values[XPOS]
						unit.values[YPOS] = lunit.values[YPOS]
						unit.values[CURSOR_ONLEVEL] = b
						
						unit.x = lunit.x
						unit.y = lunit.y
						something = true
						break
					end
				end
			end
		end
	end
end

function mapcursor_idle()
	local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
	
	local bestoption = 0
	local maincursorfound = false
	
	for i,unit in ipairs(cursors) do
		if (unit.fixed == generaldata4.values[MAINCURSOR]) then
			maincursorfound = true
		elseif (bestoption == 0) then
			bestoption = unit.fixed
		end
		
		if (unit.values[CURSOR_ONLEVEL] ~= 0) and (unit.values[CURSOR_ONLEVEL] ~= -1) and (MF_findunit_fixed(unit.values[CURSOR_ONLEVEL])) then
			local lunit = mmf.newObject(unit.values[CURSOR_ONLEVEL])
			
			unit.values[XPOS] = lunit.values[XPOS]
			unit.values[YPOS] = lunit.values[YPOS]
		elseif (unit.values[CURSOR_ONLEVEL] == -1) then
			editor.values[NAMEFLAG] = 0
			
			local targets = findallhere(unit.values[XPOS],unit.values[YPOS],unit.fixed,true)
			local limiter = false
			
			for a,b in ipairs(targets) do
				local lunit = mmf.newObject(b)
				
				if (lunit.values[COMPLETED] > 1) and lunit.visible then
					if (string.len(lunit.strings[U_LEVELFILE]) == 0) and (limiter == false) then
						unit.values[CURSOR_ONLEVEL] = b
					elseif (string.len(lunit.strings[U_LEVELFILE]) > 0) then
						unit.values[CURSOR_ONLEVEL] = b
						limiter = true
					end
				end
			end
		else
			unit.values[CURSOR_ONLEVEL] = 0
			
			if (generaldata4.values[DISPLAYLEVEL] == unit.fixed) then
				generaldata4.values[DISPLAYLEVEL] = 0
				editor.values[NAMEFLAG] = 0
			end
		end
		
		if (generaldata.values[MODE] == 0)  then
			if (generaldata.values[WINTIMER] == 0) and ((generaldata2.values[UNLOCK] < 2) or ((generaldata2.values[MAPCLEAR] == 0) and (matches == nil) and (generaldata2.values[UNLOCK] >= 2) and (generaldata2.values[UNLOCK] <= 3))) then
				unit.visible = true
			else
				unit.visible = false
			end
		end
	end
	
	if (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) and (maincursorfound == false) then
		generaldata4.values[MAINCURSOR] = bestoption
	end
end

function mapcursor_hardset(lunitid)
	local lunit = mmf.newObject(lunitid)
	
	local cursors = ws_getSelectOrEnterUnits(true,nil,nil,true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		unit.values[XPOS] = lunit.values[XPOS]
		unit.values[YPOS] = lunit.values[YPOS]
		unit.values[CURSOR_ONLEVEL] = lunitid
		
		unit.x = lunit.x
		unit.y = lunit.y
		
		unit.values[POSITIONING] = 7
	end
end

function mapcursor_levelstart()
	local cursors = ws_getSelectOrEnterUnits(true,nil,nil,true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		
		local targets = findallhere(x,y,unit.fixed,true)
		
		for a,b in ipairs(targets) do
			local lunit = mmf.newObject(b)
			
			if (lunit.values[COMPLETED] > 1) then
				unit.values[CURSOR_ONLEVEL] = b
				break
			end
		end
	end
end

function mapcursor_displayname()
	local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		if (unit.values[CURSOR_ONLEVEL] ~= 0) and (unit.values[CURSOR_ONLEVEL] ~= -1) and (MF_findunit_fixed(unit.values[CURSOR_ONLEVEL])) and unit.visible then
			local lunit = mmf.newObject(unit.values[CURSOR_ONLEVEL])
			
			local valid = true
					
			for c,d in ipairs(cursors) do
				if (d.fixed == b) then
					valid = false
					break
				end
			end
			
			if valid and (string.len(lunit.strings[U_LEVELNAME]) > 0) and (lunit.values[COMPLETED] > 1) then
				editor.values[NAMEFLAG] = 1
				generaldata4.values[DISPLAYLEVEL] = unit.fixed
				
				displaylevelname(lunit.strings[U_LEVELNAME],lunit.strings[U_LEVELFILE],2)
			end
		end
	end
end

function idleblockcheck()
	local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		if (unit.values[CURSOR_ONLEVEL] ~= 0) and (unit.values[CURSOR_ONLEVEL] ~= -1) then
			local lunit = mmf.newObject(unit.values[CURSOR_ONLEVEL])
			
			if (string.len(lunit.strings[U_LEVELFILE]) > 0) then
				return true
			end
		end
	end
	
	return false
end

function cursorcheck()
	local result = 0
	
	if (featureindex["select"] ~= nil) or (featureindex["enter"] ~= nil) then -- EDIT: also check for ENTER
		local cursors = ws_getSelectOrEnterUnits(true)
		
		if (#cursors > 0) then
			result = 1
		end
	end
	
	return result
end

function mapcursor_tofront()
	if (spritedata.values[VISION] == 0) then
		local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
		
		for i,unit in ipairs(cursors) do
			if (unit.strings[UNITTYPE] ~= "text") then
				unit.moveToFront()
			end
		end
	end
end

function hidecursor()
	local cursors = ws_getSelectOrEnterUnits(true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		unit.visible = false
	end
end

function mapcursor_setonlevel(value)
	local cursors = ws_getSelectOrEnterUnits(true,nil,nil,true) -- EDIT: also check for ENTER
	
	for i,unit in ipairs(cursors) do
		unit.values[CURSOR_ONLEVEL] = value
	end
end