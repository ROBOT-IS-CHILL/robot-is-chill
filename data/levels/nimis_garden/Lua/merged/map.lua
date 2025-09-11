

function displaysigntext(x,y,ignore)
	if (ignore ~= nil) and ignore then
		return false
	end
	
	for i=-1,1 do
		for j=-1,1 do
			if (i == 0) or (j == 0) then
				local tileid = (x + i) + (y + j) * roomsizex
				
				if (unitmap[tileid] ~= nil) then
					local stuff = findfeatureat(nil,"is","you",x + i,y + j)
					
					if (stuff ~= nil) then
						return true
					elseif (featureindex["3d"] ~= nil) then
						stuff = findfeatureat(nil,"is","3d",x + i,y + j)
						
						if (stuff ~= nil) then
							return true
						end
					elseif (featureindex["you2"] ~= nil) then
						stuff = findfeatureat(nil,"is","you2",x + i,y + j)
						
						if (stuff ~= nil) then
							return true
						end
					elseif (featureindex["alive"] ~= nil) then
						stuff = findfeatureat(nil,"is","alive",x + i,y + j)
						
						if (stuff ~= nil) then
							return true
						end
					end
				end
			end
		end
	end
	
	return false
end

function uplevel()
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
	
	--this is the only line i added to this function for visit
	visit_fullsurrounds = ""

	return oldlevel
end

function unlockeffect(dataid)
	local data = mmf.newObject(dataid)
	
	if (hiddenmap == nil) then
		local cursors = ws_getSelectOrEnterUnits(nil,nil,nil,true) -- EDIT: also get ENTER units along with SELECT ones
		local cx,cy = 0,0
		
		if (#cursors > 0) then
			local cursorunit = cursors[1]
			cx = cursorunit.values[XPOS]
			cy = cursorunit.values[YPOS]
		end
		
		hiddenmap = {}
		hiddenmap.unlocks = {}
		hiddenmap.start = {cx,cy}
	end
	
	local start = hiddenmap.start
	local x = start[1]
	local y = start[2]
	
	data.values[UNLOCKTIMER] = data.values[UNLOCKTIMER] + 1
	local timer = data.values[UNLOCKTIMER]
	local unlock = data.values[UNLOCK]
	
	if (unlock == 2) then
		if (timer == 10) then
			MF_playsound("roll")
		end
		
		if (timer > 10) and (timer < 80) and (timer % 2 == 0) then
			particles("unlock",x,y,2,{2, 4})
		end
		
		--Aiemmin tässä oli 70 enemmän (jos haluat pitkän efektin)
		
		if (timer == 70) then
			generaldata.values[SHAKE] = 15
			
			if (data.values[MAPCLEAR] == 0) then
				generaldata.values[IGNORE] = 0
			end
			
			particles("smoke",x,y,20,{0, 3})
			
			local prizeid = MF_specialcreate("Prize")
			local prize = mmf.newObject(prizeid)
			
			prize.layer = 2
			prize.values[ONLINE] = 1
			prize.values[XPOS] = Xoffset + x * tilesize * spritedata.values[TILEMULT] + tilesize * 0.5 * spritedata.values[TILEMULT]
			prize.values[YPOS] = Yoffset + y * tilesize * spritedata.values[TILEMULT] + tilesize * 0.5 * spritedata.values[TILEMULT]
			prize.values[YVEL] = -20
			prize.scaleX = 0.1
			prize.scaleY = 0.1
			
			if (data.values[MAPCLEAR] == 1) then
				MF_playsound("clear")
			else
				MF_playsound("winnery_fast")
			end
			
			MF_playsound("pop3")
			MF_stopsound("roll")
			
			if (data.values[MAPCLEAR] == 1) then
				local winid1 = MF_specialcreate("Victorytext")
				local winid2 = MF_specialcreate("Victorytext_back")
				
				local wintext1 = mmf.newObject(winid1)
				local wintext2 = mmf.newObject(winid2)
				
				wintext2.layer = 2
				wintext1.layer = 2
				
				MF_setcolour(winid1,0,3)
				MF_setcolour(winid2,1,1)
				
				wintext1.x = screenw * 0.5
				wintext1.y = screenh * 0.5
				wintext1.direction = 1
				
				wintext2.x = screenw * 0.5
				wintext2.y = screenh * 0.5 + 4
				wintext2.direction = 1
				
				if (generaldata4.values[CUSTOMFONT] == 0) and (generaldata.strings[LANG] ~= "en") then
					wintext1.visible = false
					wintext2.visible = false
					
					displaybigtext("ingame_clear",{0,3,1,3},true)
				end
			end
		end
		
		if (timer == 110) then
			local hidden,hiddenmap_ = checkhidden(x,y)
			
			if hidden then
				hiddenmap = hiddenmap_
				hiddenmap.unlocks = {}
				hiddenmap.start = {x,y}				
				
				data.values[UNLOCK] = 3
				data.values[UNLOCKTIMER] = -25
				return
			end
		end
		
		local upoint = 130
		local ox,oy = 0,0
		
		local opensound = "whoosh_alt" .. tostring(math.random(1,5))
		if (matches ~= nil) then
			opensound = "intro_flower_" .. tostring(math.random(1,7))
		end
		
		if (timer == upoint) then
			ox = 0
			oy = -1
			local unlockables = findallhere(x+ox,y+oy)
			local found = false
			
			if (#unlockables > 0) then
				for i,unitid in ipairs(unlockables) do
					local unit = mmf.newObject(unitid)
					local pgate,preq = 0,0
					local pathid = MF_findpath_id(unitid)
					
					if (pathid ~= nil) then
						local path = mmf.newObject(pathid)
						pgate = path.values[PATH_GATE]
						preq = path.values[PATH_REQUIREMENT]
					end
					
					if (unit.values[COMPLETED] == 1) and ((pgate == 0) or (preq <= 0)) and (unit.flags[LEVEL_JUSTCONVERTED] == false) then
						unit.values[COMPLETED] = 2
						found = true
					end
				end
			end
			
			if found then
				particles("hot",x+ox,y+oy,10,{0, 3})
				MF_playsound(opensound)
			else
				timer = timer + 15
			end
		end
		
		if (timer == upoint + 15) then
			ox = -1
			oy = 0
			local unlockables = findallhere(x+ox,y+oy)
			local found = false
			
			if (#unlockables > 0) then
				for i,unitid in ipairs(unlockables) do
					local unit = mmf.newObject(unitid)
					local pgate,preq = 0,0
					local pathid = MF_findpath_id(unitid)
					
					if (pathid ~= nil) then
						local path = mmf.newObject(pathid)
						pgate = path.values[PATH_GATE]
						preq = path.values[PATH_REQUIREMENT]
					end
					
					if (unit.values[COMPLETED] == 1) and ((pgate == 0) or (preq <= 0)) and (unit.flags[LEVEL_JUSTCONVERTED] == false) then
						unit.values[COMPLETED] = 2
						found = true
					end
				end
			end
			
			if found then
				particles("hot",x+ox,y+oy,10,{0, 3})
				MF_playsound(opensound)
			else
				timer = timer + 15
			end
		end
		
		if (timer == upoint + 30) then
			ox = 0
			oy = 1
			local unlockables = findallhere(x+ox,y+oy)
			local found = false
			
			if (#unlockables > 0) then
				for i,unitid in ipairs(unlockables) do
					local unit = mmf.newObject(unitid)
					local pgate,preq = 0,0
					local pathid = MF_findpath_id(unitid)
					
					if (pathid ~= nil) then
						local path = mmf.newObject(pathid)
						pgate = path.values[PATH_GATE]
						preq = path.values[PATH_REQUIREMENT]
					end
					
					if (unit.values[COMPLETED] == 1) and ((pgate == 0) or (preq <= 0)) and (unit.flags[LEVEL_JUSTCONVERTED] == false) then
						unit.values[COMPLETED] = 2
						found = true
					end
				end
			end
			
			if found then
				particles("hot",x+ox,y+oy,10,{0, 3})
				MF_playsound(opensound)
			else
				timer = timer + 15
			end
		end
		
		if (timer == upoint + 45) then
			ox = 1
			oy = 0
			local unlockables = findallhere(x+ox,y+oy)
			local found = false
			
			if (#unlockables > 0) then
				for i,unitid in ipairs(unlockables) do
					local unit = mmf.newObject(unitid)
					local pgate,preq = 0,0
					local pathid = MF_findpath_id(unitid)
					
					if (pathid ~= nil) then
						local path = mmf.newObject(pathid)
						pgate = path.values[PATH_GATE]
						preq = path.values[PATH_REQUIREMENT]
					end
					
					if (unit.values[COMPLETED] == 1) and ((pgate == 0) or (preq <= 0)) and (unit.flags[LEVEL_JUSTCONVERTED] == false) then
						unit.values[COMPLETED] = 2
						found = true
					end
				end
			end
			
			if found then
				particles("hot",x+ox,y+oy,10,{0, 3})
				MF_playsound(opensound)
			end
		end
		
		if (timer == upoint + 50) then
			if (matches == nil) then
				if (data.values[MAPCLEAR] == 0) then
					data.values[UNLOCK] = 0
					data.values[UNLOCKTIMER] = 0
					generaldata.values[IGNORE] = 0
					hiddenmap = nil
				else
					data.values[MAPCLEAR] = 0
					data.values[UNLOCK] = 4
					data.values[UNLOCKTIMER] = -30
				end
			else
				data.values[UNLOCK] = 5
				data.values[UNLOCKTIMER] = 0
			end
		end
	elseif (unlock == 3) then
		local docheck = timer % 10
		
		if (docheck == 0) and (timer > 0) then
			local newstuff = {}
			local unlocks = {}
			local currtype = 0
			
			if (#hiddenmap > 0) then
				for i,v in ipairs(hiddenmap) do
					local unit = mmf.newObject(v)
					unit.values[COMPLETED] = math.max(unit.values[COMPLETED], 1)
					local ux,uy = unit.values[XPOS],unit.values[YPOS]
					
					if (unit.strings[UNITNAME] ~= "path") then
						--MF_savelevel(v, unit.values[COMPLETED])
						currtype = 0
					else
						--MF_savepath(v)
						table.insert(unlocks, {ux, uy})
						currtype = 1
					end
					
					MF_playsound("whoosh_quiet" .. tostring(math.random(1,5)))
					particles("glow",ux,uy,5,{1, 2})
					
					local th,things = checkhidden(ux,uy)
					
					if th then
						for c,d in ipairs(things) do
							local dunit = mmf.newObject(d)
							
							local addthis = false
							
							if ((dunit.strings[UNITNAME] ~= "path") and (currtype == 0)) or ((ux == x) and (uy == y)) then
								addthis = true
							elseif (dunit.strings[UNITNAME] == "path") and (currtype == 1) then
								addthis = true
								
								if (unit.strings[UNITNAME] == "path") and (unit.values[PATH_GATE] > 0) then
									addthis = false
								end
							end
							
							if addthis then
								table.insert(newstuff, d)
							end
						end
					end
				end
			end
			
			if (#hiddenmap.unlocks > 0) then
				for i,v in ipairs(hiddenmap.unlocks) do
					local ux,uy = v[1],v[2]
					unlocklevels(ux,uy,true)
				end
			end
			
			hiddenmap = {}
			hiddenmap.unlocks = {}
			hiddenmap.start = {x,y}
			
			if (#newstuff > 0) then
				for i,v in ipairs(newstuff) do
					table.insert(hiddenmap, v)
				end
			end
			
			if (#unlocks > 0) then
				for i,v in ipairs(unlocks) do
					table.insert(hiddenmap.unlocks, v)
				end
			end
			
			if (#newstuff == 0) and (#unlocks == 0) then
				hiddenmap = {}
				hiddenmap.unlocks = {}
				hiddenmap.start = {x,y}
				data.values[UNLOCK] = 2
				data.values[UNLOCKTIMER] = 110
			end
		end
	end	
end

function mapunlock(dataid)
	local data = mmf.newObject(dataid)
	local world = generaldata.strings[WORLD]
	local level = generaldata.strings[CURRLEVEL]
	
	data.values[UNLOCKTIMER] = data.values[UNLOCKTIMER] + 1
	local timer = data.values[UNLOCKTIMER]
	local unlock = data.values[UNLOCK]
	
	if (unlock == 4) then
		if (timer == 5) then
			local unlocklevels = MF_read("level","general","unlocklevels") or ""
			local levels = MF_parsestring(unlocklevels)
			
			matches = {}
			matches.origin = level
			matches.previous = data.strings[1]
			
			for i,v in ipairs(levels) do
				if (string.len(v) > 0) then
					table.insert(matches, v)
				end
			end
		end
		
		if (timer == 10) and (#matches == 0) then
			generaldata.values[IGNORE] = 0
			data.values[MAPCLEAR] = 0
			data.values[UNLOCK] = 0
			data.values[UNLOCKTIMER] = 0
			hiddemap = nil
			matches = nil
		end
	end
	
	if (unlock == 4) or (unlock == 5) then
		if (matches ~= nil) then
			if (timer == 10) and (#matches > 0) then
				data.values[UNLOCK] = 5
				unlock = data.values[UNLOCK]
			end
		end
	end
	
	if (unlock == 5) then
		if (timer == -80) and (generaldata.values[MODE] == 0) and ((editor.values[INEDITOR] == 0) or (editor.values[INEDITOR] == 3)) then
			MF_playsound("roll")
		end
		
		if (timer > -80) and (timer < -20) and (timer % 2 == 0) and (generaldata.values[MODE] == 0) and ((editor.values[INEDITOR] == 0) or (editor.values[INEDITOR] == 3)) then
			for i,unit in ipairs(units) do
				if (unit.strings[LEVELFILE] == matches.origin) then
					local x,y = unit.values[XPOS],unit.values[YPOS]
					particles("unlock",x,y,2,{2, 4})
				end
			end
		end
		
		if (timer == -20) and (generaldata.values[MODE] == 0) and ((editor.values[INEDITOR] == 0) or (editor.values[INEDITOR] == 3)) then
			for i,unit in ipairs(units) do
				if (unit.strings[LEVELFILE] == matches.origin) then
					if (hiddenmap == nil) then
						hiddenmap = {}
						hiddenmap.unlocks = {}
					end
					
					hiddenmap.start = {unit.values[XPOS],unit.values[YPOS]}
					
					generaldata.values[SHAKE] = 15
					
					data.values[UNLOCK] = 2
					data.values[UNLOCKTIMER] = 71
					
					unit.direction = 16
					unit.values[COMPLETED] = 3
					
					local prizeid = MF_specialcreate("Prize")
					local prize = mmf.newObject(prizeid)
					
					MF_playsound_freq("winnery_fast",38000)
					MF_playsound("pop3")
					MF_stopsound("roll")
					
					local x,y = unit.values[XPOS],unit.values[YPOS]
					prize.layer = 2
					prize.values[ONLINE] = 1
					prize.values[XPOS] = Xoffset + x * tilesize + tilesize * 0.5
					prize.values[YPOS] = Yoffset + y * tilesize + tilesize * 0.5
					prize.values[YVEL] = -20
					prize.scaleX = 0.1
					prize.scaleY = 0.1
					prize.direction = 1

					particles("smoke",unit.values[XPOS],unit.values[YPOS],20,{0,3})
				end
			end
		end
		
		if (timer == 10) and (#matches == 0) then
			data.values[UNLOCK] = 6
			unlock = data.values[UNLOCK]
		end
		
		if (timer == 60) then
			generaldata.values[TRANSITIONREASON] = 10
			generaldata.values[TRANSITIONED] = 0
			MF_loop("transition",1)
		end
	end
	
	if (unlock == 6) then
		if (timer == 80) then
			generaldata.values[TRANSITIONREASON] = 10
			generaldata.values[TRANSITIONED] = 0
			MF_loop("transition",1)
		end
	end
	
	if (unlock == 5) or (unlock == 6) then
		if (generaldata.values[TRANSITIONED] == 1) and (generaldata.values[TRANSITIONREASON] == 10) and (generaldata.values[MODE] == 0) and ((editor.values[INEDITOR] == 0) or (editor.values[INEDITOR] == 3)) then
			local cursors = ws_getSelectOrEnterUnits() -- EDIT: also get ENTER units along with SELECT ones
			
			for i,cursorunit in ipairs(cursors) do
				cursorunit.visible = false
			end
			
			mapunlock_transition(dataid,cursors)
		end
	end
end

function gateindicatorcheck(x,y)
	local cursors = ws_getSelectOrEnterUnits(nil,nil,nil,true) -- EDIT: also get ENTER units along with SELECT ones
	
	for i,unit in ipairs(cursors) do
		local x_ = unit.values[XPOS]
		local y_ = unit.values[YPOS]
		
		local dx = math.abs(x_ - x)
		local dy = math.abs(y_ - y)
		
		if (dx + dy <= 1) then
			return true
		end
	end
	
	return false
end