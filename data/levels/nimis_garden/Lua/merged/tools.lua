function writerules(parent,name,x_,y_)
	--[[ 
		@mods(this) - Override reason: Custom "this" rule display. Also remove unitid display when 
			forming "this(X) is float" and "Y mimic X"
	 ]]
	local basex = x_
	local basey = y_
	local linelimit = 12
	local maxcolumns = 4
	
	local x,y = basex,basey
	
	if (#visualfeatures > 0) then
		writetext(langtext("rules_colon"),0,x,y,name,true,2,true)
	end
	
	local i_ = 1
	
	local count = 0
	local allrules = {}
	
	local custom = MF_read("level","general","customruleword")
	
	for i,rules in ipairs(visualfeatures) do
		local text = ""
		local rule = rules[1]
		
		if (#custom == 0) then
			-- EDIT: implement AMBIENT
			local target = rule[1]
			local target_ = target
			local isnot = string.sub(target, 1, 4)
			if (isnot == "not ") then
				target_ = string.sub(target, 5)
			else
				isnot = ""
			end
			if (target_ == "ambient") then -- EDIT: implement AMBIENT
				text = text .. isnot .. target_ .. " (" .. ws_ambientObject .. ") "
			else
				text = text .. target .. " "
			end
		else
			text = text .. custom .. " "
		end
		
		local conds = rules[2]
		local ids = rules[3]
		local tags = rules[4]
		
		local fullinvis = true
		for a,b in ipairs(ids) do
			for c,d in ipairs(b) do
				local dunit = mmf.newObject(d)
				
				if dunit.visible then
					fullinvis = false
				end
			end
		end
		
		if (fullinvis == false) then
			if (#conds > 0) then
				local num_this_conds = 0
				local this_cond = ""
				for a,cond in ipairs(conds) do
					local condtype = plasma_utils.real_condtype(cond[1])
					if condtype == "this" or condtype == "not this" then
						num_this_conds = num_this_conds + 1
						local pnoun_unitid = parse_this_unit_from_param_id(cond[2][1])
						local pnoun_unit = mmf.newObject(pnoun_unitid)

						if condtype == "this" then
							this_cond = pnoun_unit.strings[NAME]
						else
							this_cond = "not "..pnoun_unit.strings[NAME]
						end
					end
				end
				if num_this_conds > 0 then
					text = this_cond.." ("..rule[1]..")".." "
				end 

				for a,cond in ipairs(conds) do
					local middlecond = true
					
					if (cond[2] == nil) or ((cond[2] ~= nil) and (#cond[2] == 0)) then
						middlecond = false
					end

					local condtype = plasma_utils.real_condtype(cond[1])
					if condtype == "this" or condtype == "not this" then
					elseif middlecond then
						if (#custom == 0) then
							local target = cond[1]
							local isnot = string.sub(target, 1, 4)
							local target_ = target
							
							if (isnot == "not ") then
								target_ = string.sub(target, 5)
							else
								isnot = ""
							end
							
							if (word_names[target_] ~= nil) then
								target = isnot .. word_names[target_]
							end
							
							text = text .. target .. " "
						else
							text = text .. custom .. " "
						end
						
						if (cond[2] ~= nil) then
							if (#cond[2] > 0) then
								for c,d in ipairs(cond[2]) do
									local this_param_name = parse_this_param_and_get_raycast_units(d)
									if this_param_name then
										text = text .. this_param_name.." "
									elseif (#custom == 0) then
										local target = d
										local isnot = string.sub(target, 1, 4)
										local target_ = target
										
										if (isnot == "not ") then
											target_ = string.sub(target, 5)
										else
											isnot = ""
										end
										
										if (word_names[target_] ~= nil) then
											target = isnot .. word_names[target_]
										elseif (target_ == "ambient") then -- EDIT: implement AMBIENT
											target = isnot .. target_ .. " (" .. ws_ambientObject .. ")"
										end
										
										text = text .. target .. " "
									else
										text = text .. custom .. " "
									end
									
									if (#cond[2] > 1) and (c ~= #cond[2]) then
										text = text .. "& "
									end
								end
							end
						end
						
						if (a < #conds - num_this_conds) then
							text = text .. "& "
						end
					else
						if (#custom == 0) then
							-- EDIT: allow prefixes to have different visual names in the properties
							local target = cond[1]
							local isnot = string.sub(target, 1, 4)
							local target_ = target
							
							if (isnot == "not ") then
								target_ = string.sub(target, 5)
							else
								isnot = ""
							end
							
							if (word_names[target_] ~= nil) then
								target = isnot .. word_names[target_]
							end
							
							text = target .. " " .. text
						else
							text = custom .. " " .. text
						end
					end
				end
			end
			
			local target = rule[3]
			local isnot = string.sub(target, 1, 4)
			local target_ = target
			
			if (isnot == "not ") then
				target_ = string.sub(target, 5)
			else
				isnot = ""
			end
			
			if (word_names[target_] ~= nil) then
				target = isnot .. word_names[target_]
			elseif (target_ == "ambient") then -- EDIT: implement AMBIENT
				target = isnot .. target_ .. " (" .. ws_ambientObject .. ")"
			end
			
			if (#custom == 0) then
				text = text .. rule[2] .. " " .. target
			else
				text = text .. custom .. " " .. custom
			end
			
			for a,b in ipairs(tags) do
				if (b == "mimic") then
					text = text .. " (mimic)"
				end
			end
			
			if (allrules[text] == nil) then
				allrules[text] = 1
				count = count + 1
			else
				allrules[text] = allrules[text] + 1
			end
			i_ = i_ + 1
		end
	end
	
	local columns = math.min(maxcolumns, math.floor((count - 1) / linelimit) + 1)
	local columnwidth = math.min(screenw - f_tilesize * 2, columns * f_tilesize * 10) / columns
	
	i_ = 1
	
	local maxlimit = 4 * linelimit
	
	for i,v in pairs(allrules) do
		local text = i
		
		if (i_ <= maxlimit) then
			local currcolumn = math.floor((i_ - 1) / linelimit) - (columns * 0.5)
			x = basex + columnwidth * currcolumn + columnwidth * 0.5
			y = basey + (((i_ - 1) % linelimit) + 1) * f_tilesize * 0.8
		end
		
		if (i_ <= maxlimit-1) then
			if (v == 1) then
				writetext(text,0,x,y,name,true,2,true)
			elseif (v > 1) then
				writetext(tostring(v) .. " x " .. text,0,x,y,name,true,2,true)
			end
		end
		
		i_ = i_ + 1
	end
	
	if (i_ > maxlimit-1) then
		writetext("(+ " .. tostring(i_ - maxlimit) .. ")",0,x,y,name,true,2,true)
	end
end

function create(name,x,y,dir,oldx_,oldy_,float_,skipundo_,leveldata_,customdata)
	local oldx,oldy,float = x,y,0
	local tileid = x + y * roomsizex
	
	if (oldx_ ~= nil) then
		oldx = oldx_
	end
	
	if (oldy_ ~= nil) then
		oldy = oldy_
	end
	
	if (float_ ~= nil) then
		float = float_
	end
	
	local skipundo = skipundo_ or false
	
	local unitname = unitreference[name]

	if (unitname == nil) then
		unitname = "error"
		MF_alert("Couldn't find object for " .. tostring(name) .. "!")
	end
	
	local newunitid = MF_emptycreate(unitname,oldx,oldy)
	local newunit = mmf.newObject(newunitid)
	
	local id = newid()
	
	newunit.values[ONLINE] = 1
	newunit.values[XPOS] = x
	newunit.values[YPOS] = y
	newunit.values[DIR] = dir
	newunit.values[ID] = id
	newunit.values[FLOAT] = float
	newunit.flags[CONVERTED] = true


	if customdata ~= nil then
		persistreverts[id]=customdata[1]
		newunit.karma=customdata[2]
	end

	
	if (leveldata_ ~= nil) and (#leveldata_ > 0) then
		newunit.strings[U_LEVELFILE] = leveldata_[1]
		newunit.strings[U_LEVELNAME] = leveldata_[2]
		newunit.flags[MAPLEVEL] = leveldata_[3]
		newunit.values[VISUALLEVEL] = leveldata_[4]
		newunit.values[VISUALSTYLE] = leveldata_[5]
		newunit.values[COMPLETED] = leveldata_[6]
		
		newunit.strings[COLOUR] = leveldata_[7]
		newunit.strings[CLEARCOLOUR] = leveldata_[8]
		
		if (newunit.className == "level") then
			if (#leveldata_[1] > 0) then
				newunit.values[COMPLETED] = math.max(leveldata_[6], 2)
			else
				newunit.values[COMPLETED] = math.max(leveldata_[6], 1)
			end
			
			if (#leveldata_[7] == 0) or (#leveldata_[8] == 0) then
				newunit.strings[COLOUR] = "1,2"
				newunit.strings[CLEARCOLOUR] = "1,3"
				MF_setcolour(newunitid,1,2)
			else
				local c = MF_parsestring(leveldata_[7])
				MF_setcolour(newunitid,c[1],c[2])
			end
		elseif (#leveldata_[7] > 0) then
			local c = MF_parsestring(leveldata_[7])
			MF_setcolour(newunitid,c[1],c[2])
		end
	end
	
	newunit.flags[9] = true
	
	if (skipundo == false) then
		addundo({"create",name,id,-1,"create",x,y,dir})
	end
	
	addunit(newunitid)
	addunitmap(newunitid,x,y,newunit.strings[UNITNAME])
	dynamic(newunitid)
	
	-- Add check for ECHO units along with WORD units
	local testname = getname(newunit)
	if ((hasfeature(testname,"is","word",newunitid,x,y) ~= nil) or (hasfeature(testname,"is","echo",newunitid,x,y) ~= nil)) then
		updatecode = 1
	end
	
	return newunit.fixed,id
end

function delete(unitid,x_,y_,total_,noinside_)
	local total = total_ or false
	local noinside = noinside_ or false
	
	local check = unitid
	
	if (unitid == 2) then
		check = 200 + x_ + y_ * roomsizex
	end
	
	if (deleted[check] == nil) then
		local unit = {}
		local x,y,dir = 0,0,4
		local unitname = ""
		local insidename = ""
		
		if (unitid ~= 2) then
			unit = mmf.newObject(unitid)
			x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
			unitname = unit.strings[UNITNAME]
			insidename = getname(unit)
		else
			x,y = x_,y_
			unitname = "empty"
			insidename = "empty"
		end
		
		x = math.floor(x)
		y = math.floor(y)
		
		if (total == false) and inbounds(x,y,1) and (noinside == false) then
			local leveldata = {}
			
			if (unitid == 2) then
				dir = emptydir(x,y)
			else
				leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
			end
			
			inside(insidename,x,y,dir,unitid,leveldata)
		end
		
		if (unitid ~= 2) then
			if (spritedata.values[CAMTARGET] == unit.values[ID]) then
				changevisiontarget(unit.fixed)
			end
			
			addundo({"remove",unitname,x,y,dir,unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init,unit.originalname,unit.strings[UNITSIGNTEXT],false,unitid,unit.karma},unitid)
			unit = {}
			delunit(unitid)
			MF_remove(unitid)
			
			--MF_alert("Removed " .. tostring(unitid))
			
			if inbounds(x,y,1) then
				dynamicat(x,y)
			end
		end
		
		deleted[check] = 1
	else
		MF_alert("already deleted")
	end
end

function inside(name,x,y,dir_,unitid,leveldata_)
	local ins = {}
	local wordins = {}
	local tileid = x + y * roomsizex
	local maptile = unitmap[tileid] or {}
	local dir = dir_
	
	local leveldata = leveldata_ or {}
	
	if (dir == 4) then
		dir = fixedrandom(0,3)
	end
	
	if (featureindex[name] ~= nil) then
		for i,rule in ipairs(featureindex[name]) do
			local baserule = rule[1]
			local conds = rule[2]
			
			local target = baserule[1]
			local verb = baserule[2]
			local object = baserule[3]
			
			if (target == name) and (verb == "has") and (findnoun(object,nlist.short) or (unitreference[object] ~= nil)) then
				table.insert(ins, {object,conds})
			end
			if (target == name) and (verb == "scrawl") then
				table.insert(wordins, {object,conds})
			end
		end
	end
	
	if (#ins > 0) then
		for i,v in ipairs(ins) do
			local object = v[1]
			local conds = v[2]
			if testcond(conds,unitid,x,y) then
				if (object ~= "text") then
					for a,mat in pairs(objectlist) do
						if (a == object) and (object ~= "empty") then
							if (object ~= "all") and (string.sub(object, 1, 5) ~= "group") then
								create(object,x,y,dir,nil,nil,nil,nil,leveldata)
							elseif (object == "all") then
								createall(v,x,y,unitid,nil,leveldata)
							end
						end
					end
				else
					create("text_" .. name,x,y,dir,nil,nil,nil,nil,leveldata)
				end
			end
		end
	end
	if (#wordins > 0) then
		for i,v in ipairs(wordins) do
			local object = "text_" .. v[1]
			local conds = v[2]
			if testcond(conds,unitid,x,y) then
				if (unitreference[object] ~= nil) then
					create(object,x,y,dir,nil,nil,nil,nil,leveldata)
				end
			end
		end
	end
end

function isgone(unitid)
	if (issafe(unitid) == false) then
		local unit = mmf.newObject(unitid)
		local x,y,name = unit.values[XPOS],unit.values[YPOS],unit.strings[UNITNAME]
		
		if (unit.strings[UNITTYPE] == "text") then
			-- name = "text"
		end
		
		-- Added check for ALIVE here
		local isyou = hasfeature(name,"is","you",unitid,x,y) or hasfeature(name,"is","you2",unitid,x,y) or hasfeature(name,"is","3d",unitid,x,y) or hasfeature(name,"is","alive",unitid,x,y)
		local ismelt = hasfeature(name,"is","melt",unitid,x,y)
		local isweak = hasfeature(name,"is","weak",unitid,x,y)
		local isshut = hasfeature(name,"is","shut",unitid,x,y)
		local isopen = hasfeature(name,"is","open",unitid,x,y)
		local ismove = hasfeature(name,"is","move",unitid,x,y)
		local ispush = hasfeature(name,"is","push",unitid,x,y)
		local ispull = hasfeature(name,"is","pull",unitid,x,y)
		local eat = findfeatureat(nil,"eat",name,x,y)
		local sinks = findfeatureat(nil,"sinks",name,x,y)
		local opens = findfeatureat(nil,"opens",name,x,y)
		local melts = findfeatureat(nil,"melts",name,x,y)
		
		if (eat ~= nil) then
			for i,v in ipairs(eat) do
				if (v ~= unitid) then
					return true
				end
			end
		end
		if (sinks ~= nil) then
			for i,v in ipairs(sinks) do
				if (v ~= unitid) then
					return true
				end
			end
		end
		if (opens ~= nil) then
			for i,v in ipairs(opens) do
				if (v ~= unitid) then
					return true
				end
			end
		end
		if (meltss ~= nil) then
			for i,v in ipairs(melts) do
				if (v ~= unitid) then
					return true
				end
			end
		end

		local issink = findfeatureat(nil,"is","sink",x,y)
		
		if (issink ~= nil) then
			for i,v in ipairs(issink) do
				if (v ~= unitid) and floating(v,unitid,x,y) then
					return true
				end
			end
		end
		
		if (isyou ~= nil) then
			local isdefeat = findfeatureat(nil,"is","defeat",x,y)
			
			if (isdefeat ~= nil) then
				for i,v in ipairs(isdefeat) do
					if floating(v,unitid,x,y) then
						return true
					end
				end
			end
			
			local isdefeats = findfeatureat(nil,"defeats",name,x,y)
			
			if (isdefeats ~= nil) then
				for i,v in ipairs(isdefeats) do
					if floating(v,unitid,x,y) then
						return true
					end
				end
			end
		end
		
		if (ismelt ~= nil) then
			local ishot = findfeatureat(nil,"is","hot",x,y)
			
			if (ishot ~= nil) then
				for i,v in ipairs(ishot) do
					if floating(v,unitid,x,y) then
						return true
					end
				end
			end
		end
		
		if (isshut ~= nil) then
			local isopen_ = findfeatureat(nil,"is","open",x,y)
			
			if (isopen_ ~= nil) then
				for i,v in ipairs(isopen_) do
					if floating(v,unitid,x,y) then
						return true
					end
				end
			end
		end
		
		if (isopen ~= nil) then
			local isshut_ = findfeatureat(nil,"is","shut",x,y)
			
			if (isshut_ ~= nil) then
				for i,v in ipairs(isshut_) do
					if floating(v,unitid,x,y) then
						return true
					end
				end
			end
		end
		
		if (isweak ~= nil) then
			local things = findallhere(x,y)
			
			if (things ~= nil) then
				for i,v in ipairs(things) do
					if (v ~= unitid) and floating(v,unitid,x,y) then
						return true
					end
				end
			end
		end
	end
	
	return false
end

function floating(id1,id2,x1_,y1_,x2_,y2_)
	local empty1,empty2 = false,false
	local x1 = x1_ or 0
	local y1 = y1_ or 0
	local x2 = x2_ or x1
	local y2 = y2_ or y1
	
	local float1,float2 = -1,-1
	
	if (id1 ~= 2) then
		local unit1 = mmf.newObject(id1)
		float1 = unit1.values[FLOAT]
		local name1 = getname(unit1)
		if (hasfeature(name1,"is","tall",id1)) then
			float1 = 2
		end
	else
		local emptyfloat = hasfeature("empty","is","float",2,x1,y1)
		if (emptyfloat ~= nil) then
			float1 = 1
		else
			float1 = 0
		end
		if hasfeature("empty","is","tall",2,x1,y1) then
			float1 = 2
		end
	end
	
	if (id2 ~= 2) then
		local unit2 = mmf.newObject(id2)
		float2 = unit2.values[FLOAT]
		local name2 = getname(unit2)
		if (hasfeature(name2,"is","tall",id2)) then
			float2 = 2
		end
	else
		local emptyfloat = hasfeature("empty","is","float",2,x2,y2)
		if (emptyfloat ~= nil) then
			float2 = 1
		else
			float2 = 0
		end
		if hasfeature("empty","is","tall",2,x2,y2) then
			float2 = 2
		end
	end
	
	if (float1 == float2 or float1 == 2 or float2 == 2) then
		return true
	end
	
	return false
end

function simplecheck(x,y,noempty_,checkc)
	local obs = findobstacle(x,y)
	local noempty = noempty_ or false
	
	if (#obs > 0) then
		for i,id in ipairs(obs) do
			if (id == -1) then
				return -1
			else
				local obsunit = mmf.newObject(id)
				local obsname = getname(obsunit)
				if ((hasfeature(obsname,"is","stop",id,x,y,checkc) ~= nil)
				or (hasfeature(obsname,"is","push",id,x,y,checkc) ~= nil)
				or (hasfeature(obsname,"is","pull",id,x,y,checkc) ~= nil)
				or (hasfeature(obsname,"is","sidekick",id,x,y,checkc) ~= nil)
				)
				and (hasfeature(obsname,"is","hide",id,x,y,checkc) == nil)
				and (hasfeature(obsname,"is","phantom",id,x,y,checkc) == nil) then
					return 1
				end
			end
		end
	elseif (noempty == false) then
		if ((hasfeature("empty","is","stop",2,x,y,checkc) ~= nil)
		or (hasfeature("empty","is","push",2,x,y,checkc) ~= nil)
		or (hasfeature("empty","is","pull",2,x,y,checkc) ~= nil)
		or (hasfeature("empty","is","sidekick",2,x,y,checkc) ~= nil)
		)
		and (hasfeature("empty","is","phantom",2,x,y,checkc) == nil)
		and (hasfeature("empty","is","hide",2,x,y,checkc) == nil) then
			return 2
		end
	end
	
	return 0
end

function findfears(unitid,feartargets,x_,y_)
	-- @TODO: get rid of this once hempuli fixes the call to trypush
	local result,resultdir = false,4
	local amount = 0
	
	local ox,oy = 0,0
	local x,y = 0,0
	local name = ""
	local dir = 4
	
	if (unitid ~= 2) then
		local unit = mmf.newObject(unitid)
		x,y = unit.values[XPOS],unit.values[YPOS]
		name = getname(unit)
		dir = unit.values[DIR]
	else
		x,y = x_,y_
		name = "empty"
		dir = emptydir(x,y)
	end
	
	local feardirs = {}
	local maxfear = 0
	
	for j=0,3 do
		local i = (((dir + 2) + j) % 4) + 1
		local ndrs = ndirs[i]
		ox = ndrs[1]
		oy = ndrs[2]
		
		local dirfound = false
		local diramount = 0
		
		if (#feartargets > 0) then
			for a,v in ipairs(feartargets) do
				local foundfears = {}
				
				if (v ~= "empty") then
					foundfears = findtype({v, nil},x+ox,y+oy,unitid)
				else
					local tileid = (x + ox) + (y + oy) * roomsizex
					if (unitmap[tileid] == nil) or (#unitmap[tileid] == 0) then
						foundfears = {"a","b"}
					end
				end
				
				if (#foundfears > 0) then
					dirfound = true
					result = true
					resultdir = rotate(i-1)
					diramount = diramount + 1
				end
			end
		end
		
		if dirfound then
			feardirs[i] = diramount
			maxfear = math.max(maxfear, diramount)
		else
			feardirs[i] = 0
		end
	end
	
	local totalfeardirs = 0
	
	for i,v in ipairs(feardirs) do
		if (v >= maxfear) then
			totalfeardirs = totalfeardirs + 1
		else
			feardirs[i] = 0
		end
	end
	
	if (totalfeardirs > 0) then
		amount = maxfear
	end
	
	if (totalfeardirs > 1) then
		resultdir = dir
		local searching = true
		local tests = 0
		
		while searching do
			local problems = false
			
			if (feardirs[resultdir+1] == 1) then
				problems = true
			else
				local ndrs = ndirs[resultdir+1]
				local ox,oy = ndrs[1],ndrs[2]
				
				local obs = check(unitid,x,y,resultdir)
				
				local obsresult = 0
				for i,v in ipairs(obs) do
					if (v == 1) or (v == -1) then
						obsresult = 1
						break
					elseif (v ~= 0) and (obsresult == 0) then
						obsresult = v
					end
				end
				
				if (obsresult == 1) then
					problems = true
				elseif (obsresult ~= 0) then
					local ndrs = ndirs[resultdir+1]
					local ox,oy = ndrs[1],ndrs[2]

					local obsresult_ = trypush(obsresult,ox,oy,resultdir,false,x,y,"fear",unitid)
					
					if (obsresult_ ~= 0) then
						problems = true
					end
				end
			end
			
			if (problems == false) then
				searching = false
			else
				if (tests == 0) then
					resultdir = (resultdir - 1 + 4) % 4
				elseif (tests == 1) then
					resultdir = (resultdir + 2 + 4) % 4
				elseif (tests == 2) then
					resultdir = (resultdir + 1 + 4) % 4
				elseif (tests == 3) then
					resultdir = (resultdir - 2 + 4) % 4
				end
				
				tests = tests + 1
			end
			
			if (tests >= 4) then
				searching = false
				result = false
				resultdir = 4
			end
		end
	end
	
	return result,resultdir,amount
end

function getlevelsurrounds(levelid)
	local level = mmf.newObject(levelid)
	visit_innerlevelid = tostring(levelid)

	local loopindex = 1
	local addedids = {levelid}
	local dirids = {"r","u","l","d","dr","ur","ul","dl","o"}
	local x,y,dir = level.values[XPOS],level.values[YPOS],level.values[DIR]
	
	local result = generaldata.strings[CURRLEVEL] .. ","

	--This would be a for loop, but it's adding things to the table being looped through.
	--Specifically, if the levelsurrounds contain an object with level data, the code has to
	--get surrounds for that level too in case it gets visited.
	while #addedids >= loopindex do
		local levelid_item = addedids[loopindex]
		
		local level = mmf.newObject(levelid_item)
		result = result .. "levelseparator" .. ","
		.. tostring(levelid_item) .. ","
		.. level.strings[U_LEVELFILE] .. ","
		.. tostring(level.values[VISUALLEVEL]) .. ","
		.. tostring(level.values[VISUALSTYLE]) .. ","
		.. tostring(level.values[DIR] .. ",")

		for i,v in ipairs(dirs_diagonals) do
			result = result .. dirids[i] .. ","
			local ox,oy = v[1],v[2]
			local tileid = (level.values[XPOS] + ox) + (level.values[YPOS] + oy) * roomsizex

			if (unitmap[tileid] ~= nil) then
				if (#unitmap[tileid] > 0) then
					for a,b in ipairs(unitmap[tileid]) do
						if (b ~= levelid_item) then
							local unit = mmf.newObject(b)
							local name = getname(unit)
							
							if (string.len(unit.strings[U_LEVELFILE]) > 0) and (string.len(unit.strings[U_LEVELNAME]) > 0) and (generaldata.values[IGNORE] == 0) and (unit.values[COMPLETED] > 1) then
								result = result .. b .. ","

								local leveladded = false
								for c,d in ipairs(addedids) do
									if d == b then
										leveladded = true
									end
								end
								if leveladded == false then
									table.insert(addedids,b)
								end
							end
							result = result .. name .. ","
						end
					end
				else
					result = result .. "-" .. ","
				end
			else
				result = result .. "-" .. ","
			end
		end
		
		loopindex = loopindex + 1
	end

	-- EDIT: find all texts (including the level itself) at the level's position
	-- NOTE: this assumes that the text being echoed is of the same type as within the level (for example if BABA is a noun in the map, it's treated as a noun inside the level as well)
	
	ws_overlapping_texts = {}
	local text_ids = findtype({"text"}, x, y)
	for _,textid in ipairs(text_ids) do
		local text_unit = mmf.newObject(textid)

		--[[ 
			@Merge(Word Salad x Plasma): A few changes, mainly to account for the fact that when calling levelsurrounds(), you're often moving to a different level, meaning that unitids will not be consistent
			- use get_turning_text_interpretation() to account for turning text
			- if object overlapping an ECHO level is a pointer noun, get all of the things that the pointer noun refers to and process each of them independently
		 ]]
		if is_name_text_this(text_unit.strings[NAME]) then
			for ray_object in pairs(get_raycast_objects(textid)) do
				local unitid = plasma_utils.parse_object(ray_object)
				if unitid == 2 then
					local text_data = {"empty", 0, -1} -- If a pointer texxt is pointing to an empty tile, explicitly put in "empty" for the list of echoed texts
					table.insert(ws_overlapping_texts, text_data)
				else
					table.insert(text_ids, unitid)
				end
			end
		else
			local text_data = {get_turning_text_interpretation(textid), text_unit.values[TYPE], -1} -- The position is set to -1, so that any level obj inside the level can echo the text regardless of their position
			table.insert(ws_overlapping_texts, text_data)
		end
		-- EXPERIMENT: Allow echoing WORD units as well
		if (WS_CAN_ECHO_WORD_UNITS) then
			for _,words in ipairs(wordunits) do -- Hopefully this also works?
				-- NOTE: For consistency with how LEVEL IS ECHO handles outer text, we allow a level to echo itself if it's also WORD
				local word_unit = mmf.newObject(words[1])
				if (word_unit.values[XPOS] == x) and (word_unit.values[YPOS] == y) then
					table.insert(ws_overlapping_texts, {word_unit.strings[NAME], 0, -1}) -- WORD objects are always nouns, and like above the position is set to -1
				end
			end
		end
	

	end
	
	-- EDIT: keep the "sinful" status of a level upon entering
	ws_wasLevelSinful = level.karma
	-- EDIT: keep track of what object the current level is for AMBIENT (ideally this should be stored somewhere)
	ws_ambientObject = getname(level)
	-- EDIT: check if the level is aligned
	local columnFail, rowFail = false, false -- columnFail = object in different column, rowFail = object in different row
	for _,u in pairs(unitlists[ws_ambientObject]) do
		local unit = mmf.newObject(u)
		local ux, uy = unit.values[XPOS], unit.values[YPOS]
		if (ux ~= x) then -- Other level in a different column
			columnFail = true
		end
		if (uy ~= y) then -- Other level in a different row
			rowFail = true
		end
		if columnFail and rowFail then -- No need to keep checking
			break
		end
	end
	ws_levelAlignedRow = not rowFail
	ws_levelAlignedColumn = not columnFail

	visit_fullsurrounds = result
end

function parsesurrounds()
	local surrounds = MF_parsestring(generaldata2.strings[LEVELSURROUNDS])
	local result = {}
	local stage = 0
	
	local dirids = {"r","u","l","d","dr","ur","ul","dl","o"}
	
	for i,v in ipairs(surrounds) do
		if (i == 1) then
			result.dir = tonumber(v)
		else
			if (v == dirids[stage + 1]) then
				stage = stage + 1
			else
				local dir = dirids[stage]
				
				if (result[dir] == nil) then
					result[dir] = {}
				end
				
				table.insert(result[dir], v)
			end
		end
	end
	
	return result
end

function update(unitid,x,y,dir_)
	if (unitid ~= nil) then
		local unit = mmf.newObject(unitid)

		local unitname = unit.strings[UNITNAME]
		local dir,olddir = unit.values[DIR],unit.values[DIR]
		local tiling = unit.values[TILING]
		local unittype = unit.strings[UNITTYPE]
		local oldx,oldy = unit.values[XPOS],unit.values[YPOS]
		
		if (dir_ ~= nil) then
			dir = dir_
		end
		
		if (x ~= oldx) or (y ~= oldy) or (dir ~= olddir) then
			updateundo = true
			
			addundo({"update",unitname,oldx,oldy,olddir,x,y,dir,unit.values[ID]},unitid)
			
			local ox,oy = x-oldx,y-oldy
			
			if (math.abs(ox) + math.abs(oy) == 1) and (unit.values[MOVED] == 0) then
				unit.x = unit.x + ox * tilesize * spritedata.values[TILEMULT] * generaldata2.values[ZOOM] * 0.25
				unit.y = unit.y + oy * tilesize * spritedata.values[TILEMULT] * generaldata2.values[ZOOM] * 0.25
			end
			
			unit.values[XPOS] = x
			unit.values[YPOS] = y
			unit.values[DIR] = dir
			unit.values[MOVED] = 1
			unit.values[POSITIONING] = 0

			updateunitmap(unitid,oldx,oldy,x,y,unit.strings[UNITNAME])
			
			if (tiling == 1) then
				dynamic(unitid)
				dynamicat(oldx,oldy)
			end
			
			if (unittype == "text") then
				updatecode = 1
			end
			
			-- Add check for ECHO along with WORD
			if (featureindex["word"] ~= nil) then
				checkwordchanges(unitid,unitname)
			end
			if (featureindex["echo"] ~= nil) then
				ws_checkechochanges(unitid)
			end
		end
	else
		MF_alert("Tried to update a nil unit")
	end
end

function delunit(unitid)
	local unit = mmf.newObject(unitid)
	
	-- MF_alert("DELUNIT " .. unit.strings[UNITNAME])
	
	if (unit ~= nil) then
		local name = getname(unit)
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local unitlist = unitlists[name]
		local unitlist_ = unitlists[unit.strings[UNITNAME]] or {}
		local unittype = unit.strings[UNITTYPE]
		
		if (unittype == "text") then
			updatecode = 1
		end
		
		x = math.floor(x)
		y = math.floor(y)
		
		if (unitlist ~= nil) then
			for i,v in pairs(unitlist) do
				if (v == unitid) then
					v = {}
					table.remove(unitlist, i)
					break
				end
			end
		end
		
		if (unitlist_ ~= nil) then
			for i,v in pairs(unitlist_) do
				if (v == unitid) then
					v = {}
					table.remove(unitlist_, i)
					break
				end
			end
		end
		
		-- TÄMÄ EI EHKÄ TOIMI
		local tileid = x + y * roomsizex
		
		if (unitmap[tileid] ~= nil) then
			for i,v in pairs(unitmap[tileid]) do
				if (v == unitid) then
					v = {}
					table.remove(unitmap[tileid], i)
				end
			end
		
			if (#unitmap[tileid] == 0) then
				unitmap[tileid] = nil
			end
		end
		
		if (unittypeshere[tileid] ~= nil) then
			local uth = unittypeshere[tileid]
			
			local n = unit.strings[UNITNAME]
			
			if (uth[n] ~= nil) then
				uth[n] = uth[n] - 1
				
				if (uth[n] == 0) then
					uth[n] = nil
				end
			end
		end
		
		if (unit.strings[UNITTYPE] == "text") and (codeunits ~= nil) then
			for i,v in pairs(codeunits) do
				if (v == unitid) then
					v = {}
					table.remove(codeunits, i)
				end
			end
			
			if (unit.values[TYPE] == 5) then
				for i,v in pairs(letterunits) do
					if (v == unitid) then
						v = {}
						table.remove(letterunits, i)
					end
				end
			end
		end

		if (unit.values[TILING] > 1) and (animunits ~= nil) then
			for i,v in pairs(animunits) do
				if (v == unitid) then
					v = {}
					table.remove(animunits, i)
				end
			end
		end
		
		if (unit.values[TILING] == 1) and (tiledunits ~= nil) then
			for i,v in pairs(tiledunits) do
				if (v == unitid) then
					v = {}
					table.remove(tiledunits, i)
				end
			end
		end
		
		if (#wordunits > 0) and (unit.values[TYPE] == 0) and (unit.strings[UNITTYPE] ~= "text") then
			for i,v in pairs(wordunits) do
				if (v[1] == unitid) then
					local currentundo = undobuffer[1]
					table.insert(currentundo.wordunits, unit.values[ID])
					updatecode = 1
					v = {}
					table.remove(wordunits, i)
				end
			end
		end
		
		if (#wordrelatedunits > 0) then
			for i,v in pairs(wordrelatedunits) do
				if (v[1] == unitid) then
					local currentundo = undobuffer[1]
					table.insert(currentundo.wordrelatedunits, unit.values[ID])
					updatecode = 1
					v = {}
					table.remove(wordrelatedunits, i)
				end
			end
		end
		
		-- EDIT: store echo units (?). Will it work? Will it explode? Only time can tell
		if (#echounits > 0) and (unit.strings[UNITTYPE] ~= "text") then
			for i,v in pairs(echounits) do
				if (v[1] == unitid) then
					local currentundo = undobuffer[1]
					table.insert(currentundo.echounits, unit.values[ID])
					updatecode = 1
					v = {}
					table.remove(echounits, i)
				end
			end
		end
		
		if (#echorelatedunits > 0) then
			for i,v in pairs(echorelatedunits) do
				if (v[1] == unitid) then
					local currentundo = undobuffer[1]
					table.insert(currentundo.echorelatedunits, unit.values[ID])
					updatecode = 1
					v = {}
					table.remove(echorelatedunits, i)
				end
			end
		end
		
		if (#visiontargets > 0) then
			for i,v in pairs(visiontargets) do
				if (v == unitid) then
					local currentundo = undobuffer[1]
					--table.insert(currentundo.visiontargets, unit.values[ID])
					v = {}
					table.remove(visiontargets, i)
				end
			end
		end
	else
		MF_alert("delunit(): no object found with id " .. tostring(unitid) .. " (delunit)")
	end
		
	for i,v in ipairs(units) do
		if (v.fixed == unitid) then
			v = {}
			table.remove(units, i)
		end
	end
	
	for i,data in pairs(updatelist) do
		if (data[1] == unitid) and (data[2] ~= "convert") then
			data[2] = "DELETED"
		end
	end
end

function destroylevel(special_)
	destroylevel_check = true
	destroylevel_style = special_ or ""
	
	if (destroylevel_style == "infinity") or (destroylevel_style == "toocomplex") then
		setsoundname("removal",2)
		--Glitch code starts here.
		if (INFLOOP_LEVEL_GLITCH) and (destroylevel_style == "infinity") then
			levelconversions = {{"glitch", {}, "is"}}
		end
		--Glitch code ends here.
	elseif (destroylevel_style ~= "empty") and (destroylevel_style ~= "bonus") then
		setsoundname("removal",1)
	end
	
	MF_musicstate(1)
	generaldata2.values[NOPLAYER] = 1
end

function findgroup(grouptype_,invert_,limit_,checkedconds_)
	local result = {}
	local limit = limit_ or 0
	local invert = invert_ or false
	local grouptype = grouptype_ or "group"
	local found = {}
	local alreadyused = {}
	
	limit = limit + 1
	
	local idstring = ""
	local currmembers = {}
	local handlerecursion = false
	
	for i,v in ipairs(groupmembers) do
		local name = v[1]
		local conds = v[2]
		local gtype = v[3]
		local recursion = v[4]
		
		if (gtype == grouptype) then
			if hasconds(v) and (unitlists[name] ~= nil) then
				if (recursion == false) then
					for a,b in ipairs(unitlists[name]) do
						local unit = mmf.newObject(b)
						local x,y = unit.values[XPOS],unit.values[YPOS]
						
						if testcond(conds,b,x,y,nil,limit,checkedconds_) then
							table.insert(result, name)
							table.insert(currmembers, name)
							found[name] = 1
							idstring = idstring .. name
							break
						end
					end
				else
					handlerecursion = true
				end
			elseif (hasconds(v) == false) then
				table.insert(result, name)
				table.insert(currmembers, name)
				found[name] = 1
				idstring = idstring .. name
			end
		end
	end
	
	local reclimit = 0
	local curridstring = idstring
	
	while handlerecursion and (reclimit < 10) do
		local newidstring = idstring
		local newmembers = {}
		for i,v in ipairs(result) do
			table.insert(newmembers, v)
		end
		
		for i,v in ipairs(groupmembers) do
			local name = v[1]
			local conds = v[2]
			local gtype = v[3]
			local recursion = v[4]
			
			if recursion and (gtype == grouptype) then
				if hasconds(v) and (unitlists[name] ~= nil) then
					for a,b in ipairs(unitlists[name]) do
						local unit = mmf.newObject(b)
						local x,y = unit.values[XPOS],unit.values[YPOS]
						
						if testcond(conds,b,x,y,nil,limit,checkedconds_,nil,currmembers) then
							table.insert(newmembers, name)
							newidstring = newidstring .. name
							break
						end
					end
				elseif (hasconds(v) == false) then
					table.insert(newmembers, name)
					newidstring = newidstring .. name
				end
			end
		end
		
		--MF_alert(curridstring .. ", " .. newidstring)
		
		if (newidstring ~= curridstring) then
			currmembers = {}
			for i,v in ipairs(newmembers) do
				table.insert(currmembers, v)
			end
			curridstring = newidstring
			reclimit = reclimit + 1
		else
			for i,v in ipairs(currmembers) do
				found[v] = 1
				idstring = idstring .. v
				table.insert(result, v)
			end
			
			handlerecursion = false
		end
	end
	
	if (reclimit >= 10) then
		HACK_INFINITY = 200
		destroylevel("infinity")
		--Glitch line is here
		dolevelconversions()
		return
	end
	
	if invert then
		local actualresult = {}
		
		for a,mat in pairs(objectlist) do
			if (found[a] == nil) and (alreadyused[a] == nil) and (findnoun(a,nlist.short) == false) then
				table.insert(actualresult, a)
				alreadyused[a] = 1
			end
		end
		
		return actualresult
	end
	
	return result
end