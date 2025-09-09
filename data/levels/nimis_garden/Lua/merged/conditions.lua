

condlist.on = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	local tileid = x + y * roomsizex
	
	if (unitid ~= 2) then
		if (#params > 0) then
			for a,b in ipairs(params) do
				local pname = b
				local pnot = false
				if (string.sub(b, 1, 4) == "not ") then
					pnot = true
					pname = string.sub(b, 5)
				end

				local is_param_this, raycast_units, _, this_count = parse_this_param_and_get_raycast_units(pname)
				
				local bcode = b .. "_" .. tostring(a)
				
				if (string.sub(pname, 1, 5) == "group") then
					return false,checkedconds
				end
				
				if (unitid ~= 1) then
					if ((pname ~= "empty") and (b ~= "level")) or ((b == "level") and (alreadyfound[1] ~= nil)) then
						if (unitmap[tileid] ~= nil) then
							for c,d in ipairs(unitmap[tileid]) do
								if (d ~= unitid) and (alreadyfound[d] == nil) then
									local unit = mmf.newObject(d)
									local name_ = getname(unit)
									
									if (pnot == false) then
										if is_param_this then
											if raycast_units[d] and alreadyfound[bcode] == nil then
												alreadyfound[bcode] = 1
												alreadyfound[d] = 1
												allfound = allfound + 1
											end
										elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
											alreadyfound[bcode] = 1
											alreadyfound[d] = 1
											allfound = allfound + 1
										end
									else
										if is_param_this then
											if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
												alreadyfound[bcode] = 1
												alreadyfound[d] = 1
												allfound = allfound + 1
											end
										elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
											alreadyfound[bcode] = 1
											alreadyfound[d] = 1
											allfound = allfound + 1
										end
									end
								end
							end
						else
							print("unitmap is nil at " .. tostring(x) .. ", " .. tostring(y) .. " for object " .. tostring(name) .. " (" .. tostring(unitid) .. ")!")
						end
					elseif (pname == "empty") then
						if (pnot == false) then
							return false,checkedconds
						else
							if (unitmap[tileid] ~= nil) then
								for c,d in ipairs(unitmap[tileid]) do
									if (d ~= unitid) and (alreadyfound[d] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[d] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
						alreadyfound[bcode] = 1
						alreadyfound[1] = 1
						allfound = allfound + 1
					end
				else
					local ulist = false
					
					if is_param_this then
						if this_count > 0 then
							ulist = true
						end
					elseif (b ~= "empty") and (b ~= "level") then
						if (pnot == false) then
							if (unitlists[b] ~= nil) and (#unitlists[b] > 0) and (alreadyfound[bcode] == nil) then
								for c,d in ipairs(unitlists[b]) do
									if (alreadyfound[d] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[d] = 1
										ulist = true
										break
									end
								end
							end
						else
							for c,d in pairs(unitlists) do
								local tested = false
								
								if (c ~= pname) and (#d > 0) then
									for e,f in ipairs(d) do
										if (alreadyfound[f] == nil) and (alreadyfound[bcode] == nil) then
											alreadyfound[bcode] = 1
											alreadyfound[f] = 1
											ulist = true
											tested = true
											break
										end
									end
								end
								
								if tested then
									break
								end
							end
						end
					elseif (b == "empty") then
						local empties = findempty()
						
						if (#empties > 0) then
							for c,d in ipairs(empties) do
								if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					elseif (b == "level") then
						for c,unit in ipairs(units) do
							if (unit.className == "level") and (alreadyfound[unit.fixed] == nil) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[unit.fixed] = 1
								ulist = true
								break
							end
						end
					end
					
					if (b ~= "text") and (ulist == false) then
						if (surrounds["o"] ~= nil) then
							for c,d in ipairs(surrounds["o"]) do
								if (pnot == false) then
									if (d == pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								else
									if (d ~= pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								end
							end
						end
					end
					
					if ulist or (b == "text") then
						alreadyfound[bcode] = 1
						allfound = allfound + 1
					end
				end
			end
		else
			print("no parameters given!")
			return false,checkedconds
		end
	else
		for a,b in ipairs(params) do
			local bcode = b .. "_" .. tostring(a)
			
			if (b == "level") and (alreadyfound[bcode] == nil) then
				alreadyfound[bcode] = 1
				allfound = allfound + 1
			else
				return false,checkedconds
			end
		end
	end
	
	return (allfound == #params),checkedconds
end

condlist.near = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end

			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end

			local is_param_this, raycast_units, raycast_tileids, this_count = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2] -- <-- this last condition checks if empty is a raycast unit
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					for g=-1,1 do
						for h=-1,1 do
							if (pname ~= "empty") and not ray_unit_is_empty then
								local tileid = (x + g) + (y + h) * roomsizex
								if (unitmap[tileid] ~= nil) then
									for c,d in ipairs(unitmap[tileid]) do
										if (d ~= unitid) and (alreadyfound[d] == nil) then
											local unit = mmf.newObject(d)
											local name_ = getname(unit)
											
											if (pnot == false) then
												if is_param_this then
													if raycast_units[d] and alreadyfound[bcode] == nil then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											else
												if is_param_this then
													if not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											end
										end
									end
								end
							else
								local nearempty = false
						
								local tileid = (x + g) + (y + h) * roomsizex
								local l = map[0]
								local tile = l:get_x(x + g,y + h)
								
								local tcode = tostring(tileid) .. "e"
								
								if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and (alreadyfound[tcode] == nil) then 
									nearempty = true
								end
								
								if (g == 0) and (h == 0) then
									if (unitid == 2) then
										if (pnot == false) then
											nearempty = false
										end
									elseif (unitid ~= 1) and pnot then
										if (unitmap[tileid] == nil) or (#unitmap[tileid] <= 1) then
											nearempty = true
										end
									end
								end

								-- added "not pnot" since being near "not empty" means near any nonempty object
								if nearempty and not pnot and ray_unit_is_empty and raycast_tileids[tileid] == nil then
									nearempty = false
								end
								
								if (pnot == false) then
									if nearempty and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								else
									if (nearempty == false) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) then
						for c,d in ipairs(empties) do
							if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					for e,f in pairs(surrounds) do
						if (e ~= "dir") then
							for c,d in ipairs(f) do
								if (pnot == false) then
									if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								else
									if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.nextto = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end

			local is_param_this, raycast_units, _, this_count = parse_this_param_and_get_raycast_units(pname)
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					for g=-1,1 do
						for h=-1,1 do
							if ((h ~= 0) and (g == 0)) or ((h == 0) and (g ~= 0)) then
								if (pname ~= "empty") then
									local tileid = (x + g) + (y + h) * roomsizex
									if (unitmap[tileid] ~= nil) then
										for c,d in ipairs(unitmap[tileid]) do
											if (d ~= unitid) and (alreadyfound[d] == nil) then
												local unit = mmf.newObject(d)
												local name_ = getname(unit)
												
												if (pnot == false) then
													if is_param_this then
														if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
															alreadyfound[bcode] = 1
															alreadyfound[d] = 1
															allfound = allfound + 1
														end
													elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												else
													if is_param_this then
														if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
															alreadyfound[bcode] = 1
															alreadyfound[d] = 1
															allfound = allfound + 1
														end
													elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												end
											end
										end
									end
								else
									local nearempty = false
							
									local tileid = (x + g) + (y + h) * roomsizex
									local l = map[0]
									local tile = l:get_x(x + g,y + h)
									
									local tcode = tostring(tileid) .. "e"
									
									if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and (alreadyfound[tcode] == nil) then 
										nearempty = true
									end
									
									if (g == 0) and (h == 0) then
										if (unitid == 2) then
											if (pnot == false) then
												nearempty = false
											end
										elseif (unitid ~= 1) and pnot then
											if (unitmap[tileid] == nil) or (#unitmap[tileid] <= 1) then
												nearempty = true
											end
										end
									end
									
									if (pnot == false) then
										if nearempty and (alreadyfound[bcode] == nil) then
											alreadyfound[bcode] = 1
											alreadyfound[tcode] = 1
											allfound = allfound + 1
										end
									else
										if (nearempty == false) and (alreadyfound[bcode] == nil) then
											alreadyfound[bcode] = 1
											alreadyfound[tcode] = 1
											allfound = allfound + 1
										end
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) then
						for c,d in ipairs(empties) do
							if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					for e,f in pairs(surrounds) do
						if (e ~= "dir") and (e ~= "o") then
							for c,d in ipairs(f) do
								if (pnot == false) then
									if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								else
									if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										ulist = true
									end
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.facing = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	
	local unitid,x,y,dir,extras,surrounds,conds = cdata.unitid,cdata.x,cdata.y,cdata.dir,cdata.extras,cdata.surrounds,tostring(cdata.conds)

	if (unitid == 2) and ((checkedconds_ == nil) or (checkedconds_[conds] == nil)) then
		dir = emptydir(x,y,checkedconds)
	end
	
	local ndrs = ndirs[dir+1]
	local ox = ndrs[1]
	local oy = ndrs[2]
	
	local tileid = (x + ox) + (y + oy) * roomsizex
	
	if (#params > 0) and (dir ~= 4) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end

			local is_param_this, raycast_units, raycast_tileids = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2] -- <-- this last condition checks if empty is a 
			
			if (unitid ~= 1) then
				if not ray_unit_is_empty and (((pname ~= "empty") and (b ~= "level")) or ((b == "level") and (alreadyfound[1] ~= nil))) then
					if (stringintable(pname, extras) == false) then
						if (unitmap[tileid] ~= nil) then
							for c,d in ipairs(unitmap[tileid]) do
								if (d ~= unitid) and (alreadyfound[d] == nil) then
									local unit = mmf.newObject(d)
									local name_ = getname(unit)
									
									if (pnot == false) then
										if is_param_this then
											if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
												alreadyfound[bcode] = 1
												alreadyfound[d] = 1
												allfound = allfound + 1
											end
										elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
											alreadyfound[bcode] = 1
											alreadyfound[d] = 1
											allfound = allfound + 1
										end
									else
										if is_param_this then
											if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
												alreadyfound[bcode] = 1
												alreadyfound[d] = 1
												allfound = allfound + 1
											end
										elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
											alreadyfound[bcode] = 1
											alreadyfound[d] = 1
											allfound = allfound + 1
										end
									end
								end
							end
						end
					else
						if (pnot == false) then
							if ((pname == "right") and (dir == 0)) or ((pname == "up") and (dir == 1)) or ((pname == "left") and (dir == 2)) or ((pname == "down") and (dir == 3)) then
								if (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									allfound = allfound + 1
								end
							end
						else
							if ((pname == "right") and (dir ~= 0)) or ((pname == "up") and (dir ~= 1)) or ((pname == "left") and (dir ~= 2)) or ((pname == "down") and (dir ~= 3)) then
								if (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									allfound = allfound + 1
								end
							end
						end
					end
				elseif (pname == "empty" or ray_unit_is_empty) then
					local l = map[0]
					local tile = l:get_x(x + ox,y + oy)
					
					if (pnot == false) then
						local this_cond = not ray_unit_is_empty or (ray_unit_is_empty and raycast_tileids[tileid])
						if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and this_cond then
							if (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						end
					else
						if ((unitmap[tileid] ~= nil) and (#unitmap[tileid] > 0)) or (tile ~= 255) then
							if (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local dirids = {"r","u","l","d"}
				local dirid = dirids[dir + 1]
				
				if (surrounds[dirid] ~= nil) then
					for c,d in ipairs(surrounds[dirid]) do
						if (pnot == false) then
							if (d == pname) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						else
							if (d ~= pname) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						end
					end
				end
			end
		end
	else
		--print("no parameters given!")
		return false,checkedconds
	end
	
	return (allfound == #params),checkedconds
end

condlist.seeing = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local targets = {}
	
	local unitid,x,y,dir,conds,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.dir,tostring(cdata.conds),cdata.surrounds
	
	if (unitid == 2) then
		dir = emptydir(x,y)
	end
	
	local ndrs = ndirs[dir+1]
	local ox = ndrs[1]
	local oy = ndrs[2]
	
	local nx,ny = x,y
	local tileid = (x + ox) + (y + oy) * roomsizex
	local solid = 0
	
	if (checkedconds_ ~= nil) and (checkedconds_[tostring(conds) .. "_s_"] ~= nil) then
		return false,checkedconds,true
	end
	
	if (#params > 0) and (dir ~= 4) then
		while (solid == 0) and inbounds(nx,ny,1) do
			nx = nx + ox
			ny = ny + oy
			
			tileid = nx + ny * roomsizex
			
			if inbounds(nx,ny,1) then
				if (unitmap[tileid] ~= nil) then
					if (#unitmap[tileid] > 0) then
						local detected = false
						
						for a,b in ipairs(unitmap[tileid]) do
							local unit = mmf.newObject(b)
							local name_ = getname(unit)
							
							if (hasfeature(name_,"is","hide",b,nx,ny,checkedconds) == nil) then
								table.insert(targets, {b, name_})
								detected = true
							end
						end
						
						if (detected == false) then
							table.insert(targets, {2, "empty"})
						end
					else
						table.insert(targets, {2, "empty"})
					end
				else
					table.insert(targets, {2, "empty"})
				end
				
				solid = simplecheck(nx,ny,true,checkedconds)
			else
				solid = 1
			end
		end
		
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end

			local is_param_this, raycast_units = parse_this_param_and_get_raycast_units(pname)
			
			if (unitid ~= 1) then
				if ((pname ~= "empty") and (b ~= "level")) or ((b == "level") and (alreadyfound[1] ~= nil)) then
					for c,d_ in ipairs(targets) do
						local d = d_[1]
						
						if (d ~= unitid) and (alreadyfound[d] == nil) and (d ~= 2) then
							local name_ = d_[2]
							
							if (pnot == false) then
								if is_param_this then
									if raycast_units[d] and alreadyfound[bcode] == nil then
										alreadyfound[bcode] = 1
										alreadyfound[d] = 1
										allfound = allfound + 1
									end
								elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									allfound = allfound + 1
								end
							else
								if is_param_this then
									if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
										alreadyfound[bcode] = 1
										alreadyfound[d] = 1
										allfound = allfound + 1
									end
								elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									allfound = allfound + 1
								end
							end
						end
					end
				elseif (pname == "empty") then
					for c,d_ in ipairs(targets) do
						local d = d_[1]
						
						if (d == 2) then
							if (pnot == false) then
								if (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									allfound = allfound + 1
								end
							else
								if (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									allfound = allfound + 1
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local dirids = {"r","u","l","d"}
				local dirid = dirids[dir + 1]
				
				if (surrounds[dirid] ~= nil) then
					for c,d in ipairs(surrounds[dirid]) do
						if (pnot == false) then
							if (d == pname) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						else
							if (d ~= pname) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						end
					end
				end
			end
		end
	elseif (#params == 0) then
		print("no parameters given!")
		return false,checkedconds,true
	else
		return false,checkedconds,true
	end
	
	return (allfound == #params),checkedconds,true
end

condlist.without = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local unitcount = {}
	
	local name,unitid,notcond = cdata.name,cdata.unitid,cdata.notcond
			
	if (#params > 0) then
		for a,b in ipairs(params) do
			if (unitcount[b] == nil) then
				unitcount[b] = 0
			end
			
			unitcount[b] = unitcount[b] + 1
		end
		
		if (unitcount["level"] ~= nil) and (unitcount["level"] > 0) then
			unitcount["level"] = unitcount["level"] - 1
		end
			
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end
			
			local is_param_this, raycast_units, _, count = parse_this_param_and_get_raycast_units(pname)
			if is_param_this then
				if count == 0 or (count > 0 and raycast_units[unitid]) then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			elseif ((b ~= "level") and (b ~= "empty")) or ((b == "level") and (unitcount["level"] > 0)) then
				if (pnot == false) then
					if (alreadyfound[bcode] == nil) then
						if (unitlists[b] == nil) or (#unitlists[b] == 0) and (alreadyfound[bcode] == nil) then
							alreadyfound[bcode] = 1
							allfound = allfound + 1
						elseif (unitlists[b] ~= nil) and (#unitlists[b] > 0) then
							local found = false
							
							if (b ~= name) then
								if (#unitlists[b] < unitcount[b]) then
									found = true
								end
							else
								if (#unitlists[b] < unitcount[b] + 1) then
									found = true
								end
							end
							
							if found then
								alreadyfound[bcode] = 1
								allfound = allfound + 1
							end
						end
					end
				else
					local foundunits = 0
					local targetcount = unitcount[b]
					
					for c,d in pairs(unitlists) do
						if (c ~= pname) and (#unitlists[c] > 0) and (c ~= "text") and (string.sub(c, 1, 5) ~= "text_") then
							for e,f in ipairs(d) do
								if (f ~= unitid) and (alreadyfound[f] == nil) then
									alreadyfound[f] = 1
									foundunits = foundunits + 1
									
									if (foundunits >= targetcount) then
										break
									end
								end
							end
						end
						
						if (foundunits >= targetcount) then
							break
						end
					end
					
					if (foundunits < targetcount) and (alreadyfound[bcode] == nil) then
						alreadyfound[bcode] = 1
						allfound = allfound + 1
					end
				end
			elseif (b == "empty") then
				local empties = findempty()
				
				if (name ~= "empty") then
					if (#empties < unitcount[b]) and (alreadyfound[bcode] == nil) then
						alreadyfound[bcode] = 1
						allfound = allfound + 1
					end
				else
					if (#empties < unitcount[b] + 1) and (alreadyfound[bcode] == nil) then
						alreadyfound[bcode] = 1
						allfound = allfound + 1
					end
				end
			elseif (b == "level") then
				allfound = -99
				break
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	if notcond then
		return (allfound > 0),checkedconds
	end
			
	return (allfound == #params),checkedconds
end

condlist.above = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end
			
			local dist = roomsizey - y - 2

			local is_param_this, raycast_units, raycast_tileids, this_count = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2]
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					if (dist >= 1) then
						for g=1,dist do
							if (pname ~= "empty") and not ray_unit_is_empty then
								local tileid = x + (y + g) * roomsizex
								if (unitmap[tileid] ~= nil) then
									for c,d in ipairs(unitmap[tileid]) do
										if (d ~= unitid) and (alreadyfound[d] == nil) then
											local unit = mmf.newObject(d)
											local name_ = getname(unit)
											
											if (pnot == false) then
												if is_param_this then
													if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											else
												if is_param_this then
													if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											end
										end
									end
								end
							else
								local nearempty = false
						
								local tileid = x + (y + g) * roomsizex
								local l = map[0]
								local tile = l:get_x(x,y + g)
								
								local tcode = tostring(tileid) .. "e"
								
								if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and (alreadyfound[tcode] == nil) then 
									nearempty = true
								end

								if nearempty and ray_unit_is_empty and raycast_tileids[tileid] == nil and not pnot then
									nearempty = false
								end
								
								if (pnot == false) then
									if nearempty and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								else
									if (nearempty == false) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) and (alreadyfound[bcode] == nil) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) and (alreadyfound[bcode] == nil) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) and (alreadyfound[bcode] == nil) then
						for c,d in ipairs(unitlists[pname]) do
							if (alreadyfound[d] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					if (surrounds.d ~= nil) then
						for c,d in ipairs(surrounds.d) do
							if (pnot == false) then
								if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							else
								if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.below = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end
			
			local dist = (y - 1)

			local is_param_this, raycast_units, raycast_tileids, this_count = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2] -- <-- this last condition checks if empty is a raycast unit
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					if (y > 1) then
						for g=1,dist do
							if (pname ~= "empty") and not ray_unit_is_empty then
								local tileid = x + (y - g) * roomsizex
								if (unitmap[tileid] ~= nil) then
									for c,d in ipairs(unitmap[tileid]) do
										if (d ~= unitid) and (alreadyfound[d] == nil) then
											local unit = mmf.newObject(d)
											local name_ = getname(unit)
											
											if (pnot == false) then
												if is_param_this then
													if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											else
												if is_param_this then
													if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											end
										end
									end
								end
							else
								local nearempty = false
						
								local tileid = x + (y - g) * roomsizex
								local l = map[0]
								local tile = l:get_x(x,y - g)
								
								local tcode = tostring(tileid) .. "e"
								
								if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and (alreadyfound[tcode] == nil) then 
									nearempty = true
								end

								if nearempty and ray_unit_is_empty and raycast_tileids[tileid] == nil and not pnot then
									nearempty = false
								end
								
								if (pnot == false) then
									if nearempty and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								else
									if (nearempty == false) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) and (alreadyfound[bcode] == nil) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) and (alreadyfound[bcode] == nil) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) then
						for c,d in ipairs(empties) do
							if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					if (surrounds.u ~= nil) then
						for c,d in ipairs(surrounds.u) do
							if (pnot == false) then
								if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							else
								if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.besideright = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end
			
			local dist = (x - 1)

			local is_param_this, raycast_units, raycast_tileids, this_count = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2] -- <-- this last condition checks if empty is a raycast unit
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					if (x > 1) then
						for g=1,dist do
							if (pname ~= "empty") and not ray_unit_is_empty then
								local tileid = (x - g) + y * roomsizex
								if (unitmap[tileid] ~= nil) then
									for c,d in ipairs(unitmap[tileid]) do
										if (d ~= unitid) and (alreadyfound[d] == nil) then
											local unit = mmf.newObject(d)
											local name_ = getname(unit)
											
											if (pnot == false) then
												if is_param_this then
													if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											else
												if is_param_this then
													if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											end
										end
									end
								end
							else
								local nearempty = false
						
								local tileid = (x - g) + y * roomsizex
								local l = map[0]
								local tile = l:get_x(x - g,y)
								
								local tcode = tostring(tileid) .. "e"
								
								if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (tile == 255) and (alreadyfound[tcode] == nil) then 
									nearempty = true
								end

								if nearempty and ray_unit_is_empty and raycast_tileids[tileid] == nil then
									nearempty = false
								end
								
								if (pnot == false) then
									if nearempty and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								else
									if (nearempty == false) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) and (alreadyfound[bcode] == nil) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) and (alreadyfound[bcode] == nil) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) then
						for c,d in ipairs(empties) do
							if (alreadyfound[d] == nil) and (alreadyfound[bcode] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					if (surrounds.l ~= nil) then
						for c,d in ipairs(surrounds.l) do
							if (pnot == false) then
								if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							else
								if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.besideleft = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local unitid,x,y,surrounds = cdata.unitid,cdata.x,cdata.y,cdata.surrounds
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			
			if (string.sub(pname, 1, 5) == "group") then
				return false,checkedconds
			end
			
			local dist = roomsizex - x - 2

			local is_param_this, raycast_units, raycast_tileids, this_count = parse_this_param_and_get_raycast_units(pname)
			local ray_unit_is_empty = is_param_this and raycast_units[2] -- <-- this last condition checks if empty is a raycast unit
			
			if (unitid ~= 1) then
				if (b ~= "level") or ((b == "level") and (alreadyfound[1] ~= nil)) then
					if (dist >= 1) then
						for g=1,dist do
							if (pname ~= "empty") and not ray_unit_is_empty then
								local tileid = (x + g) + y * roomsizex
								if (unitmap[tileid] ~= nil) then
									for c,d in ipairs(unitmap[tileid]) do
										if (d ~= unitid) and (alreadyfound[d] == nil) then
											local unit = mmf.newObject(d)
											local name_ = getname(unit)
											
											if (pnot == false) then
												if is_param_this then
													if raycast_units and raycast_units[d] and alreadyfound[bcode] == nil then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ == pname) and (alreadyfound[bcode] == nil) then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											else
												if is_param_this then
													if raycast_units and not raycast_units[d] and alreadyfound[bcode] == nil and name_ ~= "text" then
														alreadyfound[bcode] = 1
														alreadyfound[d] = 1
														allfound = allfound + 1
													end
												elseif (name_ ~= pname) and (alreadyfound[bcode] == nil) and (name_ ~= "text") then
													alreadyfound[bcode] = 1
													alreadyfound[d] = 1
													allfound = allfound + 1
												end
											end
										end
									end
								end
							else
								local nearempty = false
						
								local tileid = (x + g) + y * roomsizex
								local l = map[0]
								local tile = l:get_x(x + g,y)
								
								local tcode = tostring(tileid) .. "e"
								
								if ((unitmap[tileid] == nil) or (#unitmap[tileid] == 0)) and (alreadyfound[tcode] == nil) then 
									nearempty = true
								end

								if nearempty and ray_unit_is_empty and raycast_tileids[tileid] == nil then
									nearempty = false
								end
								
								if (pnot == false) then
									if nearempty and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								else
									if (nearempty == false) and (alreadyfound[bcode] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[tcode] = 1
										allfound = allfound + 1
									end
								end
							end
						end
					end
				elseif (b == "level") and (alreadyfound[bcode] == nil) and (alreadyfound[1] == nil) then
					alreadyfound[bcode] = 1
					alreadyfound[1] = 1
					allfound = allfound + 1
				end
			else
				local ulist = false
			
				if is_param_this then
					if this_count > 0 then
						ulist = true
					end
				elseif (b ~= "empty") and (b ~= "level") then
					if (pnot == false) then
						if (unitlists[pname] ~= nil) and (#unitlists[pname] > 0) and (alreadyfound[bcode] == nil) then
							for c,d in ipairs(unitlists[pname]) do
								if (alreadyfound[d] == nil) then
									alreadyfound[bcode] = 1
									alreadyfound[d] = 1
									ulist = true
									break
								end
							end
						end
					else
						for c,d in pairs(unitlists) do
							local tested = false
							
							if (c ~= pname) and (#d > 0) and (alreadyfound[bcode] == nil) then
								for e,f in ipairs(d) do
									if (alreadyfound[f] == nil) then
										alreadyfound[bcode] = 1
										alreadyfound[f] = 1
										ulist = true
										tested = true
										break
									end
								end
							end
							
							if tested then
								break
							end
						end
					end
				elseif (b == "empty") then
					local empties = findempty()
					
					if (#empties > 0) and (alreadyfound[bcode] == nil) then
						for c,d in ipairs(unitlists[pname]) do
							if (alreadyfound[d] == nil) then
								alreadyfound[bcode] = 1
								alreadyfound[d] = 1
								ulist = true
								break
							end
						end
					end
				end
				
				if (b ~= "text") and (ulist == false) then
					if (surrounds.r ~= nil) then
						for c,d in ipairs(surrounds.r) do
							if (pnot == false) then
								if (ulist == false) and (d == pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							else
								if (ulist == false) and (d ~= pname) and (alreadyfound[bcode] == nil) then
									alreadyfound[bcode] = 1
									ulist = true
								end
							end
						end
					end
				end
				
				if ulist or (b == "text") then
					alreadyfound[bcode] = 1
					allfound = allfound + 1
				end
			end
		end
	else
		print("no parameters given!")
		return false,checkedconds
	end

	return (allfound == #params),checkedconds
end

condlist.feeling = function(params,checkedconds,checkedconds_,cdata)
	local allfound = 0
	local alreadyfound = {}
	local name,unitid,x,y,limit = cdata.name,cdata.unitid,cdata.x,cdata.y,cdata.limit
	
	if (#params > 0) then
		for a,b in ipairs(params) do
			local pname = b
			local pnot = false
			if (string.sub(b, 1, 4) == "not ") then
				pnot = true
				pname = string.sub(b, 5)
			end
			
			local bcode = b .. "_" .. tostring(a)
			local prev_GLOBAL_checking_stable = GLOBAL_checking_stable
			
			-- @mods(this) - special case to handle THIS pointing to a property
			local raycast_objects, found_letterwords = parse_this_param_and_get_raycast_infix_units(pname, "feeling")
			local raycast_props = {}
			for _, raycast_object in ipairs(raycast_objects) do
				local ray_unitid = plasma_utils.parse_object(raycast_object)
				local text_name = get_turning_text_interpretation(ray_unitid)
				raycast_props[text_name] = true
			end
			for _, letterword in ipairs(found_letterwords) do
				local word = letterword[1]
				if (string.len(word) > 5) and (string.sub(word, 1, 5) == "text_") then
                    word = string.sub(letterword[1], 6)
                end
				raycast_props[word] = true
			end
			
			if (featureindex[name] ~= nil) then
				for c,d in ipairs(featureindex[name]) do
					local drule = d[1]
					local dconds = d[2]
					
					if (checkedconds[tostring(dconds)] == nil) then
						if (pnot == false) then
							if (drule[1] == name) and (drule[2] == "is") and (drule[3] == b or raycast_props[drule[3]]) then
								checkedconds[tostring(dconds)] = 1
								
								--@mods(stable) special case with "feeling stable". Need this global set to true to refer to
								-- featureindex instead of the object's stablerules. Also, save the state of the global before setting to true
								local prev_GLOBAL_checking_stable = GLOBAL_checking_stable
								if b == "stable" then
									GLOBAL_checking_stable = true
								end
								if (alreadyfound[bcode] == nil) and testcond(dconds,unitid,x,y,nil,limit,checkedconds) then
									alreadyfound[bcode] = 1
									allfound = allfound + 1
									break
								end
							end
						else
							if (string.sub(drule[3], 1, 4) ~= "not ") then
								local obj = unitreference["text_" .. drule[3]]
								
								if (obj ~= nil) then
									local objtype = getactualdata_objlist(obj,"type")
									
									if (objtype == 2) then
										if drule[3] == "stable" then
											GLOBAL_checking_stable = true
										end
										if (drule[1] == name) and (drule[2] == "is") and (drule[3] ~= pname) then
											checkedconds[tostring(dconds)] = 1
											
											if (alreadyfound[bcode] == nil) and testcond(dconds,unitid,x,y,nil,limit,checkedconds) then
												alreadyfound[bcode] = 1
												allfound = allfound + 1
												break
											end
										end
									end
								end
							end
						end
					end
				end
			end
			GLOBAL_checking_stable = prev_GLOBAL_checking_stable
		end
	else
		return false,checkedconds,true
	end
	
	return (allfound == #params),checkedconds,true
end