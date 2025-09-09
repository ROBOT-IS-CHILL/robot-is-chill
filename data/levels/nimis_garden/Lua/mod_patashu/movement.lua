moving_units = {}


--[[ @Merge: movecommand() was merged ]]



--[[ @Merge: check() was merged ]]



--[[ @Merge: trypush() was merged ]]



--[[ @Merge: dopush() was merged ]]



--[[ @Merge: move() was merged ]]



--[[ @Merge: add_moving_units() was merged ]]


function find_copys(unitid,dir)
	--fast track
	if featureindex["copy"] == nil then return {} end
	local result = {}
	local unit = mmf.newObject(unitid)
	local unitname = getname(unit)
	local iscopy = findallfeature(nil,"copy",unitname,true)
	for _,copyid in ipairs(iscopy) do
		local copyunit = mmf.newObject(copyid)
		local copyname = getname(copyunit)
		if not hasfeature(copyname,"is","sleep",copyid) and (isstill_or_locked(copyid,copyunit.values[XPOS],copyunit.values[YPOS],dir) == false) then
			table.insert(result, copyid)
		end
	end
	return result;
end

function find_sidekicks(unitid,dir)
	--fast track
	if featureindex["sidekick"] == nil then return {} end
	local result = {}
	local unit = mmf.newObject(unitid)
	local unitname = getname(unit)
	local lazy = hasfeature(unitname,"is","lazy",unitid)
	if lazy ~= nil then
		return result;
	end
	local x,y = unit.values[XPOS],unit.values[YPOS]
	--print("find_sidekicks",x,y,dir)
	local dir90 = (dir+1) % 4;
	for i = 1,2 do
		local curdir = (dir90+2*i) % 4;
		local curdx = ndirs[curdir+1][1];
		local curdy = ndirs[curdir+1][2];
		local curx = x+curdx;
		local cury = y+curdy;
		--print("find_sidekicks is checking",curx,cury)
		local obs = findobstacle(curx,cury);
		for i,id in ipairs(obs) do
			if (id ~= -1) then
				local obsunit = mmf.newObject(id)
				local obsname = getname(obsunit)
				if hasfeature(obsname,"is","sidekick",id) and (isstill_or_locked(id,curx,cury,dir) == false) then
					table.insert(result, id);
				end
			end
		end
	end
	return result;
end

function find_sticky_pulls(unitid,dir)
	--fast track
	if featureindex["sticky"] == nil then return {} end
	--sticky units don't pull or be pulled via the normal logic - they'll happen at sidekick/copy timing because that's a lot easier for my brain to mentally model
	local result = {}
	local unit = mmf.newObject(unitid)
	local unitname = getname(unit)
	local sticky = hasfeature(unitname,"is","sticky",unitid)
	local lazy = hasfeature(unitname,"is","lazy",unitid)
	if lazy ~= nil then
		return result;
	end
	local x,y = unit.values[XPOS],unit.values[YPOS]
	--print("find_sticky_pulls",x,y,dir)
	local curdir = (dir+2) % 4;
	local curdx = ndirs[curdir+1][1];
	local curdy = ndirs[curdir+1][2];
	local curx = x+curdx;
	local cury = y+curdy;
	local obs = findobstacle(curx,cury);
	for i,id in ipairs(obs) do
		if (id ~= -1) then
			local obsunit = mmf.newObject(id)
			local obsname = getname(obsunit)

			--@mods(plasma x patashu) - get directional pull working with sticky
			arrow_prop_mod_globals.group_arrow_properties = false
			local ispull = hasfeature(obsname,"is","pull",id)
			arrow_prop_mod_globals.group_arrow_properties = true

			local _, _, ispull = do_directional_collision(dir, obsname, id, false, false, ispull, x, y, curdx, curdy, true, "pull")
			if ispull and (isstill_or_locked(id,curx,cury,dir) == false) and (sticky or hasfeature(obsname,"is","sticky",id)) then
				table.insert(result, id);
			end
		end
	end
	return result;
end

function apply_moonwalk(unitid, x, y, dir, ox, oy, reverse)
	local name = "empty"
	local sgn = reverse == true and -1 or 1
	if (unitid ~= 2) then
		local unit = mmf.newObject(unitid)
		name = getname(unit)
	end
	local rotatedness = 0;
	rotatedness = rotatedness + sgn*hasfeature_count(name,"is","moonwalk",unitid,x,y)*2;
	rotatedness = rotatedness + sgn*hasfeature_count(name,"is","drunk",unitid,x,y);
	rotatedness = rotatedness + sgn*hasfeature_count(name,"is","drunker",unitid,x,y)*0.5;
	local mag = 1;
	mag = mag + hasfeature_count(name,"is","skip",unitid,x,y);
	if (dir ~= nil and ox ~= nil and oy ~= nil) then
		dir = (dir + trunc(rotatedness)) % 4
		ox, oy = dirtooxoy(oxoytodir(ox, oy) + rotatedness)
		ox = ox * mag;
		oy = oy * mag;
		return dir, ox, oy
	elseif (dir ~= nil) then
		dir = (dir + trunc(rotatedness)) % 4
		return dir
	elseif (ox ~= nil and oy ~= nil) then
		ox, oy = dirtooxoy(oxoytodir(ox, oy) + rotatedness)
		ox = ox * mag;
		oy = oy * mag;
		return ox, oy
	else
		return nil
	end
end

function oxoytodir(ox, oy)
	ox = sign(ox)
	oy = sign(oy)
	if ox == 1 and oy == 0 then
		return 0
	elseif ox == 1 and oy == -1 then
		return 0.5
	elseif ox == 0 and oy == -1 then
		return 1
	elseif ox == -1 and oy == -1 then
		return 1.5
	elseif ox == -1 and oy == 0 then
		return 2
	elseif ox == -1 and oy == 1 then
		return 2.5
	elseif ox == 0 and oy == 1 then
		return 3
	elseif ox == 1 and oy == 1 then
		return 3.5
	end
	return nil
end

function dirtooxoy(dir)
	dir = dir % 4
	if dir == math.floor(dir) then
		local result = ndirs[dir+1]
		return result[1], result[2]
	elseif dir == 0.5 then
		return 1, -1
	elseif dir == 1.5 then
		return -1, -1
	elseif dir == 2.5 then
		return -1, 1
	elseif dir == 3.5 then
		return 1, 1
	end
	return 0, 0
end

function sign(num)
	if num > 0 then
		return 1
	elseif num < 0 then
		return -1
	else
		return 0
	end
end
function trunc(num)
	if num > 0 then
		return math.floor(num)
	elseif num < 0 then
		return math.ceil(num)
	else
		return 0
	end
end

function queue_move(unitid,ox,oy,dir,specials,reason,x,y)
	table.insert(movelist, {unitid,ox,oy,dir,specials,reason,x,y})

	--implement SIDEKICK
	--implement STICKY/PULL
	if (unitid ~= 2) then
		local unit = mmf.newObject(unitid)
		local unitname = getname(unit)
		local sidekicks = find_sidekicks(unitid, dir);
		for _,sidekickid in ipairs(sidekicks) do
			--no multiplicative cascades in sidekick - only start sidekicking if we're not already sidekicking
			local sidekick = mmf.newObject(sidekickid)
			local alreadysidekicking = false
			for _,other in ipairs(moving_units) do
				if other.unitid == sidekickid then
					alreadysidekicking = true
					break
				end
			end
			if not alreadysidekicking then
				updatedir(sidekickid, dir)
				--print("adding to moving_units",unitid,getname(sidekick))
				table.insert(moving_units, {unitid = sidekickid, reason = "sidekick", state = 0, moves = 1, dir = dir, xpos = sidekick.values[XPOS], ypos = sidekick.values[YPOS]})
			end
		end
		local others = find_sticky_pulls(unitid, dir);
		for _,otherid in ipairs(others) do
			--no multiplicative cascades in sticky/pull you get the idea by now I hope
			local other = mmf.newObject(otherid)
			local alreadymoving = false
			for _,other2 in ipairs(moving_units) do
				if other2.unitid == otherid then
					alreadymoving = true
					break
				end
			end
			if not alreadymoving then
				updatedir(otherid, dir)
				--print("adding to moving_units",unitid,getname(other))
				table.insert(moving_units, {unitid = otherid, reason = "pull", state = 0, moves = 1, dir = dir, xpos = other.values[XPOS], ypos = other.values[YPOS]})
			end
		end
	end
end

function find_entire_sticky_unit(unitid, dx, dy)
	local unit = mmf.newObject(unitid)
	local unitx = unit.values[XPOS];
	local unity = unit.values[YPOS];
	local unitname = getname(unit)
	--print("0:",unitid,unitname,unitx,unity)
	local units, pushers, pullers = {}, {}, {}
	local visited = {}
	local ignored = {}
	local unit_added = {}
	visited[tostring(unitx)..","..tostring(unity)] = unitid
	unit_added[unitid] = true
	
	--base case - add the original unit
	table.insert(units, unitid)
	
	--on with the floodfill!
	local unchecked_tiles = {{unitx, unity}}
	
	while #unchecked_tiles > 0 do
		local x, y = unchecked_tiles[1][1], unchecked_tiles[1][2]
		local cur_unitid = visited[tostring(x)..","..tostring(y)];
		local cur_unit = mmf.newObject(cur_unitid)
		--print("a:",x,y,cur_unitid)
		table.remove(unchecked_tiles, 1)
		--print("a.5:",#unchecked_tiles)
		
		--check all 4 directions
		for i = 1,4 do
			local cur_dx, cur_dy = dirtooxoy(i)
			local xx, yy = x+cur_dx, y+cur_dy
			--print("b:",cur_dx,cur_dy,xx,yy,tostring(xx)..","..tostring(yy),visited[tostring(xx)..","..tostring(yy)])
			--visit surrounding tiles if we don't know their status yet
			--print("c")
			local tileid = xx + yy * roomsizex
			local others = unitmap[tileid]
			local first = false
			if (others ~= nil) then
				for _,other in ipairs(others) do
					local other_unit = mmf.newObject(other)
					local other_name = getname(other_unit);
					--print("d:",other,other_name)
					if ((other_name == unitname) or very_sticky) and (not float_breaks_sticky or floating(other, unitid, xx, yy)) and not unit_added[other] then
						local other_is_sticky = hasfeature(other_name,"is","sticky",other)
						if other_is_sticky then
							--print("f, we did it")
							table.insert(units, other)
							unit_added[other] = true
							--print(#units)
							--we haven't expanded out from this tile yet - queue it
							if not first then
								table.insert(unchecked_tiles, {xx, yy})
								--print("f.5:",#unchecked_tiles)
								first = true
								visited[tostring(xx)..","..tostring(yy)] = other
							end
						end
					end
				end
			end
			--END iterate units on that tile
			--END visit surrounding unvisited tile
				
			--while checking the forward/backward direction, add the current unit to pushers/pullers if we know the tile ahead of/behind it is vacant
			--print("g", dx, cur_dx, dy, cur_dy, visited[tostring(xx)..","..tostring(yy)], not visited[tostring(xx)..","..tostring(yy)])
			if dx == cur_dx and dy == cur_dy and not visited[tostring(xx)..","..tostring(yy)] then
				--print("added a pusher:",cur_unitid)
				table.insert(pushers, cur_unitid)
			elseif -dx == cur_dx and -dy == cur_dy and not visited[tostring(xx)..","..tostring(yy)] then
				--print("added a puller:", cur_unitid)
				table.insert(pullers, cur_unitid)
			end
		end
		--END check all 8 directions 
		--print("final:",#unchecked_tiles)
	end
	--END check all unchecked tiles

	--failsafe: return the original unit in case we couldn't floodfill at all for whatever reason
	
	if #units == 0 then
		table.insert(units, unitid)
	end
	if #pushers == 0 then
		table.insert(pushers, unitid)
	end
	if #pullers == 0 then
		table.insert(pullers, unitid)
	end

	return units, pushers, pullers
end