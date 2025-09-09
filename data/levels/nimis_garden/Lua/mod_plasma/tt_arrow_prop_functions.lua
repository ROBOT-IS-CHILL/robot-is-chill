local PlasmaSettings = PlasmaModules.load_module("general/gui")
enable_directional_shift = not PlasmaSettings.get_toggle_setting("disable_dir_shift")

for name,display in pairs(arrow_property_display) do
	word_names[name] = display
end

arrow_prop_mod_globals = {}
function reset_arrow_properties()
	arrow_prop_mod_globals = {
		-- The reason behind this global is mainly to toggle between handling interactions that do not depend on direction ("i.e on conditions") and interactions that do depend on direction
		-- Prime example: directional you depends on direction when the player input is pressed, but doesn't depend on direction when handling a you object on a defeat object
		group_arrow_properties = true
	}
end

table.insert(mod_hook_functions["level_start"], 
    function()
		enable_directional_shift = not PlasmaSettings.get_toggle_setting("disable_dir_shift")
    end
)


function do_directional_more(full_more_units, delthese_)
    if full_more_units == nil then
        return
    end

    local full_more_units_dict = {}
	
	local more_unit_dirs = {}
	if full_more_units ~= nil then
    	for _, unit in ipairs(full_more_units) do
		full_more_units_dict[unit.fixed] = true
			more_unit_dirs[unit] = {1,2,3,4}
		end
	end

	for i=1,4 do
		local dirfeature = dirfeaturemap[i]
		local more_units = getunitswitheffect("more"..dirfeature,false,delthese)
		for j,unit in ipairs(more_units) do
			if not full_more_units_dict[unit.fixed] then
				if not more_unit_dirs[unit] then
					more_unit_dirs[unit] = {}
				end
				table.insert(more_unit_dirs[unit], i)
			end
		end
	end

	return more_unit_dirs
end

function do_directional_collision(dir, name, unitid, isstop, ispush, ispull, x, y, ox, oy, pulling, reason)
	local pulling_ = pulling or false
	local reason_ = reason or nil

    if dir ~= nil and dir >= 0 and dir <= 3 then
        local dirfeature = dirfeaturemap[dir + 1]
        local dirfeaturerotate = dirfeaturemap[rotate(dir) + 1]
        if not isstop then
            isstop = hasfeature(name,"is","stop"..dirfeaturerotate,unitid,x+ox,y+oy)
        end
		if not ispush then
			if reason_ == "fall" then
				ispush = hasfeature(name,"is","push"..dirfeaturerotate,unitid,x+ox,y+oy)
			else
				ispush = hasfeature(name,"is","push"..dirfeature,unitid,x+ox,y+oy)
			end
        end
        if not ispull then
            if pulling_ then
                ispull = hasfeature(name,"is","pull"..dirfeature,unitid,x+ox,y+oy)
            else
                ispull = hasfeature(name,"is","pull"..dirfeaturerotate,unitid,x+ox,y+oy)
            end
        end
    end

    return isstop,ispush,ispull
end

function do_directional_swap_hasfeature(dir, name, unitid, x, y)
	if dir ~= nil and dir >= 0 and dir <= 3 then
		local dirfeature = dirfeaturemap[dir + 1]
		return hasfeature(name,"is","swap"..dirfeature,unitid,x,y,{"still"})
	else
		return false
	end
end

function do_directional_swap_findfeatureat(dir, full_swap_units, x, y, ox, oy)
	local result = {}
	if dir ~= nil and dir >= 0 and dir <= 3 then
		local swapped_objs = {}
		if full_swap_units then
			for a,b in ipairs(full_swap_units) do
				swapped_objs[b] = true
			end
		end
		local dirfeature = dirfeaturemap[rotate(dir) + 1]
		local features = findfeatureat(nil,"is","swap"..dirfeature,x+ox,y+oy,{"still"})
		if features and #features > 0 then
			for i,v in ipairs(features) do
				if not swapped_objs[b] then
					table.insert(result, v)
				end
			end
		end
	end

	return result
end

function do_directional_you(dir_)
	local playersdir = {}
	local emptydir = {}
	local dirfeature = nil
	if dir_ ~= nil and dir_ >= 0 and dir_ <= 3 then
		dirfeature = dirfeaturemap[dir_ + 1]
	end
	if dirfeature ~= nil then
		playersdir,emptydir = findallfeature(nil,"is","you"..dirfeature)
	end
	return playersdir, emptydir
end

function do_directional_you2(dir_, found_normalyou2)
	local playersdir = {}
	local emptydir = {}
	local dirfeature = nil
	if dir_ ~= nil and dir_ >= 0 and dir_ <= 3 then
		dirfeature = dirfeaturemap[dir_ + 1]
	end
	if (dirfeature ~= nil) then
		playersdir,emptydir = findallfeature(nil,"is","you2"..dirfeature)
	end
	return playersdir, emptydir
end

function do_directional_you_auto(dir_, dir_2)
	local playersdir = {}
	local emptydir = {}
	local playersdir2 = {}
	local emptydir2 = {}
	local dirfeature = nil
	local dirfeature2 = nil
	if dir_ ~= nil and dir_ >= 0 and dir_ <= 3 then
		dirfeature = dirfeaturemap[dir_ + 1]
	end
	if dir_2 ~= nil and dir_2 >= 0 and dir_2 <= 3 then
		dirfeature2 = dirfeaturemap[dir_2 + 1]
	end

	if dirfeature ~= nil then
		playersdir,emptydir = findallfeature(nil,"is","you"..dirfeature)
	end
	if dirfeature2 ~= nil then
		playersdir2,emptydir2 = findallfeature(nil,"is","you2"..dirfeature2)
	end
	return playersdir, emptydir, playersdir2, emptydir2
end

function do_directional_you_level(dir_, dir_2, playerid)
	local levelmove = nil
	if dir_~= nil and dir_ >= 0 and dir_ <= 3 then
		local dirfeature = dirfeaturemap[dir_ + 1]
		if (playerid == 1) then
			levelmove = findfeature("level","is","you"..dirfeature)
		elseif (playerid == 2) then
			levelmove = findfeature("level","is","you2"..dirfeature)
	
			if (levelmove == nil) then
				levelmove = findfeature("level","is","you"..dirfeature)
			end
		elseif (playerid == 3) then
			levelmove = findfeature("level","is","you"..dirfeature) or {}
			levelmove2 = findfeature("level","is","you2"..dirfeature)
			
			if (#levelmove > 0) and (dir_ ~= nil) then
				levelmovedir = dir_
			elseif (levelmove2 ~= nil) and (dir_ ~= nil) then
				levelmovedir = dir_
			elseif (dir_2 ~= nil) then
				levelmovedir = dir_2
			end
			
			if (levelmove2 ~= nil) then
				for i,v in ipairs(levelmove2) do
					table.insert(levelmove, v)
				end
			end
			
			if (#levelmove == 0) then
				levelmove = nil
			end
		end
	end
	return levelmove
end

function do_directional_level_pushpull(dir, pull)
	local feature = "push"
	pull = pull or false
	if pull then
		feature = "pull"
	end

	local leveldir = -1
	if dir ~= nil and dir >= 0 and dir <= 3 then
		local dirfeature = dirfeaturemap[dir+1]
		local levelmovedir = findfeature("level","is",feature..dirfeature)
		if (levelmovedir ~= nil) then
			for e,f in ipairs(levelmovedir) do
				if testcond(f[2],1) then
					leveldir = dir
				end
			end
		end
	end

	return leveldir
end

function do_directional_select(dir_)
	dirfeature = nil
	if dir_ ~= nil and dir_ >= 0 and dir_ <= 3 then
		dirfeature = dirfeaturemap[dir_ + 1]
	end

	if not dirfeature then
		return {}
	else
		return getunitswitheffect("select"..dirfeature,true)
	end
end

function do_directional_shift_moveblock()
	local shifted = {}

	arrow_prop_mod_globals.group_arrow_properties = false
	local isshift = findallfeature(nil,"is","shift",true)
	-- NIMI MOD EDIT: make throw objects update their facing later in the turn like shift objects do
	local isthrow = findallfeature(nil,"is","throw",true)
	for i,v in ipairs(isthrow) do
		table.insert(isshift,v)
	end
	arrow_prop_mod_globals.group_arrow_properties = true
	
	for a,unitid in ipairs(isshift) do
		if (unitid ~= 2) and (unitid ~= 1) then
			local unit = mmf.newObject(unitid)
			local x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
			
			local things = findallhere(x,y,unitid)
			
			if (#things > 0) and (isgone(unitid) == false) then
				for e,f in ipairs(things) do
					if floating(unitid,f,x,y) and (issleep(unitid,x,y) == false) then
						local newunit = mmf.newObject(f)
						local name = newunit.strings[UNITNAME]

						-- @TODO: this small section was added by hempuli on supporting reverse. But it doesnt do anything right now...
						-- Keep watch if hempuli decides to update this
						if (featureindex["reverse"] ~= nil) then
							local turndir = unit.values[DIR]
							turndir = reversecheck(newunit.fixed,unit.values[DIR],x,y)
						end
						
						if (newunit.flags[DEAD] == false) then
							if shifted[newunit] == nil then
								local data = {
									undo_x = x,
									undo_y = y,
									horsmove = 0,
									vertmove = 0,
								}
								update_net_shift_data(unit.values[DIR], data)
								shifted[f] = data
							else
								update_net_shift_data(unit.values[DIR], shifted[f])
							end
						end
					end
				end
			end
		end
	end

	for shiftdir=0,3 do
		local dirfeature = dirfeaturemap[shiftdir + 1]
		local dirshift = findallfeature(nil,"is","shift"..dirfeature,true)
		for a,unitid in ipairs(dirshift) do
			if (unitid ~= 2) and (unitid ~= 1) then
				local unit = mmf.newObject(unitid)
				local x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
				
				local things = findallhere(x,y,unitid)
				
				if (#things > 0) and (isgone(unitid) == false) then
					for e,f in ipairs(things) do
						if floating(unitid,f,x,y) and (issleep(unitid,x,y) == false) then
							local newunit = mmf.newObject(f)
							local name = newunit.strings[UNITNAME]
							
							if (newunit.flags[DEAD] == false) then
								if shifted[f] == nil then
									local data = {
										undo_x = x,
										undo_y = y,
										horsmove = 0,
										vertmove = 0,
									}
									update_net_shift_data(shiftdir, data)
									shifted[f] = data
								else
									update_net_shift_data(shiftdir, shifted[f])
								end
							end
						end
					end
				end
			end
		end
	end

	for unitid, data in pairs(shifted) do
		local unit = mmf.newObject(unitid)
		local horsdir = -1
		local vertdir = -1
		
		if data.horsmove > 0 then
			horsdir = 0
		elseif data.horsmove < 0 then
			horsdir = 2
		end
		if data.vertmove > 0 then
			vertdir = 1
		elseif data.vertmove < 0 then
			vertdir = 3
		end

		local dir = -1
		if (horsdir > -1) and (vertdir > -1) then
			dir = vertdir
		elseif horsdir > -1 then
			dir = horsdir
		elseif vertdir > -1 then
			dir = vertdir
		end

		if dir > -1 then
			addundo({"update",name,data.undo_x,data.undo_y,unit.values[DIR],data.undo_x,data.undo_y,dir,unit.values[ID]})
			unit.values[DIR] = dir
			
			--@ Turning text --
			if is_turning_text(unit.strings[NAME]) then
				updatecode = 1
			end
			--@ Turning text --
		end
	end
end

function do_directional_shift_resolve_stacked_shifts(moving_units)
	local new_moving_units = {}
	for i,data in ipairs(moving_units) do
		if (data.reason == "shift" or data.reason == "launch" or data.reason == "throw") and data.dirshiftstate == 0 then
			if data.horsmove ~= 0 or data.vertmove ~= 0 then
				if data.horsmove > 0 then
					data.horsdir = 0
				elseif data.horsmove < 0 then
					data.horsdir = 2
				end
				if data.vertmove > 0 then
					data.vertdir = 1
				elseif data.vertmove < 0 then
					data.vertdir = 3
				end

				data.horsmove = math.abs(data.horsmove)
				data.vertmove = math.abs(data.vertmove)
				data.moves = data.horsmove + data.vertmove
				data.dir = data.horsdir
				data.dirshiftstate = 1

				table.insert(new_moving_units, data)
			end
		end
	end

	return new_moving_units
end

function do_directional_shift_update_shift_state(data, updatemovecount)
	local change_to_vertical_after = false
	if data.dirshiftstate == 1 then
		if data.horsmove <= 0 or data.horsdir == -1 then 
			data.dirshiftstate = 2
		else
			data.dir = data.horsdir
			-- updatedir(data.unitid, data.dir)
			-- dir = data.dir
			if updatemovecount then
				data.horsmove = data.horsmove - 1
			end

			if data.horsmove <= 0 then 
				data.dirshiftstate = 2
			end
		end
	end
	if data.dirshiftstate == 2 then
		if data.vertmove <= 0 or data.vertdir == -1 then 
			data.dirshiftstate = 3
		else
			data.dir = data.vertdir
			-- updatedir(data.unitid, data.dir)
			-- dir = data.dir
			if updatemovecount then
				data.vertmove = data.vertmove - 1
			end

			if data.vertmove <= 0 then 
				data.dirshiftstate = 3
			end
		end
	end

	if change_to_vertical_after then
		data.dirshiftstate = 2
	end
end

function update_net_shift_data(dir, data, value)
	if not enable_directional_shift then
		return
	end

	value = value or 1
	if data.dirshiftstate == nil then
		data.horsdir = -1
		data.vertdir = -1
		data.horsmove = 0
		data.vertmove = 0
		data.dirshiftstate = 0
	end

	if dir == 0 then
		data.horsmove = data.horsmove + value
	elseif dir == 1 then
		data.vertmove = data.vertmove + value
	elseif dir == 2 then
		data.horsmove = data.horsmove - value
	else
		data.vertmove = data.vertmove - value
	end
end

function do_directional_boom(unit)
	local ux,uy = unit.values[XPOS],unit.values[YPOS]
	local name = getname(unit)
	local iszero = true
	local out_booms = {0,0,0,0}

	for i=0,3 do
		local dirfeature = dirfeaturemap[i + 1]
		out_booms[i] = hasfeature_count(name,"is","boom"..dirfeature,unit.fixed,ux,uy)
		if out_booms[i] > 0 then
			iszero = false
		end
	end

	return out_booms, iszero
end