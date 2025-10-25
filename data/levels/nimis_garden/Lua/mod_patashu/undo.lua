
--[[ @Merge: newundo() was merged ]]



--[[ @Merge: addundo() was merged ]]



--[[ @Merge: undo() was merged ]]



--[[ @Merge: undostate() was merged ]]
 

--If gras becomes roc, then later roc becomes undo, when it disappears we want the gras to come back. This is how we code that - by scanning for the related remove event and undoing that too.
--[[function scanAndRecreateOldUnit(i, unit_id, created_from_id, ignore_no_undo)
	while (true) do
		local v = undobuffer[2][i]
		if (v == nil) then
			return
		end
		local action = v[1]
		--TODO: implement for MOUS
		if (action == "remove") then
			local old_creator_id = v[7];
			if v[7] == created_from_id then
				--no exponential cloning if gras turned into 2 rocs - abort if there's already a unit with that name on that tile
				local tile, x, y = v[2], v[3], v[4];
				local data = tiles_list[tile];
				local stuff = getUnitsOnTile(x, y, nil, true);
				for _,on in ipairs(stuff) do
					if on.name == data.name then
						return
					end
				end
				local _, new_unit = undoOneAction(turn, i, v, ignore_no_undo);
				if (new_unit ~= nil) then
					addUndo({"create", new_unit.id, true, created_from_id = unit_id})
				end
				return
			end
		end
		i = i - 1;
	end
end]]

--if water becomes roc, and roc is no undo, if we undo then the water shouldn't come back. This is how we code that - by scanning for all related create events. If we find one existing no undo byproduct and no existing non-no undo byproducts, we return false.
function turnedIntoOnlyNoUndoUnits(i, unit_id)
	local found_no_undo = false;
	local found_non_no_undo = false;
	while (true) do
		local v = undobuffer[2][i]
		if (v == nil) then
			break
		end
		local action = v[1];
		local created_from_id = v[9]
		local created_id = v[10]
		if (action == "create") and created_from_id == unit_id then
			local still_exists = mmf.newObject(created_id)
			if (still_exists ~= nil) then
				if unit_ignores_undos(created_id) then
					found_no_undo = true;
				else
					found_non_no_undo = true;
					break;
				end
			end
		end
		i = i + 1;
	end
	return not (found_non_no_undo or not found_no_undo);
end

function unit_ignores_undos(unitid)
	local still_exists = mmf.newObject(unitid)
	if (still_exists ~= nil) then
		local name = getname(still_exists)
		if resetting then
			return hasfeature(name,"is","noreset",unitid)
		else
			return hasfeature(name,"is","noundo",unitid)
		end
	end
end