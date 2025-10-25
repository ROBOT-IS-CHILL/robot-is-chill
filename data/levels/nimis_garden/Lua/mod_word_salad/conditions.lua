-- SINFUL: true if the object previously destroyed something
condlist.sinful = function(params,_,_,cdata)
	local unitid = cdata.unitid
	if unitid == 2 then -- EMPTY doesn't store karma
		return false
	elseif unitid == 1 then -- LEVEL: check level karma
		return (levelKarma == true)
	end
	local unit = mmf.newObject(unitid)
	if unit.karma then
		return true
	end
	return false
end

-- ALIGNED: true if all objects of that kind are in the same row or column
condlist.aligned = function(params,checkedconds,_,cdata)
	local unitid, name, x, y = cdata.unitid, cdata.name, cdata.x, cdata.y
	if (unitid ~= 1 and unitid ~= 2) then		
		-- Ideally we could cache the results, but there's no mod hook for the very beginning of a turn?
		local xFail, yFail = false, false
		for _,u in pairs(unitlists[name]) do
			local unit = mmf.newObject(u)
			local ux, uy = unit.values[XPOS], unit.values[YPOS]
			if (ux ~= x) then -- We found a unit in a different column
				xFail = true
			end
			if (uy ~= y) then -- We found a unit in a different row
				yFail = true
			end
			if xFail and yFail then -- We have at least one unit in a different row and one unit in a different column, so they can't be aligned
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 2 then
		local xFail, yFail = false, false
		local empties = findempty()
		for _,b in ipairs(empties) do
			local ex, ey = b % roomsizex, math.floor(b / roomsizex)
			if (ex ~= x) then -- We found an empty in a different column
				xFail = true
			end
			if (ey ~= y) then -- We found an empty in a different row
				yFail = true
			end
			if xFail and yFail then -- We have at least one empty in a different row and one empty in a different column, so they can't be aligned
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 1 then
		return ws_levelAlignedRow or ws_levelAlignedColumn,checkedconds
	end
	return false,checkedconds
end

-- ALIGNEDX: true if all objects of that kind are in the same row
condlist.alignedx = function(params,checkedconds,_,cdata)
	local unitid, name, y = cdata.unitid, cdata.name, cdata.y
	if (unitid ~= 1 and unitid ~= 2) then		
		-- Ideally we could cache the results, but there's no mod hook for the very beginning of a turn?
		for _,u in pairs(unitlists[name]) do
			local unit = mmf.newObject(u)
			local uy = unit.values[YPOS]
			if (uy ~= y) then -- One of the units is in a different row
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 2 then
		local empties = findempty()
		for _,b in ipairs(empties) do
			local ey = math.floor(b / roomsizex)
			if (ey ~= y) then -- One of the empties is in a different row
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 1 then
		return ws_levelAlignedRow,checkedconds
	end
	return false,checkedconds
end

-- ALIGNEDY: true if all objects of that kind are in the same column
condlist.alignedy = function(params,checkedconds,_,cdata)
	local unitid, name, x = cdata.unitid, cdata.name, cdata.x
	if (unitid ~= 1 and unitid ~= 2) then		
		-- Ideally we could cache the results, but there's no mod hook for the very beginning of a turn?
		for _,u in pairs(unitlists[name]) do
			local unit = mmf.newObject(u)
			local ux = unit.values[XPOS]
			if (ux ~= x) then -- One of the units is in a different column
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 2 then
		local empties = findempty()
		for _,b in ipairs(empties) do
			local ex = b % roomsizex
			if (ex ~= x) then -- One of the empties is in a different column
				return false,checkedconds
			end
		end
		return true,checkedconds
	elseif unitid == 1 then
		return ws_levelAlignedColumn,checkedconds
	end
	return false,checkedconds
end