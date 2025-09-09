

function doupdate()
	if (#updatelist > 0) then
		for i,data in pairs(updatelist) do
			if (#wordunits > 0) then
				for a,b in ipairs(wordunits) do
					if (b[1] == data[1]) then
						updatecode = 1
					end
				end
			end
			
			if (#wordrelatedunits > 0) then
				for a,b in ipairs(wordrelatedunits) do
					if (b[1] == data[1]) then
						updatecode = 1
					end
				end
			end
			
			-- EDIT: ECHO
			if (#echounits > 0) then
				for a,b in ipairs(echounits) do
					if (b[1] == data[1]) then
						updatecode = 1
					end
				end
			end
			
			if (#echorelatedunits > 0) then
				for a,b in ipairs(echorelatedunits) do
					if (b[1] == data[1]) then
						updatecode = 1
					end
				end
			end
			
			--MF_alert("Doing " .. data[2] .. " for " .. tostring(data[1]))
			
			if (data[2] == "update") then
				if (data[3] ~= nil) and (data[4] ~= nil) then
					update(data[1],data[3],data[4],data[5])
				else
					print("No meaningful update data for object " .. tostring(data[1]) .. ", " .. data[2])
				end
			elseif (data[2] == "updatedir") then
				if (data[3] ~= nil) then
					updatedir(data[1],data[3])
				else
					print("No meaningful update data for object " .. tostring(data[1]) .. ", " .. data[2])
				end
			elseif (data[2] == "convert") or (data[2] == "emptyconvert") then
				if (data[3] ~= nil) then
					doconvert(data)
				else
					print("No meaningful update data for object " .. tostring(data[1]) .. ", " .. data[2])
				end
			elseif (data[2] == "createall") then
				if (data[3] ~= nil) then
					local dolevels = data[4] or false
					createall(data[3],nil,nil,nil,dolevels)
				else
					print("No meaningful update data for object " .. tostring(data[1]) .. ", " .. data[2])
				end
			elseif (data[2] == "dofloat") then
				if (data[3] ~= nil) and (data[4] ~= nil) then
					local unit = mmf.newObject(tonumber(data[6]))
					unit.values[FLOAT] = tonumber(data[4])
					addundo({"float",data[7],data[5],data[3],data[4]})
				else
					print("No meaningful update data for object " .. tostring(data[1]) .. ", " .. data[2])
				end
			end
		end
	end
	
	updatelist = {}
end