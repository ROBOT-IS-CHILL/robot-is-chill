

function editor_currobjlist_add(objid_,build_,dopairs_,gridpos_,pairid_,pairedwith_)
	-- @mods(text splicing) - Override reason - when adding "cut" to the palette, auto-add all letterunits to the palette
	local valid = true
	local objid = tonumber(objid_) or objid_
	local build = true
	local dopairs = true
	local pairedwith = true
	local newtilename = ""

	local data = editor_objlist[objid]
	local newname = data.name or "error"
	
	if (build_ ~= nil) then
		build = build_
	end
	
	if (dopairs_ ~= nil) then
		dopairs = dopairs_
	end
	
	if (pairedwith_ ~= nil) then
		pairedwith = pairedwith_
	end

	local checking = true
	while checking and valid do
		checking = false

		for i,v in ipairs(editor_currobjlist) do
			if (v.id == objid) then
				newtilename = v.object
				valid = false
			end
		
			if (v.name == newname) then
				checking = true
				
				if (tonumber(string.sub(v.name, -1)) ~= nil) then
					local num = tonumber(string.sub(v.name, -1)) + 1
					
					newname = string.sub(newname, 1, string.len(newname)-1) .. tostring(num)
				else
					newname = newname .. "2"
				end
			end
		end
	end
	
	local newid = 0
	
	if valid then
		local id = #editor_currobjlist + 1
		newid = id
		editor_currobjlist[id] = {}
		
		local this = editor_currobjlist[id]
		this.id = objid
		this.name = newname

		if (pairid_ ~= nil) then
			this.pair = pairid_
		end
		
		local edata = objlistdata.objectreference
		local tdata = objlistdata.tilereference
		edata[objid] = 1
		
		local ogx,ogy = 0,0
		if (gridpos_ == nil) then
			local gx1,gy1 = findgridid(objid,1)
			local gx2,gy2 = findgridid(objid,0)
			
			ogx,ogy = gx1,gy1
			
			--MF_alert("Adding " .. this.name .. ", overlap: " .. tostring(gx1) .. ", " .. tostring(gy1) .. ", full: " .. tostring(gx2) .. ", " .. tostring(gy2) .. ", gridpos nil")
			
			this.grid_overlap = {gx1, gy1}
			this.grid_full = {gx2, gy2}
		else
			ogx,ogy = gridpos_[1],gridpos_[2]
			local gx2,gy2 = findgridid(objid,0)
			
			local gridreference = {}
			
			--MF_alert("Adding " .. this.name .. ", overlap: " .. tostring(ogx) .. ", " .. tostring(ogy) .. ", full: " .. tostring(gx2) .. ", " .. tostring(gy2) .. ", gridpos not nil")
			
			this.grid_overlap = {ogx, ogy}
			this.grid_full = {gx2, gy2}
			
			local gridreference = objlistdata.gridreference_overlap
			table.insert(gridreference[ogx][ogy], objid)
		end
		
		local tilename = ""
		local tileid = 0
		
		tilename = "object" .. string.sub("00" .. tostring(tileid), -3)
		
		while (tdata[tilename] ~= nil) do
			tileid = tileid + 1
			tilename = "object" .. string.sub("00" .. tostring(tileid), -3)
		end
		
		local alreadyexists = false
		this.object = tilename
		
		for i,v in pairs(tileslist) do
			if (v.name == newname) then
				local valid = true
				
				if (changes[i] ~= nil) then
					local cdata = changes[i]
					
					if (cdata.name ~= nil) and (cdata.name ~= newname) then
						valid = false
					end
				end
				
				if valid then
					alreadyexists = true
					this.object = i
					tilename = i
					local tilepos = v.tile
					this.tile = {tilepos[1], tilepos[2]}
				end
			end
		end
		
		tdata[tilename] = 1
		
		if (alreadyexists == false) then
			local d = tileslist[tilename]
			local tilepos = d.tile
			this.tile = {tilepos[1], tilepos[2]}
		end
		
		local unitid = MF_create(tilename)
		resetchanges(unitid)
		
		local colourstring = "0,3"
		if (data.colour ~= nil) then
			local c = data.colour
			colourstring = tostring(c[1]) .. "," .. tostring(c[2])
		end
		
		local activecolourstring = "0,3"
		if (data.colour_active ~= nil) then
			local c = data.colour_active
			activecolourstring = tostring(c[1]) .. "," .. tostring(c[2])
		end
		
		local argtypestring = "0"
		if (data.argtype ~= nil) then
			local c = data.argtype
			argtypestring = gettablestring(c)
		end
		
		local argextrastring = ""
		if (data.argextra ~= nil) then
			local c = data.argextra
			
			argextrastring = gettablestring(c)
		end
		
		local customobjectsstring = ""
		if (data.customobjects ~= nil) then
			local c = data.customobjects
			
			customobjectsstring = gettablestring(c)
		end
		
		local c_name = newname
		local c_image = data.sprite or data.name
		local c_colour = colourstring
		local c_tiling = data.tiling or -1
		local c_type = data.type or 0
		local c_unittype = data.unittype or "object"
		local c_activecolour = activecolourstring
		local c_root = true
		local c_layer = data.layer or 10
		local c_argtype = argtypestring
		local c_argextra = argextrastring
		local c_customobjects = customobjectsstring
		
		if (data.sprite_in_root ~= nil) then
			c_root = data.sprite_in_root
		end
		
		local changelist = {c_name, c_image, c_colour, c_tiling, c_type, c_unittype, c_activecolour, c_root, c_layer, c_argtype, c_argextra, c_customobjects}
		savechange(tilename, changelist, unitid)
		dochanges_allinstances(tilename)
		dospritechanges(tilename)
		
		MF_cleanremove(unitid)
		
		if dopairs then
			local pair_id = 0
			local so = data.paired or true
			if (data.unittype == "object") and (data.type == 0) and so then
				for i,v in pairs(editor_objlist) do
					if (v.name == "text_" .. data.name) and (v.type == 0) and (v.unittype == "text") then
						--MF_alert(this.name .. " adds " .. v.name)
						pair_id = editor_currobjlist_add(i,false,false,{ogx,ogy},id)
						this.pair = pair_id
					end
				end
			elseif (data.unittype == "text") and (data.type == 0) and so then
				local objpair = string.sub(data.name, 6)
				
				for i,v in pairs(editor_objlist) do
					if (v.name == objpair) and (v.type == 0) and (v.unittype == "object") then
						--MF_alert(this.name .. " adds " .. v.name)
						pair_id = editor_currobjlist_add(i,false,false,{ogx,ogy},id)
						this.pair = pair_id
					end
				end
			end
		end
		
		if (data.pairedwith ~= nil) and pairedwith then
			local alreadyadded = false
			
			for i,v in ipairs(editor_currobjlist) do
				if (v.name == data.pairedwith) then
					alreadyadded = true
				end
			end
			
			if (alreadyadded == false) then
				for i,v in pairs(editor_objlist) do
					if (v.name == data.pairedwith) and (v.type ~= 0) and (v.unittype == "text") then
						--MF_alert(this.name .. " adds " .. v.name)
						editor_currobjlist_add(i,false,nil,nil,nil,false)
					end
				end
			end
        end
        
        add_cut_or_pack_palette_groups(editor_currobjlist, data)
		
		newtilename = tilename
	else
		MF_alert("ID already listed! " .. tostring(objid))
	end
	
	if build then
		editor_objects_build()
	end
	
	setundo_editor()
	--MF_alert(tostring(newid) .. ", " .. tostring(newtilename) .. ", " .. tostring(newname))
	
	return newid,newtilename,newname
end