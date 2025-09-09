allrouters = {}
alloffline = {}
allrouters[1] = {}
allrouters[2] = {}
allrouters[3] = {}
allrouters[4] = {}
alloffline[1] = {}
alloffline[2] = {}
alloffline[3] = {}
alloffline[4] = {}
gettingrouters = false
gettingoffline = false
local_radius = 2
idk_what_to_name = false

table.insert(editor_objlist_order, "text_offline")
table.insert(editor_objlist_order, "text_router")
table.insert(editor_objlist_order, "text_local")
table.insert(editor_objlist_order, "text_channel1")
table.insert(editor_objlist_order, "text_channel2")
table.insert(editor_objlist_order, "text_channel3")
table.insert(editor_objlist_order, "text_offline1")
table.insert(editor_objlist_order, "text_offline2")
table.insert(editor_objlist_order, "text_offline3")
table.insert(editor_objlist_order, "text_local1")
table.insert(editor_objlist_order, "text_local2")
table.insert(editor_objlist_order, "text_local3")

editor_objlist["text_channel1"] = 
{
	name = "text_channel1",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {2, 0},
	colour_active = {2, 1},
}
editor_objlist["text_offline1"] = 
{
	name = "text_offline1",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {2, 0},
	colour_active = {2, 1},
}
editor_objlist["text_offline2"] = 
{
	name = "text_offline2",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {2, 3},
	colour_active = {2, 4},
}
editor_objlist["text_channel2"] = 
{
	name = "text_channel2",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {2, 3},
	colour_active = {2, 4},
}
editor_objlist["text_channel3"] = 
{
	name = "text_channel3",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {4, 3},
	colour_active = {4, 4},
}
editor_objlist["text_offline3"] = 
{
	name = "text_offline3",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {4, 3},
	colour_active = {4, 4},
}
editor_objlist["text_router"] = 
{
	name = "text_router",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {1, 2},
	colour_active = {1, 3},
}
editor_objlist["text_offline"] = 
{
	name = "text_offline",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_quality", "local mod"},
	tiling = -1,
	type = 2,
	layer = 19,
	colour = {6, 1},
	colour_active = {2, 3},
}

editor_objlist["text_local"] = 
{
	name = "text_local",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_condition", "local mod", "text_prefix"},
	tiling = -1,
	type = 3,
	layer = 19,
	colour = {0, 2},
	colour_active = {1, 3},
}
editor_objlist["text_local1"] = 
{
	name = "text_local1",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_condition", "local mod", "text_prefix"},
	tiling = -1,
	type = 3,
	layer = 19,
	colour = {2, 0},
	colour_active = {2, 1},
}
editor_objlist["text_local2"] = 
{
	name = "text_local2",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_condition", "local mod", "text_prefix"},
	tiling = -1,
	type = 3,
	layer = 19,
	colour = {2, 3},
	colour_active = {2, 4},
}
editor_objlist["text_local3"] = 
{
	name = "text_local3",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","text_condition", "local mod", "text_prefix"},
	tiling = -1,
	type = 3,
	layer = 19,
	colour = {4, 3},
	colour_active = {4, 4},
}


formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Mathguy", {1,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
			name = "local",
			author = "Mathguy",
			display_sprites = {"text_local", "text_local1", "text_local2", "text_local3"},
			description =
[[True if the object is within the 5x5 square area centered around a "ROUTER" or "CHANNEL X" object.

"LOCAL X" is true only if the object is within the range of a "ROUTER" object or a "CHANNEL X" object.]],
		},
		{
			name = "router",
			author = "Mathguy",
			description =
[[Creates a 5x5 square area centered around the "ROUTER" object where other objects can be considered "LOCAL" across all channels.

The "ROUTER" object itself will not be considered "LOCAL" to itself. Only to other router objects.]],
		},
		{
			name = "channel",
			thumbnail_obj = "text_channel1",
			author = "Mathguy",
			display_sprites = {"text_channel1", "text_channel2", "text_channel3"},
			description = [[Creates a 5x5 square area centered around the "CHANNEL X" object where other objects can be considered both "LOCAL" and "LOCAL X".]],
		},
		{
			name = "offline",
			author = "Mathguy",
			display_sprites = {"text_offline", "text_offline1", "text_offline2", "text_offline3"},
			description =
[[Prevents the object from being considered "LOCAL" across all channels.
	
If an object is "OFFLINE X", then the object specifically cannot be "LOCAL X".]],
		},
    })
end

condlist['local'] = function(params, checkedconds,checkedconds_, cdata)
	getallrouters()
	getallofflines()
	return isonline(cdata.unitid, cdata.x, cdata.y, 0), checkedconds
end
condlist['local1'] = function(params, checkedconds,checkedconds_, cdata)
	getallrouters()
	getallofflines()
	return isonline(cdata.unitid, cdata.x, cdata.y, 1), checkedconds
end
condlist['local2'] = function(params, checkedconds,checkedconds_, cdata)
	getallrouters()
	getallofflines()
	return isonline(cdata.unitid, cdata.x, cdata.y, 2), checkedconds
end
condlist['local3'] = function(params, checkedconds,checkedconds_, cdata)
	getallrouters()
	getallofflines()
	return isonline(cdata.unitid, cdata.x, cdata.y, 3), checkedconds
end

local copy

function getallofflines()
	if gettingoffline then
		return nil
	end
alloffline[1] = {}
alloffline[2] = {}
alloffline[3] = {}
alloffline[4] = {}
	gettingoffline = true
	getallchannelofflines(1)
	getallchannelofflines(2)
	getallchannelofflines(3)
	getallchannelofflines(0)
end

function getallchannelofflines(channel)
	local positions = {}
	local ids = {}
	local emptys = {}
	local level_is_offline = false
	if (channel == 0) then
		ids,emptys = findallfeature(nil,"is","offline")
		level_is_offline = hasfeature("level","is","offline",1, nil, nil, nil, true)
	elseif (channel == 1) then
		ids,emptys = findallfeature(nil,"is","offline1")
		level_is_offline = hasfeature("level","is","offline1",1, nil, nil, nil, true)
	elseif (channel == 2) then
		ids,emptys = findallfeature(nil,"is","offline2")
		level_is_offline = hasfeature("level","is","offline2",1, nil, nil, nil, true)
	elseif (channel == 3) then
		ids,emptys = findallfeature(nil,"is","offline3")
		level_is_offline = hasfeature("level","is","offline3",1, nil, nil, nil, true)
	end
	if (ids == nil) then
		ids = {}
	end
	if (emptys == nil) then
		emptys = {}
	end
	if (level_is_offline == true) then
		table.insert(positions,{"level"})
	end
	if (#ids ~= 0) then
		for i,v in pairs(ids) do
			if (v == 1) then
				table.insert(positions,{"level"})
			elseif (v == 2) then

			else
				local unit = mmf.newObject(v)
				table.insert(positions,{v, unit.values[XPOS],unit.values[YPOS],unit.strings[UNITNAME],})
			end
		end
	end
	if (#emptys ~= 0) then
		for i,v in pairs(emptys) do
			for i2,v2 in pairs(v) do
				table.insert(positions,{2, i2 % roomsizex,i2 // roomsizex,"empty"})
			end
		end
	end
	alloffline[channel+1] = positions
	if (channel == 0) then
		alloffline[2] = concat0(alloffline[2], positions)
		alloffline[3] = concat0(alloffline[3], positions)
		alloffline[4] = concat0(alloffline[4], positions)
	end
end
-- @Merge: reordered and made this copy() function local. Basegame also has its own copy() function. While this function is only used locally in this file
local function copy(table0)
	local empty = {}
	if (table0 == nil) then
		return {}
	end
	if (#table0 == 0) then
		return {}
	end
	for i, v in pairs(table0) do
		if (type(v) == table) then	
			empty[i] = copy(v)
		else
			empty[i] = v
		end
	end
	return empty
end
function concat0(t1_, t2_)
	local t1 = copy(t1_)
	local t2 = copy(t2_)
	for i,v in pairs(t2) do
		table.insert(t1, v)
	end
	return t1
end
function haslocal(table)
	for i,v in pairs(table) do
		if ((v[1]=="local") or (v[1]=="local1") or(v[1]=="local2") or(v[1]=="local3")) then
			return true
		end
	end
	return false
end
function isoffline(id,x,y, channel)
	if (id == 1) then
		for i, v in pairs(alloffline[channel + 1]) do
			if (v[1] == "level") then
				return true
			end
		end
	elseif (id == 2) then
		for i, v in pairs(alloffline[channel + 1]) do
			if ((v[2] == x) and (v[3] == y)) then
				return true
			end
		end
	else
		for i, v in pairs(alloffline[channel + 1]) do
			if (v[1] == id) then
				return true
			end
		end

	end
	return false
end
function isrouter(id,x,y, channel)
	if (id == 1) then
		for i, v in pairs(allrouters[channel + 1]) do
			if (v[1] == "level") then
				return true
			end
		end
	elseif (id == 2) then
		for i, v in pairs(allrouters[channel + 1]) do
			if ((v[2] == x) and (v[3] == y)) then
				return true
			end
		end
	else
		for i, v in pairs(allrouters[channel + 1]) do
			if (v[1] == id) then
				return true
			end
		end

	end
	return false
end
function getallrouters()
	if gettingrouters then
		return nil
	end
	allrouters[1] = {}
	allrouters[2] = {}
	allrouters[3] = {}
	allrouters[4] = {}
	gettingrouters = true
	getallchannelrouters(0)
	getallchannelrouters(1)
	getallchannelrouters(2)
	getallchannelrouters(3)
	if idk_what_to_name then
		gettingrouters = false
	end
end

function getallchannelrouters(channel)
	local positions = {}
	local ids = {}
	local emptys = {}
	local level_is_router = false
	if (channel == 0) then
		ids,emptys = findallfeature(nil,"is","router")
		level_is_router = hasfeature("level","is","router",1, nil, nil, nil, true)
	elseif (channel == 1) then
		ids,emptys = findallfeature(nil,"is","channel1")
		level_is_router = hasfeature("level","is","channel1",1, nil, nil, nil, true)
	elseif (channel == 2) then
		ids,emptys = findallfeature(nil,"is","channel2")
		level_is_router = hasfeature("level","is","channel2",1, nil, nil, nil, true)
	elseif (channel == 3) then
		ids,emptys = findallfeature(nil,"is","channel3")
		level_is_router = hasfeature("level","is","channel3",1, nil, nil, nil, true)
	end
	if (level_is_router == true) then
		table.insert(positions,{"level"})
	end
	if (ids == nil) then
		ids = {}
	end
	if (emptys == nil) then
		emptys = {}
	end
	if (#ids ~= 0) then
		for i,v in pairs(ids) do
			if (v == 1) then
				table.insert(positions,{"level"})
			elseif (v == 2) then

			else
				local unit = mmf.newObject(v)
				table.insert(positions,{unit.values[XPOS],unit.values[YPOS],unit.fixed})
			end
		end
	end
	if (#emptys ~= 0) then
		for i,v in pairs(emptys) do
			for i2,v2 in pairs(v) do
				table.insert(positions,{i2 % roomsizex,i2 // roomsizex, 2})
			end
		end
	end
	allrouters[channel+1] =  positions
	if (channel ~= 0) then
		allrouters[1] = concat0(allrouters[1], positions)
	end
end
function contains(table, value)
	if (table == nil) then
		return false
	end
	for i,v in pairs(table) do
		if (v==values) then
			return true
		end
	end
	return false

end

function isonline(id,x,y, channel)
	if (#(allrouters[channel + 1]) ~= 0) then
		if isoffline(id,x,y, channel) then
			return false
		end
		if (id == 1) then
			if ((#(allrouters[channel + 1]) == 1) and (allrouters[channel + 1][1][3] == 1)) then
				return false
			end
			return true
		elseif (id == 2) then
			for i,v in pairs(allrouters[channel + 1]) do
				local unit = mmf.newObject(v[3])
				local x2 = unit.values[XPOS]
				local y2 = unit.values[YPOS]
				if ((v[3] == 2) and (v[1] == x) and (v[2] == y)) then
					goto continue
				end
				if (v[1] == "level") then
					return true
				end
				if (((x-x2)>-local_radius-1) and ((x-x2)<local_radius+1) and ((y-y2)>-local_radius-1) and ((y-y2)<local_radius+1)) then
					return true
				end
				::continue::
			end
		else
			for i,v in pairs(allrouters[channel + 1]) do
				local unit = mmf.newObject(v[3])
				local x2 = unit.values[XPOS]
				local y2 = unit.values[YPOS]
				if (v[3] == id) then
					goto continue
				end
				if (v[1] == "level") then
					return true
				end
				if (((x-x2)>-local_radius-1) and ((x-x2)<local_radius+1) and ((y-y2)>-local_radius-1) and ((y-y2)<local_radius+1)) then
					return true
				end
				::continue::
			end
		end
	end
	return false
end

table.insert(mod_hook_functions["level_start"],
	function()
		allrouters[1] = {}
		allrouters[2] = {}
		allrouters[3] = {}
		allrouters[4] = {}
		alloffline[1] = {}
		alloffline[2] = {}
		alloffline[3] = {}
		alloffline[4] = {}
		gettingrouters = false
		gettingoffline = false
	end

)
table.insert(mod_hook_functions["undoed_after"],
	function()
		gettingrouters = false
		gettingoffline = false
	end

)
table.insert(mod_hook_functions["turn_end"],
	function()
		gettingrouters = false
		gettingoffline = false
	end
)
table.insert(mod_hook_functions["rule_update"],
	function()
		gettingrouters = false
		gettingoffline = false
	end
)


--[[ @Merge: code() was merged ]]
