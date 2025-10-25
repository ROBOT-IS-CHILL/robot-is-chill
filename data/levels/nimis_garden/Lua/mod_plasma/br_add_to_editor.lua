table.insert(editor_objlist_order, "text_pivot_is")
table.insert(editor_objlist_order, "text_branching_is")

table.insert(editor_objlist_order, "text_pivot_and")
table.insert(editor_objlist_order, "text_branching_and")

table.insert(editor_objlist_order, "text_pivot_has")
table.insert(editor_objlist_order, "text_branching_has")

table.insert(editor_objlist_order, "text_pivot_near")
table.insert(editor_objlist_order, "text_branching_near")

table.insert(editor_objlist_order, "text_pivot_make")
table.insert(editor_objlist_order, "text_branching_make")

table.insert(editor_objlist_order, "text_pivot_follow")
table.insert(editor_objlist_order, "text_branching_follow")

table.insert(editor_objlist_order, "text_pivot_mimic")
table.insert(editor_objlist_order, "text_branching_mimic")

table.insert(editor_objlist_order, "text_pivot_play")
table.insert(editor_objlist_order, "text_branching_play")

table.insert(editor_objlist_order, "text_pivot_eat")
table.insert(editor_objlist_order, "text_branching_eat")

table.insert(editor_objlist_order, "text_pivot_fear")
table.insert(editor_objlist_order, "text_branching_fear")

table.insert(editor_objlist_order, "text_pivot_on")
table.insert(editor_objlist_order, "text_branching_on")

table.insert(editor_objlist_order, "text_pivot_without")
table.insert(editor_objlist_order, "text_branching_without")

table.insert(editor_objlist_order, "text_pivot_facing")
table.insert(editor_objlist_order, "text_branching_facing")

table.insert(editor_objlist_order, "text_pivot_above")
table.insert(editor_objlist_order, "text_branching_above")

table.insert(editor_objlist_order, "text_pivot_below")
table.insert(editor_objlist_order, "text_branching_below")

table.insert(editor_objlist_order, "text_pivot_besideleft")
table.insert(editor_objlist_order, "text_branching_besideleft")

table.insert(editor_objlist_order, "text_pivot_besideright")
table.insert(editor_objlist_order, "text_branching_besideright")

table.insert(editor_objlist_order, "text_pivot_feeling")
table.insert(editor_objlist_order, "text_branching_feeling")

local omni_white = { {2, 2}, {2, 4} }
local omni_red   = { {4, 1}, {4, 2} }
local omni_green = { {5, 2}, {5, 4} }
local omni_blue  = { {3, 3}, {1, 4} }
local pivot_white = { {1, 3}, {1, 4} }
local pivot_red   = { {2, 2}, {2, 3} }
local pivot_green = { {4, 1}, {4, 2} }
local pivot_blue  = { {3, 2}, {3, 3} }

editor_objlist["text_branching_is"] = 
{
	name = "text_branching_is",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_is",
}

editor_objlist["text_branching_and"] = 
{
	name = "text_branching_and",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 6,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_and",
}

editor_objlist["text_branching_has"] = 
{
	name = "text_branching_has",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_has",
}

editor_objlist["text_branching_fear"] = 
{
	name = "text_branching_fear",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_red[1],
	colour_active = omni_red[2],
	pairedwith = "text_fear",
}

editor_objlist["text_branching_make"] = 
{
	name = "text_branching_make",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_make",
}

editor_objlist["text_branching_follow"] = 
{
	name = "text_branching_follow",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_green[1],
	colour_active = omni_green[2],
	pairedwith = "text_follow",
}

editor_objlist["text_branching_mimic"] = 
{
	name = "text_branching_mimic",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_red[1],
	colour_active = omni_red[2],
	pairedwith = "text_mimic",
}

editor_objlist["text_branching_play"] = 
{
	name = "text_branching_play",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_green[1],
	colour_active = omni_green[2],
	pairedwith = "text_play",
}

editor_objlist["text_branching_eat"] = 
{
	name = "text_branching_eat",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = omni_red[1],
	colour_active = omni_red[2],
	pairedwith = "text_eat",
}

editor_objlist["text_branching_near"] = 
{
	name = "text_branching_near",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_near",
}

editor_objlist["text_branching_on"] = 
{
	name = "text_branching_on",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_on",
}
editor_objlist["text_branching_without"] = 
{
	name = "text_branching_without",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_without",
}
editor_objlist["text_branching_facing"] = 
{
	name = "text_branching_facing",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_facing",
	argextra = {"right","up","left","down"},
}
editor_objlist["text_branching_above"] = 
{
	name = "text_branching_above",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_blue[1],
	colour_active = omni_blue[2],
	pairedwith = "text_above",
}
editor_objlist["text_branching_below"] = 
{
	name = "text_branching_below",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_blue[1],
	colour_active = omni_blue[2],
	pairedwith = "text_below",
}
editor_objlist["text_branching_besideleft"] = 
{
	name = "text_branching_besideleft",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_blue[1],
	colour_active = omni_blue[2],
	pairedwith = "text_besideleft",
}
editor_objlist["text_branching_besideright"] = 
{
	name = "text_branching_besideright",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_blue[1],
	colour_active = omni_blue[2],
	pairedwith = "text_besideright",
}
editor_objlist["text_branching_feeling"] = 
{
	name = "text_branching_feeling",
	sprite_in_root = false,
	unittype = "text",
	tags = {"omni text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = omni_white[1],
	colour_active = omni_white[2],
	pairedwith = "text_feeling",
	argtype = {2},
}

-- Pivot text
editor_objlist["text_pivot_is"] = 
{
	name = "text_pivot_is",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_is",
}

editor_objlist["text_pivot_and"] = 
{
	name = "text_pivot_and",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 6,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_and",
}

editor_objlist["text_pivot_has"] = 
{
	name = "text_pivot_has",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_has",
}

editor_objlist["text_pivot_fear"] = 
{
	name = "text_pivot_fear",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_red[1],
	colour_active = pivot_red[2],
	pairedwith = "text_fear",
}

editor_objlist["text_pivot_make"] = 
{
	name = "text_pivot_make",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_make",
}

editor_objlist["text_pivot_follow"] = 
{
	name = "text_pivot_follow",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_green[1],
	colour_active = pivot_green[2],
	pairedwith = "text_follow",
}

editor_objlist["text_pivot_mimic"] = 
{
	name = "text_pivot_mimic",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_red[1],
	colour_active = pivot_red[2],
	pairedwith = "text_mimic",
}

editor_objlist["text_pivot_play"] = 
{
	name = "text_pivot_play",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_green[1],
	colour_active = pivot_green[2],
	pairedwith = "text_play",
}

editor_objlist["text_pivot_eat"] = 
{
	name = "text_pivot_eat",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_verb"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = pivot_red[1],
	colour_active = pivot_red[2],
	pairedwith = "text_eat",
}

editor_objlist["text_pivot_near"] = 
{
	name = "text_pivot_near",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_near",
}

editor_objlist["text_pivot_on"] = 
{
	name = "text_pivot_on",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_on",
}
editor_objlist["text_pivot_without"] = 
{
	name = "text_pivot_without",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_without",
}
editor_objlist["text_pivot_facing"] = 
{
	name = "text_pivot_facing",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_facing",
	argextra = {"right","up","left","down"},
}
editor_objlist["text_pivot_above"] = 
{
	name = "text_pivot_above",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = {3, 2},
	colour_active = pivot_blue[2],
	pairedwith = "text_above",
}
editor_objlist["text_pivot_below"] = 
{
	name = "text_pivot_below",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = {3, 2},
	colour_active = pivot_blue[2],
	pairedwith = "text_below",
}
editor_objlist["text_pivot_besideleft"] = 
{
	name = "text_pivot_besideleft",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = {3, 2},
	colour_active = pivot_blue[2],
	pairedwith = "text_besideleft",
}
editor_objlist["text_pivot_besideright"] = 
{
	name = "text_pivot_besideright",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = {3, 2},
	colour_active = pivot_blue[2],
	pairedwith = "text_besideright",
}
editor_objlist["text_pivot_feeling"] = 
{
	name = "text_pivot_feeling",
	sprite_in_root = false,
	unittype = "text",
	tags = {"pivot text", "plasma's mods", "text", "abstract", "text_condition"},
	tiling = -1,
	type = 7,
	layer = 20,
	colour = pivot_white[1],
	colour_active = pivot_white[2],
	pairedwith = "text_feeling",
	argtype = {2},
}

formatobjlist()