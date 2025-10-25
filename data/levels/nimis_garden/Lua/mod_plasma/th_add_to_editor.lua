table.insert(editor_objlist_order, "text_this")
table.insert(editor_objlist_order, "text_that")
table.insert(editor_objlist_order, "text_these")
table.insert(editor_objlist_order, "text_those")
table.insert(editor_objlist_order, "text_block")
table.insert(editor_objlist_order, "text_relay")
table.insert(editor_objlist_order, "text_pass")

editor_objlist["text_this"] = 
{
	name = "text_this",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_noun", "pointer nouns"},
	tiling = 0,
	type = 0,
	layer = 20,
	colour = {0, 1},
    colour_active = {0, 3},
}
editor_objlist["text_that"] = 
{
	name = "text_that",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_noun", "pointer nouns"},
	tiling = 0,
	type = 0,
	layer = 20,
	colour = {0, 1},
    colour_active = {0, 3},
}
editor_objlist["text_these"] = 
{
	name = "text_these",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_noun", "pointer nouns"},
	tiling = 0,
	type = 0,
	layer = 20,
	colour = {0, 1},
    colour_active = {0, 3},
}
editor_objlist["text_those"] = 
{
	name = "text_those",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_noun", "pointer nouns"},
	tiling = 0,
	type = 0,
	layer = 20,
	colour = {0, 1},
    colour_active = {0, 3},
}
editor_objlist["text_block"] = 
{
	name = "text_block",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_quality", "pointer nouns"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {2, 1},
    colour_active = {2, 2},
}
editor_objlist["text_relay"] = 
{
	name = "text_relay",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_quality", "pointer nouns"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {5, 2},
	colour_active = {5, 4},
}
editor_objlist["text_pass"] = 
{
	name = "text_pass",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract", "text_quality", "pointer nouns"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {4, 3},
    colour_active = {4, 4},
}

formatobjlist()

local DirTextDisplay = PlasmaModules.load_module("general/directional_text_display")
DirTextDisplay:register_directional_text_prefix("this")
DirTextDisplay:register_directional_text_prefix("that")
DirTextDisplay:register_directional_text_prefix("these")
DirTextDisplay:register_directional_text_prefix("those")