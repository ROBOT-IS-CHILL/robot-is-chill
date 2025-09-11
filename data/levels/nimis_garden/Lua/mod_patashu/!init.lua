--If true, MOONWALK and related properties (DRUNK, DRUNKER, SKIP) apply to PUSH, PULL, SHIFT and LAUNCH in addition to basically everything else. Defaults to true.
very_drunk = true
--If true, two things at different float values can't stick together. Defaults to true.
float_breaks_sticky = true
--If true, two things with different names can stick together. Defaults to false.
very_sticky = false

print("Start of !init.lua")

function apply_patashu_settings(settings_dict)
	for setting_name, value in pairs(settings_dict) do
		if setting_name == "very_drunk" then
			very_drunk = value
		elseif setting_name == "float_breaks_sticky" then
			float_breaks_sticky = value
		elseif setting_name == "very_sticky" then
			very_sticky = value
		end
	end
end

table.insert(objlistdata.alltags, "patashu")

table.insert(editor_objlist_order, "text_slip")
editor_objlist["text_slip"] = {
	name = "text_slip",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_slide")
editor_objlist["text_slide"] = {
	name = "text_slide",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_hates")
editor_objlist["text_hates"] = {
	name = "text_hates",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {5, 0},
	colour_active = {5, 1},
}

table.insert(editor_objlist_order, "text_likes")
editor_objlist["text_likes"] = {
	name = "text_likes",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {5, 0},
	colour_active = {5, 1},
}

table.insert(editor_objlist_order, "text_sidekick")
editor_objlist["text_sidekick"] = {
	name = "text_sidekick",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {6, 0},
	colour_active = {6, 1},
}

table.insert(editor_objlist_order, "text_lazy")
editor_objlist["text_lazy"] = {
	name = "text_lazy",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {6, 1},
	colour_active = {6, 2},
}

table.insert(editor_objlist_order, "text_moonwalk")
editor_objlist["text_moonwalk"] = {
	name = "text_moonwalk",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_drunk")
editor_objlist["text_drunk"] = {
	name = "text_drunk",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_drunker")
editor_objlist["text_drunker"] = {
	name = "text_drunker",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_skip")
editor_objlist["text_skip"] = {
	name = "text_skip",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_tall")
editor_objlist["text_tall"] = {
	name = "text_tall",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {0, 0},
	colour_active = {0, 1},
}

table.insert(editor_objlist_order, "text_oneway")
editor_objlist["text_oneway"] = {
	name = "text_oneway",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 3},
	colour_active = {1, 4},
}

table.insert(editor_objlist_order, "text_copy")
editor_objlist["text_copy"] = {
	name = "text_copy",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {2, 1},
	colour_active = {2, 2},
}

table.insert(editor_objlist_order, "text_reset")
editor_objlist["text_reset"] = {
	name = "text_reset",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {3, 0},
	colour_active = {3, 1},
	layer = 20,
}

table.insert(editor_objlist_order, "text_noundo")
editor_objlist["text_noundo"] = {
	name = "text_noundo",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {5, 2},
	colour_active = {5, 3},
}

table.insert(editor_objlist_order, "text_noreset")
editor_objlist["text_noreset"] = {
	name = "text_noreset",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {5, 2},
	colour_active = {5, 3},
}

table.insert(editor_objlist_order, "text_stops")
editor_objlist["text_stops"] = {
	name = "text_stops",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {5, 0},
	colour_active = {5, 1},
}

table.insert(editor_objlist_order, "text_pushes")
editor_objlist["text_pushes"] = {
	name = "text_pushes",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {6, 0},
	colour_active = {6, 1},
}

table.insert(editor_objlist_order, "text_pulls")
editor_objlist["text_pulls"] = {
	name = "text_pulls",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {6, 1},
	colour_active = {6, 2},
}

table.insert(editor_objlist_order, "text_sinks")
editor_objlist["text_sinks"] = {
	name = "text_sinks",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {1, 2},
	colour_active = {1, 3},
}

table.insert(editor_objlist_order, "text_defeats")
editor_objlist["text_defeats"] = {
	name = "text_defeats",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {2, 0},
	colour_active = {2, 1},
}

table.insert(editor_objlist_order, "text_opens")
editor_objlist["text_opens"] = {
	name = "text_opens",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {6, 1},
	colour_active = {2, 4},
}

table.insert(editor_objlist_order, "text_shifts")
editor_objlist["text_shifts"] = {
	name = "text_shifts",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {1, 2},
	colour_active = {1, 3},
}

--[[table.insert(editor_objlist_order, "text_teles")
editor_objlist["text_teles"] = {
	name = "text_teles",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {1, 2},
	colour_active = {1, 4},
}]]

table.insert(editor_objlist_order, "text_swaps")
editor_objlist["text_swaps"] = {
	name = "text_swaps",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {3, 0},
	colour_active = {3, 1},
}

table.insert(editor_objlist_order, "text_melts")
editor_objlist["text_melts"] = {
	name = "text_melts",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	operatortype = "verb",
	colour = {2, 2},
	colour_active = {2, 3},
}

table.insert(editor_objlist_order, "text_topple")
editor_objlist["text_topple"] = {
	name = "text_topple",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {3, 0},
	colour_active = {3, 1},
}

table.insert(editor_objlist_order, "text_zoom")
editor_objlist["text_zoom"] = {
	name = "text_zoom",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

table.insert(editor_objlist_order, "text_yeet")
editor_objlist["text_yeet"] = {
    name = "text_yeet",
    sprite_in_root = false,
    unittype = "text",
    tags = {"text", "text_verb", "patashu"},
    tiling = -1,
    type = 1,
    layer = 20,
    colour = {0, 2},
    colour_active = {0, 3},
}

table.insert(editor_objlist_order, "text_launch")
editor_objlist["text_launch"] = {
	name = "text_launch",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

formatobjlist()

table.insert(editor_objlist_order, "text_send")
editor_objlist["text_send"] = {
	name = "text_send",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {1, 2},
	colour_active = {1, 3},
}

table.insert(editor_objlist_order, "text_sticky")
editor_objlist["text_sticky"] = {
	name = "text_sticky",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "patashu"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {6, 1},
	colour_active = {2, 4},
}

table.insert(editor_objlist_order, "text_print")
editor_objlist["text_print"] = {
	name = "text_print",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	argtype = {0, 2},
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

table.insert(editor_objlist_order, "text_scrawl")
editor_objlist["text_scrawl"] = {
	name = "text_scrawl",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_verb", "patashu"},
	tiling = -1,
	type = 1,
	argtype = {0, 2},
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Patashu", {3,1} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
{
	name = "slip",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When an object moves onto something that is "SLIP", the object will involuntarily move one step in the direction they're facing, but only once per turn.]]
},
{
	name = "slide",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When an object moves onto something that is "SLIDE", the object will instantly move in the direction they're facing. This can happen multiple times within a single turn as long as the object is on a "SLIDE" object without hitting a wall.]]
},
{
	name = "hates",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[If "X HATES Y", then X cannot move onto any objects of type Y.

- If X is on something that it "HATES", then X tries to move in the direction that it's facing.

- "X HATES LEVEL" causes X to be unable to move.]]
},
{
	name = "likes",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[If "X LIKES Y", then X will only move onto objects of type Y.]]
},
{
	name = "sidekick",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When a "SIDEKICK" object has a different object move sideways to it, the "SIDEKICK" object move along with the moving object.

- The "SIDEKICK" object acts like stop when being pushed against it.]]
},
{
	name = "lazy",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Makes an object unwilling to PUSH, PULL, or SIDEKICK other objects, treating those objects equivalent to "STOP" objects.]]
},
{
	name = "moonwalk",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When an object moves, its velocity gets redirected 180 degrees counterclockwise. The object will turn to face towards the direction it came from.

- Unlike "REVERSE", the "moonwalk" object will change its velocity when pushed or pulled.

- The effects can stack with other directional modifiers.]]
},
{
	name = "drunk",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When an object moves, its velocity gets redirected 90 degrees counterclockwise. The object will turn to face towards the direction it came from.

- Unlike "REVERSE", the "drunk" object will change its velocity when pushed or pulled.

- The effects can stack with other directional modifiers.]]
},
{
	name = "drunker",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When an object moves, its velocity gets redirected 45 degrees counterclockwise, meaning that it can move diagonally. The object will turn to face towards the direction it came from.

- Unlike "REVERSE", the "drunker" object will change its velocity when pushed or pulled.

- The effects can stack with other directional modifiers.]]
},
{
	name = "skip",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[For every space an object moves, the object will move more cells.

- An object that is SKIP once will skip one cell for every cell it moves and will never enter the cells it skips.

- An object that is SKIP several times will skip cells equal to the number of times it is SKIP.]]
},
{
	name = "tall",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [["Tall" objects interact with both objects that are "float" and objects that are not "float".]]
},
{
	name = "oneway",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Makes the object acts as "STOP" only in the direction it is facing. It has no effect in the other directions.]]
},
{
	name = "copy",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Makes an object mimic the same movements of another object.]]
},
{
	name = "reset",
	author = "Patashu",
    group = "Pata Redux Mods",
	description =
[[When a "YOU" object touches a "RESET" object, the level goes back to its starting state as if the level has been restarted.
	
Any NORESET objects retain their position and state after a RESET.]]
},
{
	name = "noundo",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Prevents the object from being affected by changes to the object from undoing.]]
},
{
	name = "noreset",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = 
[[Prevents the effects of "RESET" by retaining position and state of "NORESET" objects.

"NORESET" does not apply if the level is restarted from the pause menu (or through a shortcut button).]]
},
{
	name = "verb_props",
	author = "Patashu",
    group = "Pata Redux Mods",
	display_name = "Verb Properties",
	thumbnail_obj = "text_pushes",
	display_sprites = {"text_pushes", "text_pulls", "text_stops", "text_sinks", "text_defeats", "text_opens", "text_shifts", "text_swaps", "text_melts"},
	description =
[[Verb versions of common properties. Makes an object type have the corresponding property only for another object type.

For instance: if "WATER SINKS BABA", then water can sink baba but not other objects.]],
},
{
	name = "topple",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[**DEPRECATED**. It is discouraged to use this word. Its behavior in its original implementation proved to be inconsistent and janky with a heavy reliance on priority as a core function. 

Original Description: When a stack of topplers are on a tile, they eject themselves in the facing direction to form a line. ]]
},
{
	name = "zoom",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When a "ZOOM" object moves, it keeps moving in the same direction until prevented.]]
},
{
	name = "launch",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Like "SHIFT/SHIFTS", but it sends the target hurtling as far away as possible.
- Happens in the same take as "SHIFT/SHIFTS"]]
},
{
	name = "send",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Whenever a unit steps onto something that is "SEND", it immediately moves again in the "SEND" object's direction. This can happen multiple times within a single turn.]]
},
{
	name = "sticky",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[Objects of the same type that touch each other will stick to each other, moving as if they were one unit.

By default, two objects of the same type have to be on the same float level to stick. This can be disabled in mega modpack settings.]]
},
{
	name = "print",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[The targeted object creates instances of whatever text is the right of "print".
	
Think of as if "make" and "write" were combined into one word.]]
},
{
	name = "scrawl",
	author = "Patashu",
    group = "Pata Redux Mods",
	description = [[When the targeted object is destroyed, it creates instances of whatever text is the right of "scrawl".
	
Think of as if "has" and "write" were combined into one word.]]
},

    })
end


print("End of !init.lua")