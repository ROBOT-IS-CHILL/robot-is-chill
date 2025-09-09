THROW_PUSH = true

function apply_shrugsimontiger_settings(settings_dict)
	for setting_name, value in pairs(settings_dict) do
		if setting_name == "throw_push" then
			THROW_PUSH = value
		end
	end
end

table.insert(editor_objlist_order, "text_throw")
editor_objlist["text_throw"] = {
	name = "text_throw",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "movement", "shrugsimontiger"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {3, 0},
	colour_active = {3, 1},
}

table.insert(editor_objlist_order, "text_unstack")
editor_objlist["text_unstack"] = {
	name = "text_unstack",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text", "text_quality", "movement", "shrugsimontiger"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {3, 0},
	colour_active = {3, 1},
}

formatobjlist()

if keys.IS_WORD_GLOSSARY_PRESENT then
	keys.WORD_GLOSSARY_FUNCS.register_author("shrugsimontiger", {1, 4})
	keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
		{
			name = "throw",
			author = "shrugsimontiger",
			description =
[[Like Shift, but it shifts things 2 tiles. The tile the shifted object skips over is completely ignored, kind of like Skip. Additionally, if a Throw object tries to push, the pushed object also gets thrown (this part can be disabled in the settings).]]
		}
	})
end

if keys.IS_WORD_GLOSSARY_PRESENT then
	keys.WORD_GLOSSARY_FUNCS.register_author("shrugsimontiger", {1, 4})
	keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
		{
			name = "unstack",
			author = "shrugsimontiger",
			description =
[[If multiple Unstack objects with the same name are stacked, they try their best to unstack themselves without causing any priority jank.

- Ex 1: if Arrow Is Unstack, there are 3 arrows stacked, and they're all facing down, one donesn't move, one moves 1 space, and one moves 2 spaces.

- Ex 2: if Arrow Is Unstack, there are 3 arrows stacked, two are facing right and one is facing down, the down-facing one and one of the right-facing ones move 1 space, and the other right-facing one moves 2 spaces.]]
		}
	})
end