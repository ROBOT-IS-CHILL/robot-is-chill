local str = debug.getinfo(1).source:sub(2)
local dir = str:match("(.*/)")

local load_order = {
    "blocks.lua",
	"br_add_to_editor.lua",
	"br_branching_text.lua",
	"conditions.lua",
	"editor_objectlist_func.lua",
	"features.lua",
	"fl_filler_text.lua",
	"gd_guard.lua",
	"glossary.lua",
	"mapcursor.lua",
	"movement.lua",
	"rules.lua",
	"sb_stable.lua",
	"syntax.lua",
	"th_add_to_editor.lua",
	"th_testcond_this.lua",
	"th_text_this.lua",
	"tools.lua",
	"ts_palette_groups.lua",
	"ts_text_splicing.lua",
	"tt_add_to_editor.lua",
	"tt_arrow_prop_functions.lua",
	"tt_turning_text_functions.lua",
	"undo.lua"
}

for _, file in ipairs(load_order) do
    print("[Mega Modpack] Loading "..file)
    dofile(dir..file)
end