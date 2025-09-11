local str = debug.getinfo(1).source:sub(2)
local dir = str:match("(.*/)")

local load_order = {
    "!add_words.lua",
	"!global_vars.lua",
	"!settings.lua",
	"!utils.lua",
	"blocks.lua",
	"clears.lua",
	"conditions.lua",
	"convert.lua",
	"effects.lua",
	"letterunits.lua",
	"load.lua",
	"map.lua",
	"mapcursor.lua",
	"movement.lua",
	"rules.lua",
	"syntax.lua",
	"tools.lua",
	"undo.lua",
	"update.lua"
}

for _, file in ipairs(load_order) do
    print("[Mega Modpack] Loading "..file)
    dofile(dir..file)
end