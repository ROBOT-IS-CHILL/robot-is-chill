local str = debug.getinfo(1).source:sub(2)
local dir = str:match("(.*/)")

local load_order = {
    "!init.lua",
	"blocks.lua",
	"clears.lua",
	"convert.lua",
	"effects.lua",
	"movement.lua",
	"syntax.lua",
	"tools.lua",
	"undo.lua"
}

for _, file in ipairs(load_order) do
    print("[Mega Modpack] Loading "..file)
    dofile(dir..file)
end