plasma_modpack_version = "1.5.16"

br_prefix = "branching_"
br_prefix_len = string.len(br_prefix)
pivot_prefix = "pivot_"
pivot_prefix_len = string.len(pivot_prefix)
pf_filler_text_type = 11

dirfeaturemap = {"right", "up", "left", "down"}

local str = debug.getinfo(1).source:sub(2)
pf_lua_dir = str:match("(.*/)")

arrow_properties = {
    you=true,
    you2=true,
    push=true,
    pull=true,
    swap=true,
    stop=true,
    more=true,
	shift=true,
	select=true,
	boom=true,
    offset=true, --@Merge(offset and nuhuh x plasma) add support for turning offset and nuhuh
    nuhuh=true,
}

turning_word_names = {
    fall=true, 
    nudge=true, 
    locked=true, 
    dir=true, 
    you=true,
    you2=true,
    push=true,
    pull=true,
    swap=true,
    stop=true,
    shift=true,
    more=true,
    select=true,
    boom=true,
    beside=true,
    offset=true, --@Merge(offset and nuhuh x plasma) add support for turning offset and nuhuh
    nuhuh=true,
}

branching_text_names = {
    is = true,
    has = true,
    near = true,
    make = true,
    follow = true,
    mimic = true,
    play = true,
    eat = true,
    fear = true,
    on = true,
    without = true,
    facing = true,
    above = true,
    below = true,
    besideleft = true,
    besideright = true,
    feeling = true,
    ["and"] = true
}

arrow_property_display = {
    youright="you (right)",
    youup="you (up)",
    youleft="you (left)",
    youdown="you (down)",
    you2right="you2 (right)",
    you2up="you2 (up)",
    you2left="you2 (left)",
    you2down="you2 (down)",
    pushright="push (right)",
    pushup="push (up)",
    pushleft="push (left)",
    pushdown="push (down)",
    pullright="pull (right)",
    pullup="pull (up)",
    pullleft="pull (left)",
    pulldown="pull (down)",
    swapright="swap (right)",
    swapup="swap (up)",
    swapleft="swap (left)",
    swapdown="swap (down)",
    stopright="stop (right)",
    stopup="stop (up)",
    stopleft="stop (left)",
    stopdown="stop (down)",
    moreright="more (right)",
    moreup="more (up)",
    moreleft="more (left)",
	moredown="more (down)",
	shiftright="shift (right)",
	shiftup="shift (up)",
	shiftleft="shift (left)",
	shiftdown="shift (down)",
	selectright="select (right)",
	selectup="select (up)",
	selectleft="select (left)",
	selectdown="select (down)",
	boomright="boom (right)",
	boomup="boom (up)",
	boomleft="boom (left)",
	boomdown="boom (down)",
	offsetright="offset (right)", --@Merge(offset and nuhuh x plasma) add support for turning offset and nuhuh
	offsetup="offset (up)",
	offsetleft="offset (left)",
	offsetdown="offset (down)",
}

baba_font_consts = {
    letter_w = 8,
    letter_h = 24,
    letter_spacing = 2,
    total_letter_w = 10,
    button_w = 24,
}

table.insert(objlistdata.alltags, "plasma's mods")
table.insert(objlistdata.alltags, "arrow properties")
table.insert(objlistdata.alltags, "turning text")
table.insert(objlistdata.alltags, "pivot text")
table.insert(objlistdata.alltags, "omni text")
table.insert(objlistdata.alltags, "filler text")
table.insert(objlistdata.alltags, "pointer nouns")