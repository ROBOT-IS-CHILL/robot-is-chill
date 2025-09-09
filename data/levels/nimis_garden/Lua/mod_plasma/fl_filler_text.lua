local new_words = {
    "text_uhh",
    "text_hmm",
    "text_ellipsis",
    "text_so",
    "text_actually",
    "text_really",
    "text_well",
    "text_oh",
    "text_ok",
    "text_yknow",
    "text_like",
    "text_just",
    "text_mmm",
    "text_ah",
}

for i, word in ipairs(new_words) do
    table.insert(editor_objlist_order, word)
end

for i, word in ipairs(new_words) do
    editor_objlist[word] = 
    {
        name = word,
        sprite_in_root = false,
        unittype = "text",
        tags = {"filler text", "plasma's mods", "text", "abstract"},
        tiling = -1,
        type = 11,
        layer = 20,
        colour = {0, 1},
        colour_active = {0, 3},
    }
end

formatobjlist()