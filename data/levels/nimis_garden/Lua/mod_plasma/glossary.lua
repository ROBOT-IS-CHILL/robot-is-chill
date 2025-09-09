-- Support for the Word Glossary
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("PlasmaFlare", nil, "$1,4Plasma$3,4flare")
    keys.WORD_GLOSSARY_FUNCS.register_custom_text_type(11, "Filler")
    
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
{
    name = "turning_text",
    thumbnail_obj = "text_turning_fall",
    display_name = "turning text",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Variant of various texts that can change their meaning based on the direction they are facing.]],
    display_sprites = {"text_turning_fall","text_turning_nudge","text_turning_dir","text_turning_locked","text_turning_you", "text_turning_you2", "text_turning_push","text_turning_pull","text_turning_swap","text_turning_more","text_turning_stop","text_turning_shift","text_turning_select","text_turning_boom","text_turning_beside"}
},
{
    name = "arrow_you",
    thumbnail_obj = "text_youright",
    display_name = "directional you",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Variant of "YOU" that allows the player to move the object the direction of the arrow. Objects that are directional YOU can still trigger "WIN", and will be destroyed on "DEFEAT" object, like normal "YOU" objects.]],
    display_sprites = {"text_youup", "text_youright", "text_youleft", "text_youdown",}
},
{
    name = "arrow_you2",
    thumbnail_obj = "text_you2right",
    display_name = "directional you2",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Variant of "YOU2" that allows the player to move the object the direction of the arrow. Objects that are directional YOU2 can still trigger "WIN", and will be destroyed on "DEFEAT" object, like normal "YOU2" objects.]],
    display_sprites = {"text_you2up", "text_you2right", "text_you2left", "text_you2down",}
},
{
    name = "arrow_stop",
    thumbnail_obj = "text_stopright",
    display_name = "directional stop",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Makes an object stop other incoming objects in the direction of the indicated arrow.]],
    display_sprites = {"text_stopup", "text_stopright", "text_stopleft", "text_stopdown",}
},
{
    name = "arrow_push",
    thumbnail_obj = "text_pushright",
    display_name = "directional push",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Allows an object to be pushed in the direction of the indicated arrow.]],
    display_sprites = {"text_pushup", "text_pushright", "text_pushleft", "text_pushdown",}
},
{
    name = "arrow_pull",
    thumbnail_obj = "text_pullright",
    display_name = "directional pull",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Allows an object to be pulled in the direction of the indicated arrow.]],
    display_sprites = {"text_pullup", "text_pullright", "text_pullleft", "text_pulldown",}
},
{
    name = "arrow_swap",
    thumbnail_obj = "text_swapright",
    display_name = "directional swap",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Allows an object to move in the direction of the arrow through SWAP-like interactions.]],
    display_sprites = {"text_swapup", "text_swapright", "text_swapleft", "text_swapdown",}
},
{
    name = "arrow_shift",
    thumbnail_obj = "text_shiftright",
    display_name = "directional shift",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[Makes an object able to move other objects in the direction of the arrow.
    
Unlike normal "SHIFT", the direction that the object is facing does not affect the direction of the movement.]],
    display_sprites = {"text_shiftup", "text_shiftright", "text_shiftleft", "text_shiftdown",}
},
{
    name = "arrow_more",
    thumbnail_obj = "text_moreright",
    display_name = "directional more",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Makes an object duplicate itself, creating a new copy adjacent to the original in the direction of the arrow.]],
    display_sprites = {"text_moreup", "text_moreright", "text_moreleft", "text_moredown",}
},
{
    name = "arrow_select",
    thumbnail_obj = "text_selectright",
    display_name = "directional select",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[Allows an object to travel on paths in the direction of the indicated arrow. The object can still select levels to enter like a normal "SELECT" object.]],
    display_sprites = {"text_selectup", "text_selectright", "text_selectleft", "text_selectdown",}
},
{
    name = "arrow_boom",
    thumbnail_obj = "text_boomright",
    display_name = "directional boom",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[Makes an object instantly explode and destroy other objects 1 tile in the direction of the arrow.

- Stacking multiple arrow booms in the same direction will increase the range of explosion.
    - Ex: an object that is 3x boomright will destroy objects within 3 tiles to the right of the boomed object. ]],
    display_sprites = {"text_boomup", "text_boomright", "text_boomleft", "text_boomdown",}
},


{
    name = "omni",
    thumbnail_obj = "text_branching_is",
    display_name = "omni text",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[When parsing reaches an omni text, parsing is split into both horizontal and vertical directions, starting from the omni text.]],
    display_sprites = {"text_branching_is", "text_branching_and", "text_branching_has", "text_branching_near", "text_branching_make", "text_branching_follow", "text_branching_mimic", "text_branching_eat", "text_branching_fear", "text_branching_on", "text_branching_without", "text_branching_facing", "text_branching_above", "text_branching_below", "text_branching_feeling", "text_branching_besideright"}
},
{
    name = "pivot",
    thumbnail_obj = "text_pivot_is",
    display_name = "pivot text",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = [[When parsing reaches a pivot text, the direction of parsing is switched between horizontal and vertical directions, starting from the pivot text.]],
    display_sprites = {"text_pivot_is", "text_pivot_and", "text_pivot_has", "text_pivot_near", "text_pivot_make", "text_pivot_follow", "text_pivot_mimic", "text_pivot_eat", "text_pivot_fear", "text_pivot_on", "text_pivot_without", "text_pivot_facing", "text_pivot_above", "text_pivot_below", "text_pivot_feeling", "text_pivot_besideright"}
},
{
    name = "filler",
    thumbnail_obj = "text_ellipsis",
    display_name = "filler text",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[Random meaningless texts that can extend the length of normal rules. 

Example:
Baba is uhh well hmm ... y'know ... you ]],
    display_sprites = {"text_ellipsis", "text_uhh","text_hmm","text_so","text_actually","text_really","text_well","text_oh","text_ok","text_yknow","text_like","text_just","text_mmm","text_ah"
    }
},
{
    name = "this",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[A pointer noun.
Refers to the closest object in front of the "THIS" text itself.

- When used like "THIS IS X", the specific object refered by the pointer noun will have X applied.

- When used like "X IS THIS", the pointer noun will interpret what it points to as a property or noun to transform into.

- When used like "X on THIS is Y", the infix condition will use the specific object refered by the pointer noun as a parameter.]],
},
{
    name = "that",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[A pointer noun.
Refers to the farthest object in front of the "THAT" text itself.

- When used like "THAT IS X", the specific object refered by the pointer noun will have X applied.

- When used like "X IS THAT", the pointer noun will interpret what it points to as a property or noun to transform into.

- When used like "X on THAT is Y", the infix condition will use the specific object refered by the pointer noun as a parameter.]],
},
{
    name = "these",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[A pointer noun.
Refers to all objects between 2 different "THESE" texts that are facing each other. 

- Only looks at the first "THESE" in front of it for validity.

- When used like "THESE IS X", all individual objects refered by the pointer noun will have X applied.

- When used like "X IS THESE", the pointer noun will interpret what it points to as properties and/or nouns to transform into.

- When used like "X on THESE is Y", the infix condition will use all individual objects refered by the pointer noun as a parameter.]],
},
{
    name = "those",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[A pointer noun.
Refers to all objects connected adjacently from the object directly in front of the "THOSE" text.

- When used like "THOSE IS X", all individual objects refered by the pointer noun will have X applied.

- When used like "X IS THOSE", the pointer noun will interpret what it points to as properties and/or nouns to transform into.

- When used like "X on THOSE is Y", the infix condition will use all individual objects refered by the pointer noun as a parameter.]],
},
{
    name = "block",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[Limits which objects can be referred by pointer nouns (THIS, THAT, THESE, THOSE).

While a pointer noun selects a "BLOCK" object, all targeted objects won't be affected by rules formed with the pointer noun.]],
},
{
    name = "pass",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[Limits which objects can be referred by pointer nouns (THIS, THAT, THESE, THOSE).

"PASS" objects will be ignored by pointer nouns as if the "PASS" objects are empty space.

- As a consequence, "EMPTY" is inheritly "PASS". It is similar to how all texts are PUSH]],
},
{
    name = "relay",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[Limits which objects can be referred by pointer nouns (THIS, THAT, THESE, THOSE).

A "RELAY" object can redirect the targeting of pointer nouns in the direction the "RELAY" object is facing.

- "RELAY" objects do not get selected by pointer nouns]],
},
{
    name = "cut",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[Gives an object the ability to split a text block into individual letters.

- The effect happens when the "CUT" object walks into a text block.

- Letters cannot be cut.

- When a text block is cut, its letters are extracted out in the direction of the cut. Letter extraction can stop early at the first solid object encountered.

- If a text block gets cut from occupying the same space as a "CUT" object, letters are extracted in the direction the text block is facing.
]],
},
{
    name = "pack",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description = 
[[Gives an object the ability to convert a line of letters into a valid text block.
- The effect can be done when the "PACK" object walks into a line of letters.

- You can only PACK letters, not regular text blocks.

- The line of letters must be arranged to spell out a valid word in the the "PACK" object's direction.
    - Ex: If the player wants to form "SINK" by packing leftward, the letters should be arranged like "KNIS".
]],
},
{
    name = "stable",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[The moment an object becomes STABLE, all rules applied to the object get preserved. While an object is STABLE, the object cannot gain or lose any preserved rules from manipulating text blocks. Only when the object stops becoming STABLE does it lose its preserved rules.

- Hover your mouse over a STABLE object to view its preserved rules.]],
},
{
    name = "guard",
    author = "PlasmaFlare",
    group = "Plasma's Mods",
    description =
[[Allows objects to sacrifice themselves in order to save another object from being destroyed.

Ex: If "keke GUARD baba" is formed and a baba walks into a "SINK" object, all kekes get destroyed instead of baba. If there are no kekes, baba will be destroyed as normal.

- "GUARD" can be chained (Ex: "X guard Y guard Z" means that if X is about to get destroyed, Z will get destroyed instead). This can be disabled in the modpack settings.]],
},

})
end