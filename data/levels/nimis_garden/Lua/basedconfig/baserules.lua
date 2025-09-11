-- These are some extra configurations
local mod_config = {
    -- Displays a lua error message on level start, showing which sentences in baserules.lua were invalid.
    -- You can disable this if you want to avoid all the excess error messages. But invalid sentences will still be ignored.
    report_invalid_sentences = true,

    -- When set to true, this disables the game from automatically inserting "text is push", "level is stop", and "cursor is select" as baserules.
    -- Note that this affects the levelpack as a whole, and cannot be configured per level. If you want some levels to have the above rules,
    -- configure "level_baserules" at the bottom of this file.
    disable_normal_baserules = false,
}

--[[ 
    persist_baserules are always applied in every level in the levelpack. Just put in your list of sentences below.
    Example:
    local persist_baserules = {
        "baba is keke",
        "keke make me",
        "me is blue and pink",
    }

    NOTE: There is a bug where adding group-related rules as persist baserule doesn't work without having at least one text object in the level.
    The fix would require some lua function overriding in rules.lua, which I don't want to do for compatability reasons. There might be other rules that also have this
    problem. So to be on the safe side, make sure you have at least one text object in you level.
 ]]
local persist_baserules = {
    
}



--[[
    levelpack_baserules are just like persist_baserules where every sentence is applied to every level in the levelpack.
    However, each "set" of sentences is applied only when the game detects the rule "level is X", where X can be any string you want.

    The format for each entry in levelpack_baserules is:
        <text block name> = {
            <sentence 1>,
            <sentence 2>,
            <sentence 3>,
            ...
        }
    Where <text block name> does not have "text_" prepended ("push", "shift", "ice" etc).

    Note: Only full rules are allowed. So no "baba is keke is push", where it would've been parsed as two sentences in game.

    The below example will apply "baba is sleep and pet" and "level is pink" when "level is lovebaba" is formed. A similar thing happens when you form "level is poem"

        local levelpack_baserules = {
            lovebaba = {
                "baba is sleep and pet",
                "all near baba is love",
                "level is pink",
            },
            poem = {
                "rose is red",
                "violet is blue",
                "flag is win",
                "baba is you",
            },
        }
 ]]
local levelpack_baserules = {
}




--[[ 
    level baserules are baserules that only apply to specific levels in your levelpack.
    The format for each entry is:
        [<level name>] = {
            <text block name> = {
                <sentence 1>,
                <sentence 2>,
                <sentence 3>,
                ...
            }
        }
    Where <level name> is CASE-SENSITIVE and can be EITHER: 
        - the name of the level ingame (Ex: "the return of scenic pond", "skull house", "prison")
        - the name of the .ld file (excluding ".ld")
    The rest of the format is exactly the same as global baserules.
    
    Note: level baserules will be used instead of global baserules when playing a level that you specified in the variable below.

    The below example applies "baba is green" when "level is baserule1" is formed when playing a level named "woah". 
    It also applies "keke is purple" when "level is baserule1" is formed when playing a level whose .ld file is "23level.ld". 

        local level_baserules = {
            ["woah"] = {
                baserule1 = {
                    "baba is green"
                }
            },
            ["23level"] = {
                baserule1 = {
                    "keke is purple"
                }
            }
        }
 ]]
local level_baserules = {

        ["seafloor shipwreck"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "fence is group",
                "line is group",
                "bog is group",
                "text on cliff is hide",
                "group is float"
            }
        },

        ["midnight snack"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "fence is group",
                "line is group",
                "bog is group",
                "fort is group",
                "grass is hide",
                "text on grass is hide",
                "group is float"
            }
        },

        ["desired turns"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "group is group",
                "fence is group",
                "line is group",
                "bog is group",
                "group is still",
                "text on spike is hide",
                "spike is hide",
                "group is float"
            }
        },

        ["magician's scramble"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "fence is group",
                "line is group",
                "bog is group",
                "brick is group",
                "group is still",
                "group is group",
                "group is pass",
                "text on grass is hide",
                "group is float"
            }
        },

        ["prickly weeds"] = {
            cluebaserule = {
                "group2 not feeling reveal is hide",
                "fence is group2",
                "line is group2",
                "grass is group2",
                "fort is group2",
                "text on bucket is hide",
                "bucket is hide",
                "group2 is float"
            }
        },

        ["le chateau"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "bog is group",
                "line is group",
                "hedge is group",
                "brick is group",
                "group is still",
                "text on fire is hide",
                "fire is hide",
                "group is float"
            }
        },

        ["hedge hopping"] = {
            cluebaserule = {
                "group2 not feeling reveal is hide",
                "bog is group2",
                "line is group2",
                "brick is group2",
                "group2 is still",
                "text on flower is hide",
                "flower is hide",
                "group2 is float"
            }
        },

        ["build-a-bridge"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "cliff is group",
                "line is group",
                "fence is group",
                "group is still",
                "text on book is hide",
                "book is hide",
                "group is float"
            }
        },

        ["force through"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "bog is group",
                "line is group",
                "fence is group",
                "blob is group",
                "text on brick is hide",
                "brick is hide",
                "group is float"
            }
        },

        ["mining with explosives"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "grass is group",
                "line is group",
                "ice is group",
                "hedge is group",
                "group is still",
                "group is safe",
                "group is group",
                "text on lava is hide",
                "group is float"
            }
        },

        ["chutes"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "line is group",
                "fence is group",
                "cloud is group",
                "group is still",
                "text on cliff is hide",
                "group is float"
            }
        },

        ["taken off the list"] = {
            cluebaserule = {
                "group not feeling reveal is hide",
                "line is group",
                "bog is group",
                "brick is group",
                "group is still",
                "group is float",
                "text on bed is hide",
                "bed is hide",
            }
        },

        ["where it all began"] = {
            gardenbaserule = {
                "bog is donut",
                "banana is bucket",
                "arrow is book",
                "bee is bomb",
                "arm is burger",

                "bog is group5",
                "banana is group5",
                "arrow is group5",
                "bee is group5",
                "arm is group5",

                "group4 is not group5",
                "group5 is group5",
                "group5 is hide",

                "not lonely angler is group",
                "not lonely fairy is group",
                "not lonely amoeba is group",
                "not lonely magic is group",
                "not lonely cactus is group",
                "not lonely fruit is group",
                "not lonely fox is group",
                "not lonely armour is group",
                "not lonely plane is group",
                "not lonely rocket is group",
                "not lonely what is group",
                "not lonely pickaxe is group",
                
                "group is black",
                "group is best",

                "me not without group is group3",
                "group3 not without group2 is box",
                "box is move",
                "box is me",
                "idle box is still",
                "idle box is box",
                "group3 without group2 is empty"

            }
        }
}

-- Ignore this last part. It's needed to load all the baserules into the mod
return mod_config, levelpack_baserules, level_baserules, persist_baserules