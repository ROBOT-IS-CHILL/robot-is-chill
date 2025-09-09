local CutPackWordVerify = {}

local function get_special_cut_mappings()
    local cut_mappings = {
        fallright =      "fall",
        fallleft =       "fall",
        fallup =         "fall",
        falldown =       "fall",
        lockedright =    "locked",
        lockedleft =     "locked",
        lockedup =       "locked",
        lockeddown =     "locked",
        nudgeleft =      "nudge",
        nudgeup =        "nudge",
        nudgeright =     "nudge",
        nudgedown =      "nudge",
        ellipsis =       "ooo",    
        besideright =    "beside",    
        besideleft =     "beside",    
    }

    -- Arrow properties
    for arrow_prop,_ in pairs(arrow_properties) do
        cut_mappings[arrow_prop.."right"] = arrow_prop
        cut_mappings[arrow_prop.."left"] = arrow_prop
        cut_mappings[arrow_prop.."up"] = arrow_prop
        cut_mappings[arrow_prop.."down"] = arrow_prop
    end
    -- Turning text
    for turning_prop, _ in pairs(turning_word_names) do
        if turning_prop ~= "dir" and turning_prop ~= "beside" then
            cut_mappings["turning_"..turning_prop] = turning_prop
        end
    end
    -- Omni text
    for branching_text, _ in pairs(branching_text_names) do
        cut_mappings[br_prefix..branching_text] = branching_text
        cut_mappings[pivot_prefix..branching_text] = branching_text
    end

    return cut_mappings
end

local function get_special_pack_mappings()
    local pack_mappings = {
        ooo = "ellipsis"
    }

    for turning_prop, _ in pairs(turning_word_names) do
        pack_mappings["turning"..turning_prop] = "turning_"..turning_prop
    end
    for branching_text, _ in pairs(branching_text_names) do
        pack_mappings["omni"..branching_text] = br_prefix..branching_text
        pack_mappings["pivot"..branching_text] = pivot_prefix..branching_text
    end

    return pack_mappings
end


local function get_special_pack_directional_mappings()
    local dir_pack_mappings = {
        fall = {
            [0] = "fallright",
            [1] = "fallup",
            [2] = "fallleft",
            [3] = "fall",
        },
        locked = {
            [0] = "lockedright",
            [1] = "lockedup",
            [2] = "lockedleft",
            [3] = "lockeddown",
        },
        nudge = {
            [0] = "nudgeright",
            [1] = "nudgeup",
            [2] = "nudgeleft",
            [3] = "nudgedown",
        },
        beside = {
            [0] = "besideright",
            [2] = "besideleft",
        },
    }
    for arrow_prop,_ in pairs(arrow_properties) do
        dir_pack_mappings[arrow_prop] = {
            [0] = arrow_prop.."right",
            [1] = arrow_prop.."up",
            [2] = arrow_prop.."left",
            [3] = arrow_prop.."down",
        }
    end

    return dir_pack_mappings
end

local function get_valid_characters()
    local valid_chars = {}
    for text_name, object in pairs(objectpalette) do
        local type = getactualdata_objlist(object, "type")
        local unittype = getactualdata_objlist(object, "unittype")
        
        if type == 5 and unittype == "text" then
            if string.sub(text_name, 1, 5) == "text_" then
                local character = string.sub(text_name, 6)
                if string.sub(character, 1, 5) ~= "text_" then -- Prevent metatext letters from being counted
                    valid_chars[character] = true
                end
            end
        end
    end
    return valid_chars
end


-- Local variables
local valid_characters = {}
local special_cut_mappings = {}
local special_pack_mappings = {}
local special_pack_directional_mappings = {}
local directional_packs_without_normal_variants = {
    fall=true,
    nudge=true,
    locked=true,
    beside=true,
}

local function initialize_word_verify()
    valid_characters = get_valid_characters()
    special_cut_mappings = get_special_cut_mappings()
    special_pack_mappings = get_special_pack_mappings()
    special_pack_directional_mappings = get_special_pack_directional_mappings()
end


-- Mod hook inserts
table.insert(mod_hook_functions["level_start"],
    function()
        initialize_word_verify()
    end
)


--[[
    Given a name of a text that will be cut, return the output text that will be produced. Normally this will be the text name
    itself. But special cases are defined in special_cut_mappings.
 ]]
function CutPackWordVerify:get_cut_text(name, text_dir, cut_direction)
    -- Note: dir is currently not used, but keeping it here just in case I want the cutting to depend on direction

    local t = special_cut_mappings[name]
    if t then return t end

    -- THIS
    local pointer_noun = is_name_text_this(name, false)
    if pointer_noun then
        return pointer_noun
    end

    -- Turning dir
    if name == "turning_dir" then
        if text_dir == 0 then return "right"
        elseif text_dir == 1 then return "up"
        elseif text_dir == 2 then return "left"
        elseif text_dir == 3 then return "down"
        end
    elseif name == "turning_beside" then
        if text_dir == 0 then return "beside"
        elseif text_dir == 1 then return "above"
        elseif text_dir == 2 then return "beside"
        elseif text_dir == 3 then return "below"
        end
    end

    for c in name:gmatch"." do
        if not valid_characters[c] or not unitreference["text_"..c] then
            return nil
        end
    end

    return name
end


--[[
    Given a sequence of letters that will be packed as a single string, return the name of the potential text block the packing might
    produce. Normally this will be the string itself, but special cases are defined in special_pack_mappings and
    special_pack_directional_mappings. Supports special mappings when packed at certain directions.

    Note that the output is not guaranteed to be in the palette.
 ]]
function CutPackWordVerify:get_pack_text(name, dir)
    -- Note: dir is currently not used, but keeping it here just in case I want the cutting to depend on direction

    --[[
        Special case priorities:
        1) If there's a normal special mapping for a text name, return that mapping
        2) If there's a directional special mapping:
            - If the normal variant is in the palette, return the normal variant
            - Otherwise return the directional variant
        3) Return the literal text name, regardless of if it is in the palette
     ]]

    if special_pack_mappings[name] then
        return special_pack_mappings[name]
    elseif special_pack_directional_mappings[name] then
        local dir_pack_mapped_text = special_pack_directional_mappings[name][dir]

        if not directional_packs_without_normal_variants[name] and is_text_in_palette(name) then
            -- If there is a normal variant of the directional variant, Normal variants take precedence over directional variants.
            -- Ex: if "shift" and "shiftup" are in the palette, packing SHIFT upwards will yield "shift"
            return name
        end

        if not is_text_in_palette(dir_pack_mapped_text) then
            dir_pack_mapped_text = nil

            -- This entire loop determines if there is only one directional variant defined in the palette:
            --  - if there is only one defined, return that variant
            --  - otherwise, return the literal name, which resolves to not packing the text
            for testdir = 0,3 do
                local test_mapped_text = special_pack_directional_mappings[name][testdir]
                if is_text_in_palette(test_mapped_text) then
                    if dir_pack_mapped_text == nil then
                        dir_pack_mapped_text = test_mapped_text
                    else
                        return name
                    end
                end
            end
            return dir_pack_mapped_text
        else
            return dir_pack_mapped_text
        end
    end

    -- Note. I'm going to leave "THIS" to the technical names, so text_this2323 requires "THIS2323" to pack. There's too many possibilities for "THIS<some string>".
    -- I would have to scan through the entire palette for a single THIS text. I don't want to be *that* thorough.

    return name
end

return CutPackWordVerify