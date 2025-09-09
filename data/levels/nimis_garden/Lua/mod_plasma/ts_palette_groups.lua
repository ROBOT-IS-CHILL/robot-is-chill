local editor_objlist_pack_groups = {}
local editor_objlist_multi_pairing_indexes = {}
local editor_objlist_pack_group_map = {}

function initialize_palette_groups()
    editor_objlist_pack_groups = {
        {"text_besideright", "text_besideleft"},
        {"text_fallright", "text_fallup", "text_fallleft", "text_fall"},
        {"text_nudgeright", "text_nudgeup", "text_nudgeleft", "text_nudgedown"},
        {"text_lockedright", "text_lockedup", "text_lockedleft", "text_lockeddown"},
    }
    editor_objlist_pack_group_map["text_turning_fall"]   = 2
    editor_objlist_pack_group_map["text_turning_nudge"]  = 3
    editor_objlist_pack_group_map["text_turning_locked"] = 4

    for arrow_prop,_ in pairs(arrow_properties) do
        local arrow_prop_text = "text_"..arrow_prop
        table.insert(editor_objlist_pack_groups, {
            arrow_prop_text.."right", arrow_prop_text.."up", arrow_prop_text.."left", arrow_prop_text.."down"
        })

        if turning_word_names[arrow_prop] then
            editor_objlist_pack_group_map["text_turning_"..arrow_prop] = #editor_objlist_pack_groups
        end
        
    end

    -- Turning dir case
    table.insert(editor_objlist_pack_groups, {
        "text_right", "text_up", "text_left", "text_down"
    })
    editor_objlist_pack_group_map["text_turning_dir"] = #editor_objlist_pack_groups
    
    table.insert(editor_objlist_pack_groups, {
        "text_besideright", "text_above", "text_besideleft", "text_below"
    })
    editor_objlist_pack_group_map["text_turning_beside"] = #editor_objlist_pack_groups


    local multi_pair_texts = {"text_cut"}
    for _, name in ipairs(multi_pair_texts) do
        editor_objlist_multi_pairing_indexes[name] = {}
    end

    -- Store indexes of letterunits in editor_objectlist so that we can reference them faster
    for i, v in pairs(editor_objlist) do
        if v.unittype == "text" and string.sub(v.name, 1, 5) == "text_" then
            local textname = string.sub(v.name, 6)
            if v.type == 5 and textname ~= "sharp" and textname ~= "flat" then
                table.insert(editor_objlist_multi_pairing_indexes["text_cut"], v.name)
            end
        end
    end

    for i, pack_group in ipairs(editor_objlist_pack_groups) do
        for _, object in ipairs(pack_group) do
            editor_objlist_pack_group_map[object] = i
        end
    end
end
initialize_palette_groups()


local function is_object_in_editor_palette(checkname)
    for i,v in ipairs(editor_currobjlist) do
        if (v.name == checkname) then
            return true
        end
    end
    return false
end

local function add_object_to_editor_palatte(objname)
    local index = editor_objlist_reference[objname]
    if index == nil then
        return false
    end
    if (#editor_currobjlist >= 150) then
        return false
    end
    editor_currobjlist_add(index,false,nil,nil,nil,false)
    return true
end

function add_cut_or_pack_palette_groups(editor_currobjlist, data)
    if editor_objlist_multi_pairing_indexes[data.name] then
        for _, objname in ipairs(editor_objlist_multi_pairing_indexes[data.name]) do
            if add_object_to_editor_palatte(objname) == false then
                return
            end
        end
    end

    if editor_objlist_pack_group_map[data.name] then
        if is_object_in_editor_palette("text_pack") then
            local pack_group = editor_objlist_pack_groups[editor_objlist_pack_group_map[data.name]]
            for _, object in ipairs(pack_group) do
                if add_object_to_editor_palatte(object) == false then
                    return
                end
            end
        end
    elseif data.name == "text_pack" then
        for object_key, pack_group_index in pairs(editor_objlist_pack_group_map) do
            if is_object_in_editor_palette(object_key) then
                local pack_group = editor_objlist_pack_groups[pack_group_index]
                for _, object in ipairs(pack_group) do
                    if add_object_to_editor_palatte(object) == false then
                        return
                    end
                end
            end
        end
    end
end