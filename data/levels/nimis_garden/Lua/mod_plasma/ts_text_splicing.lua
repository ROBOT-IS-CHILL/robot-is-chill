table.insert(editor_objlist_order, "text_cut")

editor_objlist["text_cut"] = 
{
	name = "text_cut",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {5, 2},
    colour_active = {5, 4},
}
table.insert(editor_objlist_order, "text_pack")

editor_objlist["text_pack"] = 
{
	name = "text_pack",
	sprite_in_root = false,
	unittype = "text",
	tags = {"plasma's mods", "text", "abstract"},
	tiling = -1,
	type = 2,
	layer = 20,
	colour = {2, 2},
    colour_active = {2, 3},
}

formatobjlist()

local utils = PlasmaModules.load_module("general/utils")
local cut_pack_word_verify = PlasmaModules.load_module("cutpack/word_verify")
-- Global variables

splice_mod_globals = {}
local function reset_splice_mod_globals()
    splice_mod_globals = {
        -- This prevents packing via pulling
        calling_push_check_on_pull = false, 
        record_packed_text = false, -- Flag for indicating when to update processed_pack_texts
    }
end
reset_splice_mod_globals()


-- Local variables
local cut_texts = {} -- a record of all texts that were destroyed via cut when we call handle_cut_text
local pack_texts = {} -- Keeps track of which texts have already been packed. This is used to prevent letter duplication via packing
local exclude_from_cut_blocking = {} -- list of unit ids that are excluded from checking its solidity when creating the letter units after a cut
-- flag for indicating inside check() and therefore inside check_text_packing() if we are calling check() when we are handling pull.


-- Mod hook inserts
table.insert(mod_hook_functions["command_given"], 
    function()
        exclude_from_cut_blocking = {}
        cut_texts = {}
        processed_pack_texts = {}
    end
)

-- Note: commented out since delete() has a parameter to disable has. But maybe there's a reason we still need this
-- local function delete_without_triggering_has(unitid)
--     if unitid ~= 2 then
--         local unit = mmf.newObject(unitid)
--         -- All of this to delete the cut text without triggering HAS
--         addundo({"remove",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init},unitid)
--         delunit(unit.fixed)
--         MF_remove(unit.fixed)
--     end
-- end

function is_text_in_palette(textname)
    if textname == nil then
        return false
    end

    -- Due to weird legacy systems of object indexing, we have to check if the current
    -- packed text name's unit reference (i.e "object034") refers to the actual text object
    local realname = unitreference["text_"..textname]
    if realname == nil then
        return false
    end
    local dname = getactualdata_objlist(realname,"name")

    return dname == "text_"..textname 
end

function reset_splice_mod_vars_per_take()
    exclude_from_cut_blocking = {}
    cut_texts = {}
    pack_excluded_texts = {}
    processed_pack_texts = {}

    splice_mod_globals.calling_push_check_on_pull = false
    splice_mod_globals.record_packed_text = false
end

function add_moving_units_to_exclude_from_cut_blocking(moving_units)
    for i,data in ipairs(moving_units) do
        exclude_from_cut_blocking[data.unitid] = true
    end
end

function check_text_cutting(cutterunitid, textunitid, pulling, cutter_pushed_against, x, y, levelcut)
    if textunitid == 2 then
        return false
    end

    local textunit = mmf.newObject(textunitid)
    if pulling or textunit.strings[UNITTYPE] ~= "text" or (textunit.values[TYPE] == 5 and #textunit.strings[NAME] == 1) then
        return false
    end

    if not x or not y then
        x = textunit.values[XPOS]
        y = textunit.values[YPOS]
    end

    if issafe(textunit.fixed) then
        return false
    end

    if levelcut then
        if not floating_level(textunitid) then
            return false
        end
    else
        if not floating(cutterunitid,textunitid,x,y) then
            return false
        end
    end

    -- check that all characters in the text are valid
    local name = textunit.strings[NAME]
    if not cut_pack_word_verify:get_cut_text(name, textunit.values[DIR]) then 
        return false 
    end

    exclude_from_cut_blocking[textunitid] = true
    pack_excluded_texts[textunitid] = true
    local data = {
        cut_text = textunitid,
        cutter_pushed_against = cutter_pushed_against,
        cutterunitid = cutterunitid,
        cut_text_start_x = x,
        cut_text_start_y = y,
    }
    return data
end

function handle_text_cutting(data, cut_direction)
    utils.debug_assert(data.cut_text > 2)

    -- This is to prevent stacked cut objects cutting the same text
    if cut_texts[data.cut_text] then
        return
    end

    local cut_text_unit = mmf.newObject(data.cut_text)
    local bname = cut_text_unit.strings[NAME]
    local leveldata = {cut_text_unit.strings[U_LEVELFILE],cut_text_unit.strings[U_LEVELNAME],cut_text_unit.flags[MAPLEVEL],cut_text_unit.values[VISUALLEVEL],cut_text_unit.values[VISUALSTYLE],cut_text_unit.values[COMPLETED],cut_text_unit.strings[COLOUR],cut_text_unit.strings[CLEARCOLOUR]}
    local x = cut_text_unit.values[XPOS]
    local y = cut_text_unit.values[YPOS]

    if data.cutter_pushed_against then
        if data.cutterunitid == 2 then
            x = data.cut_text_start_x
            y = data.cut_text_start_y
        elseif data.cutterunitid ~= -1 then
            local cutterunit = mmf.newObject(data.cutterunitid)
            x = cutterunit.values[XPOS]
            y = cutterunit.values[YPOS]
        end
    end

    local dirvec = dirs[cut_direction+1]
    local ox = dirvec[1]
    local oy = dirvec[2]

    local outstr = cut_pack_word_verify:get_cut_text(bname, cut_text_unit.values[DIR], cut_direction)
    if outstr then
        if cut_direction == 1 or cut_direction == 2 then
            outstr = outstr:reverse()
        end

        arrow_prop_mod_globals.group_arrow_properties = false

        for c in outstr:gmatch"." do
            local obs = findobstacle(x+ox,y-oy)
            local valid = true
            if (#obs > 0) then
                for a,b in ipairs(obs) do
                    if not exclude_from_cut_blocking[b] then
                        if (b == -1) then
                            valid = false
                        elseif (b ~= 0) and (b ~= -1) then
                            local bunit = mmf.newObject(b)
                            local obsname = getname(bunit)
                            
                            local obsstop = hasfeature(obsname,"is","stop",b,x+ox,y+oy)
                            local obspush = hasfeature(obsname,"is","push",b,x+ox,y+oy)
                            local obspull = hasfeature(obsname,"is","pull",b,x+ox,y+oy)

                            obsstop, obspush, obspull = do_directional_collision(cut_direction, obsname, b, obsstop, obspush, obspull, x, y, ox, oy, false)
                            
                            if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) then
                                valid = false
                                break
                            end
                        end
                    end
                end
            else
                local obsstop = hasfeature("empty","is","stop",2,x+ox,y+oy)
                local obspush = hasfeature("empty","is","push",2,x+ox,y+oy)
                local obspull = hasfeature("empty","is","pull",2,x+ox,y+oy)

                obsstop, obspush, obspull = do_directional_collision(cut_direction, "empty", b, obsstop, obspush, obspull, x, y, ox, oy, false)
                
                if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) then
                    valid = false
                end
            end

            if valid then
                -- objectlist[c] = 1 -- Commented out due to error when "X is all". But is there a need to add the lettertext to the objectlist
                local newunitid = create("text_"..c, x + ox, y - oy, cut_direction, x, y, nil, nil, leveldata)

                exclude_from_cut_blocking[newunitid] = true
                ox = ox + dirvec[1]
                oy = oy + dirvec[2]
            else
                break
            end
        end

        arrow_prop_mod_globals.group_arrow_properties = true

        local cutterunit_name = ""
        if data.cutterunitid == 1 then
            cutterunit_name = "level"
        elseif data.cutterunitid == 2 then
            cutterunit_name = "empty"
        else
            local cutterunit = mmf.newObject(data.cutterunitid)
            cutterunit_name = getname(cutterunit)
        end
        if hasfeature(cutterunit_name, "is", "weak") then
            if cutterunit_name == "level" then
                destroylevel()
            else
                delete(data.cutterunitid)
            end
        end
        
        local pmult,sound = checkeffecthistory("cut")
        MF_particles("destroy",x,y,8 * pmult,0,3,1,1)
        generaldata.values[SHAKE] = 3

        -- deleting without triggering has and ignoring guard @mods(guard)(param)
        GLOBAL_disable_guard_checking = true
        delete(data.cut_text, nil, nil, nil, true, true)
        GLOBAL_disable_guard_checking = false

        cut_texts[data.cut_text] = true
        setsoundname("removal",1,sound)
    end
end

function handle_level_cutting()
    exclude_from_cut_blocking = {}
    local cut_textunits = {} 
    for a,unitid in ipairs(codeunits) do
        local data = check_text_cutting(1, unitid, false, false, nil, nil, true)
        if data then
            table.insert(cut_textunits, data)
        end
    end
    for _, cut_entry in ipairs(cut_textunits) do
        local textunit = mmf.newObject(cut_entry.cut_text)
        handle_text_cutting(cut_entry, textunit.values[DIR])
    end
    exclude_from_cut_blocking = {}
end

local function text_packing_get_letter(unitid, x, y, dir, packer_pushed_against)
    local unit = mmf.newObject(unitid)
    local collisions, obstacle_list, specials = check(unitid, x, y, dir, false, "pack")
    local letterunitid = nil
    local valid = true
    for _, obs in ipairs(collisions) do
        if obs == 1 or obs == -1 then
            valid = false
            break
        end
        if (obs ~= 2 and obs ~= 0) then
            local obsunit = mmf.newObject(obs)
            if obsunit.strings[UNITTYPE] == "text" and obsunit.values[TYPE] == 5 then
                if not letterunitid then
                    letterunitid = obs
                else
                    valid = false
                    break
                end
            else
                valid = false
                break
            end
        end
    end
    if not valid or not letterunitid then
        return false
    end
    if processed_pack_texts[letterunitid] then
        return false
    end

    return letterunitid
end

function check_text_packing(packerunitid, textunitid, dir, pulling, packer_pushed_against, packer_x, packer_y)
    if textunitid == 2 then
        return false
    end
    if splice_mod_globals.calling_push_check_on_pull then
        return false
    end
    if issafe(textunitid) then
        return false
    end

    local textunit = mmf.newObject(textunitid)
    if pulling or textunit.strings[UNITTYPE] ~= "text" or textunit.values[TYPE] ~= 5 then
        return false
    end

    local reverse = dir == 1 or dir == 2
    local dirvec = dirs[dir+1]
    
    local x = nil
    local y = nil
    local check_unitid = nil
    if packerunitid == -1 then
        x = textunit.values[XPOS]
        y = textunit.values[YPOS]
        check_unitid = textunitid
    else
        if packerunitid == 2 then
            x = packer_x
            y = packer_y
        else
            local packerunit = mmf.newObject(packerunitid)
            if packerunit.strings[UNITTYPE] == "text" and packerunit.values[TYPE] == 5 then
                -- NOTE: disable any letterunits from packing for now. Actually making this work seems like a lot of wrangling with the movement system
                return false
            end
            x = packerunit.values[XPOS]
            y = packerunit.values[YPOS]
        end
        check_unitid = packerunitid
    end

    local ox = 0
    local oy = 0
    local letterunits = {}
    local found_letters = ""
    local packed_text_pos = {textunit.values[XPOS],textunit.values[YPOS]}

    if not packer_pushed_against then
        packed_text_pos[1] = packed_text_pos[1] + dirvec[1]
        packed_text_pos[2] = packed_text_pos[2] - dirvec[2]
    end

    local letterwidths = {}
    local processed_first_level_pack = false
    while true do
        local letterunitid = text_packing_get_letter(check_unitid, x+ox, y-oy, dir, packer_pushed_against)
        if not letterunitid then
            break
        end
        if pack_excluded_texts[letterunitid] then
            break
        end
        if issafe(letterunitid) or not floating(letterunitid, check_unitid) then
            break
        end

        local letterunit = mmf.newObject(letterunitid)

        if packerunitid == -1 and not processed_first_level_pack then
            processed_first_level_pack = true
            local rx, ry = dirvec[1] * -1, dirvec[2] * -1
            local first_letterunitid = text_packing_get_letter(letterunitid, letterunit.values[XPOS], letterunit.values[YPOS], rotate(dir), packer_pushed_against)
            if not first_letterunitid then
                return false
            end
            if pack_excluded_texts[first_letterunitid] then
                return false
            end
            if not floating_level(first_letterunitid) then
                return false
            end

            local first_letterunit = mmf.newObject(first_letterunitid)
            found_letters = found_letters..first_letterunit.strings[NAME]
            table.insert(letterunits, first_letterunitid)
            table.insert(letterwidths, #first_letterunit.strings[NAME])
        end

        if reverse then
            found_letters = letterunit.strings[NAME]..found_letters
            table.insert(letterunits, 1, letterunitid)
            table.insert(letterwidths, 1, #letterunit.strings[NAME])
        else
            found_letters = found_letters..letterunit.strings[NAME]
            table.insert(letterunits, letterunitid)
            table.insert(letterwidths, #letterunit.strings[NAME])
        end

        check_unitid = letterunitid
        ox = ox + dirvec[1]
        oy = oy + dirvec[2]
    end

    if #letterunits == 0 then
        return false
    end

    local length = #found_letters
    local packed_text_name = ""
    for i=1,length do
        packed_text_name = cut_pack_word_verify:get_pack_text(found_letters, dir)
        if #found_letters > 1 and is_text_in_palette(packed_text_name)then
            -- Commented out due to error when "X is all". This isn't the direct cause, but might cause other cases. But is there a need to add this to the objectlist
            -- objectlist["text_"..packed_text_name] = 1
            break
        end
        
        if reverse then
            if #letterwidths > 0 then
                for c=1,letterwidths[1] do
                    found_letters = found_letters:sub(2)
                end
            end
            table.remove(letterunits, 1)
            table.remove(letterwidths, 1)
        else
            if #letterwidths > 0 then
                for c=1,letterwidths[#letterwidths] do
                    found_letters = found_letters:sub(1,-2)
                end
            end
            table.remove(letterunits, #letterunits)
            table.remove(letterwidths, #letterwidths)
        end
    end

    if #letterunits <= 1 or #found_letters <= 1 or packed_text_name == "" then
        return false
    end
    
    if splice_mod_globals.record_packed_text then
        for _, letter in ipairs(letterunits) do
            processed_pack_texts[letter] = true
        end
    end
    data = {
        letterunits = letterunits,
        packed_text_name = packed_text_name,
        packed_text_pos = packed_text_pos,
        packerunitid = packerunitid,
        packer_pushed_against = packer_pushed_against,
    }
    return data
end

function handle_text_packing(unitid, dir, pack_entry)
    if pack_entry then
        local old_x = nil
        local old_y = nil
        if unitid ~= -1 and unitid ~= 2 then
            local firstunit = mmf.newObject(unitid)
            old_x = firstunit.values[XPOS]
            old_y = firstunit.values[YPOS]
        end
        if pack_entry.packer_pushed_against then
            old_x, old_y = pack_entry.packed_text_pos[1], pack_entry.packed_text_pos[2]
        end
        for _,letterunit in ipairs(pack_entry.letterunits) do
            local u = mmf.newObject(letterunit)
            u.values[EFFECT] = 1

            local pmult,sound = checkeffecthistory("smoke")
            MF_particles("eat",u.values[XPOS],u.values[YPOS],5 * pmult,0,3,1,1)

            -- deleting without triggering has and ignoring guard @mods(guard)(param)
            GLOBAL_disable_guard_checking = true
            delete(letterunit, nil, nil, nil, true)
            GLOBAL_disable_guard_checking = false
        end
        local newunitid = create("text_"..pack_entry.packed_text_name, pack_entry.packed_text_pos[1], pack_entry.packed_text_pos[2], dir, old_x, old_y, nil, nil, nil)
        local newunit = mmf.newObject(newunitid)
        newunit.values[EFFECT] = 1

        local c1,c2 = getcolour(newunitid)
        local pmult,sound = checkeffecthistory("bling")
        MF_particles("bling",pack_entry.packed_text_pos[1],pack_entry.packed_text_pos[2],5 * pmult,c1,c2,1,1)
        generaldata.values[SHAKE] = 3
        setsoundname("turn",9,sound)
    end 
end