-- collects all turning text unit at the start of every turn. Also keeps track of their starting direction + if they have any directional rules applied to them
local turning_units = {}
local is_sound_played = false
local DirTextDisplay = PlasmaModules.load_module("general/directional_text_display")

turning_text_mod_globals = {}

function clear_turning_text_mod()
    turning_units = {}
    is_sound_played = false

    turning_text_mod_globals = {
        final_turning_unit_dir = {},
        
        -- We keep this global to prevent calling update_raycast_units while we call code()
        -- when we process turning text in movecommand
        tt_executing_code = false,        
    }
end

for name,_ in pairs(turning_word_names) do
    DirTextDisplay:register_directional_text("turning_"..name)
end

table.insert( mod_hook_functions["turn_end"], 
    function()
        local play_rule_sound = false
        for i,unitid in ipairs(codeunits) do
            local unit = mmf.newObject(unitid)
            if (is_turning_text(unit.strings[NAME])) then
                for i,b in ipairs(turning_units) do
                    local id,init_dir,prev_has_rule,prev_active = b[1],b[2],b[3],b[4]
                    if (unitid == id and unit.values[DIR] ~= init_dir) then
                        if (unit.active and prev_active) then
                            play_rule_sound = true
                            local x = unit.values[XPOS]
                            local y = unit.values[YPOS]
                            
                            local c1,c2 = getcolour(unitid,"active")
                            MF_particles("bling",x,y,10,c1,c2,1,1)
                            break
                        end
                    end
                end
            end
        end
        if (play_rule_sound and not is_sound_played) then
            local pmult,sound = checkeffecthistory("rule")
            local rulename = "rule" .. tostring(math.random(1,5)) .. sound
            MF_playsound(rulename)
            is_sound_played = true
        end
    end
)


table.insert( mod_hook_functions["command_given"],
    function()
        turning_units = {}
        turning_text_mod_globals.final_turning_unit_dir = {}
        is_sound_played = false

        for i,unitid in ipairs(codeunits) do
            local unit = mmf.newObject(unitid)
            local name = unit.strings[NAME]
            local unitname = getname(unit)

            if (is_turning_text(name)) then
                local ur = hasfeature_count(unitname,"is","right",unit.fixed)
                local uu = hasfeature_count(unitname,"is","up",unit.fixed)
                local ul = hasfeature_count(unitname,"is","left",unit.fixed)
                local ud = hasfeature_count(unitname,"is","down",unit.fixed)

                local has_dir_rule = (ur > 0) or (uu > 0) or (ul > 0) or (ud > 0)
                
                table.insert(turning_units, {unitid, unit.values[DIR], has_dir_rule, unit.active})
            end
        end
    end
)

function is_turning_text(name)
    return string.len(name) > 8 and string.sub(name, 1,8) == "turning_" and turning_word_names[string.sub(name, 9)]
end

local function parse_turning_text(name)
    local is_turning_text = false
    local word = nil
    if string.len(name) > 8 and string.sub(name, 1,8) == "turning_" then
        local word = string.sub(name, 9)
        if turning_word_names[word] then
            return word
        end
    end
    return nil
end

-- mainly copied from statusblock(). Given a set of right/up/left/down rules applied, figure out the resulting dir
local function eval_dir_rule(currdir, ur,uu,ul,ud)
    local fdir = nil

    if (ur > 0) or (uu > 0) or (ul > 0) or (ud > 0) then
        if (ur > uu) and (ur > ul) and (ur > ud) then
            fdir = 0
        elseif (uu > ur) and (uu > ul) and (uu > ud) then
            fdir = 1
        elseif (ul > ur) and (ul > uu) and (ul > ud) then
            fdir = 2
        elseif (ud > ur) and (ud > ul) and (ud > uu) then
            fdir = 3
        elseif (currdir == 3) then
            if (ul > 0) and (ul >= uu) and (ul >= ur) then
                fdir = 2
            elseif (uu > 0) and (uu > ul) and (uu >= ur) then
                fdir = 1
            elseif (ur > 0) and (ur > ul) and (ur > uu) then
                fdir = 0
            end
        elseif (currdir == 2) then
            if (uu > 0) and (uu >= ur) and (uu >= ul) then
                fdir = 1
            elseif (ur > 0) and (ur > uu) and (ur >= ud) then
                fdir = 0
            elseif (ud > 0) and (ud > ur) and (ud > uu) then
                fdir = 3
            end
        elseif (currdir == 1) then
            if (ur > 0) and (ur >= ul) and (ur >= ud) then
                fdir = 0
            elseif (ud > 0) and (ud > ur) and (ud >= ul) then
                fdir = 3
            elseif (ul > 0) and (ul > ur) and (ul > ud) then
                fdir = 2
            end
        elseif (currdir == 0) then
            if (ud > 0) and (ud >= ul) and (ud >= uu) then
                fdir = 3
            elseif (ul > 0) and (ul > ud) and (ul >= uu) then
                fdir = 2
            elseif (uu > 0) and (uu > ud) and (uu > ul) then
                fdir = 1
            end
        end
    end

    return fdir
end

-- injected into codecheck(). "Reinterprets" the meaning of turning text based on its direction and adds it to the feature list
-- This is however hijacked to mainly provide a slightly cheaty way of making turning text instantly respond to changes in the set of r/u/l/d rules. 
function get_turning_text_interpretation(turning_text_unitid)
    local turning_text_unit = mmf.newObject(turning_text_unitid)
    local turn_word = parse_turning_text(turning_text_unit.strings[NAME])
    local v_name = turning_text_unit.strings[NAME]

    if turn_word then
        local dir = turning_text_unit.values[DIR]
        local dirstring = ""
        local dir_str_map = {"right", "up", "left", "down"}

        local final_dir = turning_text_mod_globals.final_turning_unit_dir[turning_text_unit.fixed]
        if final_dir ~= nil then
            dir = final_dir
        end

        dirstring = dir_str_map[dir+1]

        if turn_word == "dir" then
            turn_word = ""
        end
        v_name = turn_word..dirstring
        
        if v_name == "falldown" then
            v_name = "fall"
        end

        if turn_word == "beside" then
            if dirstring == "right" then
                v_name = "besideright"  
            elseif dirstring == "up" then
                v_name = "above"  
            elseif dirstring == "left" then
                v_name = "besideleft"  
            elseif dirstring == "down" then
                v_name = "below"  
            end
        end
    end
    return v_name
end

-- After doing the movement code in movecommand(), determine the final directions of each turning text.
-- This is made more complicated since I want to support "text is turning_dir" remaining the same while pushing the 
-- sentence and instant updating of turning text after moving
function finalize_turning_text_dir()
    for i,v in ipairs(turning_units) do
        local unitid, init_dir, pre_has_dir_rule = v[1],v[2],v[3]
        local unit = mmf.newObject(unitid)

        if unit.strings[NAME] == "turning_dir" then
            local unitname = getname(unit)

            local r = hasfeature_count(unitname,"is","right",unit.fixed)
            local u = hasfeature_count(unitname,"is","up",unit.fixed)
            local l = hasfeature_count(unitname,"is","left",unit.fixed)
            local d = hasfeature_count(unitname,"is","down",unit.fixed)

            local curr_has_dir_rule = (r > 0) or (u > 0) or (l > 0) or (d > 0)
            local move_dir = unit.values[DIR]
            local final_dir = nil

            if not pre_has_dir_rule and not curr_has_dir_rule then
                final_dir = move_dir
            elseif not pre_has_dir_rule and curr_has_dir_rule then
                final_dir = move_dir
            elseif pre_has_dir_rule and not curr_has_dir_rule then
                final_dir = move_dir
            elseif pre_has_dir_rule and curr_has_dir_rule then
                final_dir = init_dir
            end
            
            if final_dir == nil then
                final_dir = move_dir
            end

            updatedir(unit.fixed, final_dir)
            updatecode = 1
        end
    end
    
    code() --Question: instead of calling code yet again, why not just populate a local table with turning dir information? Its not like the set of normal rules is gonna change during this time.
    -- Answer: you need testcond to work with turning dir
    
    for i,v in ipairs(turning_units) do
        local unitid, init_dir, pre_has_dir_rule = v[1],v[2],v[3]
        local unit = mmf.newObject(unitid)
        local unitname = getname(unit)

        if unit.strings[NAME] == "turning_dir" then
            local r = hasfeature_count(unitname,"is","right",unit.fixed)
            local u = hasfeature_count(unitname,"is","up",unit.fixed)
            local l = hasfeature_count(unitname,"is","left",unit.fixed)
            local d = hasfeature_count(unitname,"is","down",unit.fixed)


            local final_dir = init_dir
            if r > 0 or l > 0 or u > 0 or d > 0 then
                final_dir = eval_dir_rule(init_dir, r,u,l,d)
            else
                final_dir = unit.values[DIR] -- should be the initial moved direction
            end
            turning_text_mod_globals.final_turning_unit_dir[unit.fixed] = final_dir
        end

        updatecode = 1
    end
end