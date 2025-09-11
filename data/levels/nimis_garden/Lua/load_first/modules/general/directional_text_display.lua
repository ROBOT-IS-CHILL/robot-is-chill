local DirTextDisplay = {
    directional_text_names = {},
    directional_text_prefixes = {},
}

function DirTextDisplay:register_directional_text(name)
    self.directional_text_names[name] = true
end
function DirTextDisplay:register_directional_text_prefix(name)
    table.insert(self.directional_text_prefixes, name)
end

function DirTextDisplay:set_tt_display_direction(unit, dir)
    dir = dir or nil
    local is_tt = self.directional_text_names[unit.strings[NAME]]
    if not is_tt then
        for i, prefix in pairs(self.directional_text_prefixes) do
            if string.sub(unit.strings[NAME], 1, #prefix) == prefix then
                is_tt = true
                break
            end
        end
    end
    if is_tt then
        if dir == nil then
            unit.direction = (unit.values[DIR] * 8) % 32
        else
            unit.direction = (dir * 8) % 32
        end
    end
end

-- @TODO(Optimize): maybe we can optimize having to loop through all of codeunits just to find texts with a specific name
table.insert( mod_hook_functions["level_start"],
    function()
        for i,unitid in ipairs(codeunits) do
            local unit = mmf.newObject(unitid)
            DirTextDisplay:set_tt_display_direction(unit)
        end
    end
)

table.insert( mod_hook_functions["turn_end"], 
    function()
        for i,unitid in ipairs(codeunits) do
            local unit = mmf.newObject(unitid)
            DirTextDisplay:set_tt_display_direction(unit)
        end
    end
)

table.insert( mod_hook_functions["undoed_after"],
    function()
        for i,unitid in ipairs(codeunits) do
            local unit = mmf.newObject(unitid)
            DirTextDisplay:set_tt_display_direction(unit)
        end
    end
)

return DirTextDisplay