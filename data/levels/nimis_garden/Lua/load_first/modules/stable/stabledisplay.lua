local utils = PlasmaModules.load_module("general/utils")
local RaycastBank = PlasmaModules.load_module("this/raycast_bank")

local StableDisplay = {
    stable_indicators = {},
    --[[ 
        object -> unitid of indicator: unitid
    ]]

    stable_this_indicators = {},

    stablestate = nil,
}
StableDisplay.__index = StableDisplay

local LEVEL_OBJ = utils.make_object(1)
local LETTER_HEIGHT = 24
local LETTER_WIDTH = 8
local LETTER_SPACING = 2
local LINE_SPACING = LETTER_HEIGHT - 4
local MARGIN = 12
local PADDING = 4

function StableDisplay:new(stablestate)
    local new_stabledisplay = {}
    setmetatable(new_stabledisplay, self)
    new_stabledisplay:reset()

    new_stabledisplay.stablestate = stablestate
    return new_stabledisplay
end

function StableDisplay:reset()
    for _, indicator_id in pairs(self.stable_indicators) do
        MF_cleanremove(indicator_id)
    end
    for ray_unit_id, indicator_id in pairs(self.stable_this_indicators) do
        MF_cleanremove(indicator_id)
    end
    self.stable_indicators = {}
    self.stable_this_indicators = {}
end

local function make_stable_indicator()
    local indicator_id = MF_create("customsprite")
    local indicator = mmf.newObject(indicator_id)

    indicator.values[ONLINE] = 1
    indicator.layer = 2
    indicator.direction = 26
    indicator.values[ZLAYER] = 23
    MF_loadsprite(indicator_id,"stable_indicator_0",26,true)
    MF_setcolour(indicator_id,3,3)
    return indicator_id
end

local function update_stable_indicator(object, indicator_id)
    local unitid, _, _, tileid = utils.parse_object(object)
    local indicator_unit = mmf.newObject(indicator_id)

    if unitid == 2 then
        local nx = math.floor(tileid % roomsizex)
        local ny = math.floor(tileid / roomsizex)
        local indicator_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
        indicator_unit.values[XPOS] = nx * indicator_tilesize + Xoffset + (indicator_tilesize / 2)
        indicator_unit.values[YPOS] = ny * indicator_tilesize + Yoffset + (indicator_tilesize / 2)
    else
        local unit = mmf.newObject(unitid)
        indicator_unit.values[XPOS] = unit.x
        indicator_unit.values[YPOS] = unit.y
        indicator_unit.visible = unit.visible
    end

    indicator_unit.scaleX = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
    indicator_unit.scaleY = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]

    if (generaldata.values[DISABLEPARTICLES] ~= 0 or generaldata5.values[LEVEL_DISABLEPARTICLES] ~= 0) then
        -- Just to hide it
        indicator_unit.values[XPOS] = -20
        indicator_unit.values[YPOS] = -20
    end
end

function StableDisplay:add_stableunit(object)
    if not self.stable_indicators[object] and object ~= LEVEL_OBJ then
        local indicator_id = make_stable_indicator()
        self.stable_indicators[object] = indicator_id
    end
end

function StableDisplay:remove_stableunit(object)
    if self.stable_indicators[object] and object ~= LEVEL_OBJ then
        MF_cleanremove(self.stable_indicators[object])
        self.stable_indicators[object] = nil
    end
end

function StableDisplay:update_stable_indicators()
    for object, indicator_id in pairs(self.stable_indicators) do
        update_stable_indicator(object, indicator_id)
    end
end

function StableDisplay:show_stablerule_display()
    local mouse_x, mouse_y = MF_mouse()
    MF_letterclear("stablerules")

    local displayed_objects = {}
    local half_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT] / 2

    local level_mouse_x = mouse_x - Xoffset
    local level_mouse_y = mouse_y - Yoffset
    local tile_scale = (f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT])
    local grid_x = math.floor(level_mouse_x / tile_scale)
    local grid_y = math.floor(level_mouse_y / tile_scale)
    local mouse_tileid = grid_x + grid_y * roomsizex

    local unit_x = nil
    local unit_y = nil
    local empty_tileid = nil
    
    for object, _ in pairs(self.stablestate.objects) do
        local unitid, _, _, tileid = utils.parse_object(object)
        if unitid == 2 then
            if mouse_tileid == tileid then
                unit_x = mouse_x - (level_mouse_x % tile_scale) + tile_scale / 2
                unit_y = mouse_y - (level_mouse_y % tile_scale) + tile_scale / 2
                table.insert(displayed_objects, object)
            end
        else
            local unit = mmf.newObject(unitid)
            if unit and unit.visible and mouse_x >= unit.x - half_tilesize and mouse_x < unit.x + half_tilesize and mouse_y >= unit.y - half_tilesize and mouse_y < unit.y + half_tilesize then
                table.insert(displayed_objects, object)
                unit_x = unit.x
                unit_y = unit.y
            end
        end
    end

    if #displayed_objects > 0 or empty_tileid then
        self:write_stable_rules(displayed_objects, unit_x, unit_y, empty_tileid)
    else
        for ray_unit_id, indicator_id in pairs(self.stable_this_indicators) do
            MF_cleanremove(indicator_id)
            self.stable_this_indicators[ray_unit_id] = nil
        end
    end
end

function StableDisplay:write_stable_rules(obj_list, x, y, empty_tileid)
    if generaldata2.values[INPAUSEMENU] == 1 then
        return -- Don't display the stablerules when in the pause menu
    end
    
    local ruleids = {}
    local ruleid_count = 0
    for _, object in ipairs(obj_list) do
        for ruleid, ruleid_data in pairs(self.stablestate.objects[object].ruleids) do
            if not ruleids[ruleid] then
                ruleid_count = ruleid_count + 1
            end
            ruleids[ruleid] = ruleid_data.stack_count
        end
    end

    -- Determine final X
    local list_width = 0
    local found_ray_unit_ids = {}
    for ruleid, stack_count in pairs(ruleids) do
        local display = self.stablestate.rules[ruleid].display

        if stack_count > 1 then
            display = table.concat({stack_count, " x ", display})
        end

        list_width = math.max(list_width, LETTER_WIDTH * #display + LETTER_SPACING * (#display - 1))

        for _, stable_this_id in ipairs(self.stablestate.rules[ruleid].stable_this_ids) do
            for ray_object in pairs(RaycastBank:get_raycast_objects(stable_this_id)) do
                local ray_unitid, x, y = utils.parse_object(ray_object)
                local ind_x, ind_y
                local indicator_id
                
                if not self.stable_this_indicators[ray_object] then
                    indicator_id = make_stable_indicator()
                    MF_setcolour(indicator_id,4,2)
                else
                    indicator_id = self.stable_this_indicators[ray_object]
                end
                found_ray_unit_ids[ray_object] = true
                self.stable_this_indicators[ray_object] = indicator_id
                
                if ray_unitid == 2 then
                    local indicator_tilesize = f_tilesize * generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
                    ind_x = x * indicator_tilesize + Xoffset + (indicator_tilesize / 2)
                    ind_y = y * indicator_tilesize + Yoffset + (indicator_tilesize / 2)
                else
                    local ray_unit = mmf.newObject(ray_unitid)
                    ind_x = ray_unit.x
                    ind_y = ray_unit.y
                end
                
                local indicator = mmf.newObject(indicator_id)
                indicator.values[XPOS] = ind_x
                indicator.values[YPOS] = ind_y

                indicator.scaleX = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]
                indicator.scaleY = generaldata2.values[ZOOM] * spritedata.values[TILEMULT]

            end
        end
    end
    for ray_unit_id, indicator_id in pairs(self.stable_this_indicators) do
        if not found_ray_unit_ids[ray_unit_id] then
            MF_cleanremove(indicator_id)
            self.stable_this_indicators[ray_unit_id] = nil
        end
    end

    local x_lower_bound = Xoffset
    local x_upper_bound = Xoffset + f_tilesize * roomsizex * spritedata.values[TILEMULT] * generaldata2.values[ZOOM]

    local final_x = x
    if final_x - list_width/2 < x_lower_bound then
        final_x = x_lower_bound + list_width/2
    elseif final_x + list_width/2 > x_upper_bound then
        final_x = x_upper_bound - list_width/2
    end
    
    -- Determine final Y
    local y_lower_bound = Yoffset
    local y_upper_bound = Yoffset + f_tilesize * roomsizey * spritedata.values[TILEMULT] * generaldata2.values[ZOOM]
    local list_height = LINE_SPACING * ruleid_count
    
    local final_y = y + (f_tilesize + 4) * generaldata2.values[ZOOM]
    if final_y + list_height > y_upper_bound then
        final_y = y - list_height
        if final_y - LINE_SPACING/2 < y_lower_bound then
            final_y = y - list_height/2
        end
    end

    -- Write the rules 
    local y_offset = 0
    for ruleid, stack_count in pairs(ruleids) do
        local display = self.stablestate.rules[ruleid].display
        local color = {3,3}

        if stack_count > 1 then
            display = table.concat({stack_count, " x ", display})
        end

        -- Create the text "outline". (Hacky but does the job. Though if there's a more supported way to do this I'm all ears)
        for outline_x = -2, 2, 2 do
            for outline_y = -2, 2, 2 do
                writetext(display,-1, final_x + outline_x, final_y + y_offset + outline_y,"stablerules",true,2,true, {0, 4})
            end
        end
        writetext(display,-1, final_x, final_y + y_offset,"stablerules",true,2,true, color)

        y_offset = y_offset + LINE_SPACING
    end
end

return StableDisplay