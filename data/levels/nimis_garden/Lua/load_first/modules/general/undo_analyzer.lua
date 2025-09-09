local Undo_Analyzer = {
    names_updated = {},
    objects_updated = {},
    tileids_updated = {},
    analyzers = {}
}

local plasma_utils = PlasmaModules.load_module("general/utils")

function Undo_Analyzer:new(o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    o:reset()
    return o
end

function Undo_Analyzer:reset()
    self.names_updated = {}
    self.objects_updated = {}
    self.tileids_updated = {}
end

-- Define all analyzer instances here
Undo_Analyzer.analyzers.raycast_analyzer = Undo_Analyzer:new()

local function add_updated_item(update_dict_name, item, undo_style)
    for t, analyzer in pairs(Undo_Analyzer.analyzers) do
        local update_dict = analyzer[update_dict_name]
        if update_dict[item] == nil then
            update_dict[item] = {}
        end
        update_dict[item][undo_style] = true
    end
end

function Undo_Analyzer.analyze_undo_line(line)
    local style = line[1]
    if (style == "update") then
        local uid = line[9]
        if (paradox[uid] == nil) then
            local unitid = getunitid(line[9])
            local unit = mmf.newObject(unitid)

            local old_tileid = plasma_utils.tileid_from_coords(line[6],line[7])
            local new_tileid = plasma_utils.tileid_from_coords(line[3],line[4])

            add_updated_item("objects_updated", uid, style)
            add_updated_item("names_updated", unit.strings[UNITNAME], style)
            add_updated_item("tileids_updated", old_tileid, style)
            add_updated_item("tileids_updated", new_tileid, style)
        end
    elseif (style == "remove") then
        local uid = line[6]
        local baseuid = line[7] or -1
        
        if (paradox[uid] == nil) and (paradox[baseuid] == nil) then
            local x,y = line[3],line[4]
            local name = line[2]
            
            local tileid = plasma_utils.tileid_from_coords(x,y)
            
            add_updated_item("objects_updated", uid, style)
            add_updated_item("names_updated", name, style)
            add_updated_item("tileids_updated", tileid, style)
        end
    elseif (style == "create") then
        local uid = line[3]
        local baseid = line[4]
        local source = line[5]
        
        if (paradox[uid] == nil) then
            local name = line[2]
            local x,y = line[6], line[7]

            local tileid = plasma_utils.tileid_from_coords(x,y)
            
            add_updated_item("objects_updated", uid, style)
            add_updated_item("names_updated", name, style)
            add_updated_item("tileids_updated", tileid, style)
        end
    elseif (style == "backset") then
        local uid = line[3]
        local name = line[2]

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
    elseif (style == "done") then
        local uid = line[6]
        local name = line[2]
        local tileid = plasma_utils.tileid_from_coords(line[3],line[4])

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
        add_updated_item("tileids_updated", tileid, style)
    elseif (style == "float") then
        local uid = line[3]
					
        if (paradox[uid] == nil) then
            local unitid = getunitid(uid)
            
            if (unitid ~= nil) and (unitid ~= 0) then
                local unit = mmf.newObject(unitid)
                local name = unit.strings[UNITNAME]

                add_updated_item("objects_updated", uid, style)
                add_updated_item("names_updated", name, style)
            end
        end
    elseif (style == "levelupdate") then
    elseif (style == "maprotation") then
    elseif (style == "mapdir") then
    elseif (style == "mapcursor") then
        local uid = line[10]
        local unitid = getunitid(line[10])

        if (unitid ~= nil) then
            local unit = mmf.newObject(unitid)
            local name = unit.strings[UNITNAME]

            local old_tileid = plasma_utils.tileid_from_coords(line[3],line[4])
            local new_tileid = plasma_utils.tileid_from_coords(line[7],line[8])

            add_updated_item("objects_updated", uid, style)
            add_updated_item("names_updated", name, style)
            add_updated_item("tileids_updated", old_tileid, style)
            add_updated_item("tileids_updated", new_tileid, style)
        end
    elseif (style == "colour") then
        local uid = line[2]
        local unitid = getunitid(uid)
        local unit = mmf.newObject(unitid)
        local name = unit.strings[UNITNAME]

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
        print("test")
    elseif (style == "broken") then
        local uid = line[3]
        local name = line[4]

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
    elseif (style == "bonus") then
    elseif (style == "followed") then
        local uid = line[3]
        local name = line[5]

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
    elseif (style == "startvision") then
    elseif (style == "stopvision") then
    elseif (style == "visiontarget") then
    elseif (style == "holder") then
        local uid = line[2]
        local unitid = getunitid(uid)
        local unit = mmf.newObject(unitid)
        local name = unit.strings[UNITNAME]

        add_updated_item("objects_updated", uid, style)
        add_updated_item("names_updated", name, style)
    end
end

return Undo_Analyzer