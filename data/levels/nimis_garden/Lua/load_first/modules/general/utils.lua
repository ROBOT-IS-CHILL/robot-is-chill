local utils = {}
local DEBUG_TEST = false

utils = {
    debug_assert = function(expr, err_msg)
        if DEBUG_TEST then
            if not expr then
                if not err_msg then
                    err_msg = ""
                end
                error("Assertion failed: "..err_msg.."\n"..debug.traceback(), 2)
            end
        end
    end,

    debug_error = function(error_msg)
        if DEBUG_TEST then
            error(error_msg.." "..debug.traceback())
        else
            error(error_msg)
        end
    end,

    try_call = function(func, ...)
        local status, err = pcall(func, ...)
        if not status then
            local callinfo = debug.getinfo(2)
            timedmessage("[Plasma Error: Please report this!].")
            timedmessage("[File]: "..callinfo.short_src, 0, 1)
            timedmessage("[Line]: "..tostring(callinfo.currentline), 0, 2)
            if DEBUG_TEST then
                print(err)
                print(debug.traceback())
            end
        end

        return status
    end,

    make_object = function(unitid, x, y, nil_if_invalid)
        if not unitid then 
            if nil_if_invalid then
                return nil
            else
                utils.debug_error("Provided nil unitid")
            end
        end
        if unitid == 2 then
            if not x then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Provided nil x coord for empty object") 
                end
            end
            if not y then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Provided nil y coord for empty object") 
                end
            end
            return -(200 + x + y * roomsizex) -- JAAAAAAANK
        elseif unitid == 1 then
            return -1
        else
            local unit = mmf.newObject(unitid)
            if not unit then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Invalid unit for unitid "..tostring(unitid)) 
                end
            end
            return unit.values[ID]
        end
    end,

    parse_object = function(object, nil_if_invalid)
        if not object then utils.debug_error("Provided nil object") end
        if object <= -200 then
            local tileid = (-object) - 200
            local x = tileid % roomsizex
            local y = math.floor(tileid / roomsizex)
            return 2, x, y, tileid
        elseif object == -1 then
            return 1
        else
            local unitid = MF_getfixed(object)
            if not unitid then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Cannot find unitid of object: "..tostring(object)) 
                end
            end

            local unit = mmf.newObject(unitid)
            if not unit then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Cannot find unit of object: "..tostring(object)) 
                end
            end

            return unitid, unit.values[XPOS], unit.values[YPOS], unit.values[XPOS] + unit.values[YPOS] * roomsizex
        end
    end,

    parse_object_full = function(object, nil_if_invalid)
        if not object then 
            if nil_if_invalid then
                return nil
            else
                utils.debug_error("Provided nil object") 
            end
        end
        if object <= -200 then
            local tileid = (-object) - 200
            local x = tileid % roomsizex
            local y = math.floor(tileid / roomsizex)
            return {
                unitid = 2,
                name = "empty",
                x = x,
                y = y,
                tileid = tileid,
                unittype = "object",
                texttype = 0
            }
        elseif object == -1 then
            return {
                unitid = 1,
                name = "level",
                x = 0,
                y = 0,
                tileid = 0,
                unittype = "object",
                texttype = 0
            }
        else
            local unitid = MF_getfixed(object)
            if not unitid then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Cannot find unitid of object: "..tostring(object)) 
                end
            end

            local unit = mmf.newObject(unitid)
            if not unit then 
                if nil_if_invalid then
                    return nil
                else
                    utils.debug_error("Cannot find unit of object: "..tostring(object)) 
                end
            end

            return {
                unitid = unitid,
                name = unit.strings[NAME],
                x = unit.values[XPOS],
                y = unit.values[YPOS],
                tileid = unit.values[XPOS] + unit.values[YPOS] * roomsizex,
                unittype = unit.strings[UNITTYPE],
                texttype = unit.values[TYPE]
            }
        end
    end,

    object_exists = function(object)
        if not object then return false end
        if object <= -200 then
            return true
        elseif object == -1 then
            return true
        else
            local unitid = MF_getfixed(object)
            if not unitid then return false end

            local unit = mmf.newObject(unitid)
            if not unit then return false end
            
            return true
        end
    end,

    objectstring = function(object)
        local unitid, x, y = utils.parse_object(object)
        if unitid == 1 then
            return "(Level)"
        elseif unitid == 2 then
            return string.format("(Empty at %d,%d)", x, y)
        else
            local unit = mmf.newObject(unitid)
            if not unit then return "Deleted object with unitid: "..unitid end
            return string.format("(%s with id %d at %d,%d | unitid %s)", unit.strings[UNITNAME], unit.values[ID], unit.values[XPOS], unit.values[YPOS], tostring(unitid))
        end
    end,

    unitstring = function(unitid)
        local unit = mmf.newObject(unitid)
        if not unit then return "Deleted unitid: "..unitid end

        return utils.objectstring(utils.make_object(unitid, unit.values[XPOS], unit.values[YPOS]))
    end,

    deep_copy_table = function(table)
        local copy = {}
        for k,v in pairs(table) do
            if type(v) == "table" then
                v = utils.deep_copy_table(v)
            end
            copy[k] = v
        end
    
        return copy
    end,

    get_deleted_unitid_key = function(object)
        local unitid, x, y = utils.parse_object(object)
        local deleted_unitid = unitid
        if unitid == 1 then
            return nil
        elseif unitid == 2 then
            -- JANK WARNING!!! This formula is apparently how the game determines the key for marking an empty to be "deleted".
            deleted_unitid = 200 + x + y * roomsizex
        end
        return deleted_unitid
    end,

    condsort = function(a,b)
        if a[1] ~= b[1] then
            return a[1] < b[1]
        else
            if #a[2] ~= #b[2] then
                return #a[2] < #b[2]
            else
                local param_a = utils.deep_copy_table(a[2])
                local param_b = utils.deep_copy_table(b[2])

                table.sort(param_a)
                table.sort(param_b)

                for i = 1, #param_a do
                    if param_a[i] ~= param_b[i] then
                        return param_a[i] < param_b[i]
                    end
                end
            end
        end
    end,

    serialize_feature = function(feature)
        local tokens = {}
        local baserule = feature[1]
        for i, word in ipairs(baserule) do
            tokens[#tokens + 1] = word
            if i ~= #baserule then
                tokens[#tokens + 1] = " "
            end
        end
        
        if #feature[2] > 0 then
            tokens[#tokens + 1] = " | "
            local conds = utils.deep_copy_table(feature[2])
            table.sort(conds, utils.condsort)

            for j, cond in ipairs(conds) do
                tokens[#tokens + 1] = cond[1]
                tokens[#tokens + 1] = "["
                for i, param in ipairs(cond[2]) do
                    tokens[#tokens + 1] = "("
                    tokens[#tokens + 1] = param
                    tokens[#tokens + 1] = ")"

                    if i ~= #cond[2] then
                        tokens[#tokens + 1] = ","
                    end
                end

                -- Serialization with THIS as a cond
                -- if is_name_text_this(cond[1]) then
                --     local this_text_unitid = parse_this_unit_from_param_id(cond[2][1])
                --     for ray_object in pairs(get_raycast_objects(this_text_unitid)) do
                --         local ray_unitid, _, _, ray_tileid = utils.parse_object(ray_object)
                --         if ray_unitid == 2 then
                --             tokens[#tokens + 1] = "empty{"..tostring(ray_tileid).."}"
                --         else
                --             local ray_unit = mmf.newObject(ray_unitid)
                --             tokens[#tokens + 1] = tostring(ray_unit.values[ID])
                --         end
                --     end
                -- end

                tokens[#tokens + 1] = "]"
                if j ~= #conds then
                    tokens[#tokens + 1] = " && "
                end
            end
        end

        -- for _, tag in ipairs(feature[4]) do 
        --     tokens[#tokens + 1] = " #"
        --     tokens[#tokens + 1] = tag
        -- end
        return table.concat(tokens)
    end,

    serialize_typedata = function(typedata)
        local tokens = {}
        tokens[#tokens + 1] = typedata[1]
        
        if #typedata[2] > 0 then
            tokens[#tokens + 1] = " | "
            local conds = utils.deep_copy_table(typedata[2])
            table.sort(conds, utils.condsort)
            
            for i, cond in ipairs(conds) do
                tokens[#tokens + 1] = cond[1]
                tokens[#tokens + 1] = "["
                for _, param in ipairs(cond[2]) do
                    tokens[#tokens + 1] = "("
                    tokens[#tokens + 1] = param
                    tokens[#tokens + 1] = ")"

                    if i ~= #cond[2] then
                        tokens[#tokens + 1] = ","
                    end
                end
                tokens[#tokens + 1] = "]"

                if i ~= #conds then
                    tokens[#tokens + 1] = " && "
                end
            end
            tokens[#tokens + 1] = ")"
        end
        return table.concat(tokens)
    end,

    tileid_from_coords = function(x, y)
        return x + y * roomsizex
    end,

    coords_from_tileid = function(tileid)
        return math.floor(tileid % roomsizex), math.floor(tileid / roomsizex)
    end,

    real_condtype = function(condtype)
        local _, _, type = string.find(condtype, "%(*([%a%d%s]+)%)*")
        return type
    end,
}

return utils