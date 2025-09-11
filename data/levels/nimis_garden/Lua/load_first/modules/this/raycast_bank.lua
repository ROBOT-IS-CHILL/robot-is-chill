local utils = PlasmaModules.load_module("general/utils")
local RAY_ID_BASE = -50

local RaycastBank = {
    curr_ray_id = RAY_ID_BASE,
    objects = {},
    --[[ 
        ray_id -> {
            objects: [list of raycast units], 
            tileids: int
        }
     ]]
    free_ray_ids = {}
}

RaycastBank.__index = RaycastBank

function RaycastBank:reset()
    self.objects = {}
    self.free_ray_ids = {}
    self.curr_ray_id = RAY_ID_BASE
end

function RaycastBank:evaluate_and_store_pnouns_in_conds(conds)
    local newconds = {}
    local ray_ids = {}
    for i, cond in ipairs(conds) do
        local condtype = cond[1]
        local real_condtype = utils.real_condtype(condtype)
        local params = cond[2]

        if real_condtype == "this" or real_condtype == "not this" then
            local this_unitid = parse_this_unit_from_param_id(params[1])
            local ray_id = self:register_ray_objects_from_pnoun(this_unitid)
            table.insert(newconds, {condtype, { ray_id } })
        else
            local new_params = {}
            for a,b in ipairs(params) do
                local pname = b
                local isnot_prefix = ""
                if (string.sub(b, 1, 4) == "not ") then
                    pname = string.sub(b, 5)
                    isnot_prefix = "not "
                end
                local this_param_name,_,_,_,this_unitid = parse_this_param_and_get_raycast_units(pname)
                if this_param_name then
                    local ray_id = self:register_ray_objects_from_pnoun(this_unitid)
                    local this_param = make_this_param(isnot_prefix..this_param_name, tostring(ray_id))
                    table.insert(new_params, this_param)
                    table.insert(ray_ids, ray_id)
                else
                    table.insert(new_params, b)
                end
            end
            table.insert(newconds, {condtype, new_params})
        end
    end
    return newconds, ray_ids
end

function RaycastBank:register_ray_objects_from_pnoun(pnoun_unitid)
    local raycast_objects, ray_count = get_raycast_objects(pnoun_unitid)
    local raycast_tileids = get_raycast_tileid(pnoun_unitid)

    local ray_id = self:register_ray_objects(raycast_objects, raycast_tileids, ray_count)
    return ray_id
end

function RaycastBank:register_ray_objects(ray_objects, ray_tileids, count)
    local new_id = self.curr_ray_id

    if #self.free_ray_ids > 0 then
        new_id = table.remove(self.free_ray_ids)
    else
        self.curr_ray_id = self.curr_ray_id - 1
    end

    self.objects[new_id] = {
        objects = ray_objects,
        tileids = ray_tileids,
        count = count
    }


    return new_id
end

function RaycastBank:revoke_ray_id(ray_id)
    if self:is_valid_ray_id(ray_id) then
        self.objects[ray_id] = nil
        table.insert(self.free_ray_ids, ray_id)
    end
end

function RaycastBank:get_raycast_objects(ray_id)
    local ray_objects = self.objects[ray_id]
    if not ray_objects then
        return {}, 0
    else
        local count = 0
        local existing_objects = {}
        for ray_object in pairs(ray_objects.objects) do
            if utils.object_exists(ray_object) then
                existing_objects[ray_object] = true
                count = count + 1
            end
        end
        return existing_objects, count
    end
end

function RaycastBank:get_raycast_tileids(ray_id)
    local ray_objects = self.objects[ray_id]
    if not ray_objects then
        return {}
    else
        return ray_objects.tileids
    end 
end

function RaycastBank:is_valid_ray_id(id)
    return tonumber(id) and self.objects[tonumber(id)] ~= nil
end

-- Injection reason: reset the Raycast Bank. We do it in clearunits instead of levelstart since we want to sync with stablestate and THIS
local old_clearunits = clearunits
function clearunits(...)
    local ret = old_clearunits(...)
    RaycastBank:reset()
    return ret
end

return RaycastBank