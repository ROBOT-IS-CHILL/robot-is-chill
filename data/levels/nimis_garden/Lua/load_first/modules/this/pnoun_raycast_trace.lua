local RaycastTrace = {
    tileids = {},

    --[[ 
        hasfeature_checks = [
            {
                params = {<name>, is, block, <unitid>}, 
                last_result = false
            }
        ],
    ]]
    hasfeature_checks = {},

    --[[ 
        all_is_feature_checks = [
            {
                parameters = {<name>, is, block, <unitid>}, 
                last_result = false
            }
        ]
    ]]
    all_is_feature_checks = {},
}

function RaycastTrace:new(o)
    o = o or {}
    setmetatable(o, self) -- setmetatable() here so that o can call RaycastTrace's methods
    self.__index = self -- When accessing RaycastTrace statically, access the "RaycastTrace" table above

    -- Call clear() to initialize o's own copy of tileids. This is so that when we get o.tileids, 
    -- it accesses o's own copy instead of the global "RaycastTrace" table's copy.
    o:clear()

    return o
end

function RaycastTrace:clear()
    self.tileids = {}
    self.hasfeature_checks = {}
    self.all_is_feature_checks = {}
end

function RaycastTrace:add_tileid(tileid)
    self.tileids[tileid] = true
end

function RaycastTrace:add_hasfeature_check(pnoun_unitid, params, last_result, is_nontrivial)
    if is_nontrivial then
        table.insert(self.hasfeature_checks, {
            pnoun_unitid = pnoun_unitid,
            params = params,
            last_result = last_result
        })
    end
end

function RaycastTrace:add_all_is_feature_check(params, last_result)
    table.insert(self.all_is_feature_checks, {
        params = params,
        last_result = last_result
    })
end

function RaycastTrace:add_traces(other_raycast_trace)
    for other_tileid in pairs(other_raycast_trace.tileids) do
        self.tileids[other_tileid] = true
    end
    for _, other_featurecheck in ipairs(other_raycast_trace.hasfeature_checks) do
        table.insert(self.hasfeature_checks, other_featurecheck)
    end
    for _, other_featurecheck in ipairs(other_raycast_trace.all_is_feature_checks) do
        table.insert(self.all_is_feature_checks, other_featurecheck)
    end
end

function RaycastTrace:is_tileid_recorded(tileid)
    return self.tileids[tileid] ~= nil
end

function RaycastTrace:retest_features_for_testcond_change(curr_pnoun_ref)
    for _, featurecheck in ipairs(self.all_is_feature_checks) do
        local new_result = findfeature(table.unpack(featurecheck.params)) ~= nil
        if new_result ~= featurecheck.last_result then
            -- print(table.concat(featurecheck.params, " "), featurecheck.last_result, new_result)
            return true
        end
    end

    for _, featurecheck in ipairs(self.hasfeature_checks) do
        curr_pnoun_ref[0] = featurecheck.pnoun_unitid
        local new_result = hasfeature(table.unpack(featurecheck.params))
        if new_result == nil then
            new_result = false
        end
        if new_result ~= featurecheck.last_result then
            -- print(table.concat(featurecheck.params, " ")..plasma_utils.unitstring(featurecheck.params[4]), featurecheck.last_result, new_result)
            return true
        end
    end

    return false
end

function RaycastTrace:call_hasfeature_with_trace(pnoun_unitid, params)
    local result, is_nontrivial = hasfeature(table.unpack(params))
    if result == nil then
        result = false
    end
    self:add_hasfeature_check(pnoun_unitid, params, result, is_nontrivial)
    return result
end

function RaycastTrace:call_findfeature_with_trace(params)
    local result = findfeature(table.unpack(params)) ~= nil
    self:add_all_is_feature_check(params, result)
    return result
end

function RaycastTrace:evaluate_raycast_property(pnoun_unitid, name, property, unitid, x, y)
    local is_x = self:call_hasfeature_with_trace(pnoun_unitid, {name, "is", property, unitid, x, y})
    local is_not_x = self:call_hasfeature_with_trace(pnoun_unitid, {name, "is", "not "..property, unitid, x, y})

    return is_x and not is_not_x
end

return RaycastTrace