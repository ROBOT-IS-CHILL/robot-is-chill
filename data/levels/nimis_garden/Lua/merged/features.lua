

function findfeature(rule1,rule2,rule3)
	--[[
		@Mods(turning text) - Override reason: when looking for "X is stop", if the global   
								"arrow_prop_mod_globals.group_arrow_properties" is true, then include stopright, stopleft, etc 
								as part of the result. The global "arrow_prop_mod_globals.group_arrow_properties" is set to 
								false whenever we want to search specifically for a non arrow property
	]] 
	--								   
	local options = {}
	local result = {}
	local rule = ""
	
	if (rule1 ~= nil) then
		rule = rule1 .. " "
	end
	
	if (rule2 ~= nil) then
		rule = rule .. rule2 .. " "
	end
	
	if (rule3 ~= nil) then
		rule = rule .. rule3
	end
	
	if (featureindex[rule1] ~= nil) then
		for i,rules in ipairs(featureindex[rule1]) do
			local rule = rules[1]
			local conds = rules[2]
			
			if (conds[1] ~= "never") then
				if (rule[1] == rule1) and (rule[2] == rule2) then
					local baserule = {rule[1],rule[2],rule[3]}
					table.insert(options, {baserule,conds})
				end
			end
		end
	end
	
	if (featureindex[rule3] ~= nil) and (featureindex[rule1] == nil) then
		for i,rules in ipairs(featureindex[rule3]) do
			local rule = rules[1]
			local conds = rules[2]
			
			if (conds[1] ~= "never") then
				if (rule[3] == rule3) and (rule[2] == rule2) then
					local baserule = {rule[1],rule[2],rule[3]}
					table.insert(options, {baserule,conds})
				end
			end
        end
	end
    --@Turning Text --------------------
    if arrow_prop_mod_globals.group_arrow_properties and arrow_properties[rule3] then
        for i,dirfeature in ipairs(dirfeaturemap) do
            if featureindex[rule3..dirfeature] ~= nil then
                for i,rules in ipairs(featureindex[rule3..dirfeature]) do
                    local rule = rules[1]
                    local conds = rules[2]
                    
                    if (conds[1] ~= "never") then
                        if (rule[3] == rule3..dirfeature) and (rule[2] == rule2) then
                            local baserule = {rule[1],rule[2],rule3}
                            table.insert(options, {baserule,conds})
                        end
                    end
                end
            end
        end
    end 
    ----------------------
	
	if (rule1 == nil) and (rule3 == nil) and (rule2 ~= nil) then
		if (featureindex[rule2] ~= nil) then 
			for i,rules in ipairs(featureindex[rule2]) do
				local usable = false
				local rule = rules[1]
				local conds = rules[2]

				if (conds[1] ~= "never") then
					for a,mat in pairs(objectlist) do
						if (a == rule[3]) then
							usable = true
						end
					end
					
					for a,mat in ipairs(customobjects) do
						if (mat == rule[3]) then
							usable = true
						end
					end
					
					if (rule[2] == rule2) and usable then
						local baserule = {rule[1],rule[2],rule[3]}
						table.insert(options, {baserule,conds})
					end
				end
			end
		end
	end
	
	for i,rules in ipairs(options) do
		local words = {}
		local baserule = rules[1]
		
		for a,b in ipairs(baserule) do
			table.insert(words, b)
		end
		
		if (#words >= 3) then
			local one = words[3]
			local two = words[2] .. " " .. words[3]
			local three = words[1] .. " " .. words[2] .. " " .. words[3]

			if (one == rule) or (two == rule) or (three == rule) or ((rule2 == words[2]) and (rule1 == nil) and (rule3 == nil)) then				
				table.insert(result, {baserule[1], rules[2]})
			end
		end
	end
	
	if (#result > 0) then
		return result
	else
		return nil
	end
end

function hasfeature(rule1,rule2,rule3,unitid,x,y,checkedconds,ignorebroken_)
	local ignorebroken = false
	if (ignorebroken_ ~= nil) then
		ignorebroken = ignorebroken_
	end

	--[[ 
		@mods(THIS) - Override reason - add "is_nontrivial_check", which is set to true if the call to hasfeature() evaluated conditions.
		This is used for the raycast tracer, which records all calls to hasfeature() when we are determining raycast units. If for instance
		hasfeature("baba", "is", "block") is called and the featureindex does not have a "baba is block" rule, it would be counted as trivial,
		and therefore not worthy of recording for the raycast tracer. This is especially important for performance since we look at the raycast trace
		data every time we call code().

		Another note: for the purposes of reducing the amount of hasfeature checks in raycast tracer, 
		we also consider any features with only the THIS condition to be "trivial".
	]]
	local is_nontrivial_check = false
	--[[ 
		@Mods(turning text) - Override reason: when looking for "X is stop", if 
			"arrow_prop_mod_globals.group_arrow_properties" is true, then include stopright, stopleft, etc as part of the
			result. The global "arrow_prop_mod_globals.group_arrow_properties" is set to false whenever we want to 
			search specifically for a non arrow property.
	 ]]
	if (rule1 ~= nil) and (rule2 ~= nil) and (rule3 ~= nil) then
		if (featureindex[rule1] ~= nil) then
			for i,rules in ipairs(featureindex[rule1]) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (conds[1] ~= "never") then
					if (rule[1] == rule1) and (rule[2] == rule2) and (rule[3] == rule3) then
						if #conds > 0 and not (#conds == 1 and (conds[1][1] == "this" or conds[1][1] == "not this")) then
							is_nontrivial_check = true
						end
						if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
							return true, is_nontrivial_check
						end
					end
				end
			end
		end
	
		if (string.sub(rule1,1,5) == "text_") and (featureindex["text"] ~= nil) then
			for i,rules in ipairs(featureindex["text"]) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (conds[1] ~= "never") then
					if (rule[1] == "text") and (rule[2] == rule2) and (rule[3] == rule3) then
						if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
							return true
						end
					end
				end
			end
		end
	end
	
	if (rule3 ~= nil) and (rule2 ~= nil) and (rule1 ~= nil) then
		if (featureindex[rule3] ~= nil) then
			for i,rules in ipairs(featureindex[rule3]) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (conds[1] ~= "never") then
					if (rule[1] == rule1) and (rule[2] == rule2) and (rule[3] == rule3) then
						if #conds > 0 and not (#conds == 1 and (conds[1][1] == "this" or conds[1][1] == "not this")) then
							is_nontrivial_check = true
						end
						if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
							return true, is_nontrivial_check
						end
					end
				end
			end
		end
		
		-- @Turning Text -----------------------------
		if arrow_prop_mod_globals.group_arrow_properties and arrow_properties[rule3] then
			for i,dirfeature in ipairs(dirfeaturemap) do
				if (featureindex[rule3..dirfeature] ~= nil) and (rule2 ~= nil) and (rule1 ~= nil) then
					for i,rules in ipairs(featureindex[rule3..dirfeature]) do
						local rule = rules[1]
						local conds = rules[2]
						
						if (conds[1] ~= "never") then
							if (rule[1] == rule1) and (rule[2] == rule2) and (rule[3] == rule3..dirfeature) then
								if #conds > 0 and not (#conds == 1 and (conds[1][1] == "this" or conds[1][1] == "not this")) then
									is_nontrivial_check = true
								end
								if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
									return true
								end
							end
						end
					end
				end
			end
		end
		----------------------------------
	
		if (string.sub(rule3,1,5) == "text_") and (featureindex["text"] ~= nil) then
			for i,rules in ipairs(featureindex["text"]) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (conds[1] ~= "never") then
					if (rule[1] == rule1) and (rule[2] == rule2) and (rule[3] == "text") then
						if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
							return true
						end
					end
				end
			end
		end
	end
	
	if (featureindex[rule2] ~= nil) and (rule1 ~= nil) and (rule3 == nil) then
		local usable = false
		
		if (featureindex[rule1] ~= nil) then
			for i,rules in ipairs(featureindex[rule1]) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (conds[1] ~= "never") then
					for a,mat in pairs(objectlist) do
						if (a == rule[1]) then
							usable = true
							break
						end
					end
					
					if (rule[1] == rule1) and (rule[2] == rule2) and usable then
						if #conds > 0 and not (#conds == 1 and (conds[1][1] == "this" or conds[1][1] == "not this")) then
							is_nontrivial_check = true
						end
						if testcond(conds,unitid,x,y,nil,nil,checkedconds,ignorebroken) then
							return true
						end
					end
				end
			end
		end
	end
	
	return nil, is_nontrivial_check
end

function getunitswitheffect(rule3,nolevels_,ignorethese_,checkedconds)
	--[[ 
		@Mods(turning text) - Override reason: when looking for "X is stop", if the global 
			"arrow_prop_mod_globals.group_arrow_properties" is true, then include stopright, stopleft, etc as part of the 
			result. The global "arrow_prop_mod_globals.group_arrow_properties" is set to false whenever we want to search specifically for a non arrow property.
	 ]]
	local group = {}
	local result = {}
	local ignorethese = ignorethese_ or {}
	
	local nolevels = nolevels_ or false
	
	if (featureindex[rule3] ~= nil) then
		for i,v in ipairs(featureindex[rule3]) do
			local rule = v[1]
			local conds = v[2]
			
			if (rule[2] == "is") and (conds[1] ~= "never") and (findnoun(rule[1],nlist.brief) == false) then
				table.insert(group, {rule[1], conds})
			end
		end
		
		for i,v in ipairs(group) do
			if (v[1] ~= "empty") then
				local name = v[1]
				local fgroupmembers = unitlists[name]
				
				local valid = true
				
				if (name == "level") and nolevels then
					valid = false
				end
				
				if (fgroupmembers ~= nil) and valid then
					for a,b in ipairs(fgroupmembers) do
						if testcond(v[2],b,nil,nil,nil,nil,checkedconds) then
							local unit = mmf.newObject(b)
							
							if (unit.flags[DEAD] == false) then
								valid = true
								
								for c,d in ipairs(ignorethese) do
									if (d == b) then
										valid = false
										break
									end
								end
								
								if valid then
									table.insert(result, unit)
								end
							end
						end
					end
				end
			else
				--table.insert(result, {2, v[2]})
			end
		end
    end
    
    -- @Turning Text -----------------------------
    if arrow_prop_mod_globals.group_arrow_properties and arrow_properties[rule3] then
        for i,dirfeature in ipairs(dirfeaturemap) do
            if (featureindex[rule3..dirfeature] ~= nil) then
                for i,v in ipairs(featureindex[rule3..dirfeature]) do
                    local rule = v[1]
                    local conds = v[2]
                    
                    if (rule[2] == "is") and (conds[1] ~= "never") and (rule[1] ~= "all") and (rule[1] ~= "group") then
                        table.insert(group, {rule[1], conds})
                    end
                end
                
                for i,v in ipairs(group) do
                    if (v[1] ~= "empty") then
                        local name = v[1]
                        local groupmembers = unitlists[name]
                        
                        local valid = true
                        
                        if (name == "level") and nolevels then
                            valid = false
                        end
                        
                        if (groupmembers ~= nil) and valid then
                            for a,b in ipairs(groupmembers) do
                                if testcond(v[2], b) then
                                    local unit = mmf.newObject(b)
                                    
                                    if (unit.flags[DEAD] == false) then
                                        valid = true
                                        
                                        for c,d in ipairs(ignorethese) do
                                            if (d == b) then
                                                valid = false
                                                break
                                            end
                                        end
                                        
                                        if valid then
                                            table.insert(result, unit)
                                        end
                                    end
                                end
                            end
                        end
                    else
                        --table.insert(result, {2, v[2]})
                    end
                end
            end
        end
    end
    -- @Turning Text -----------------------------
	
	return result
end