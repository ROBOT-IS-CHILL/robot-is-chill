-- OVERRIDE: add ECHO particles (monkey patching goes brrr)
local effects_old = effects
-- @Merge(injection)
effects = function(timer) 
	effects_old(timer)
	-- EDIT: Add ECHO particles
	local echornd = math.random(2,3)
	doeffect(timer,nil,"echo","glow",1,4,25,{0,echornd},"leveledge")
	doeffect(timer,nil,"echo","glow",1,5,100,{0,1},"leveledge")
end

-- OVERRIDE: add "leveledge" special rule so that LEVEL IS ECHO spawns particles around the edge

--[[ @Merge: doeffect() was merged ]]
