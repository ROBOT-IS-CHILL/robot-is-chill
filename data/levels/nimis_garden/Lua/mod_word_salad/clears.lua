-- OVERRIDE: clear ECHO related stuff
local clearunits_old = clearunits
-- @Merge(injection)
clearunits = function(restore_)
	-- EDIT: clear all echo units
	echounits = {}
	echorelatedunits = {}
	echomap = {}

	clearunits_old(restore_)
end

-- OVERRIDE: clear ECHO related stuff
local clear_old = clear
-- @Merge(injection)
clear = function()
	-- EDIT: clear all echo units
	echounits = {}
	echorelatedunits = {}
	echomap = {}

	clear_old()
end