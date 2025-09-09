-- OVERRIDE: clear echo maps
local init_old = init
init = function(tilemapid,roomsizex_,roomsizey_,tilesize_,Xoffset_,Yoffset_,generaldataid,generaldataid2,generaldataid3,generaldataid4,generaldataid5,spritedataid,vardataid,screenw_,screenh_)
	init_old(tilemapid,roomsizex_,roomsizey_,tilesize_,Xoffset_,Yoffset_,generaldataid,generaldataid2,generaldataid3,generaldataid4,generaldataid5,spritedataid,vardataid,screenw_,screenh_)
	-- EDIT: clear echo lists
	echounits = {}
	echorelatedunits = {}
	echomap = {} 
end