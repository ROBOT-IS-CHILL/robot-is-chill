

function moveblock(onlystartblock_)
	-- @Mods(Turning Text) - Override reason: directional shift updates the shifted objects direction here
	local onlystartblock = onlystartblock_ or false
	
	local isshift,istele = {},{}
	local isfollow = findfeature(nil,"follow",nil,true)
	
	if (onlystartblock == false) then
		isshift = findallfeature(nil,"is","shift",true)
		istele = findallfeature(nil,"is","tele",true)
		-- NIMI MOD EDIT: make throw objects update their facing later in the turn like shift objects do
		local isthrow = findallfeature(nil,"is","throw",true)
		for i,v in ipairs(isthrow) do
			table.insert(isshift,v)
		end
	end
	
	local doned = {}
	
	if (isfollow ~= nil) then
		for h,j in ipairs(isfollow) do
			local allfollows = findall(j)
			
			if (#allfollows > 0) then
				for k,l in ipairs(allfollows) do
					if (issleep(l) == false) then
						local unit = mmf.newObject(l)
						local x,y,name,dir = unit.values[XPOS],unit.values[YPOS],unit.strings[UNITNAME],unit.values[DIR]
						local unitrules = {}
						local followedfound = false
						
						if (unit.strings[UNITTYPE] == "text") then
							name = "text"
						end
						
						if (featureindex[name] ~= nil) then					
							for a,b in ipairs(featureindex[name]) do
								local baserule = b[1]
								local conds = b[2]
								
								local verb = baserule[2]
								
								if (verb == "follow") then
									if testcond(conds,l) then
										table.insert(unitrules, b)
									end
								end
							end
						end
						
						local follow = xthis(unitrules,name,"follow")
						
						if (#follow > 0) and (unit.flags[DEAD] == false) then
							local distance = 9999
							local targetdir = -1
							local stophere = false
							local highesttarget = false
							local counterclockwise = false
							
							local priorityfollow = -1
							local priorityfollowdir = -1
							
							local highpriorityfollow = -1
							local highpriorityfollowdir = -1
							
							for i,v in ipairs(follow) do
								local these = findall({v})
								
								if (#these > 0) and (stophere == false) then
									for a,b in ipairs(these) do
										if (b ~= unit.fixed) and (stophere == false) then
											local funit = mmf.newObject(b)
											
											local fx,fy = funit.values[XPOS],funit.values[YPOS]
											
											local xdir = fx-x
											local ydir = fy-y
											local dist = math.abs(xdir) + math.abs(ydir)
											local fdir = -1
											
											if (math.abs(xdir) <= math.abs(ydir)) then
												if (ydir >= 0) then
													fdir = 3
												else
													fdir = 1
												end
											else
												if (xdir > 0) then
													fdir = 0
												else
													fdir = 2
												end
											end
											
											if (dist <= distance) and (dist > 0) then
												distance = dist
												targetdir = fdir
												
												--MF_alert(name .. ": suggested dir " .. tostring(targetdir))
												
												if (dist == 1) then
													if (unit.followed ~= funit.values[ID]) then
														local ndrs = ndirs[dir + 1]
														local ox,oy = ndrs[1],ndrs[2]
														
														priorityfollow = funit.values[ID]
														priorityfollowdir = targetdir
														
														if (x + ox == fx) and (y + oy == fy) then
															highpriorityfollow = funit.values[ID]
															highpriorityfollowdir = targetdir
															highesttarget = true
															--MF_alert(tostring(unit.fixed) .. " moves forward: " .. tostring(dir) .. ", " .. tostring(targetdir))
														elseif (highesttarget == false) then
															local turnl = (dir + 1 + 4) % 4
															local ndrsl = ndirs[turnl + 1]
															local oxl,oyl = ndrsl[1],ndrsl[2]
															
															if (x + oxl == fx) and (y + oyl == fy) then
																highpriorityfollow = funit.values[ID]
																highpriorityfollowdir = targetdir
																counterclockwise = true
																--MF_alert(tostring(unit.fixed) .. " turns left: " .. tostring(dir) .. ", " .. tostring(turnl) .. ", " .. tostring(targetdir))
															elseif (counterclockwise == false) then
																local turnr = (dir - 1 + 4) % 4
																local ndrsr = ndirs[turnr + 1]
																local oxr,oyr = ndrsr[1],ndrsr[2]
																
																if (x + oxr == fx) and (y + oyr == fy) then
																	highpriorityfollow = funit.values[ID]
																	highpriorityfollowdir = targetdir
																	--MF_alert(tostring(unit.fixed) .. " turns right: " .. tostring(dir) .. ", " .. tostring(turnr) .. ", " .. tostring(targetdir))
																end
															end
														end
													else
														followedfound = true
														stophere = true
														break
													end
												end
											end
										end
									end
									
									if stophere then
										break
									end
								end
								
								if stophere then
									break
								end
							end
							
							if (followedfound == false) then
								if (highpriorityfollow > -1) then
									if (onlystartblock == false) then
										addundo({"followed",unit.values[ID],unit.followed,highpriorityfollow,unit.strings[UNITNAME]},unit.fixed)
									end
									unit.followed = highpriorityfollow
									targetdir = highpriorityfollowdir
									stophere = true
									followedfound = true
								elseif (priorityfollow > -1) then
									if (onlystartblock == false) then
										addundo({"followed",unit.values[ID],unit.followed,priorityfollow,unit.strings[UNITNAME]},unit.fixed)
									end
									unit.followed = priorityfollow
									targetdir = priorityfollowdir
									stophere = true
									followedfound = true
								elseif (unit.followed > -1) then
									if (onlystartblock == false) then
										addundo({"followed",unit.values[ID],unit.followed,0,unit.strings[UNITNAME]},unit.fixed)
									end
									unit.followed = -1
								end
							end
			
							if (targetdir >= 0) then
								--MF_alert(unit.strings[UNITNAME] .. " faces to " .. tostring(targetdir))
								updatedir(unit.fixed,targetdir,onlystartblock)
							end
						end
					end
				end
			end
		end
	end
	
	if (onlystartblock == false) then
		local isback = findallfeature(nil,"is","back",true)
		
		for i,unitid in ipairs(isback) do
			local unit = mmf.newObject(unitid)
			
			local undooffset = #undobuffer - unit.back_init
			
			local undotargetid = undooffset * 2 + 1
			
			if (undotargetid <= #undobuffer) and (unit.back_init > 0) and (unit.flags[DEAD] == false) then
				local currentundo = undobuffer[undotargetid]
				
				particles("wonder",unit.values[XPOS],unit.values[YPOS],1,{3,0})
				
				updateundo = true
				
				if (currentundo ~= nil) then
					for a,line in ipairs(currentundo) do
						local style = line[1]
						
						if (style == "update") and (line[9] == unit.values[ID]) then
							local uid = line[9]
							
							if (paradox[uid] == nil) then
								local ux,uy = unit.values[XPOS],unit.values[YPOS]
								local oldx,oldy = line[6],line[7]
								local x,y,dir = line[3],line[4],line[5]
								
								local ox = x - oldx
								local oy = y - oldy
								
								--[[
								Enable this to make the Back effect relative to current position
								x = ux + ox
								y = uy + oy
								]]--
								
								--MF_alert(unit.strings[UNITNAME] .. " is being updated from " .. tostring(ux) .. ", " .. tostring(uy) .. ", offset " .. tostring(ox) .. ", " .. tostring(oy))
								
								if (ox ~= 0) or (oy ~= 0) then
									addaction(unitid,{"update",x,y,dir})
								else
									addaction(unitid,{"updatedir",dir})
								end
								updateundo = true
								
								if (objectdata[unitid] == nil) then
									objectdata[unitid] = {}
								end
								
								local odata = objectdata[unitid]
								
								odata.tele = 1
							else
								particles("hot",line[3],line[4],1,{1, 1})
								updateundo = true
							end
						elseif (style == "create") and (line[3] == unit.values[ID]) then
							local uid = line[4]
							
							--MF_alert(unit.strings[UNITNAME] .. " back: " .. tostring(uid) .. ", " .. tostring(line[3]))
							
							if (paradox[uid] == nil) then
								local name = unit.strings[UNITNAME]
								
								local delname = {}
								
								for b,bline in ipairs(currentundo) do
									--MF_alert(" -- " .. bline[1] .. ", " .. tostring(bline[6]))
									
									if (bline[1] == "remove") and (bline[6] == uid) then
										local x,y,dir,levelfile,levelname,vislevel,complete,visstyle,maplevel,colour,clearcolour,followed,back_init = bline[3],bline[4],bline[5],bline[8],bline[9],bline[10],bline[11],bline[12],bline[13],bline[14],bline[15],bline[16],bline[17]
										
										local newname = bline[2]
										
										local newunitname = ""
										local newunitid = 0
										
										local ux,uy = unit.values[XPOS],unit.values[YPOS]
										
										newunitname = unitreference[newname]
										newunitid = MF_emptycreate(newunitname,ux,uy)
										
										local newunit = mmf.newObject(newunitid)
										newunit.values[ONLINE] = 1
										newunit.values[XPOS] = ux
										newunit.values[YPOS] = uy
										newunit.values[DIR] = dir
										newunit.values[ID] = bline[6]
										newunit.flags[9] = true
										
										newunit.strings[U_LEVELFILE] = levelfile
										newunit.strings[U_LEVELNAME] = levelname
										newunit.flags[MAPLEVEL] = maplevel
										newunit.values[VISUALLEVEL] = vislevel
										newunit.values[VISUALSTYLE] = visstyle
										newunit.values[COMPLETED] = complete
										
										newunit.strings[COLOUR] = colour
										newunit.strings[CLEARCOLOUR] = clearcolour
										
										if (newunit.className == "level") then
											MF_setcolourfromstring(newunitid,colour)
										end
										
										addunit(newunitid,true)
										addunitmap(newunitid,x,y,newunit.strings[UNITNAME])
										dynamic(unitid)
										
										newunit.followed = followed
										newunit.back_init = back_init
										
										if (newunit.strings[UNITTYPE] == "text") then
											updatecode = 1
										end
										
										local undowordunits = currentundo.wordunits
										local undowordrelatedunits = currentundo.wordrelatedunits
										
										if (#undowordunits > 0) then
											for a,b in ipairs(undowordunits) do
												if (b == bline[6]) then
													updatecode = 1
												end
											end
										end
										
										if (#undowordrelatedunits > 0) then
											for a,b in ipairs(undowordrelatedunits) do
												if (b == bline[6]) then
													updatecode = 1
												end
											end
										end
										
										-- EDIT: ECHO once more
										local undoechounits = currentundo.echounits
										local undoechorelatedunits = currentundo.echorelatedunits
										
										if (#undoechounits > 0) then
											for a,b in ipairs(undoechounits) do
												if (b == bline[6]) then
													updatecode = 1
												end
											end
										end
										
										if (#undoechorelatedunits > 0) then
											for a,b in ipairs(undoechorelatedunits) do
												if (b == bline[6]) then
													updatecode = 1
												end
											end
										end
										
										table.insert(delname, {newunit.strings[UNITNAME], bline[6], newunit.values[XPOS], newunit.values[YPOS], newunit.values[DIR]})
									end
								end
								
								addundo({"remove",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init,unit.originalname,unit.strings[UNITSIGNTEXT],false,unitid,unit.karma})
								
								for a,b in ipairs(delname) do
									MF_alert("added undo for " .. b[1] .. " with ID " .. tostring(b[2]))
									addundo({"create",b[1],b[2],b[2],"back",b[3],b[4],b[5]})
								end
								
								delunit(unitid)
								dynamic(unitid)
								MF_specialremove(unitid,2)
							end
						end
					end
				end
			end
		end
		
		doupdate()
		
		for i,unitid in ipairs(istele) do
			if (isgone(unitid) == false) then
				local unit = mmf.newObject(unitid)
				-- METATEXT
				local name = getname(unit)
				local x,y = unit.values[XPOS],unit.values[YPOS]
			
				local targets = findallhere(x,y)
				local telethis = false
				local telethisx,telethisy = 0,0
				
				if (#targets > 0) then
					for i,v in ipairs(targets) do
						local vunit = mmf.newObject(v)
						local thistype = vunit.strings[UNITTYPE]
						local vname = vunit.strings[UNITNAME]
						
						local targetvalid = isgone(v)
						local targetstill = hasfeature(vname,"is","still",v,x,y)
						-- Luultavasti ei väliä onko kohde tuhoutumassa?
						
						if (targetstill == nil) and floating(v,unitid,x,y) and (vunit.flags[DEAD] == false) then
							local targetname = getname(vunit)
							if (objectdata[v] == nil) then
								objectdata[v] = {}
							end
							
							local odata = objectdata[v]
							
							if (odata.tele == nil) then
								if (targetname ~= name) and (v ~= unitid) then
									local teles = istele
									
									if (#teles > 1) then
										local teletargets = {}
										local targettele = 0
										
										for a,b in ipairs(teles) do
											local tele = mmf.newObject(b)
											local telename = getname(tele)
											
											if (b ~= unitid) and (telename == name) and (tele.flags[DEAD] == false) then
												table.insert(teletargets, b)
											end
										end
										
										if (#teletargets > 0) then
											local randomtarget = fixedrandom(1, #teletargets)
											targettele = teletargets[randomtarget]
											local limit = 0
											
											while (targettele == unitid) and (limit < 10) do
												randomtarget = fixedrandom(1, #teletargets)
												targettele = teletargets[randomtarget]
												limit = limit + 1
											end
											
											odata.tele = 1
											
											local tele = mmf.newObject(targettele)
											local tx,ty = tele.values[XPOS],tele.values[YPOS]
											local vx,vy = vunit.values[XPOS],vunit.values[YPOS]
										
											update(v,tx,ty)
											
											local pmult,sound = checkeffecthistory("tele")
											
											MF_particles("glow",vx,vy,5 * pmult,1,4,1,1)
											MF_particles("glow",tx,ty,5 * pmult,1,4,1,1)
											setsoundname("turn",6,sound)
										end
									end
								end
							end
						end
					end
				end
			end
		end
	
	if enable_directional_shift then
		--@Turning Text(shift)
		do_directional_shift_moveblock()
	else
			--[[
				@Note from plasma: as weird as this may sound, I think that the entirety of this shift code below is redundant in vanilla.
				The code to handle updating directions of shifted objects is already handled in movecommand() with this code snippet:
					if (state == 0) and (data.reason == "shift") and (data.unitid ~= 2) then
						updatedir(data.unitid, data.dir)
						dir = data.dir
					end 
				As such, I decided to not bother with removing calling do_directional_shift_moveblock() if enable_directional_shift == true
			]]
		for a,unitid in ipairs(isshift) do
			if (unitid ~= 2) and (unitid ~= 1) then
				local unit = mmf.newObject(unitid)
				local x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
				
				local things = findallhere(x,y,unitid)
				
				if (#things > 0) and (isgone(unitid) == false) then
					for e,f in ipairs(things) do
						if floating(unitid,f,x,y) and (issleep(unitid,x,y) == false) then
							local newunit = mmf.newObject(f)
							local name = newunit.strings[UNITNAME]
							
							if (featureindex["reverse"] ~= nil) then
								local turndir = unit.values[DIR]
								turndir = reversecheck(newunit.fixed,unit.values[DIR],x,y)
							end
							
							if (newunit.flags[DEAD] == false) then
								addundo({"update",name,x,y,newunit.values[DIR],x,y,unit.values[DIR],newunit.values[ID]})
								newunit.values[DIR] = unit.values[DIR]
								
								--@ Turning text --
								if is_turning_text(newunit.strings[NAME]) then
									updatecode = 1
								end
								--@ Turning text --
							end
						end
					end
				end
			end
		end
		
		doupdate()
	end
	end
end

function block(small_)
	-- @mods(turning text) - Override reason - handle collision-based directional properties
	-- @mods(text splicing) - Override reason - handle overlap-based cutting
	local delthese = {}
	local doned = {}
	local unitsnow = #units
	local removalsound = 1
	local removalshort = ""
	
	local small = small_ or false
	
	local doremovalsound = false

	--@Turning Text ----------------
	arrow_prop_mod_globals.group_arrow_properties = false
	----------------------
	
	if (small == false) then
		if (generaldata2.values[ENDINGGOING] == 0) then
			local isdone = getunitswitheffect("done",false,delthese)
			
			for id,unit in ipairs(isdone) do
				table.insert(doned, unit)
			end
			
			if (#doned > 0) then
				setsoundname("turn",10)
			end
			
			for i,unit in ipairs(doned) do
				updateundo = true
				
				local ufloat = unit.values[FLOAT]
				local ded = unit.flags[DEAD]
				
				unit.values[FLOAT] = 2
				unit.values[EFFECTCOUNT] = math.random(-10,10)
				unit.values[POSITIONING] = 7
				unit.flags[DEAD] = true
				
				local x,y = unit.values[XPOS],unit.values[YPOS]
				
				if (spritedata.values[VISION] == 1) and (unit.values[ID] == spritedata.values[CAMTARGET]) then
					updatevisiontargets()
				end
				
				if (ufloat ~= 2) and (ded == false) then
					addundo({"done",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.fixed,ufloat,unit.originalname})
				end
				
				delunit(unit.fixed)
				dynamicat(x,y)
			end
		end
		
		local ismore = getunitswitheffect("more",false,delthese)
		--@Turning Text(more)
		ismore = do_directional_more(ismore, delthese)

		--@plasma(more): changed format of "ismore" to be unit -> list of directions to duplicate unit from MORE
		for unit,dirs in pairs(ismore) do
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local unitid = unit.fixed
			local name = unit.strings[UNITNAME]
			local doblocks = {}
			
			for _, i in ipairs(dirs) do
				local drs = ndirs[i]
				ox = drs[1]
				oy = drs[2]
				
				local valid = true
				local obs = findobstacle(x+ox,y+oy)
				local tileid = (x+ox) + (y+oy) * roomsizex
				
				if (#obs > 0) then
					for a,b in ipairs(obs) do
						if (b == -1) then
							valid = false
						elseif (b ~= 0) and (b ~= -1) then
							local bunit = mmf.newObject(b)
							local obsname = bunit.strings[UNITNAME]
							
							local obsstop = hasfeature(obsname,"is","stop",b,x+ox,y+oy) or (featureindex["stops"] ~= nil and hasfeature(obsname,"stops",name,b,x+ox,y+oy)) or hasfeature(obsname,"is","sidekick",b,x+ox,y+oy) or (featureindex["hates"] ~= nil and hasfeature(name,"hates",obsname,unitid,x,y)) or (hasfeature(obsname,"is","oneway",b) and oxoytodir(ox,oy) == rotate(bunit.values[DIR]))
							if (obsstop == false) then
								obsstop = nil
							end
							local obspush = hasfeature(obsname,"is","push",b,x+ox,y+oy) or (featureindex["pushes"] ~= nil and hasfeature(name,"pushes",obsname,unitid,x,y))
							if (obspush == false) then
								obspush = nil
							end
							local obspull = hasfeature(obsname,"is","pull",b,x+ox,y+oy) or (featureindex["pulls"] ~= nil and hasfeature(name,"pulls",obsname,unitid,x,y))
							if (obspull == false) then
								obspull = nil
							end

							obsstop, obspush, obspull = do_directional_collision(i-1, obsname, b, obsstop, obspush, obspull, x,y,ox,oy, false)
							
							if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) or (obsname == name) then
								valid = false
								break
							end
						end
					end
				else
					local obsstop = hasfeature("empty","is","stop",2,x+ox,y+oy) or (featureindex["stops"] ~= nil and hasfeature("empty","stops",name,2,x+ox,y+oy)) or hasfeature("empty","is","sidekick",2,x+ox,y+oy) or (featureindex["hates"] ~= nil and hasfeature(name,"hates","empty",unitid,x,y))
					if (obsstop == false) then
						obsstop = nil
					end
					local obspush = hasfeature("empty","is","push",2,x+ox,y+oy) or (featureindex["pushes"] ~= nil and hasfeature(name,"pushes","empty",unitid,x,y))
					if (obspush == false) then
						obspush = nil
					end
					local obspull = hasfeature("empty","is","pull",2,x+ox,y+oy) or (featureindex["pulls"] ~= nil and hasfeature(name,"pulls","empty",unitid,x,y))
					if (obspull == false) then
						obspull = nil
					end

					obsstop, obspush, obspull = do_directional_collision(i-1, "empty", 2, obsstop, obspush, obspull, x,y,ox,oy, false)
					
					if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) then
						valid = false
					end
				end
				
				if valid then
					local newunit = copy(unit.fixed,x+ox,y+oy)
				end
			end
		end

	end
	arrow_prop_mod_globals.group_arrow_properties = true

	
	-- EDIT: Implement REPENT (cleanse the karma status of sinful objects)
	local isrepent = getunitswitheffect("repent",false,delthese)
	for id,unit in ipairs(isrepent) do
		if unit.karma then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local pmult,sound = checkeffecthistory("repent")
			MF_particles("unlock",x,y,5 * pmult,5,2,1,1)
			ws_setKarma(unit.fixed, false)
		end
	end
	
	-- EDIT: Destroy units by KARMA
	local iskarma = getunitswitheffect("karma",false,delthese) 
	for id,unit in ipairs(iskarma) do
		if unit.karma and (issafe(unit.fixed) == false) then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local pmult,sound = checkeffecthistory("karma")
			MF_particles("unlock",x,y,5 * pmult,2,2,1,1)
			removalshort = sound
			removalsound = 1
			generaldata.values[SHAKE] = 4
			table.insert(delthese, unit.fixed)
		end
	end
	
	-- EDIT: Implement LEVEL IS ENTER (disabled in the editor)
	-- LEVEL IS ENTER will attempt to enter the first opened inner level it finds
	-- Code *HEAVILY* based on the VISIT mod by btd456creeper
	if (editor.values[E_INEDITOR] == 0) then
		local isenter = getunitswitheffect("enter",false,delthese)
		local enterablelevels = {}
		for _,unit in ipairs(isenter) do
			if (WS_CAN_ENTER_ANY or (unit.strings[UNITNAME] == "level")) then
				-- Check if the level is open and points to a valid level
				if (string.len(unit.strings[U_LEVELFILE]) > 0) and (string.len(unit.strings[U_LEVELNAME]) > 0) and (generaldata.values[IGNORE] == 0) and (unit.values[COMPLETED] > 1) then
					table.insert(enterablelevels, unit)
				end
			end
		end
		if (#enterablelevels > 1) then -- DESTROY!! the level if trying to enter more than one level at once (something about tearing the fabric of reality i guess)
			destroylevel()
			destroylevel_do()
		else
			for _,unit in ipairs(enterablelevels) do
				-- I have no idea what's going on in here tbh, it works though?
				levelfile = unit.strings[U_LEVELFILE]
				levelnum = unit.values[VISUALLEVEL]
				leveltype = unit.values[VISUALSTYLE]

				findpersists() -- @Merge(Word Salad x Persist): support for persisting objects via LEVEL IS ENTER

				getlevelsurrounds(unit.fixed) -- Update the level surrounds (hopefully this doesn't break the game)
				sublevel(levelfile,tonumber(levelnum),tonumber(leveltype))
	
				----> These next six lines are just to switch the level while displaying the proper effects.
				----> I don't know what most of these lines do, they're taken from an example Hempuli posted once.
				----> (from: btd456creeper)
				-- I also have no idea what these do
				generaldata.values[TRANSITIONREASON] = 9
				generaldata.values[IGNORE] = 1
				generaldata3.values[STOPTRANSITION] = 1
				generaldata2.values[UNLOCK] = 0
				generaldata2.values[UNLOCKTIMER] = 0
				MF_loop("transition",1)
				
				break
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound,true)
	
	local isplay = getunitswithverb("play",delthese)
	
	for id,ugroup in ipairs(isplay) do
		local sound_freq = ugroup[1]
		local sound_units = ugroup[2]
		local sound_name = ugroup[3]
		
		if (#sound_units > 0) then
			local ptunes = play_data.tunes
			local pfreqs = play_data.freqs
			
			local tune = "beep"
			local freq = pfreqs[sound_freq] or 24000
			
			if (ptunes[sound_name] ~= nil) then
				tune = ptunes[sound_name]
			end
			
			-- MF_alert(sound_name .. " played at " .. tostring(freq) .. " (" .. sound_freq .. ")")
			
			MF_playsound_freq(tune,freq)
			setsoundname("turn",11,nil)
			
			if (sound_name ~= "empty") then
				for a,unit in ipairs(sound_units) do
					local x,y = unit.values[XPOS],unit.values[YPOS]
					
					MF_particles("music",unit.values[XPOS],unit.values[YPOS],1,0,3,3,1)
				end
			end
		end
	end
	
	if (generaldata.strings[WORLD] == "museum") then
		local ishold = getunitswitheffect("hold",false,delthese)
		local holders = {}
		
		for id,unit in ipairs(ishold) do
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local tileid = x + y * roomsizex
			holders[unit.values[ID]] = 1
			
			if (unitmap[tileid] ~= nil) then
				local water = findallhere(x,y)
				
				if (#water > 0) then
					for a,b in ipairs(water) do
						if floating(b,unit.fixed,x,y) then
							if (b ~= unit.fixed) then
								local bunit = mmf.newObject(b)
								addundo({"holder",bunit.values[ID],bunit.holder,unit.values[ID],},unitid)
								bunit.holder = unit.values[ID]
							end
						end
					end
				end
			end
		end
		
		for i,unit in ipairs(units) do
			if (unit.holder ~= nil) and (unit.holder ~= 0) then
				if (holders[unit.holder] ~= nil) then
					local unitid = getunitid(unit.holder)
					local bunit = mmf.newObject(unitid)
					local x,y = bunit.values[XPOS],bunit.values[YPOS]
					
					update(unit.fixed,x,y,unit.values[DIR])
				else
					addundo({"holder",unit.values[ID],unit.holder,0,},unitid)
					unit.holder = 0
				end
			else
				unit.holder = 0
			end
		end
	end
	
	local issink = getunitswitheffect("sink",false,delthese)
	
	for id,unit in ipairs(issink) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local tileid = x + y * roomsizex
		
		if (unitmap[tileid] ~= nil) then
			local water = findallhere(x,y)
			local sunk = false
			
			if (#water > 0) then
				for a,b in ipairs(water) do
					if floating(b,unit.fixed,x,y) then
						if (b ~= unit.fixed) then
							local dosink = true
							
							for c,d in ipairs(delthese) do
								if (d == unit.fixed) or (d == b) then
									dosink = false
								end
							end
							
							local safe1 = issafe(b)
							local safe2 = issafe(unit.fixed)
							
							if safe1 and safe2 then
								dosink = false
							end
							
							if dosink then
								generaldata.values[SHAKE] = 3
								
								if (safe1 == false) then
									table.insert(delthese, b)
									if safe2 or is_unit_guarded(unit.fixed) then -- EDIT: implement karma for SINK
										delthese,removalshort,removalsound = ws_karma(x,y,"sink",b,delthese,removalshort,removalsound)
									end
								end
								
								local pmult,sound = checkeffecthistory("sink")
								removalshort = sound
								removalsound = 3
								local c1,c2 = getcolour(unit.fixed)
								MF_particles("destroy",x,y,15 * pmult,c1,c2,1,1)
								
								if (b ~= unit.fixed) and (safe2 == false) then
									sunk = true
									if safe1 or is_unit_guarded(b) then -- EDIT: implement karma for SINK
										delthese,removalshort,removalsound = ws_karma(x,y,"sink2",unit.fixed,delthese,removalshort,removalsound)
									end
								end
							end
						end
					end
				end
			end
			
			if sunk then
				table.insert(delthese, unit.fixed)
			end
		end
	end
	
	local issinks = getunitswithverb("sinks",delthese)
	local issinksed = {}
	
	for id,ugroup in ipairs(issinks) do
		local v = ugroup[1]
		
		if (ugroup[3] ~= "empty") then
			for a,unit in ipairs(ugroup[2]) do
				local x,y = unit.values[XPOS],unit.values[YPOS]
				local things = findtype({v,nil},x,y,unit.fixed)
				local sunk = false

				-- @mods(patashu) - apparently in patashu's modpack, sinks doesn't work with safe. So additing that here.
				local sinker_is_safe = issafe(unit.fixed)

				if (#things > 0) then
					for a,b in ipairs(things) do
						if (issafe(b) == false) and floating(b,unit.fixed,x,y) and (b ~= unit.fixed) and (issinksed[b] == nil) then
							generaldata.values[SHAKE] = 4
							table.insert(delthese, b)
							sunk = true
							issinksed[b] = 1
							
							local pmult,sound = checkeffecthistory("sink")
							removalshort = sound
							removalsound = 3
							local c1,c2 = getcolour(unit.fixed)
							MF_particles("destroy",x,y,15 * pmult,c1,c2,1,1)

							if sinker_is_safe or is_unit_guarded(unit.fixed) then
								ws_setKarma(unit.fixed)
							end
						end
					end
				end
				if sunk and not sinker_is_safe then
					table.insert(delthese, unit.fixed)
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local isboom = getunitswitheffect("boom",false,delthese)
	-- @mods(turning text)
	arrow_prop_mod_globals.group_arrow_properties = false
	
	for id,unit in ipairs(isboom) do
		local ux,uy = unit.values[XPOS],unit.values[YPOS]
		local sunk = false
		local doeffect = true
		
		if (issafe(unit.fixed) == false) then
			sunk = true
		else
			doremovalsound = true
		end
		
		-- @mods(turning text)
		arrow_prop_mod_globals.group_arrow_properties = false
		local name = unit.strings[UNITNAME]
		local count = hasfeature_count(name,"is","boom",unit.fixed,ux,uy)
		local dim = math.min(count - 1, math.max(roomsizex, roomsizey))
		arrow_prop_mod_globals.group_arrow_properties = true

		local dir_booms, dir_iszero = do_directional_boom(unit)
		
		local locs = {}
		if count > 0 then -- @mods(turning text) - need this since you can go into the for loop without "X is boom". It could have only directional booms
			if (dim <= 0) then
				table.insert(locs, {0,0})
			else
				for g=-dim,dim do
					for h=-dim,dim do
						table.insert(locs, {g,h})
					end
				end
			end
		end
		
		local baseradius = dim
		if dim <= 0 then
			baseradius = 0
		end
		-- left boom
		for i=1,dir_booms[2] do
			table.insert(locs, {-(baseradius+i),0})
		end
		-- right boom
		for i=1,dir_booms[0] do
			table.insert(locs, {(baseradius+i),0})
		end
		-- up boom
		for i=1,dir_booms[1] do
			table.insert(locs, {0,-(baseradius+i)})
		end
		-- down boom
		for i=1,dir_booms[3] do
			table.insert(locs, {0,(baseradius+i)})
		end
		
		for a,b in ipairs(locs) do
			local g = b[1]
			local h = b[2]
			local x = ux + g
			local y = uy + h
			local tileid = x + y * roomsizex
			
			if (unitmap[tileid] ~= nil) and inbounds(x,y,1) then
				local water = findallhere(x,y)
				
				if (#water > 0) then
					for e,f in ipairs(water) do
						if floating(f,unit.fixed,x,y) then
							if (f ~= unit.fixed) then -- EDIT: set KARMA for BOOM objects, unless they're REPENT (can the code be better? probably)
								local doboom = true
								
								for c,d in ipairs(delthese) do
									if (d == f) then
										doboom = false
									elseif (d == unit.fixed) then
										sunk = false
									end
								end
								
								if (sunk == false or is_unit_guarded(unit.fixed)) and (issafe(f) == false) and not ws_isrepent(unit.fixed,ux,uy) then
									ws_setKarma(unit.fixed)
								end
								
								if doboom and (issafe(f) == false) then
									table.insert(delthese, f)
									MF_particles("smoke",x,y,4,0,2,1,1)
								end
							end
						end
					end
				end
			end
		end
		
		if doeffect then
			generaldata.values[SHAKE] = 6
			local pmult,sound = checkeffecthistory("boom")
			removalshort = sound
			removalsound = 1
			local c1,c2 = getcolour(unit.fixed)
			MF_particles("smoke",ux,uy,15 * pmult,c1,c2,1,1)
		end
		
		if sunk then
			table.insert(delthese, unit.fixed)
		end
	end
	arrow_prop_mod_globals.group_arrow_properties = true
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local iscut = getunitswitheffect("cut",false,delthese)
	
	-- Note: because we do not want CUT to trigger HAS, we need to avoid calling delete(), which is done in handledels().
	-- Fortunetly, handledels returns an empty table as the new delthese, meaning that each block of populating delthese and then
	-- calling handledels() is self contained, if in an awkward and repetative way. Since handle_text_cutting() does its own
	-- deletion without triggering HAS, it should be fine given that the above is true. 
	for id,unit in ipairs(iscut) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local texts = findtext(x,y)

        for _, textunitid in ipairs(texts) do
            if textunitid ~= unit.fixed then
				local textunit = mmf.newObject(textunitid)
				local cutdata = check_text_cutting(unit.fixed, textunitid, false)
				if cutdata then
                    local dir = textunit.values[DIR]
                    handle_text_cutting(cutdata, dir)
                end            
            end
        end
    end
	
	local isweak = getunitswitheffect("weak",false,delthese)
	
	for id,unit in ipairs(isweak) do
		if (issafe(unit.fixed) == false) and (unit.new == false) then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local stuff = findallhere(x,y)
			
			if (#stuff > 0) then
				for i,v in ipairs(stuff) do
					if floating(v,unit.fixed,x,y) then
						local vunit = mmf.newObject(v)
						local thistype = vunit.strings[UNITTYPE]
						if (v ~= unit.fixed) then
							local pmult,sound = checkeffecthistory("weak")
							MF_particles("destroy",x,y,5 * pmult,0,3,1,1)
							removalshort = sound
							removalsound = 1
							generaldata.values[SHAKE] = 4
							table.insert(delthese, unit.fixed)
							delthese,removalshort,removalsound = ws_karma(x,y,"weak",unit.fixed,delthese,removalshort,removalsound) -- EDIT: implement karma for WEAK overlap
							break
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound,true)
	
	local ismelt = getunitswitheffect("melt",false,delthese)
	
	for id,unit in ipairs(ismelt) do
		local hot = findfeature(nil,"is","hot")
		local x,y = unit.values[XPOS],unit.values[YPOS]
		
		if (hot ~= nil) then
			for a,b in ipairs(hot) do
				local lava = findtype(b,x,y,0)
			
				if (#lava > 0) and (issafe(unit.fixed) == false) then
					for c,d in ipairs(lava) do
						if floating(d,unit.fixed,x,y) then
							local pmult,sound = checkeffecthistory("hot")
							MF_particles("smoke",x,y,5 * pmult,0,1,1,1)
							generaldata.values[SHAKE] = 5
							removalshort = sound
							removalsound = 9
							table.insert(delthese, unit.fixed)
							delthese,removalshort,removalsound = ws_karma(x,y,"melt",unit.fixed,delthese,removalshort,removalsound) -- EDIT: implement karma for MELT
							break
						end
					end
				end
			end
		end
	end
	
	local ismelts = getunitswithverb("melts",delthese)
	
	for id,ugroup in ipairs(ismelts) do
		local v = ugroup[1]
		
		if (ugroup[3] ~= "empty") then
			for a,unit in ipairs(ugroup[2]) do
				local x,y = unit.values[XPOS],unit.values[YPOS]
				local things = findtype({v,nil},x,y,0)
				
				if (#things > 0) then
					for a,b in ipairs(things) do
						if (issafe(b) == false) and floating(b,unit.fixed,x,y) then
							local pmult,sound = checkeffecthistory("hot")
							MF_particles("smoke",x,y,5 * pmult,0,1,1,1)
							generaldata.values[SHAKE] = 5
							removalshort = sound
							removalsound = 9
							table.insert(delthese, b)

							-- @mods(word salad x patashu): implement karma for MELTS. The thing that "melts" another is to blame.
							ws_setKarma(unit.fixed)
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound,true)
	
	local isyou = getunitswitheffect("you",false,delthese)
	local isyou2 = getunitswitheffect("you2",false,delthese)
	local isyou3 = getunitswitheffect("3d",false,delthese)
	local isyou4 = getunitswitheffect("alive",false,delthese) -- EDIT: add ALIVE units
	
	for i,v in ipairs(isyou2) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou3) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou4) do
		table.insert(isyou, v)
	end
	
	for id,unit in ipairs(isyou) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local defeat = findfeature(nil,"is","defeat")
		local defeats = findfeature(nil,"defeats",getname(unit));
		
		if (defeat ~= nil) then
			for a,b in ipairs(defeat) do
				if (b[1] ~= "empty") then
					local skull = findtype(b,x,y,0)
					
					if (#skull > 0) and (issafe(unit.fixed) == false) then
						for c,d in ipairs(skull) do
							local doit = false
							
							if (d ~= unit.fixed) then
								if floating(d,unit.fixed,x,y) then
									local kunit = mmf.newObject(d)
									local kname = kunit.strings[UNITNAME]
									
									local weakskull = hasfeature(kname,"is","weak",d)
									
									if (weakskull == nil) or ((weakskull ~= nil) and issafe(d)) then
										doit = true
									end
								end
							else
								doit = true
							end
							
							if doit then
								local pmult,sound = checkeffecthistory("defeat")
								MF_particles("destroy",x,y,5 * pmult,0,3,1,1)
								generaldata.values[SHAKE] = 5
								removalshort = sound
								removalsound = 1
								table.insert(delthese, unit.fixed)
								delthese,removalshort,removalsound = ws_karma(x,y,"defeat",unit.fixed,delthese,removalshort,removalsound) -- EDIT: implement karma for DEFEAT
							end
						end
					end
				end
			end
		end
		if (defeats ~= nil) then
			for a,b in ipairs(defeats) do
				if (b[1] ~= "empty") then
					local skull = findtype(b,x,y,0)
					
					if (#skull > 0) and (issafe(unit.fixed) == false) then
						for c,d in ipairs(skull) do
							local doit = false
							
							if (d ~= unit.fixed) then
								if floating(d,unit.fixed,x,y) then
									local kunit = mmf.newObject(d)
									local kname = getname(kunit)
									
									local weakskull = hasfeature(kname,"is","weak",d)
									
									if (weakskull == nil) or ((weakskull ~= nil) and issafe(d)) then
										doit = true
									end
								end
							else
								doit = true
							end
							
							if doit then
								local pmult,sound = checkeffecthistory("defeat")
								MF_particles("destroy",x,y,5 * pmult,0,3,1,1)
								generaldata.values[SHAKE] = 5
								removalshort = sound
								removalsound = 1
								table.insert(delthese, unit.fixed)

								-- @mods(word salad x patashu): implement karma for DEFEATS. The thing that "defeats" another is to blame.
								ws_setKarma(d)
							end
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local isshut = getunitswitheffect("shut",false,delthese)
	
	for id,unit in ipairs(isshut) do
		local open = findfeature(nil,"is","open")
		local x,y = unit.values[XPOS],unit.values[YPOS]
		
		if (open ~= nil) then
			for i,v in ipairs(open) do
				local key = findtype(v,x,y,0)
				
				if (#key > 0) then
					local doparts = false
					for a,b in ipairs(key) do -- EDIT: add KARMA for OPEN/SHUT
						if (b ~= 0) and floating(b,unit.fixed,x,y) then
							local isKeyUnsafe = (b ~= unit.fixed) and (issafe(b) == false)
							local isDoorUnsafe = (issafe(unit.fixed) == false)
							
							if isDoorUnsafe then
								generaldata.values[SHAKE] = 8
								table.insert(delthese, unit.fixed)
								doparts = true
								online = false
							elseif isKeyUnsafe and not ws_isrepent(unit.fixed,x,y) then -- Door is safe, key isn't
								ws_setKarma(unit.fixed) -- Set the karma of the door unless it's REPENT
							end
							
							if isKeyUnsafe then
								table.insert(delthese, b)
								doparts = true
							elseif isDoorUnsafe and not ws_isrepent(b,x,y) then -- Key is safe, door isn't
								ws_setKarma(b) -- Set the karma of the key unless it's REPENT
							end
							
							if doparts then
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("turn",7,sound)
								MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
							end
							
							break
						end
					end
				end
			end
		end
	end
	
	local isopens = getunitswithverb("opens",delthese)
	
	for id,ugroup in ipairs(isopens) do
		local v = ugroup[1]
		
		if (ugroup[3] ~= "empty") then
			for a,unit in ipairs(ugroup[2]) do
				local x,y = unit.values[XPOS],unit.values[YPOS]
				local things = findtype({v,nil},x,y,0)
				local sunk = false
				if (#things > 0) then
					local doparts = false
					for a,b in ipairs(things) do  -- @mods(word salad x patashu): add KARMA for OPENs/SHUT
						if (b ~= 0) and floating(b,unit.fixed,x,y) then
							local isKeyUnsafe = (b ~= unit.fixed) and (issafe(b) == false)
							local isDoorUnsafe = (issafe(unit.fixed) == false)
							
							if isDoorUnsafe then
								generaldata.values[SHAKE] = 8
								table.insert(delthese, unit.fixed)
								doparts = true
								online = false
							elseif isKeyUnsafe then -- Door is safe, key isn't
								ws_setKarma(unit.fixed)
							end
							
							if isKeyUnsafe then
								table.insert(delthese, b)
								doparts = true
							elseif isDoorUnsafe then -- Key is safe, door isn't
								ws_setKarma(b)
							end
							
							if doparts then
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("turn",7,sound)
								MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
							end
							
							break
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local iseat = getunitswithverb("eat",delthese)
	local iseaten = {}
	
	for id,ugroup in ipairs(iseat) do
		local v = ugroup[1]
		
		if (ugroup[3] ~= "empty") then
			for a,unit in ipairs(ugroup[2]) do
				local x,y = unit.values[XPOS],unit.values[YPOS]
				local things = findtype({v,nil},x,y,unit.fixed)
				
				if (#things > 0) then
					for a,b in ipairs(things) do
						if (issafe(b) == false) and floating(b,unit.fixed,x,y) and (b ~= unit.fixed) then -- EDIT: implement karma for EAT
							if (iseaten[b] == nil) then
							generaldata.values[SHAKE] = 4
							table.insert(delthese, b)
							
							iseaten[b] = 1
							
							local pmult,sound = checkeffecthistory("eat")
							MF_particles("eat",x,y,5 * pmult,0,3,1,1)
							removalshort = sound
							removalsound = 1
						end
							delthese,removalshort,removalsound = ws_setKarmaOrDestroy(x,y,unit.fixed,delthese,removalshort,removalsound) -- This already checks for REPENT
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	if (small == false) then
		local ismake = getunitswithverb("make",delthese)
		
		for id,ugroup in ipairs(ismake) do
			local v = ugroup[1]
			
			for a,unit in ipairs(ugroup[2]) do
				local x,y,dir,name = 0,0,4,""
				
				local leveldata = {}
				
				if (ugroup[3] ~= "empty") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					name = getname(unit)
					leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
				else
					x = math.floor(unit % roomsizex)
					y = math.floor(unit / roomsizex)
					name = "empty"
					dir = emptydir(x,y)
				end
				
				if (dir == 4) then
					dir = fixedrandom(0,3)
				end
				
				local exists = false
				
				if (v ~= "text") and (v ~= "all") then
					for b,mat in pairs(objectlist) do
						if (b == v) then
							exists = true
						end
					end
				else
					exists = true
				end
				
				if exists then
					local domake = true
					
					if (name ~= "empty") then
						local thingshere = findallhere(x,y)
						
						if (#thingshere > 0) then
							for a,b in ipairs(thingshere) do
								local thing = mmf.newObject(b)
								local thingname = thing.strings[UNITNAME]
								
								if (thing.flags[CONVERTED] == false) and ((thingname == v) or ((thing.strings[UNITTYPE] == "text") and (v == "text"))) then
									domake = false
								end
							end
						end
					end
					
					if domake then
						if (findnoun(v,nlist.short) == false) then
							create(v,x,y,dir,x,y,nil,nil,leveldata)
						elseif (v == "text") then
							if (name ~= "text") and (name ~= "all") then
								create("text_" .. name,x,y,dir,x,y,nil,nil,leveldata)
								updatecode = 1
							end
						elseif (string.sub(v, 1, 5) == "group") then
							--[[
							local mem = findgroup(v)
							
							for c,d in ipairs(mem) do
								local thishere = findtype({d},x,y,nil,true)
								
								if (#thishere == 0) then
									create(d,x,y,dir,x,y,nil,nil,leveldata)
								end
							end
							]]--
						end
					end
				end
			end
		end
		
		local isprint = getunitswithverb("print",delthese)
		
		for id,ugroup in ipairs(isprint) do
			local v = ugroup[1]
			v = "text_" .. v
			
			for a,unit in ipairs(ugroup[2]) do
				local x,y,dir,name = 0,0,4,""
				
				local leveldata = {}
				
				if (ugroup[3] ~= "empty") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					name = getname(unit)
					leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
				else
					x = math.floor(unit % roomsizex)
					y = math.floor(unit / roomsizex)
					name = "empty"
					dir = emptydir(x,y)
				end
				
				if (dir == 4) then
					dir = fixedrandom(0,3)
				end
				
				if unitreference[v] ~= nil then
					local domake = true
					
					if (name ~= "empty") then
						local thingshere = findallhere(x,y)
						
						if (#thingshere > 0) then
							for a,b in ipairs(thingshere) do
								local thing = mmf.newObject(b)
								local thingname = thing.strings[UNITNAME]
								
								if (thing.flags[CONVERTED] == false) and ((thingname == v) or ((thing.strings[UNITTYPE] == "text") and (v == "text"))) then
									domake = false
								end
							end
						end
					end
					
					if domake then
						if (findnoun(v,nlist.short) == false) then
							create(v,x,y,dir,x,y,nil,nil,leveldata)
						elseif (v == "text") then
							if (name ~= "text") and (name ~= "all") then
								create("text_" .. name,x,y,dir,x,y,nil,nil,leveldata)
								updatecode = 1
							end
						elseif (string.sub(v, 1, 5) == "group") then
							--[[
							local mem = findgroup(v)
							
							for c,d in ipairs(mem) do
								local thishere = findtype({d},x,y,nil,true)
								
								if (#thishere == 0) then
									create(d,x,y,dir,x,y,nil,nil,leveldata)
								end
							end
							]]--
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	isyou = getunitswitheffect("you",false,delthese)
	isyou2 = getunitswitheffect("you2",false,delthese)
	isyou3 = getunitswitheffect("3d",false,delthese)
	isyou4 = getunitswitheffect("alive",false,delthese) -- EDIT: add ALIVE units
	
	for i,v in ipairs(isyou2) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou3) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou4) do
		table.insert(isyou, v)
	end
	
	for id,unit in ipairs(isyou) do
		if (unit.flags[DEAD] == false) and (delthese[unit.fixed] == nil) then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			
			if (small == false) then
				local bonus = findfeature(nil,"is","bonus")
				
				if (bonus ~= nil) then
					for a,b in ipairs(bonus) do
						if (b[1] ~= "empty") then
							local flag = findtype(b,x,y,0)
							
							if (#flag > 0) then
								for c,d in ipairs(flag) do
									if floating(d,unit.fixed,x,y) then
										local pmult,sound = checkeffecthistory("bonus")
										MF_particles("bonus",x,y,10 * pmult,4,1,1,1)
										removalshort = sound
										removalsound = 2
										MF_playsound("bonus")
										MF_bonus(1)
										addundo({"bonus",1})
										
										if (issafe(d,x,y) == false) then
											generaldata.values[SHAKE] = 5
											table.insert(delthese, d)
											delthese,removalshort,removalsound = ws_karma(x,y,"bonus",d,delthese,removalshort,removalsound) -- EDIT: add karma for BONUS
										end
									end
								end
							end
						end
					end
				end
				
				local ending = findfeature(nil,"is","end")
				
				if (ending ~= nil) then
					for a,b in ipairs(ending) do
						if (b[1] ~= "empty") then
							local flag = findtype(b,x,y,0)
							
							if (#flag > 0) then
								for c,d in ipairs(flag) do
									if floating(d,unit.fixed,x,y) and (generaldata.values[MODE] == 0) then
										if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) then
											MF_particles("unlock",x,y,10,1,4,1,1)
											MF_end(unit.fixed,d)
											break
										elseif (editor.values[INEDITOR] ~= 0) then
											local pmult = checkeffecthistory("win")
									
											MF_particles("win",x,y,10 * pmult,2,4,1,1)
											MF_end_single()
											MF_win()
											break
										else
											local pmult = checkeffecthistory("win")
											
											local mods_run = do_mod_hook("levelpack_end", {})
											
											if (mods_run == false) then
												MF_particles("win",x,y,10 * pmult,2,4,1,1)
												MF_end_single()
												MF_win()
												MF_credits(1)
											end
											break
										end
									end
								end
							end
						end
					end
				end
			end
			
			local reset = findfeature(nil,"is","reset")

			if (reset ~= nil) and not doreset then
				for a,b in ipairs(reset) do
					if (b[1] ~= "empty") then
						local flag = findtype(b,x,y,0)
						if (#flag > 0) then
							for c,d in ipairs(flag) do
								if floating(d,unit.fixed,x,y) then
									doreset = true
									break
								end
							end
						end
					end
				end
			end

			--Roughly copied from the Win code, this handles when a You object touches a Visit object, triggering a visit.
			local shoulddovisit = false
			local dovisitdir
			local visit = findfeature(nil,"is","visit")
			if (visit ~= nil) then
				for a,b in ipairs(visit) do
					if (b[1] ~= "empty") then
						local flag = findtype(b,x,y,0)
						if (#flag > 0) then
							for c,d in ipairs(flag) do
								if floating(d,unit.fixed,x,y) and (hasfeature(b[1],"is","done",d,x,y) == nil) and (hasfeature(b[1],"is","end",d,x,y) == nil) then
									shoulddovisit = true
									dovisitdir = (mmf.newObject(d).values[DIR])
									break
								end
							end
						end
					end
					if shoulddovisit then
						break
					end
				end
			end
			if shoulddovisit then
				dovisit(dovisitdir)
				break
			end

			local win = findfeature(nil,"is","win")
			
			if (win ~= nil) then
				for a,b in ipairs(win) do
					if (b[1] ~= "empty") then
						local flag = findtype(b,x,y,0)
						if (#flag > 0) then
							for c,d in ipairs(flag) do
								if floating(d,unit.fixed,x,y) and (hasfeature(b[1],"is","done",d,x,y) == nil) and (hasfeature(b[1],"is","end",d,x,y) == nil) then
									local pmult = checkeffecthistory("win")
									
									MF_particles("win",x,y,10 * pmult,2,4,1,1)
									MF_win()
									break
								end
							end
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	for i,unit in ipairs(units) do
		if (inbounds(unit.values[XPOS],unit.values[YPOS],1) == false) then
			--MF_alert("DELETED!!!")
			table.insert(delthese, unit.fixed)
		end
	end
	
	GLOBAL_disable_guard_checking = true
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	GLOBAL_disable_guard_checking = false
	
	if (small == false) then
		local iscrash = getunitswitheffect("crash",false,delthese)
		
		if (#iscrash > 0) then
			HACK_INFINITY = 200
			destroylevel("infinity")
			return
		end
	end
	
	if doremovalsound then
		setsoundname("removal",removalsound,removalshort)
	end
end

function levelblock()
	--[[ 
		@mods(text splicing) - Override reason: provide hook for handling level cutting. Levelblock is mainly for handling "X on level" interactions.
		@mods(guard) - Override reason: handle level guarding
	]]
	local unlocked = false
	local things = {}
	local donethings = {}
	local delthese = {}
	local edelthese = {}
	local emptythings = {}
	local level_obj = plasma_utils.make_object(1)
	
	if (destroylevel_check == false) then
		if (featureindex["level"] ~= nil) then
			for i,v in ipairs(featureindex["level"]) do
				table.insert(things, v)
			end
		end
		
		if (featureindex["empty"] ~= nil) then
			for i,v in ipairs(featureindex["empty"]) do
				local rule = v[1]
				
				if (rule[1] == "empty") and ((rule[2] == "is") or (rule[2] == "eat") or (rule[2] == "defeats") or (rule[2] == "melts") or (rule[2] == "opens")) then
					table.insert(emptythings, v)
				end
			end
		end
		
		local lstill = isstill_or_locked(1,nil,nil,mapdir)
		local lsleep = issleep(1)
		local lsafe = issafe(1)
		local lkarma = hasfeature("level","is","karma",1) or false -- EDIT: check if the level is karma or repent
		local lrepent = hasfeature("level","is","repent",1) or false
		local emptybonus = false
		local emptydone = false
		
		local ewintiles = {}
		local eendtiles = {}
		
		local levelteledone = 0
		
		if (#emptythings > 0) then
			for i=1,roomsizex-2 do
				for j=1,roomsizey-2 do
					local tileid = i + j * roomsizex
					
					if (unitmap[tileid] == nil) or (#unitmap[tileid] == 0) then
						local esafe = issafe(2,i,j)
						
						--MF_alert(tostring(i) .. ", " .. tostring(j))
						local keypair = ""
						local visitpair = ""
						local winpair = ""
						local hotpair = ""
						local defeatpair = ""
						local bonuspair = ""
						local endpair = ""
						
						local canmelt = false
						local candefeat = false
						local canvisit = false
						local canwin = false
						local canbonus = false
						local canend = false
						
						local unlock = false
						local visiting = false
						local victory = false
						local melt = false
						local defeat = false
						local bonus = false
						local ending = false
						local emptyboom = false
						
						local visitdir = 0
						
						for a,rules in ipairs(emptythings) do
							local rule = rules[1]
							local conds = rules[2]
							
							if (rule[2] == "melts" and rule[3] == "empty" and testcond(conds,2,i,j)) then
								melt = true
							end
							if (rule[2] == "opens" and rule[3] == "empty" and testcond(conds,2,i,j)) then
								unlock = true
							end
							if (rule[2] == "defeats" and rule[3] == "empty") and testcond(conds,2,i,j) then
								if (string.len(defeatpair) == 0) then
									defeatpair = "you"
								elseif (defeatpair == "defeat") then
									defeat = true
								end
							elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
								candefeat = true
								canwin = true
								
								if (string.len(defeatpair) == 0) then
									defeatpair = "defeat"
								elseif (defeatpair == "you") then
									defeat = true
								end
							end
							
							if (rule[2] == "is") then
								if (rule[3] == "open") and testcond(conds,2,i,j) then
									if (string.len(keypair) == 0) then
										keypair = "shut"
									elseif (keypair == "open") then
										unlock = true
									end
								elseif (rule[3] == "shut") and testcond(conds,2,i,j) then
									if (string.len(keypair) == 0) then
										keypair = "open"
									elseif (keypair == "shut") then
										unlock = true
									end
								end
								
								if (rule[3] == "melt") and testcond(conds,2,i,j) then
									canmelt = true
									
									if (string.len(hotpair) == 0) then
										hotpair = "hot"
									elseif (hotpair == "melt") then
										melt = true
									end
								elseif (rule[3] == "hot") and testcond(conds,2,i,j) then
									if (string.len(hotpair) == 0) then
										hotpair = "melt"
									elseif (hotpair == "hot") then
										melt = true
									end
								end
								
								if (rule[3] == "defeat") and testcond(conds,2,i,j) then
									if (string.len(defeatpair) == 0) then
										defeatpair = "you"
									elseif (defeatpair == "defeat") then
										defeat = true
									end
								elseif ws_isPlayerProp(rule[3]) and testcond(conds,2,i,j) then -- EDIT: Replace long check with "Is player property" function
									candefeat = true
									canwin = true
									
									if (string.len(defeatpair) == 0) then
										defeatpair = "defeat"
									elseif (defeatpair == "you") then
										defeat = true
									end
								end
								
								if (rule[3] == "win") and testcond(conds,2,i,j) then
									if (string.len(winpair) == 0) then
										winpair = "you"
									elseif (winpair == "win") then
										victory = true
									end
								elseif ws_isPlayerProp(rule[3]) and testcond(conds,2,i,j) then -- EDIT: Same as above
									candefeat = true
									canwin = true
									
									if (string.len(winpair) == 0) then
										winpair = "win"
									elseif (winpair == "you") then
										victory = true
									end
								end
								
								if (rule[3] == "visit") and testcond(conds,2,i,j) then
									if (string.len(visitpair) == 0) then
										visitpair = "you"
									elseif (visitpair == "visit") then
										visiting = true
										visitdir = emptydir(i,j,true)
									end
								elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
									candefeat = true
									canvisit = true
									
									if (string.len(visitpair) == 0) then
										visitpair = "visit"
									elseif (visitpair == "you") then
										visiting = true
										visitdir = emptydir(i,j,true)
									end
								end

								if (rule[3] == "bonus") and testcond(conds,2,i,j) then
									if (string.len(bonuspair) == 0) then
										bonuspair = "you"
									elseif (bonuspair == "bonus") then
										bonus = true
									end
									
									canbonus = true
								elseif ws_isPlayerProp(rule[3]) and testcond(conds,2,i,j) then -- EDIT: Same as above
									if (string.len(bonuspair) == 0) then
										bonuspair = "bonus"
									elseif (bonuspair == "you") then
										bonus = true
									end
								end
								
								if (rule[3] == "end") and testcond(conds,2,i,j) then
									if (string.len(endpair) == 0) then
										endpair = "you"
									elseif (bonuspair == "end") then
										ending = true
									end
									
									canend = true
								elseif ws_isPlayerProp(rule[3]) and testcond(conds,2,i,j) then -- EDIT: Same as above
									if (string.len(endpair) == 0) then
										endpair = "end"
									elseif (endpair == "you") then
										ending = true
									end
								end
								
								if (rule[3] == "done") and testcond(conds,2,i,j) then
									emptydone = true
								end
								
								if (rule[3] == "boom") and testcond(conds,2,i,j) then
									emptyboom = true
								end
								
								if (keypair == "shut") and (hasfeature("level","is","shut",1,i,j) ~= nil) and floating_level(2,i,j) then
									unlock = true
								elseif (keypair == "open") and (hasfeature("level","is","open",1,i,j) ~= nil) and floating_level(2,i,j) then
									unlock = true
								end
								
								if canmelt and (hasfeature("level","is","hot",1,i,j) ~= nil) and floating_level(2,i,j) then
									melt = true
								end
								
								if candefeat and (hasfeature("level","is","defeat",1,i,j) ~= nil) and floating_level(2,i,j) then
									defeat = true
								end
								
								if canvisit and (hasfeature("level","is","visit",1,i,j) ~= nil) and floating_level(2,i,j) then
									visiting = true
									visitdir = mapdir
								end

								if canwin and (hasfeature("level","is","win",1,i,j) ~= nil) and floating_level(2,i,j) then
									victory = true
								end
								
								if canbonus and ws_isLevelPlayer(i,j) and floating_level(2,i,j) then -- EDIT: Replaced long check with function
									bonus = true
								end
								
								if canend and ws_isLevelPlayer(i,j) and floating_level(2,i,j) then -- EDIT: Same as above
									ending = true
								end
								
								if victory then
									table.insert(ewintiles, {i,j})
								end
								
								if ending then
									table.insert(eendtiles, {i,j})
								end
							elseif (rule[2] == "eat") and (rule[3] == "level") and (lsafe == false) then
								if testcond(conds,2,i,j) and floating_level(2,i,j) then
									local pmult,sound = checkeffecthistory("eat")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							elseif (rule[2] == "melts") and (rule[3] == "level") and (lsafe == false) then
								if testcond(conds,2,i,j) and floating_level(2,i,j) then
									local pmult,sound = checkeffecthistory("hot")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							elseif (rule[2] == "opens") and (rule[3] == "level") and (lsafe == false) then
								if testcond(conds,2,i,j) and floating_level(2,i,j) then
									local pmult,sound = checkeffecthistory("unlock")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							elseif (rule[2] == "defeats") and (rule[3] == "level") and (lsafe == false) and ((hasfeature("level","is","you",1,i,j) ~= nil) or (hasfeature("level","is","you2",1,i,j) ~= nil) or (hasfeature("level","is","3d",1,i,j) ~= nil)) then
								if testcond(conds,2,i,j) and floating_level(2,i,j) then
									local pmult,sound = checkeffecthistory("defeat")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
						
						if emptyboom then
							local count = hasfeature_count("empty","is","boom",2,i,j)
							local dim = math.min(count - 1, math.max(roomsizex, roomsizey))
		
							local locs = {}
							if (dim <= 0) then
								table.insert(locs, {0,0})
							else
								for g=-dim,dim do
									for h=-dim,dim do
										table.insert(locs, {g,h})
									end
								end
							end
							
							for a,b in ipairs(locs) do
								local g = b[1]
								local h = b[2]
								local x = i + g
								local y = j + h
								local tileid = x + y * roomsizex
								
								if (unitmap[tileid] ~= nil) and inbounds(x,y,1) then
									local water = findallhere(x,y)
									
									if (#water > 0) then
										for e,f in ipairs(water) do
											if floating(f,2,x,y) then
												local doboom = true
												
												for c,d in ipairs(delthese) do
													if (d == f) then
														doboom = false
													end
												end
												
												if doboom and (issafe(f) == false) then
													table.insert(delthese, f)
													MF_particles("smoke",x,y,4,0,2,1,1)
												end
											end
										end
									end
								end
							end
							
							local pmult,sound = checkeffecthistory("boom")
							MF_particles("smoke",i,j,2 * pmult,0,3,1,1)
							setsoundname("removal",1)
							
							if (esafe == false) then
								table.insert(edelthese, {i,j})
							end
						end
						
						local alive = true
						
						if unlock and (esafe == false) and alive then
							setsoundname("turn",7)
							
							if (math.random(1,4) == 1) then
								MF_particles("unlock",i,j,1,2,4,1,1)
							end
							
							alive = false
							
							table.insert(edelthese, {i,j})
						end
						
						if melt and (esafe == false) and alive then
							setsoundname("turn",9)
							
							if (math.random(1,4) == 1) then
								MF_particles("smoke",i,j,1,0,1,1,1)
							end
							
							alive = false
							table.insert(edelthese, {i,j})
						end
						
						if defeat and (esafe == false) and alive then
							setsoundname("turn",1)
							
							if (math.random(1,4) == 1) then
								MF_particles("destroy",i,j,1,0,3,1,1)
							end
							
							alive = false
							table.insert(edelthese, {i,j})
						end
						
						if bonus and (esafe == false) then
							if alive then
							setsoundname("turn",2)
							
							if (math.random(1,4) == 1) then
								MF_particles("win",i,j,1,4,2,1,1)
							end
							
								alive = false
								table.insert(edelthese, {i,j})
							end
							
							if (emptybonus == false) then
								MF_playsound("bonus")
								MF_bonus(1)
								addundo({"bonus",1})
								emptybonus = true
							end
						end
						
						if visiting and alive then
							dovisit(visitdir)
							return
						end
						
						if victory and alive then
							MF_win()
							return
						end
						
						if ending and alive and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
							if (editor.values[INEDITOR] ~= 0) then
								MF_end_single()
								MF_win()
								return
							else
								MF_end_single()
								MF_win()
								MF_credits(1)
								return
							end
						end
					end
				end
			end
		end
		
		if emptydone then
			local donenum = math.random(1,4)
			MF_playsound("done" .. tostring(donenum))
		end
		
		for a,b in ipairs(delthese) do
			local bunit = mmf.newObject(b)
			delete(b,bunit.values[XPOS],bunit.values[YPOS])
		end
		
		for a,b in ipairs(edelthese) do
			delete(2,b[1],b[2])
		end
		
		if (#ewintiles > 0) then
			for a,b in ipairs(ewintiles) do
				local i,j = b[1],b[2]
				local tileid = i + j * roomsizex
				if (unitmap[tileid] == nil) or (#unitmap[tileid] == 0) then
					MF_win()
					return
				end
			end
		end
		
		if (#eendtiles > 0) and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
			for a,b in ipairs(eendtiles) do
				local i,j = b[1],b[2]
				local tileid = i + j * roomsizex
				if (unitmap[tileid] == nil) or (#unitmap[tileid] == 0) then
					if (editor.values[INEDITOR] ~= 0) then
						MF_end_single()
						MF_win()
						return
					else
						MF_end_single()
						MF_win()
						MF_credits(1)
						return
					end
				end
			end
		end
		
		if (#things > 0) then
			for i,rules in ipairs(things) do
				local rule = rules[1]
				local conds = rules[2]
				
				--MF_alert(rule[1] .. " " .. rule[2] .. " " .. rule[3] .. ", " .. tostring(testcond(conds,1)))
				
				-- EDIT: cleanse the level karma if it's REPENT
				if levelKarma and lrepent then
					ws_setLevelKarma(false)
				end
				
				if levelKarma and lkarma and (lsafe == false) then -- EDIT: destroy KARMA levels if the karma flag is true and the repent flag is false
					destroylevel()
					return
				end
				
				if (rule[2] == "eat") then
					local eaten = {}
					
					if (rule[1] == "level") and testcond(conds,1) then
						local target = rule[3]
						
						if (target ~= "all") and (target ~= "empty") then
							local dothese = {}
							
							if (string.sub(target, 1, 5) ~= "group") then
								dothese = {target}
							else
								dothese = findgroup(target)
							end
							
							local destroyedSomething = false
							
							for c,d in ipairs(dothese) do
								if (unitlists[d] ~= nil) then
									if (d == "level") and (#unitlists["level"] > 0) and (lsafe == false) then
										local pmult,sound = checkeffecthistory("eat")
										setsoundname("removal",1,sound)
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
									
									for a,unitid in ipairs(unitlists[d]) do -- EDIT: set level karma when destroying something (LEVEL EAT X)
										if (issafe(unitid) == false) then
											destroyedSomething = true
											table.insert(eaten, unitid)
										end
									end
									
									if destroyedSomething and not lrepent then -- Do nothing if level is REPENT; destroy the level if it's KARMA and not SAFE; set karma status otherwise
										local is_guarded = ack_endangered_unit(level_obj)
										if lkarma and (lsafe == false) and not is_guarded then
											destroylevel()
											return
										else
											ws_setLevelKarma()
										end
									end
								end
							end
						elseif (target == "empty") then
							local empties = findempty()
							
							for a,b in ipairs(empties) do
								local x = b % roomsizex
								local y = math.floor(b / roomsizex)
								
								generaldata.values[SHAKE] = 4
							
								local pmult,sound = checkeffecthistory("eat")
								MF_particles("eat",x,y,5 * pmult,0,3,1,1)
								setsoundname("removal",1,sound)
								
								delete(2,x,y)
							end
						end
					elseif (rule[1] ~= "level") and (rule[3] == "level") then
						local dothese = {}
						if (findnoun(rule[1]) == false) then
							dothese = findall({rule[1],conds},nil,true)
						elseif (rule[1] == "empty") then
							dothese = findempty(conds,true)
						end
							
						if (#dothese > 0) and (lsafe == false) then
							local pmult,sound = checkeffecthistory("eat")
							setsoundname("removal",1,sound)
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
								return
							end
						end
					end
						
					for a,b in ipairs(eaten) do
						local bunit = mmf.newObject(b)
						local x,y = bunit.values[XPOS],bunit.values[YPOS]
						generaldata.values[SHAKE] = 4
						
						local pmult,sound = checkeffecthistory("eat")
						MF_particles("eat",x,y,5 * pmult,0,3,1,1)
						setsoundname("removal",1,sound)
						
						delete(b,x,y)
					end
				end
				
				--TODO for all of these: check floating and safe
				--base off of level eat x/x eat level?
				if (rule[1] == "level") and (rule[2] == "melts") and testcond(conds,1) then
					if (rule[3] == "level" and lsafe == false) then
						local pmult,sound = checkeffecthistory("hot")
						setsoundname("removal",1,sound)
						local is_guarded = ack_endangered_unit(level_obj)
						if not is_guarded then
							destroylevel()
							return
						end
					else
						local eaten = {}
						local target = rule[3]
							
						if (target ~= "all") and (target ~= "empty") then
							local dothese = {}
							
							if (string.sub(target, 1, 5) ~= "group") then
								dothese = {target}
							else
								dothese = findgroup(target)
							end
							
							for c,d in ipairs(dothese) do
								if (unitlists[d] ~= nil) then
									for a,unitid in ipairs(unitlists[d]) do
										if (issafe(unitid) == false) then
											table.insert(eaten, unitid)
										end
									end
								end
							end
						elseif (target == "empty") then
							local empties = findempty()
							
							for a,b in ipairs(empties) do
								local x = b % roomsizex
								local y = math.floor(b / roomsizex)

								local pmult,sound = checkeffecthistory("hot")
								MF_particles("smoke",x,y,5 * pmult,0,1,1,1)
								setsoundname("removal",1,sound)
									
								delete(2,x,y)
							end
						end
						
						for a,b in ipairs(eaten) do
							local bunit = mmf.newObject(b)
							local x,y = bunit.values[XPOS],bunit.values[YPOS]
							generaldata.values[SHAKE] = 4
							
							local pmult,sound = checkeffecthistory("eat")
							MF_particles("eat",x,y,5 * pmult,0,3,1,1)
							setsoundname("removal",1,sound)
							
							delete(b,x,y)
						end
					end
				end
				if (rule[1] == "level") and (rule[2] == "opens") and testcond(conds,1) then
					if (rule[3] == "level" and lsafe == false) then
						local pmult,sound = checkeffecthistory("unlock")
						setsoundname("removal",1,sound)
						local is_guarded = ack_endangered_unit(level_obj)
						if not is_guarded then
							destroylevel()
							return
						end
					else
						local eaten = {}
						local target = rule[3]
							
						if (target ~= "all") and (target ~= "empty") then
							local dothese = {}
							
							if (string.sub(target, 1, 5) ~= "group") then
								dothese = {target}
							else
								dothese = findgroup(target)
							end
							
							for c,d in ipairs(dothese) do
								if (unitlists[d] ~= nil) then
									for a,unitid in ipairs(unitlists[d]) do
										if (issafe(unitid) == false) then
											table.insert(eaten, unitid)
										end
										if lsafe == false then
											local pmult,sound = checkeffecthistory("unlock")
											setsoundname("removal",1,sound)
											local is_guarded = ack_endangered_unit(level_obj)
											if not is_guarded then
												destroylevel()
												return
											end
										end
									end
								end
							end
						elseif (target == "empty") then
							local empties = findempty()
							
							for a,b in ipairs(empties) do
								local x = b % roomsizex
								local y = math.floor(b / roomsizex)

								local pmult,sound = checkeffecthistory("unlock")
								MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
								setsoundname("removal",1,sound)
									
								delete(2,x,y)
								if lsafe == false then
									local pmult,sound = checkeffecthistory("unlock")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
						
						for a,b in ipairs(eaten) do
							local bunit = mmf.newObject(b)
							local x,y = bunit.values[XPOS],bunit.values[YPOS]
							generaldata.values[SHAKE] = 4
							
							local pmult,sound = checkeffecthistory("unlock")
							MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
							setsoundname("removal",1,sound)
							
							delete(b,x,y)
						end
					end
				end
				if (rule[1] == "level") and (rule[2] == "defeats") and testcond(conds,1) then
					if rule[3] == "level" and lsafe == false and (hasfeature("level","is","you",1) ~= nil or hasfeature("level","is","you2",1) ~= nil or hasfeature("level","is","3d",1) ~= nil) then
						local pmult,sound = checkeffecthistory("defeat")
						setsoundname("removal",1,sound)
						local is_guarded = ack_endangered_unit(level_obj)
						if not is_guarded then
							destroylevel()
							return
						end
					else
						local eaten = {}
						local target = rule[3]
							
						if (target ~= "all") and (target ~= "empty") then
							local dothese = {}
							
							if (string.sub(target, 1, 5) ~= "group") then
								dothese = {target}
							else
								dothese = findgroup(target)
							end
							
							for c,d in ipairs(dothese) do
								if (unitlists[d] ~= nil) then
									for a,unitid in ipairs(unitlists[d]) do
										local unit = mmf.newObject(unitid)
										local name = getname(unit)
										if (issafe(unitid) == false) and (hasfeature(name,"is","you",unitid,x,y) or hasfeature(name,"is","you2",unitid,x,y) or hasfeature(name,"is","3d",unitid,x,y)) then
											table.insert(eaten, unitid)
										end
									end
								end
							end
						elseif (target == "empty") then
							local empties = findempty()
							
							for a,b in ipairs(empties) do
								if (hasfeature("empty","is","you",2,x,y) or hasfeature("empty","is","you2",2,x,y) or hasfeature("empty","is","3d",2,x,y)) then
									local x = b % roomsizex
									local y = math.floor(b / roomsizex)

									local pmult,sound = checkeffecthistory("defeat")
									MF_particles("eat",x,y,5 * pmult,0,3,1,1)
									setsoundname("removal",1,sound)
										
									delete(2,x,y)
								end
							end
						end
						
						for a,b in ipairs(eaten) do
							local bunit = mmf.newObject(b)
							local x,y = bunit.values[XPOS],bunit.values[YPOS]
							generaldata.values[SHAKE] = 4
							
							local pmult,sound = checkeffecthistory("defeat")
							MF_particles("eat",x,y,5 * pmult,0,3,1,1)
							setsoundname("removal",1,sound)
							
							delete(b,x,y)
						end
					end
				end
				if (rule[1] == "level") and (rule[2] == "sinks") and (rule[3] ~= level) and testcond(conds,1) then
					local target = rule[3]
					local eaten = {}
						
					if (target ~= "all") and (target ~= "empty") then
						local dothese = {}
						
						if (string.sub(target, 1, 5) ~= "group") then
							dothese = {target}
						else
							dothese = findgroup(target)
						end
						
						for c,d in ipairs(dothese) do
							if (unitlists[d] ~= nil) then
								for a,unitid in ipairs(unitlists[d]) do
									if (issafe(unitid) == false) then
										table.insert(eaten, unitid)
									end
									if lsafe == false then
										local pmult,sound = checkeffecthistory("sink")
										setsoundname("removal",1,sound)
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
								end
							end
						end
					end
					
					for a,b in ipairs(eaten) do
						local bunit = mmf.newObject(b)
						local x,y = bunit.values[XPOS],bunit.values[YPOS]
						generaldata.values[SHAKE] = 4
						
						local pmult,sound = checkeffecthistory("sink")
						MF_particles("destroy",i,j,1,0,3,1,1)
						setsoundname("removal",1,sound)
						
						delete(b,x,y)
					end
				end
				if (rule[1] ~= "level") and (rule[3] == "level") and (rule[2] == "melts") and testcond(conds,1) and lsafe == false then
					local target = rule[1]
						
					if (target ~= "all") and (target ~= "empty") then
						local dothese = {}
						
						if (string.sub(target, 1, 5) ~= "group") then
							dothese = {target}
						else
							dothese = findgroup(target)
						end
						
						for c,d in ipairs(dothese) do
							if (unitlists[d] ~= nil) then
								for a,unitid in ipairs(unitlists[d]) do
									local pmult,sound = checkeffecthistory("hot")
									setsoundname("removal",1,sound)

									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
					elseif (target == "empty") then
						local empties = findempty()
						
						for a,b in ipairs(empties) do
							local pmult,sound = checkeffecthistory("hot")
							setsoundname("removal",1,sound)

							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
							end
						end
					end
				end
				if (rule[1] ~= "level") and (rule[3] == "level") and (rule[2] == "opens") and testcond(conds,1) then
					local eaten = {}
					local target = rule[1]
						
					if (target ~= "all") and (target ~= "empty") then
						local dothese = {}
						
						if (string.sub(target, 1, 5) ~= "group") then
							dothese = {target}
						else
							dothese = findgroup(target)
						end
						
						for c,d in ipairs(dothese) do
							if (unitlists[d] ~= nil) then
								for a,unitid in ipairs(unitlists[d]) do
									if (issafe(unitid) == false) then
										table.insert(eaten, unitid)
									end
									if lsafe == false then
										local pmult,sound = checkeffecthistory("unlock")
										setsoundname("removal",1,sound)
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
								end
							end
						end
					elseif (target == "empty") then
						local empties = findempty()
						
						for a,b in ipairs(empties) do
							local x = b % roomsizex
							local y = math.floor(b / roomsizex)

							local pmult,sound = checkeffecthistory("unlock")
							MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
							setsoundname("removal",1,sound)
								
							delete(2,x,y)
							if lsafe == false then
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("removal",1,sound)
								local is_guarded = ack_endangered_unit(level_obj)
								if not is_guarded then
									destroylevel()
									return
								end
							end
						end
					end
					
					for a,b in ipairs(eaten) do
						local bunit = mmf.newObject(b)
						local x,y = bunit.values[XPOS],bunit.values[YPOS]
						generaldata.values[SHAKE] = 4
						
						local pmult,sound = checkeffecthistory("unlock")
						MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
						setsoundname("removal",1,sound)
						
						delete(b,x,y)
					end
				end
				if (rule[1] ~= "level") and (rule[3] == "level") and (rule[2] == "defeats") and testcond(conds,1) and lsafe == false and (hasfeature("level","is","you",1) ~= nil or hasfeature("level","is","you2",1) ~= nil or hasfeature("level","is","3d",1) ~= nil) then
					local target = rule[1]
						
					if (target ~= "all") and (target ~= "empty") then
						local dothese = {}
						
						if (string.sub(target, 1, 5) ~= "group") then
							dothese = {target}
						else
							dothese = findgroup(target)
						end
						
						for c,d in ipairs(dothese) do
							if (unitlists[d] ~= nil) then
								for a,unitid in ipairs(unitlists[d]) do
									local pmult,sound = checkeffecthistory("defeat")
									setsoundname("removal",1,sound)
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
					elseif (target == "empty") then
						local empties = findempty()
						
						for a,b in ipairs(empties) do
							local pmult,sound = checkeffecthistory("defeat")
							setsoundname("removal",1,sound)
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
							end
						end
					end
				end
				if (rule[1] ~= "level") and (rule[3] == "level") and (rule[2] == "sinks") and testcond(conds,1) then
					local eaten = {}
					local target = rule[1]
						
					if (target ~= "all") and (target ~= "empty") then
						local dothese = {}
						
						if (string.sub(target, 1, 5) ~= "group") then
							dothese = {target}
						else
							dothese = findgroup(target)
						end
						
						for c,d in ipairs(dothese) do
							if (unitlists[d] ~= nil) then
								for a,unitid in ipairs(unitlists[d]) do
									if (issafe(unitid) == false) then
										table.insert(eaten, unitid)
									end
									if lsafe == false then
										local pmult,sound = checkeffecthistory("sink")
										setsoundname("removal",1,sound)
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
								end
							end
						end
					end
					
					for a,b in ipairs(eaten) do
						local bunit = mmf.newObject(b)
						local x,y = bunit.values[XPOS],bunit.values[YPOS]
						generaldata.values[SHAKE] = 4
						
						local pmult,sound = checkeffecthistory("sink")
						MF_particles("destroy",i,j,1,0,3,1,1)
						setsoundname("removal",1,sound)
						
						delete(b,x,y)
					end
				end
				
				if (rule[1] == "level") and (rule[2] == "is") and testcond(conds,1) then
					local action = rule[3]
					
					if ws_isPlayerProp(action) then -- EDIT: Replace long check with "Is player property" function
						local defeats = findfeature(nil,"is","defeat")
						local visits = findfeature(nil,"is","visit")
						local wins = findfeature(nil,"is","win")
						local ends = findfeature(nil,"is","end")
						local bonus = findfeature(nil,"is","bonus")
						
						if (defeats ~= nil) then
							for a,b in ipairs(defeats) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then
										for c,d in ipairs(allyous) do
											if (issafe(1) == false) and floating_level(d) then
												local is_guarded = ack_endangered_unit(level_obj)
												if not is_guarded then
													destroylevel()
													return
												end
											end
										end
									end
								elseif testcond(b[2],1) and (lsafe == false) then
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
						
						if ((#findallfeature("empty","is","defeat") > 0)) and floating_level(2) and (lsafe == false) then
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
								return
							end
						end
						
						local canwin = false
						local canvisit = false
						local canend = false
						local canbonus = false
						
						if (wins ~= nil) then
							for a,b in ipairs(wins) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then
										for c,d in ipairs(allyous) do
											if floating_level(d) then
												canwin = true
											end
										end
									end
								elseif testcond(b[2],1) then
									canwin = true
								end
							end
						end
						
						if (visits ~= nil) then
							for a,b in ipairs(visits) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then
										for c,d in ipairs(allyous) do
											if floating_level(d) then
												canvisit = true
											end
										end
									end
								elseif testcond(b[2],1) then
									canvisit = true
								end
							end
						end

						if (ends ~= nil) then
							for a,b in ipairs(ends) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then
										for c,d in ipairs(allyous) do
											if floating_level(d) then
												canend = true
											end
										end
									end
								elseif testcond(b[2],1) then
									canend = true
								end
							end
						end

						if (bonus ~= nil) then
							for a,b in ipairs(bonus) do
								local allbonus = findall(b)
								
								if (#allbonus > 0) then
									local destroyedSomething = false -- EDIT: add karma when level collects bonus
									for c,d in ipairs(allbonus) do
										if (issafe(d) == false) and floating_level(d) then
											local unit = mmf.newObject(d)
											
											local pmult,sound = checkeffecthistory("bonus")
											MF_particles("bonus",unit.values[XPOS],unit.values[YPOS],10 * pmult,4,1,1,1)
											MF_playsound("bonus")
											canbonus = true
											generaldata.values[SHAKE] = 2
											setsoundname("removal",2,sound)
											delete(d)
											destroyedSomething = true
										end
									end
										
									if destroyedSomething and not lrepent then  -- Do nothing if level is REPENT; destroy the level if it's KARMA and not SAFE; set karma status otherwise
										local is_guarded = ack_endangered_unit(level_obj)
										if lkarma and (lsafe == false) and not is_guarded then
											destroylevel()
											return
										else
											ws_setLevelKarma()
										end
									end
								end
							end
						end
						
						if (#findallfeature("empty","is","win") > 0) and floating_level(2) then -- EDIT: Removed double check
							canwin = true
						end
						
						if ((#findallfeature("empty","is","visit") > 0) or (#findallfeature("empty","is","visit") > 0)) and floating_level(2) then
							canvisit = true
						end
						
						if (#findallfeature("empty","is","end") > 0) and floating_level(2) then -- EDIT: Removed double check
							canend = true
						end
						
						if canvisit then
							dovisit(mapdir)
							return
						end

						if canbonus then
							MF_bonus(1)
							addundo({"bonus",1})
						end
						
						if canwin then
							MF_win()
							return
						end
						
						if canend and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
							if (editor.values[INEDITOR] ~= 0) then
								MF_end_single()
								MF_win()
								return
							else
								MF_end_single()
								MF_win()
								MF_credits(1)
								return
							end
						end
					elseif (action == "defeat") then
						local yous = ws_findPlayers() -- EDIT: Replaced long repeated sequence with function
						
						if (yous4 ~= nil) then
							for i,v in ipairs(yous4) do
								table.insert(yous, v)
							end
						end
						
						if (yous ~= nil) then
							for a,b in ipairs(yous) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then -- EDIT: set level karma when destroying something (LEVEL IS DEFEAT)
										local destroyedSomething = false
										for c,d in ipairs(allyous) do
											if (issafe(d) == false) and floating_level(d) then
												destroyedSomething = true
												local unit = mmf.newObject(d)
												
												local pmult,sound = checkeffecthistory("defeat")
												MF_particles("destroy",unit.values[XPOS],unit.values[YPOS],5 * pmult,0,3,1,1)
												setsoundname("removal",1,sound)
												generaldata.values[SHAKE] = 2
												delete(d)
											end
										end
										
										if destroyedSomething and not lrepent then  -- Do nothing if level is REPENT; destroy the level if it's KARMA and not SAFE; set karma status otherwise
											local is_guarded = ack_endangered_unit(level_obj)
											if lkarma and (lsafe == false) and not is_guarded then
												destroylevel()
												return
											else
												ws_setLevelKarma()
											end
										end
									end
								elseif testcond(b[2],1) and (lsafe == false) then
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
					elseif (action == "weak") then
						for i,unit in ipairs(units) do
							local name = unit.strings[UNITNAME]
							if (unit.strings[UNITTYPE] == "text") then
								name = "text"
							end
							
							if floating_level(unit.fixed) and (lsafe == false) then
								local is_guarded = ack_endangered_unit(level_obj)
								if not is_guarded then
									destroylevel()
									return
								end
							end
						end
					elseif (action == "hot") then
						local melts = findfeature(nil,"is","melt")
						
						if (melts ~= nil) then
							for a,b in ipairs(melts) do
								local allmelts = findall(b)
								
								if (#allmelts > 0) then -- EDIT: set level karma when destroying something (LEVEL IS HOT)
									local destroyedSomething = false
									for c,d in ipairs(allmelts) do
										if (issafe(d) == false) and floating_level(d) then
											destroyedSomething = true
											local unit = mmf.newObject(d)
											
											local pmult,sound = checkeffecthistory("hot")
											MF_particles("smoke",unit.values[XPOS],unit.values[YPOS],5 * pmult,0,1,1,1)
											generaldata.values[SHAKE] = 2
											setsoundname("removal",9,sound)
											delete(d)
										end
									end
									
									if destroyedSomething and not lrepent then  -- Do nothing if level is REPENT; destroy the level if it's KARMA and not SAFE; set karma status otherwise
										if lkarma and (lsafe == false) then
											destroylevel()
											return
										else
											ws_setLevelKarma()
										end
									end
								end
							end
						end
					elseif (action == "melt") then
						local hots = findfeature(nil,"is","hot")
						
						if (hots ~= nil) and (lsafe == false) then
							for a,b in ipairs(hots) do
								local doit = false
								
								if (b[1] ~= "level") then
									local allhots = findall(b)
									
									for c,d in ipairs(allhots) do
										if floating_level(d) then
											doit = true
										end
									end
								elseif testcond(b[2],1) then
									doit = true
								end
								
								if doit then
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
							end
						end
						
						if (#findallfeature("empty","is","hot") > 0) and floating_level(2) and (lsafe == false) then
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
								return
							end
						end
					elseif (action == "open") then
						local shuts = findfeature(nil,"is","shut")
						
						local openthese = {}
						
						if (shuts ~= nil) then
							for a,b in ipairs(shuts) do
								local doit = false
								
								if (b[1] ~= "level") then
									local allshuts = findall(b)
									
									for c,d in ipairs(allshuts) do
										if floating_level(d) then
											doit = true
											
											if (issafe(d) == false) then
												table.insert(openthese, d)
											end
										end
									end
								elseif testcond(b[2],1) then
									doit = true
								end
								
								if doit then
									if (lsafe == false) then
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
								end
							end
						end
						
						if (#openthese > 0) then
							if not lrepent then
								ws_setLevelKarma() -- EDIT: set level karma when destroying something (LEVEL IS OPEN) if level isn't REPENT
							end
							generaldata.values[SHAKE] = 8
							
							for a,b in ipairs(openthese) do
								local bunit = mmf.newObject(b)
								local bx,by = bunit.values[XPOS],bunit.values[YPOS]
								
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("turn",7,sound)
								MF_particles("unlock",bx,by,15 * pmult,2,4,1,1)
								
								delete(b)
								deleted[b] = 1
							end
						end
						
						if (#findallfeature("empty","is","shut") > 0) and floating_level(2) and (lsafe == false) then
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
								return
							end
						end
					elseif (action == "shut") then
						local opens = findfeature(nil,"is","open")
						
						local openthese = {}
						
						if (opens ~= nil) then
							for a,b in ipairs(opens) do
								local doit = false
								
								if (b[1] ~= "level") then
									local allopens = findall(b)
									
									for c,d in ipairs(allopens) do
										if floating_level(d) then
											doit = true
											
											if (issafe(d) == false) then
												table.insert(openthese, d)
											end
										end
									end
								elseif testcond(b[2],1) then
									doit = true
								end
								
								if doit then
									if (lsafe == false) then
										local is_guarded = ack_endangered_unit(level_obj)
										if not is_guarded then
											destroylevel()
											return
										end
									end
								end
							end
						end
						
						if (#openthese > 0) then
							if not lrepent then
								ws_setLevelKarma() -- EDIT: set level karma when destroying something (LEVEL IS SHUT) if level isn't REPENT
							end
							generaldata.values[SHAKE] = 8
							
							for a,b in ipairs(openthese) do
								local bunit = mmf.newObject(b)
								local bx,by = bunit.values[XPOS],bunit.values[YPOS]
								
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("turn",7,sound)
								MF_particles("unlock",bx,by,15 * pmult,2,4,1,1)
								
								delete(b)
								deleted[b] = 1
							end
						end
						
						if (#findallfeature("empty","is","open") > 0) and floating_level(2) and (lsafe == false) then
							local is_guarded = ack_endangered_unit(level_obj)
							if not is_guarded then
								destroylevel()
								return
							end
						end
					elseif (action == "sink") then
						local openthese = {}
						
						for a,unit in ipairs(units) do
							local name = unit.strings[UNITNAME]
							
							if (unit.strings[UNITTYPE] == "text") then
								name = "text"
							end
							
							if floating_level(unit.fixed) then
								if (lsafe == false) then
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
								
								if (issafe(unit.fixed) == false) then
									table.insert(openthese, unit.fixed)
								end
							end
						end
						
						if (#openthese > 0) then -- EDIT: set level karma when destroying something (LEVEL IS SINK) if level isn't REPENT
							if not lrepent then
							ws_setLevelKarma()
							end
							generaldata.values[SHAKE] = 3
							
							for a,b in ipairs(openthese) do
								local bunit = mmf.newObject(b)
								local bx,by = bunit.values[XPOS],bunit.values[YPOS]
								
								local pmult,sound = checkeffecthistory("sink")
								setsoundname("removal",3,sound)
								local c1,c2 = getcolour(b)
								MF_particles("destroy",bx,by,15 * pmult,c1,c2,1,1)
								
								delete(b)
								deleted[b] = 1
							end
						end
					elseif (action == "boom") then
						local openthese = {}
						
						for a,unit in ipairs(units) do
							local name = unit.strings[UNITNAME]
							
							if (unit.strings[UNITTYPE] == "text") then
								name = "text"
							end
							
							if floating_level(unit.fixed) then
								if (lsafe == false) then
									local is_guarded = ack_endangered_unit(level_obj)
									if not is_guarded then
										destroylevel()
										return
									end
								end
								
								if (issafe(unit.fixed) == false) then
									table.insert(openthese, unit.fixed)
								end
							end
						end
						
						if (#openthese > 0) then -- EDIT: set level karma when destroying something (LEVEL IS BOOM) if level isn't REPENT
							if not lrepent then
							ws_setLevelKarma()
							end
							generaldata.values[SHAKE] = 3
							
							for a,b in ipairs(openthese) do
								local bunit = mmf.newObject(b)
								local bx,by = bunit.values[XPOS],bunit.values[YPOS]
								
								local pmult,sound = checkeffecthistory("boom")
								setsoundname("removal",1,sound)
								MF_particles("smoke",bx,by,15 * pmult,0,2,1,1)
								
								delete(b)
								deleted[b] = 1
							end
						end
					elseif (action == "cut") then
						handle_level_cutting()
					elseif (action == "done") then
						local doned = {}
						for a,unit in ipairs(units) do
							table.insert(doned, unit)
						end
						
						updateundo = true
						
						for a,unit in ipairs(doned) do
							addundo({"done",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.fixed,unit.values[FLOAT]})
							
							unit.values[FLOAT] = 2
							unit.values[EFFECTCOUNT] = math.random(-10,10)
							unit.values[POSITIONING] = 7
							unit.flags[DEAD] = true
							
							delunit(unit.fixed)
						end
						
						MF_playsound("doneall_c")
					elseif (action == "bonus") then
						local yous = ws_findPlayers() -- EDIT: Replaced long repeated sequence with function
						
						if (yous4 ~= nil) then
							for i,v in ipairs(yous4) do
								table.insert(yous, v)
							end
						end
						
						if (yous ~= nil) then
							for a,b in ipairs(yous) do
								if (b[1] ~= "level") then
									local allyous = findall(b)
									
									if (#allyous > 0) then
										for c,d in ipairs(allyous) do
											if floating_level(d) then
												bonusget = true
												
												if (lsafe == false) then
												destroylevel("bonus")
												return
											end
										end
									end
									end
								elseif testcond(b[2],1) then
									bonusget = true
									
									if (lsafe == false) then
										destroylevel("bonus")
										return
									end
								end
							end
						end
						
						if ws_areTherePlayerEmpties() and floating_level(2) and (lsafe == false) then -- Replaced alive empty check with function
							bonusget = true
							
							if (lsafe == false) then
								destroylevel("bonus")
								return
							end
						end

						if bonusget then
							MF_playsound("bonus")
							MF_bonus(1)
							addundo({"bonus",1})
						end
					elseif (action == "visit") then
						local yous = findfeature(nil,"is","you")
						local yous2 = findfeature(nil,"is","you2")
						local yous3 = findfeature(nil,"is","3d")
						
						if (yous == nil) then
							yous = {}
						end
						
						if (yous2 ~= nil) then
							for i,v in ipairs(yous2) do
								table.insert(yous, v)
							end
						end
						
						if (yous3 ~= nil) then
							for i,v in ipairs(yous3) do
								table.insert(yous, v)
							end
						end
						
						local canvisit = false
						
						if (yous ~= nil) then
							for a,b in ipairs(yous) do
								local allyous = findall(b)
								local doit = false
								
								for c,d in ipairs(allyous) do
									if floating_level(d) then
										doit = true
									end
								end
								
								if doit then
									canvisit = true
								end
							end
						end
						
						local emptyyou = false
						if ((#findallfeature("empty","is","you") > 0) or (#findallfeature("empty","is","you2") > 0) or (#findallfeature("empty","is","3d") > 0)) and floating_level(2) then
							emptyyou = true
						end
						
						if (hasfeature("level","is","you",1) ~= nil) or (hasfeature("level","is","you2",1) ~= nil) or (hasfeature("level","is","3d",1) ~= nil) or emptyyou then
							canvisit = true
						end
						
						if canvisit then
							dovisit(mapdir)
							return
						end
					elseif (action == "win") then
						local yous = ws_findPlayers() -- Replaced long repeated sequence with function
						
						if (yous4 ~= nil) then
							for i,v in ipairs(yous4) do
								table.insert(yous, v)
							end
						end
						
						local canwin = false
						
						if (yous ~= nil) then
							for a,b in ipairs(yous) do
								local allyous = findall(b)
								local doit = false
								
								for c,d in ipairs(allyous) do
									if floating_level(d) then
										doit = true
									end
								end
								
								if doit then
									canwin = true
									if action == "win" then
									for c,d in ipairs(allyous) do
										local unit = mmf.newObject(d)
										local pmult,sound = checkeffecthistory("win")
										MF_particles("win",unit.values[XPOS],unit.values[YPOS],10 * pmult,2,4,1,1)
									end
								end
							end
						end
						end
						
						local emptyyou = false
						if ws_areTherePlayerEmpties() and floating_level(2) then -- Replaced alive empty check with function
							emptyyou = true
						end
						
						if ws_isLevelPlayer() or emptyyou then -- Replaced long check with function
							canwin = true
						end
						
						if canwin then
							if action == "win" then MF_win() else doreset = true end
							return
						end
					elseif (action == "end") then
						local yous = ws_findPlayers() -- Replaced long repeated sequence with function
						
						if (yous4 ~= nil) then
							for i,v in ipairs(yous4) do
								table.insert(yous, v)
							end
						end
						
						local canend = false
						
						if (yous ~= nil) then
							for a,b in ipairs(yous) do
								local allyous = findall(b)
								local doit = false
								
								for c,d in ipairs(allyous) do
									if floating_level(d) then
										doit = true
									end
								end
								
								if doit then
									canend = true
									for c,d in ipairs(allyous) do
										local unit = mmf.newObject(d)
										local pmult,sound = checkeffecthistory("win")
										MF_particles("win",unit.values[XPOS],unit.values[YPOS],10 * pmult,2,4,1,1)
									end
								end
							end
						end
						
						local emptyyou = false
						if ws_areTherePlayerEmpties() and floating_level(2) then -- Replaced alive empty check with function
							emptyyou = true
						end
						
						if ws_isLevelPlayer() or emptyyou then -- Replaced long check with function
							canend = true
						end
						
						if canend and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
							if (editor.values[INEDITOR] ~= 0) then
								MF_end_single()
								MF_win()
								break
							else
								MF_end_single()
								MF_win()
								MF_credits(1)
								break
							end
						end
					elseif (action == "tele") and (levelteledone < 3) and (lstill == false) then
						levelteledone = levelteledone + 1
						
						for a,unit in ipairs(units) do
							local x,y = unit.values[XPOS],unit.values[YPOS]
							
							local tx,ty = fixedrandom(1,roomsizex-2),fixedrandom(1,roomsizey-2)
							
							if floating_level(unit.fixed) then
								update(unit.fixed,tx,ty)
								
								local pmult,sound = checkeffecthistory("tele")
								MF_particles("glow",x,y,5 * pmult,1,4,1,1)
								MF_particles("glow",tx,ty,5 * pmult,1,4,1,1)
								setsoundname("turn",6,sound)
							end
						end
					elseif (action == "move") then
						local dir = mapdir
						
						if (featureindex["reverse"] ~= nil) then
							dir = reversecheck(1,dir)
						end
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,dir,dir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "chill") then
						local dir = fixedrandom(0,3)
						addundo({"mapdir",mapdir,dir})
						mapdir = dir
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,dir,dir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "nudgeright") then
						local dir = 0
						
						if (featureindex["reverse"] ~= nil) then
							dir = reversecheck(1,dir)
						end
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "nudgeup") then
						local dir = 1
						
						if (featureindex["reverse"] ~= nil) then
							dir = reversecheck(1,dir)
						end
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "nudgeleft") then
						local dir = 2
						
						if (featureindex["reverse"] ~= nil) then
							dir = reversecheck(1,dir)
						end
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "nudgedown") then
						local dir = 3
						
						if (featureindex["reverse"] ~= nil) then
							dir = reversecheck(1,dir)
						end
						
						local drs = ndirs[dir + 1]
						local ox,oy = drs[1],drs[2]
						
						if (lstill == false) and (lsleep == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
							MF_scrollroom(ox * tilesize,oy * tilesize)
							updateundo = true
						end
					elseif (action == "fall") then
						local drop = 20
						local dir = mapdir
						
						local ox = 0
						local oy = 1
						
						if (featureindex["reverse"] ~= nil) then
							dir,ox,oy = reversecheck(1,dir,nil,nil,ox,oy)
						end
						
						if (lstill == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
							MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
							updateundo = true
						end
					elseif (action == "fallright") then
						local drop = 35
						local dir = mapdir
						
						local ox = 1
						local oy = 0
						
						if (featureindex["reverse"] ~= nil) then
							dir,ox,oy = reversecheck(1,dir,nil,nil,ox,oy)
						end
						
						if (lstill == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
							MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
							updateundo = true
						end
					elseif (action == "fallup") then
						local drop = 20
						local dir = mapdir
						
						local ox = 0
						local oy = -1
						
						if (featureindex["reverse"] ~= nil) then
							dir,ox,oy = reversecheck(1,dir,nil,nil,ox,oy)
						end
						
						if (lstill == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
							MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
							updateundo = true
						end
					elseif (action == "fallleft") then
						local drop = 35
						local dir = mapdir
						
						local ox = -1
						local oy = 0
						
						if (featureindex["reverse"] ~= nil) then
							dir,ox,oy = reversecheck(1,dir,nil,nil,ox,oy)
						end
						
						if (lstill == false) then
							addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
							MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
							updateundo = true
						end
					elseif (rule[3] == "turn") then
						local newmapdir = (mapdir - 1 + 4) % 4
						local newmaprotation = ((mapdir + 1 + 4) % 4) * 90
						
						updateundo = true
						
						addundo({"maprotation",maprotation,newmaprotation,newmapdir})
						addundo({"mapdir",mapdir,newmapdir})
						maprotation = newmaprotation
						mapdir = newmapdir
						MF_levelrotation(maprotation)
					elseif (rule[3] == "deturn") then
						local newmapdir = (mapdir + 1 + 4) % 4
						local newmaprotation = ((mapdir + 1 + 4) % 4) * 90
						
						updateundo = true
						
						addundo({"maprotation",maprotation,newmaprotation,newmapdir})
						addundo({"mapdir",mapdir,newmapdir})
						maprotation = newmaprotation
						mapdir = newmapdir
						MF_levelrotation(maprotation)
					elseif (action == "empty") then
						destroylevel("empty")
					end
				end
			end
		end
		
		if (featureindex["done"] ~= nil) then
			for i,v in ipairs(featureindex["done"]) do
				table.insert(donethings, v)
			end
		end
		
		if (#donethings > 0) and (generaldata.values[WINTIMER] == 0) then
			for i,rules in ipairs(donethings) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (rule[1] == "all") and (rule[2] == "is") and (rule[3] == "done") then
					local targets = findallfeature(nil,"is","done",true)
					local found = false
					
					local levelunits_ = {}
					
					for a,v in ipairs(targets) do
						local unit = mmf.newObject(v)
						
						if (unit.className ~= "level") then
							found = true
							break
						end
					end
					
					if (objectlist["level"] ~= nil) then
						for a,unit in ipairs(units) do
							if (unit.className == "level") then
								table.insert(levelunits_, unit.fixed)
							end
						end
					end
					
					if found then
						if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) and (editor.values[INEDITOR] == 0) then
							MF_playsound("doneall_c")
							MF_allisdone()
						elseif (editor.values[INEDITOR] ~= 0) and (#targets >= #units - #codeunits - #levelunits_) then
							local pmult = checkeffecthistory("win")
							
							MF_playsound("doneall_c")
							MF_done_single()
							MF_win()
							break
						elseif (#targets >= #units - #codeunits - #levelunits_) then
							local pmult = checkeffecthistory("win")
							
							local mods_run = do_mod_hook("levelpack_done", {})
							
							if (mods_run == false) then
								MF_playsound("doneall_c")
								MF_done_single()
								MF_win()
								-- plays custom music for the all is done ending!
								-- in audio channel 2, so technically the empty theme is still playing (but whatever!)
								MF_playmusic("Dogeiscut - Best", 1, 2, 0)
								MF_credits(2)
							end
							break
						end
					end
				end
			end
		end
		
		if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) and (generaldata.strings[CURRLEVEL] == "305level") then
			local numfound = false
			
			if (featureindex["image"] ~= nil) then
				for i,v in ipairs(featureindex["image"]) do
					local rule = v[1]
					local conds = v[2]
					
					if (rule[1] == "image") and (rule[2] == "is") and (#conds == 0) then
						local num = rule[3]
						
						local nums = {
							one = {1, "image_desc_1"},
							two = {2, "image_desc_2"},
							three = {3, "image_desc_3"},
							four = {4, "image_desc_4"},
							five = {5, "image_desc_5"},
							six = {6, "image_desc_6"},
							seven = {7, "image_desc_7"},
							eight = {8, "image_desc_8"},
							nine = {9, "image_desc_9"},
							ten = {10, "image_desc_10"},
							fourteen = {11, "image_desc_11"},
							sixteen = {12, "image_desc_12"},
							minusone = {13, "image_desc_13"},
							minustwo = {14, "image_desc_14"},
							minusthree = {15, "image_desc_15"},
							minusten = {16, "image_desc_16"},
							win = {0, "win"}
						}
						
						if (nums[num] ~= nil) then
							local data = nums[num]
							
							if (data[2] ~= "win") then
								MF_setart(data[1], langtext(data[2],true))
								numfound = true
							else
								local yous = findallfeature(nil,"is","you",true)
								local yous2 = findallfeature(nil,"is","you2",true)
								local yous3 = findallfeature(nil,"is","3d",true)
								local yous4 = findallfeature(nil,"is","alive",true)
								
								if (#yous2 > 0) then
									for a,b in ipairs(yous2) do
										table.insert(yous, b)
									end
								end
								
								if (#yous3 > 0) then
									for a,b in ipairs(yous3) do
										table.insert(yous, b)
									end
								end
								
								if (#yous4 > 0) then
									for a,b in ipairs(yous4) do
										table.insert(yous, b)
									end
								end
								
								for a,b in ipairs(yous) do
									local unit = mmf.newObject(b)
									local x,y = unit.values[XPOS],unit.values[YPOS]
									
									if (x > roomsizex - 16) then
										local pmult = checkeffecthistory("win")
										
										MF_particles("win",x,y,10 * pmult,2,4,1,1)
										MF_win()
										break
									end
								end
							end
						end
					end
				end
			end
				
			if (numfound == false) then
				MF_setart(0,"")
			end
		end
		
		if unlocked then
			setsoundname("turn",7)
		end
	end
	
	if (#units >= unitlimit) then
		HACK_INFINITY = 200
		destroylevel("toocomplex")
		return
	end

	guard_checkpoint("levelblock")
end

function effectblock()
	local levelhide = nil
	
	if (featureindex["level"] ~= nil) then
		levelhide = hasfeature("level","is","hide",1)
		
		local isred = hasfeature("level","is","red",1)
		local isblue = hasfeature("level","is","blue",1)
		local isgreen = hasfeature("level","is","green",1)
		local islime = hasfeature("level","is","lime",1)
		local isyellow = hasfeature("level","is","yellow",1)
		local ispurple = hasfeature("level","is","purple",1)
		local ispink = hasfeature("level","is","pink",1)
		local isrosy = hasfeature("level","is","rosy",1)
		local isblack = hasfeature("level","is","black",1)
		local isgrey = hasfeature("level","is","grey",1)
		local issilver = hasfeature("level","is","silver",1)
		local iswhite = hasfeature("level","is","white",1)
		local isbrown = hasfeature("level","is","brown",1)
		local isorange = hasfeature("level","is","orange",1)
		local iscyan = hasfeature("level","is","cyan",1)
	
		local colours = {isred, isorange, isyellow, islime, isgreen, iscyan, isblue, ispurple, ispink, isrosy, isblack, isgrey, issilver, iswhite, isbrown}
		local ccolours = {{2,2},{2,3},{2,4},{5,3},{5,2},{1,4},{3,2},{3,1},{4,1},{4,2},{0,4},{0,1},{0,2},{0,3},{6,1}}
		
		leveldata.colours = {}
		local c1,c2 = -1,-1
		
		for a=1,#ccolours do
			if (colours[a] ~= nil) then
				local c = ccolours[a]
				
				if (#leveldata.colours == 0) then
					c1 = c[1]
					c2 = c[2]
				end
				
				table.insert(leveldata.colours, {c[1],c[2]})
			end
		end
		
		if (#leveldata.colours == 1) then
			if (c1 > -1) and (c2 > -1) then
				if (c1 == 0) and (c2 == 4) then
					MF_backcolour(c1, c2)
				else
					MF_backcolour_dim(c1, c2)
				end
			end
		elseif (#leveldata.colours == 0) then
			MF_backcolour(0, 4)
		end
	else
		MF_backcolour(0, 4)
	end
	
	local resetcolour = {}
	local updatecolour = {}
	
	for i,unit in ipairs(units) do
		unit.new = false
		
		if (levelhide == nil) then
			unit.visible = true
		else
			unit.visible = false
		end
		
		if (unit.className ~= "level") then			
			local name = unit.strings[UNITNAME]
			local isred = hasfeature(name,"is","red",unit.fixed)
			local isblue = hasfeature(name,"is","blue",unit.fixed)
			local islime = hasfeature(name,"is","lime",unit.fixed)
			local isgreen = hasfeature(name,"is","green",unit.fixed)
			local isyellow = hasfeature(name,"is","yellow",unit.fixed)
			local ispurple = hasfeature(name,"is","purple",unit.fixed)
			local ispink = hasfeature(name,"is","pink",unit.fixed)
			local isrosy = hasfeature(name,"is","rosy",unit.fixed)
			local isblack = hasfeature(name,"is","black",unit.fixed)
			local isgrey = hasfeature(name,"is","grey",unit.fixed)
			local issilver = hasfeature(name,"is","silver",unit.fixed)
			local iswhite = hasfeature(name,"is","white",unit.fixed)
			local isbrown = hasfeature(name,"is","brown",unit.fixed)
			local isorange = hasfeature(name,"is","orange",unit.fixed)
			local iscyan = hasfeature(name,"is","cyan",unit.fixed)
			
			unit.colours = {}
			
			local colours = {isred, isorange, isyellow, islime, isgreen, iscyan, isblue, ispurple, ispink, isrosy, isblack, isgrey, issilver, iswhite, isbrown}
			local ccolours = {{2,2},{2,3},{2,4},{5,3},{5,2},{1,4},{3,2},{3,1},{4,1},{4,2},{0,4},{0,1},{0,2},{0,3},{6,1}}
			
			local c1,c2,ca = -1,-1,-1
			
			unit.flags[PHANTOM] = false
			local isphantom = hasfeature(name,"is","phantom",unit.fixed)
			if (isphantom ~= nil) then
				unit.flags[PHANTOM] = true
			end
			
			for a=1,#ccolours do
				if (colours[a] ~= nil) then
					local c = ccolours[a]
					
					if (#unit.colours == 0) then
						c1 = c[1]
						c2 = c[2]
						ca = a
					end
					
					table.insert(unit.colours, c)
				end
			end
			
			if (#unit.colours == 1) then
				if (c1 > -1) and (c2 > -1) and (ca > 0) then
					MF_setcolour(unit.fixed,c1,c2)
					unit.colour = {c1,c2}
					unit.values[A] = ca
				end
			elseif (#unit.colours == 0) then
				if (unit.values[A] > 0) and (math.floor(unit.values[A]) == unit.values[A]) then
					if (unit.strings[UNITTYPE] ~= "text") or (unit.active == false) then
						setcolour(unit.fixed)
					else
						setcolour(unit.fixed,"active")
					end
					unit.values[A] = 0
				end
			else
				unit.values[A] = ca
				
				if (unit.strings[UNITTYPE] == "text") then
					local curr = (unit.currcolour % #unit.colours) + 1
					local c = unit.colours[curr]
					
					unit.colour = {c[1],c[2]}
					MF_setcolour(unit.fixed,c[1],c[2])
				end
			end
		end
	end
	
	if (levelhide == nil) then
		local ishide = findallfeature(nil,"is","hide",true)
		
		for i,unitid in ipairs(ishide) do
			local unit = mmf.newObject(unitid)
			
			unit.visible = false
		end
	end
end

function findplayer(undoing)
	local playerfound = false
	local playerfound_3d = false
	local vesselsfound = false
	
	local noundo = false
	if (undoing ~= nil) and (undoing == 2) then
		noundo = true
	end
	
	local players1 = findfeature(nil,"is","you")
	local players2 = findfeature(nil,"is","you2")
	local players3 = findfeature(nil,"is","3d")
	local players4 = findfeature(nil,"is","alive") -- Get all ALIVE features
	
	local vessels = findfeature(nil,"is","vessel") or {} -- Get all VESSEL features (used to keep music if option is enabled)
	local vessels2 = findfeature(nil,"is","vessel2") or {}
	
	local players = {}
	if (players1 ~= nil) then
		for i,v in ipairs(players1) do
			table.insert(players, v)
		end
	end
	
	if (players2 ~= nil) then
		for i,v in ipairs(players2) do
			table.insert(players, v)
		end
	end
	
	if (players4 ~= nil) then -- Add ALIVE features to the list of 2D players
		for i,v in ipairs(players4) do
			table.insert(players, v)
		end
	end
	
	local limit = #players
	
	if (players3 ~= nil) then
		for i,v in ipairs(players3) do
			if (v[1] ~= "empty") then
				table.insert(players, v)
			end
		end
	end
	
	if (#players > 0) then
		for i,v in ipairs(players) do
			if (v[1] ~= "level") and (v[1] ~= "empty") then
				local allplayers = findall(v)
				
				if (#allplayers > 0) then
					playerfound = true
					
					if (i > limit) then
						local unitid = allplayers[1]
						local unit = mmf.newObject(unitid)
						
						playerfound_3d = true
						
						if (spritedata.values[CAMTARGET] == 0) or (spritedata.values[CAMTARGET] == 0.5) then
							spritedata.values[CAMTARGET] = unit.values[ID]
						end
					end
				end
			elseif (v[1] == "level") then
				if testcond(v[2],1) then
					playerfound = true
				end
			elseif (v[1] == "empty") then
				local empties = findempty(v[2],true)
				
				if (#empties > 0) then
					playerfound = true
				end
			end
		end
	end
	
	if (playerfound_3d == false) and playerfound then
		visionmode(0,nil,noundo)
	end
	
	-- EDIT: check if there are VESSEL units
	for _,v in ipairs(vessels2) do
		table.insert(vessels,v)
	end
	
	if (#vessels > 0) then
		for i,v in ipairs(vessels) do
			if (v[1] ~= "level") and (v[1] ~= "empty") then
				local allvessels = findall(v)
				
				if (#allvessels > 0) then
					vesselsfound = true
				end
			elseif (v[1] == "level") then
				if testcond(v[2],1) then
					vesselsfound = true
				end
			elseif (v[1] == "empty") then
				local empties = findempty(v[2],true)
				
				if (#empties > 0) then
					vesselsfound = true
				end
			end
		end
	end
	--
	
	if playerfound or (MUSIC_WHEN_ONLY_VESSELS and vesselsfound) then
		MF_musicstate(0)
		generaldata2.values[NOPLAYER] = 0
	else
		if (generaldata2.values[NOPLAYER] == 0) then
			dolog("no_you","event")
		end
		
		MF_musicstate(1)
		generaldata2.values[NOPLAYER] = 1
	end
end