function effects(timer)
	doeffect(timer,nil,"win","unlock",1,2,20,{2,4})
	doeffect(timer,nil,"reset","unlock",1,2,20,{3,1})
	doeffect(timer,nil,"best","unlock",6,30,2,{2,4})
	doeffect(timer,nil,"router","unlock",2,2,20,{4,4})
	doeffect(timer,nil,"channel1","unlock",2,2,20,{2,1})
	doeffect(timer,nil,"channel2","unlock",2,2,20,{3,4})
	doeffect(timer,nil,"channel3","unlock",2,2,20,{4,3})
	doeffect(timer,nil,"tele","glow",1,5,20,{1,4})
	doeffect(timer,nil,"hot","hot",1,80,10,{0,1})
	doeffect(timer,nil,"bonus","bonus",1,2,20,{4,1})
	doeffect(timer,nil,"wonder","wonder",1,10,5,{0,3})
	doeffect(timer,nil,"sad","tear",1,2,20,{3,2})
	doeffect(timer,nil,"sleep","sleep",1,2,60,{3,2})
	doeffect(timer,nil,"broken","error",3,10,8,{2,2})
	doeffect(timer,nil,"pet","pet",1,0,50,{3,1},"nojitter")
	
	doeffect(timer,nil,"power","electricity",2,5,8,{2,4})
	doeffect(timer,nil,"power2","electricity",2,5,8,{5,4})
	doeffect(timer,nil,"power3","electricity",2,5,8,{4,4})
	
	local rnd = math.random(2,4)
	doeffect(timer,nil,"end","unlock",1,1,10,{1,rnd},"inwards")

	doeffect(timer,nil,"visit","unlock",1,2,8,{0,3},"visitrule")

	doeffect(timer,nil,"visit","unlock",1,2,8,{0,3},"visitrule")
	
	do_mod_hook("effect_always")
end

function doeffect(timer,word2_,word3,particle,count,chance,timing,colour,specialrule_,layer_)
	local zoom = generaldata2.values[ZOOM]
	
	local specialrule = specialrule_ or ""
	local layer = layer_ or 1
	local word2 = word2_ or "is"
	
	if (timer % timing == 0) then
		local this = findfeature(nil,word2,word3)
		
		local c1 = colour[1]
		local c2 = colour[2]
		
		if (this ~= nil) then
			for k,v in ipairs(this) do
				if (v[1] ~= "empty") and (v[1] ~= "all")--[[ and (v[1] ~= "level")--]] then -- EDIT: fix a basegame bug where level objects don't emit particles
					local these = findall(v,true)
					
					if (#these > 0) then
						for a,b in ipairs(these) do
							local unit = mmf.newObject(b)
							local x,y = unit.values[XPOS],unit.values[YPOS]
							local udir = unit.values[DIR] * math.pi / 2
							
							if (word3 == "broken") then
								if (unit.strings[UNITTYPE] == "text") then
									c1,c2 = getcolour(b,"active")
								else
									c1,c2 = getcolour(b)
								end
							end
							
							if unit.visible then
								for i=1,count do
									local partid = 0
									
									if (chance > 1) then
										if (math.random(chance) == 1) then
											if (specialrule ~= "nojitter") then
												partid = MF_particle(particle,x,y,c1,c2,layer)
											else
												partid = MF_staticparticle(particle,x,y,c1,c2,layer)
											end
										end
									else
										if (specialrule ~= "nojitter") then
											partid = MF_particle(particle,x,y,c1,c2,layer)
										else
											partid = MF_staticparticle(particle,x,y,c1,c2,layer)
										end
									end
									
									if (partid ~= nil) and (specialrule == "inwards") and (partid ~= 0) then
										local part = mmf.newObject(partid)
										
										part.values[ONLINE] = 2
										local midx = math.floor(roomsizex * 0.5)
										local midy = math.floor(roomsizey * 0.5)
										local mx = x + 0.5 - midx
										local my = y + 0.5 - midy
										
										local dir = 0 - math.atan2(my, mx)
										local dist = math.sqrt(my ^ 2 + mx ^ 2)
										local roomrad = math.rad(generaldata2.values[ROOMROTATION])
										
										mx = Xoffset + (midx + math.cos(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
										my = Yoffset + (midy - math.sin(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
										
										part.x = mx + math.random(0 - tilesize * 1.5 * zoom,tilesize * 1.5 * zoom)
										part.y = my + math.random(0 - tilesize * 1.5 * zoom,tilesize * 1.5 * zoom)
										part.values[XPOS] = part.x
										part.values[YPOS] = part.y
										
										dir = math.pi - math.atan2(part.y - my, part.x - mx)
										dist = math.sqrt((part.y - my) ^ 2 + (part.x - mx) ^ 2)
										part.values[XVEL] = math.cos(dir) * (dist * 0.2)
										part.values[YVEL] = 0 - math.sin(dir) * (dist * 0.2)
									end

									if (partid ~= nil) and (specialrule == "visitrule") and (partid ~= 0) then
										local part = mmf.newObject(partid)
										
										part.values[ONLINE] = 2
										local midx = math.floor(roomsizex * 0.5)
										local midy = math.floor(roomsizey * 0.5)
										local mx = x + 0.5 - midx
										local my = y + 0.5 - midy
										
										local dir = 0 - math.atan2(my, mx)
										local dist = math.sqrt(my ^ 2 + mx ^ 2)
										local roomrad = math.rad(generaldata2.values[ROOMROTATION])
										
										mx = Xoffset + (midx + math.cos(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
										my = Yoffset + (midy - math.sin(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
										
										part.x = mx + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
										part.y = my + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
										part.values[XPOS] = part.x
										part.values[YPOS] = part.y
										
										part.values[XVEL] = math.cos(udir) * 10
										part.values[YVEL] = math.sin(udir) * -10
									end
								end
							end
						end
					end
				end -- EDIT: same bugfix, originally this was a elseif rather than two if
				if ((v[1] == "empty") or (v[1] == "level")) then
					local ignorebroken = false
					if (word3 == "broken") then
						ignorebroken = true
					end

					local cause = v[1]
					
					-- EDIT: Spawn particles around the level edge if the special rule is "leveledge"
					if (specialrule == "leveledge") and (v[1] == "level") and testcond(v[2],1) then
						for i=0,roomsizex-1 do
							if (chance > 0) then
								if (math.random(chance*2) == 1) then
									partid = MF_particle(particle,i,0,c1,c2,layer)
								end
								if (math.random(chance*2) == 1) then
									partid = MF_particle(particle,i,roomsizey-1,c1,c2,layer)
								end
							end
						end
						for j=1,roomsizey-2 do
							if (chance > 0) then
								if (math.random(chance*2) == 1) then
									partid = MF_particle(particle,0,j,c1,c2,layer)
								end
								if (math.random(chance*2) == 1) then
									partid = MF_particle(particle,roomsizex-1,j,c1,c2,layer)
								end
							end
						end
					elseif (v[1] ~= "level") or ((v[1] == "level") and testcond(v[2],1)) then
						for i=1,roomsizex-2 do
							for j=1,roomsizey-2 do
								local tileid = i + j * roomsizex
								
								if (unitmap[tileid] == nil) or ((unitmap[tileid] ~= nil) and (#unitmap[tileid] == 0)) then
									if (v[1] ~= "empty") or ((v[1] == "empty") and testcond(v[2],2,i,j,nil,nil,nil,ignorebroken)) then
										local partid = 0
										-- EDIT: reduce level particles when the "reducedlvl" rule is used
										local skipthis = false
										if ((v[1] == "level") and (specialrule == "reducedlvl") and (math.random(1,12) < 12)) then
											skipthis = true
										end
										
										if (not skipthis) then
											if (chance > 1) then
												if (math.random(chance) == 1) then
													if (specialrule ~= "nojitter") then
														partid = MF_particle(particle,i,j,c1,c2,layer)

														if (partid ~= nil) and (specialrule == "visitrule") and (partid ~= 0) then
															local udir
															if cause == "level" then
																udir = mapdir * math.pi / 2
															elseif cause == "empty" then
																udir = math.random(0,3) * math.pi / 2
															end
															
															local part = mmf.newObject(partid)
															
															part.values[ONLINE] = 2
															local midx = math.floor(roomsizex * 0.5)
															local midy = math.floor(roomsizey * 0.5)
															local mx = i + 0.5 - midx
															local my = j + 0.5 - midy
															
															local dir = 0 - math.atan2(my, mx)
															local dist = math.sqrt(my ^ 2 + mx ^ 2)
															local roomrad = math.rad(generaldata2.values[ROOMROTATION])
															
															mx = Xoffset + (midx + math.cos(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
															my = Yoffset + (midy - math.sin(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
															
															part.x = mx + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
															part.y = my + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
															part.values[XPOS] = part.x
															part.values[YPOS] = part.y
															
															part.values[XVEL] = math.cos(udir) * 10
															part.values[YVEL] = math.sin(udir) * -10
														end
													else
														partid = MF_staticparticle(particle,i,j,c1,c2,layer)
													end
												end
											else
												if (specialrule ~= "nojitter") then
													partid = MF_particle(particle,i,j,c1,c2,layer)
													
													if (partid ~= nil) and (specialrule == "visitrule") and (partid ~= 0) then
														local udir
														if cause == "level" then
															udir = mapdir * math.pi / 2
														elseif cause == "empty" then
															udir = math.random(0,3) * math.pi / 2
														end
														
														local part = mmf.newObject(partid)
														
														part.values[ONLINE] = 2
														local midx = math.floor(roomsizex * 0.5)
														local midy = math.floor(roomsizey * 0.5)
														local mx = i + 0.5 - midx
														local my = j + 0.5 - midy
														
														local dir = 0 - math.atan2(my, mx)
														local dist = math.sqrt(my ^ 2 + mx ^ 2)
														local roomrad = math.rad(generaldata2.values[ROOMROTATION])
														
														mx = Xoffset + (midx + math.cos(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
														my = Yoffset + (midy - math.sin(dir + roomrad) * dist * zoom) * tilesize * spritedata.values[TILEMULT]
														
														part.x = mx + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
														part.y = my + math.random(0 - tilesize * 0.5 * zoom,tilesize * 0.5 * zoom)
														part.values[XPOS] = part.x
														part.values[YPOS] = part.y
														
														part.values[XVEL] = math.cos(udir) * 10
														part.values[YVEL] = math.sin(udir) * -10
													end
												else
													partid = MF_staticparticle(particle,i,j,c1,c2,layer)
												end
											end
										end
									end
								end
							end
						end
					end
				end
			end
		end
	end
end