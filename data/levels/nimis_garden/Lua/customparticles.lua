particletypes.sandstorm =
{
	amount = 15,
	animation = 1,
	colour = {6, 2},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
			
			unit.values[XVEL] = math.random(-80,-40) * -0.5
			unit.values[YVEL] = math.random(5,15) * 0.2
				
			unit.scaleX = math.abs(unit.values[XVEL]) * 0.025
			unit.scaleY = math.abs(unit.values[XVEL]) * 0.025
			unit.flags[INFRONT] = true
		end,
}
particletypes.yellowstars =
{
	amount = 40,
	animation = {6, 7},
	colour = {3, 4},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
				
			unit.values[XVEL] = ((unit.direction - 6) + math.random(0,5) * 0.1) + 0.05
			--unit.values[YVEL] = math.random(40,80) * 0.05
				
			if (unit.direction == 7) then
				MF_setcolour(unitid,2,4)
					
				unit.strings[COLOUR] = tostring(2) .. "," .. tostring(4)
			end
				
			local size = math.random(2,5)
			unit.scaleX = size * 0.2
			unit.scaleY = size * 0.2
		end,
}
particletypes.snowyclouds = 
{
	amount = 50,
	animation = {1, 2, 12},
	colour = {0, 0},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
		
			if (unit.direction == 1) then --snow particles
				MF_setcolour(unitid,0,3)
				unit.strings[COLOUR] = tostring(0) .. "," .. tostring(3)
				
				unit.values[XVEL] = math.random(-50,-10) * -0.27
				unit.values[YVEL] = math.abs(unit.values[XVEL]) * (math.random(5,15) * 0.08)
				
				unit.scaleX = math.abs(unit.values[XVEL]) * 0.10
				unit.scaleY = math.abs(unit.values[XVEL]) * 0.10
				unit.flags[INFRONT] = true
			else 						--cloud particles
				unit.scaleX = 1 + math.random(-30,30) * 0.01
				unit.scaleY = unit.scaleX * 0.9
				
				unit.values[YVEL] = 0 - unit.scaleX
				unit.values[XVEL] = 0 - unit.scaleX
				--randomly invisible so theres less of them (bad way to do this but idc!!!)
				if(math.random(0,5) < 4) then
					unit.scaleX=0
					unit.scaleY=0
				end
			end
		end,
}
particletypes.coloursmoke = 
{
	amount = 30,
	animation = 3,
	colour = {1, 0},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
			
			unit.angle = math.random(0,359)
			
			unit.scaleX = 1 + math.random(-30,30) * 0.01
			unit.scaleY = unit.scaleX
			
			unit.values[YVEL] = -1
			unit.values[DIR] = math.random(-25,25) * 0.05

			local coloury1 = (math.random(0,1) * 2) + 1
			local coloury2 = math.random(0,1)
			MF_setcolour(unitid,coloury1,coloury2)
			unit.strings[COLOUR] = tostring(coloury1) .. "," .. tostring(coloury2)
		end,
}

particletypes.paravoid =
{
	amount = 30,
	animation = 11,
	colour = {1, 0},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
			
			if math.random(1,2) == 2 then
				MF_setcolour(unitid,5,0)
					
				unit.strings[COLOUR] = "5,0"
			end
			local rand = (math.random(8,18) * 0.35)
			unit.values[YVEL] = math.random(-50,20) * 0.075
			unit.values[XVEL] = math.random(-50,20) * 0.075
			unit.scaleX = 1.6 * rand
			unit.scaleY = 0.2 * rand
			MF_setshader(unitid,"add")
		end
}

particletypes.void =
{
	amount = 144,
	animation = 11,
	colour = {6, 4},
	extra = 
		function(unitid)
			local unit = mmf.newObject(unitid)
			
			unit.values[YVEL] = math.random(-3,-1)
			unit.values[XVEL] = math.random(-50,50) * 0.01
			unit.angle = math.random(0,89)
			unit.scaleX = 8 * (unit.values[YVEL] * -0.3)
			unit.scaleY = unit.values[YVEL] * -0.3
		end
}