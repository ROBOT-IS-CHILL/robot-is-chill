past_pause = "9" -- key to pause past replay.
past_go = "0" -- key fast-forward or go forward one step.
--[[ List of valid keys (from values.lua)
"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
"1","2","3","4","5","6","7","8","9","0"
"Right","Up","Left","Down"
".",",","-","+","´","'","¨","§","<"
"Control","Shift","Return","Esc","Space","Backspace" ]]--

-- Here we add past to the object list

table.insert(editor_objlist_order, "text_past")

-- This defines the exact data for it (note that since the sprite is specific to this levelpack, sprite_in_root must be false!)

editor_objlist["text_past"] = {
  name = "text_past",
  unittype = "text",
  sprite_in_root = false,
  tags = {"text","text_condition","text_prefix","abstract","past"},
  tiling = -1,
  type = 3,
  layer = 20,
  colour = {1, 1},
  colour_active = {3, 2},
  advanced = true,
}

-- After adding new objects to the list, formatobjlist() must be run to setup everything correctly.

formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
  keys.WORD_GLOSSARY_FUNCS.register_author("EmilyEmmi", {3,1} )
  keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
      {
          name = "past",
          author = "EmilyEmmi",
          description =
[[When a "PAST" rule is made, a replay of the player's inputs will start. During the replay, rules made using "PAST" will be applied on every turn, potentially changing the current present to a different outcome.

Rules using "NOT PAST" will be disabled during a past replay.]],
      }
  })
end

--[[ Adds the rules for the past condition, it is true if we are running the past rules
This allows Not Past to function ]]--
condlist["past"] = function(params,checkedconds,checkedconds_,cdata)
	local result = doingpast == true
	return result, checkedconds
end

--[[ 
  @mods(past) - An alternate system to storing the "donepast" property per unit. This stores unit.values[ID]'s
  of all texts involved in a past rule. unit.values[ID] is more effective since it (should be) consistent across
  undos.

  This fixes an issue with the original past mod, where forming a past rule using transformed/created text causes an infinite
  past replay loop. The original past mod relied on storing a boolean, called "donepast", within the unit itself. donepast is 
  used to mark which texts to not restart the past replay on if that text is used to form a past rule. This variable
  persists as long as the containing unit persists. When starting a past replay, it continuosly calls undo() until the undo stack
  is empty before replaying the events. And when undoing a "create" event, it *deletes* the object instead!
  SO, donepast gets deleted! And therefore the infinite loop happens!
 ]]
donepast_units = {}

function add_donepast_unit(unit)
  donepast_units[unit.values[ID]] = true
end
function remove_donepast_unit(unit)
  donepast_units[unit.values[ID]] = nil
end
function has_donepast_unit(unit)
  return donepast_units[unit.values[ID]] ~= nil
end

-- This will reset certain functions upon level start/restart
local function resetstuff()
  undowasupdated = false
  amundoing = false
  keyssofar = {}
  redokeys = {}
  doingpast = false
  inputstatus = {false,1}
  pastrules = {}
  prepastrules = {}
  startpoint = 1
  donepast_units = {}
end
table.insert( mod_hook_functions["level_start"],resetstuff)
table.insert( mod_hook_functions["level_restart"],resetstuff)
-- This adds the keys that need to be replayed
table.insert( mod_hook_functions["undoed"],
  function()
    if doingpast == false then
      table.insert(redokeys,keyssofar[#keyssofar])
      table.remove(keyssofar, #keyssofar)
    end
    amundoing = true
  end
)
-- Adds past rules to rule list.
table.insert( mod_hook_functions["rule_baserules"],
  function()
    if doingpast and #pastrules > 0 then
      prepastrules = {}
      for i,rules in ipairs(pastrules) do
        table.insert(prepastrules,rules)
        local rule = rules[1]
        local conds = rules[2]
        local ids = rules[3]
        local tags = rules[4]
        local newconds = {}
        local newtags = {}

        for c,d in ipairs(conds) do
          table.insert(newconds, d)
        end

        for c,d in ipairs(tags) do
          table.insert(newtags, d)
        end

        table.insert(newtags, "past")

        if (ids ~= nil) then
          local idlist = {}

          if (#ids > 0) then
            for a,b in ipairs(ids) do
              table.insert(idlist, b)
            end
          end

          if (#idlist > 0) then
            for a,d in ipairs(idlist) do
              for c,b in ipairs(d) do
                if (b ~= 0) then
                  local bunit = mmf.newObject(b)

                  add_donepast_unit(bunit)
                end
              end
            end
          end
        end

        local newword1 = rule[1]
        local newword2 = rule[2]
        local newword3 = rule[3]

        local newrule = {newword1, newword2, newword3}
        addoption(newrule,newconds,{},true,nil,newtags)
      end
    end
  end
)

-- Keeps track of key presses.
function past_addkey(keyid,player,keyid2)
	if undowasupdated or doingpast then
		table.insert(keyssofar,{keyid,player,keyid2})
    undowasupdated = false
	end
end

-- Keeps a variable so the past_addkey function knows when to run.

--[[ @Merge: newundo() was merged ]]


--Now handles direct keyids, turns off during past turns, and add keys.

--[[ @Merge: command() was merged ]]


-- Turns off auto during past turns and adds keys.

--[[ @Merge: command_auto() was merged ]]


--[[ The other custom function, and this is a big one.
Replays all inputs if a past rule exists, adding the past rules into the replay.
Note that RNG events will happen differently in the replay ]]--
function dopast()
  local runpast = false
  amundoing = false
	if #pastrules <= 100 then
		if doingpast == false then
			pastrules = {}
		end
		for i,rules in ipairs(visualfeatures) do
			local rule = rules[1]
			local conds = rules[2]
			local ids = rules[3]
			local tags = rules[4]
      local valid = false

			for a,b in ipairs(conds) do
				if b[1] == "past" then
					valid = true
				end
      end
      for c,d in ipairs(tags) do
        if (d == "past") or (d == "mimic") then
          valid = false
          break
        end
      end
			if valid == true then
  			local newconds = {}
  			local newtags = {}
  			local valid = true

  			for c,d in ipairs(conds) do
  				table.insert(newconds, d)
  			end

  			for c,d in ipairs(tags) do
  				table.insert(newtags, d)
  			end

  			table.insert(newtags, "past")

  			local newword1 = rule[1]
  			local newword2 = rule[2]
  			local newword3 = rule[3]

  			local newrule = {newword1, newword2, newword3}
  			local toinsert = {newrule,newconds,ids,newtags}

        if #prepastrules > 0 then
          for i,rules in ipairs(prepastrules) do
            local same = comparerules(newrule,rules[1])
            if same then
              valid = false
              break
            end
          end
        else
          valid = false
        end

  			if (ids ~= nil) and not valid then
  				local idlist = {}

  				if (#ids > 0) then
  					for a,b in ipairs(ids) do
  						table.insert(idlist, b)
  					end
  				end

  				if (#idlist > 0) then
  					for a,d in ipairs(idlist) do
              if valid then
                break
              end
  						for c,b in ipairs(d) do
  							if (b ~= 0) then
  								local bunit = mmf.newObject(b)

                  if not has_donepast_unit(bunit) then
  									valid = true
                    break
  								end
  							end
  						end
  					end
  				end
  			end

  			if valid then
  				runpast = true
  				table.insert(pastrules,toinsert)
  			end
			end
		end
		if runpast == true then
      if doingpast == false then
        prepastrules = {}
      end
      updatecode = 1
      if doingpast == false and startpoint == 1 then
        local backupbuffer = {}
        if #undobuffer > 1 then
  				generaldata2.strings[TURNSOUND] = "silent"
  				undo()
  			end
        newundo(true)
        updateundo = true
        doundo = true
        addundo({"maprotation",maprotation})
        addundo({"mapdir",mapdir})
        addundo({"levelupdate",Xoffset,Yoffset})
        for id,unit in pairs(units) do
          addundo({"remove",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init},id)
        end
        newundo()
        for thisundo,currentundo in ipairs(undobuffer) do
          table.insert(backupbuffer,currentundo)
        end
        table.remove(undobuffer,1)
        undobuffer[1] = {}
        while #undobuffer > 1 do
  				generaldata2.strings[TURNSOUND] = "silent"
  				undo()
  			end
        for thisundo,currentundo in ipairs(backupbuffer) do
          table.insert(undobuffer,currentundo)
        end
        startpoint = #undobuffer
      else
  			while #undobuffer > startpoint do
  				generaldata2.strings[TURNSOUND] = "silent"
  				undo()
  			end
      end
			toredo = {}
			for i,v in ipairs(redokeys) do
				table.insert(toredo, redokeys[#redokeys - (i - 1)])
			end
			doingpast = true
      updatecode = 1
      code()
      effectblock()
      animate()
      pastthing(#toredo)
      pasttimemax = 15 - (#toredo / 10)
		end
	end
	if doingpast then
    updatecode = 1
    if toredo[1] ~= nil and not runpast then
      if doreset then
        --@mods(past x patashu) - if we hit a reset object during a past replay, stop the replay by clearing toredo
        toredo = {}
      else
        table.remove(toredo,1)
      end
    end
    if toredo[1] == nil then
      MF_letterclear("pasttime")
      MF_letterclear("pastrules")
      doingpast = false
    end
    pasttimer = 0
    if inputstatus[1] == false then
      inputstatus[1] = true
    end
  end
end

-- Prevents past rules from happening again.

--[[ @Merge: postrules() was merged ]]


-- This inputs keys if particles are enabled, to create the replay
table.insert( mod_hook_functions["always"],
  function()
    if doingpast and generaldata.values[MODE] == 0 and generaldata.values[IGNORE] ~= 1 then
      generaldata.values[IGNORE] = 2
      pasttimer = pasttimer + 1
      if (inputstatus[1] == true and (pasttimer > pasttimemax or MF_keydown(past_go))) or inputstatus[1] == "go" then
        if #pastrules <= 100 then
          pastthing(#toredo - 1)
          pasttimer = 0
          local v = toredo[1]
    			local keyid = v[1]
    			local pastplayer = v[2]
    			local keyid2 = 4
    			if v[3] ~= nil then
    				keyid2 = v[3]
    			end
    			if toredo ~= {} then
            if inputstatus[1] == true then
              inputstatus[1] = false
            elseif inputstatus[1] == "go" then
              inputstatus[1] = "stop"
            end
    			  command(nil,pastplayer,keyid,keyid2)
    			end
          if #toredo == 0 then
            for i=1,#undobuffer - startpoint + 1 do
              table.remove(undobuffer,2)
            end
            startpoint = 1
            table.remove(undobuffer,1)
            generaldata.values[IGNORE] = 0

            if doreset then
              --@mods(past x patashu) - if we hit a reset object during a past replay, reset the level like how patashu
              -- does it instead of doing other undo stuff with past
              resetlevel()
              MF_update()
            else
              updateundo = true
              doundo = true
              for id,unit in pairs(units) do
                addundo({"create",unit.strings[UNITNAME],unit.values[ID],-1,"create",unit.values[XPOS],unit.values[YPOS],unit.values[DIR], true}) --@mods(past) - the extra "true" argument marks this undo line as being created during a past replay
              end
              newundo()
            end
          end
        else
          for i=1,#undobuffer - startpoint do
            table.remove(undobuffer,2)
          end
          startpoint = 1
          table.remove(undobuffer,1)
          generaldata.values[IGNORE] = 0
          amundoing = false
          doingpast = false
          inputstatus = {false,1}
          pastrules = {}
          toredo = {}
          MF_letterclear("pasttime")
          MF_letterclear("pastrules")
          destroylevel("toocomplex")
          command("idle")
          table.remove(undobuffer,2)
        end
      end
      if MF_keydown(past_pause) then
        if inputstatus[2] == 0 then
          if inputstatus[1] ~= "stop" then
            inputstatus = {"stop",1}
            pastthing(#toredo)
          else
            pasttimer = pasttimemax
            inputstatus = {true,1}
          end
        end
      elseif MF_keydown(past_go) and inputstatus[1] == "stop" then
        if inputstatus[2] == 0 then
          inputstatus = {"go",1}
        end
      elseif inputstatus[2] == 1 then
        inputstatus[2] = 0
      end
    end
  end
)

-- Show turns left and rules in past replay.
function pastthing(turnsleft)
  MF_letterclear("pasttime")
  MF_letterclear("pastrules")
  if inputstatus[1] == "stop" or inputstatus[1] == "go" then
    writetext("(paused, press " .. past_go .. " to step. "..past_pause.." to continue replay)",-1,screenw * 0.5 - 12,screenh * 0.5 - 75,"pasttime",true,2,true,{3,2})
  else
    writetext("(hold " .. past_go .. " to fast forward. "..past_pause.." to pause replay)",-1,screenw * 0.5 - 12,screenh * 0.5 - 75,"pasttime",true,2,true,{3,2})
  end
  writetext("past turns left:" .. turnsleft,-1,screenw * 0.5 - 12,screenh * 0.5 - 60,"pasttime",true,2,true,{3,2})
  writepastrules("past","pastrules",427.0,216.0)
end
function writepastrules(parent,name,x_,y_)
	local basex = x_
	local basey = y_
	local linelimit = 12
	local maxcolumns = 4

	local x,y = basex,basey

	if (#pastrules > 0) then
		writetext(langtext("rules_colon"),0,x,y,name,true,1,true,{3,2})
	end

	local i_ = 1

	local count = 0
	local allrules = {}

	for i,rules in ipairs(pastrules) do
		local text = ""
		local rule = rules[1]

		text = text .. rule[1] .. " "

		local conds = rules[2]
		local ids = rules[3]
		local tags = rules[4]

		local fullinvis = true
		for a,b in ipairs(ids) do
			for c,d in ipairs(b) do
				local dunit = mmf.newObject(d)

				if dunit.visible then
					fullinvis = false
				end
			end
		end

		if (fullinvis == false) then
			if (#conds > 0) then
				for a,cond in ipairs(conds) do
					local middlecond = true

					if (cond[2] == nil) or ((cond[2] ~= nil) and (#cond[2] == 0)) then
						middlecond = false
					end

					if middlecond then
						text = text .. cond[1] .. " "

						if (cond[2] ~= nil) then
							if (#cond[2] > 0) then
								for c,d in ipairs(cond[2]) do
									text = text .. d .. " "

									if (#cond[2] > 1) and (c ~= #cond[2]) then
										text = text .. "& "
									end
								end
							end
						end

						if (a < #conds) then
							text = text .. "& "
						end
					else
						text = cond[1] .. " " .. text
					end
				end
			end

			local target = rule[3]
			local isnot = string.sub(target, 1, 4)
			local target_ = target

			if (isnot == "not ") then
				target_ = string.sub(target, 5)
			else
				isnot = ""
			end

			if (word_names[target_] ~= nil) then
				target = isnot .. word_names[target_]
			end

			text = text .. rule[2] .. " " .. target

			for a,b in ipairs(tags) do
				if (b == "mimic") then
					text = text .. " (mimic)"
				end
			end

			if (allrules[text] == nil) then
				allrules[text] = 1
				count = count + 1
			else
				allrules[text] = allrules[text] + 1
			end
			i_ = i_ + 1
		end
	end

	local columns = math.min(maxcolumns, math.floor((count - 1) / linelimit) + 1)
	local columnwidth = math.min(screenw - f_tilesize * 2, columns * f_tilesize * 10) / columns

	i_ = 1

	local maxlimit = 4 * linelimit

	for i,v in pairs(allrules) do
		local text = i

		if (i_ <= maxlimit) then
			local currcolumn = math.floor((i_ - 1) / linelimit) - (columns * 0.5)
			x = basex + columnwidth * currcolumn + columnwidth * 0.5
			y = basey + (((i_ - 1) % linelimit) + 1) * f_tilesize * 0.8
		end

		if (i_ <= maxlimit-1) then
			if (v == 1) then
				writetext(text,0,x,y,name,true,1,true,{3,2})
			elseif (v > 1) then
				writetext(tostring(v) .. " x " .. text,0,x,y,name,true,1,true,{3,2})
			end
		end

		i_ = i_ + 1
	end

	if (i_ > maxlimit-1) then
		writetext("(+ " .. tostring(i_ - maxlimit) .. ")",0,x,y,name,true,1,true,{3,2})
	end
end
