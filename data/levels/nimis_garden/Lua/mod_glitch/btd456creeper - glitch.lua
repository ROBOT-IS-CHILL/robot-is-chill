--If true, "infinite loop"-ed levels transform into glitch objects.
--This only works if there is a glitch object in the world's palette (like with normal transforms).
--This also makes cursors not get destroyed by glitch objects.
--Change the "true" to "false" if you want to disable this behavior.
INFLOOP_LEVEL_GLITCH = true

function apply_btdcreeper_glitch_mod_settings(settings_dict)
	for setting_name, value in pairs(settings_dict) do
		if setting_name == "infloop_level_glitch" then
			INFLOOP_LEVEL_GLITCH = value
		end
	end
end


--A global table for handling objects and texts that are affected by glitch objects.
glitchtable = {}

--do not refer to me
--for practical purposes, i do not exist
--(Make "glitch" a special noun so it isn't included in "All" or "Not X" statements.)
table.insert(nlist.full, "glitch")
table.insert(nlist.short, "glitch")
table.insert(nlist.brief, "glitch")

--Add the glitch object to the editor.
--This object's functionality is entirely code-based, so no text is needed.
table.insert(editor_objlist_order, "glitch")
editor_objlist["glitch"] = 
{
	name = "glitch",
	sprite_in_root = false,
	unittype = "object",
	tags = {"special", "btd456creeper mods"},
	tiling = 1,
	type = 0,
	layer = 30,
	colour = {0, 3},
}
formatobjlist()

-- @Merge: Word Glossary Mod support
if keys.IS_WORD_GLOSSARY_PRESENT then
    keys.WORD_GLOSSARY_FUNCS.register_author("Btd456creeper", {0,3} )
    keys.WORD_GLOSSARY_FUNCS.add_entries_to_word_glossary({
        {
            name = "glitch",
			thumbnail_obj = "glitch",
			author = "Btd456creeper",
			description = 
[[An anomaly.
- Objects that overlap a glitch instantly get destroyed.

- Texts near a glitch that form a rule turn into more glitches.

- Glitches cannot be manipulated by rules.]],
        }
    })
end

--Color-changing effect for the glitch object, as well as making it unmovable and indestructible.
table.insert(mod_hook_functions["rule_baserules"],
	function()
		addbaserule("glitch","is","red")
        addbaserule("glitch","is","lime")
        addbaserule("glitch","is","cyan")
		addbaserule("glitch","is","safe")
		addbaserule("glitch","is","still")
		addbaserule("glitch","is","glitchprop")
		addbaserule("glitch","is","block")
	end
)

--Give it Broken particles too.
table.insert(mod_hook_functions["effect_always"],
	function()
		doeffect(0,nil,"glitchprop","error",1,20,1,{2,2})
        doeffect(0,nil,"glitchprop","error",1,20,1,{5,4})
        doeffect(0,nil,"glitchprop","error",1,20,1,{4,4})
	end
)

--Mark everything on the same tile as a glitch object, except for itself, by adding its ID to a table.
--Marked objects will be deleted later, and spread the glitch instead of parsing in rules.
function doglitchmarks()
	for i,glitchid in ipairs(findall({"glitch"})) do
		local glitch = mmf.newObject(glitchid)
		for j,goner in ipairs(findallhere(glitch.values[XPOS],glitch.values[YPOS])) do
			if (goner ~= glitch.fixed and (INFLOOP_LEVEL_GLITCH == false or not hasfeature(getname(mmf.newObject(goner)),"is","select",goner))) then
				table.insert(glitchtable, goner)
			end
		end
	end
end

--Delete every marked object.
function doglitchdelete()
	if (#glitchtable > 0) then
		setsoundname("removal",1)
	end
	GLOBAL_disable_guard_checking = true -- @Merge(plasma x glitch) Make it so that guarded objects will still be destroyed by the glitch. No one can escape the corruption.
	handledels(glitchtable)
	GLOBAL_disable_guard_checking = false
end

--Implement the last two functions to run before/after code().
local oldcode = code
function code(alreadyrun_)
	glitchtable = {}
	doglitchmarks()
	oldcode(alreadyrun_)
	doglitchdelete()
end


--Checkglitchrule: Returns true if at least one text/object making up a rule is marked as glitched.
function checkglitchrule(rule3)
	for i,textunitid in ipairs(rule3) do
		for j,v in ipairs(glitchtable) do
				if (v == textunitid) then
					return true
				end
			end
		end
	return false
end

--Spreadglitches: Creates new glitch objects at the locations of the given objects, by ID.
function spreadglitches(rule)
	for i,word in ipairs(rule) do
		for j,spreadunitid in ipairs(word[3]) do
			local spreadunit = mmf.newObject(spreadunitid)
			local x = spreadunit.values[XPOS]
			local y = spreadunit.values[YPOS]
			if (#findunitat("glitch",x,y) == 0) then
				create("glitch",x,y,0)
			end
		end
	end

	randcolor = fixedrandom(0,2)
	for i,glitch in ipairs(findall({"glitch"})) do
		mmf.newObject(glitch).currcolour = randcolor
	end
end

--Override to docode to put in the above code for glitches affecting rules.

--[[ @Merge: docode() was merged ]]


--If an infinite loop happens and the relevant setting is true, transform the level into a glitch object.

--[[ @Merge: destroylevel() was merged ]]


--Override to dolevelconversions.
--This function usually checks if the level isn't destroyed before transforming it.
--Since we want destroying the level via infinite loop to do something, this must be changed.

--[[ @Merge: dolevelconversions() was merged ]]


--Override to findgroup, adding a single line to call dolevelconversions.
--Apparently, this type of infinite loop is special somehow and doesn't transform the level unless this is added.

--[[ @Merge: findgroup() was merged ]]
