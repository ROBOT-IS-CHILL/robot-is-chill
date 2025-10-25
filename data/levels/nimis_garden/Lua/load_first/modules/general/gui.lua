local modpack_name = string.format("Plasmaflare's Mods V%s", plasma_modpack_version)
local modpack_name_with_color = string.format("$1,4Plasma$3,4flare$0,3's Mods V%s", plasma_modpack_version)
local PlasmaSettings = {}

MF_loadsound("gd")

local function write_modpack_version()
    local x = screenw - (baba_font_consts.total_letter_w * (#modpack_name - 2))
    local y = f_tilesize * 1.5
    writetext(modpack_name_with_color, -1, x, y, "level")
end

local plasma_modpack_settings = {
    disable_dir_shift = {
        display = "Disable net shifting",
        value = false,
        buttonfunc = "pfdirshift",
        tooltip = "Disables SHIFT logic change where it combines SHIFT sources to one overall SHIFT movement. THIS DISABLES DIRECTIONAL SHIFT."
    },
    disable_stable_display = {
        display = "Disable stable text display",
        value = false,
        buttonfunc = "pfstabledisplay",
        tooltip = "Disables showing a list of rules when hovering a STABLE object with mouse."
    },
    disable_guard_chain = {
        display = "Disable guard chaining",
        value = false,
        buttonfunc = "pfchainguard",
        tooltip = "Disables recursive guarding. This makes GUARD simpler to figure out in puzzles but reduces complex guarding interactions."
    },
}
local settings_order = {"disable_dir_shift", "disable_stable_display", "disable_guard_chain"}

local gd = nil
function PlasmaSettings.get_toggle_setting(setting)
    if not plasma_modpack_settings[setting] then
        error(string.format("Plasma modpack: setting not defined %s", setting))
    end

    local setting = gettoggle(MF_read("world","Plasma Mods",setting))
    return setting == 1
end

local function make_plasma_button(buttonfunc, name, buttonid, label, x, y, selected, tooltip)
    local width = (#label + 4) * baba_font_consts.total_letter_w
    local scale = width / baba_font_consts.button_w
    local final_x = x + width / 2
    createbutton(buttonfunc,final_x,y,2,scale,1,label,name,3,2,buttonid, nil, selected, tooltip)
end

local function display_modpack_setting_button()
    if generaldata.values[MODE] == 5 then --"So it doesn't run when you start the pack outside of the editor" - Thanks metatext mod!
        local buttonstring = "Plasma Modpack Settings"
        local x = screenw-( (#buttonstring + 5) * baba_font_consts.total_letter_w)
        local y = f_tilesize * 1.5
        make_plasma_button("pfsettings", "level", menufuncs.level.button, buttonstring, x, y, false)
    end
end

local old_level_func = menufuncs.level.enter
menufuncs.level.enter = function(...)
    old_level_func(...)
    display_modpack_setting_button()
end

-- display_modpack_setting_button()

local structure = {}
table.insert(structure, {{"pfreturn"}})
for _, setting_name in ipairs(settings_order) do
    local data = plasma_modpack_settings[setting_name]
    table.insert(structure, {{data.buttonfunc}})
end

menufuncs.pfsettings = {
    button = "PlasmaModpackMenu",
    escbutton = "pfreturn",
    slide = {1,0},
    enter = function(parent,name,buttonid,extra)
        MF_letterclear("leveltext")
        MF_cursorvisible(0) -- Letting this be off until Hempuli fixes "escbutton" field not working

        local dynamic_structure = {}

        writetext("Plasma Modpack Settings", -1, screenw * 0.5, f_tilesize, name, true)
        writetext("$4,4---------------------------------------------------------------------------------$0,3", -1, 20, f_tilesize * 2, name)
        local version_x = screenw-(#modpack_name * 10) - 20
        local version_y = screenh - f_tilesize
        writetext(modpack_name_with_color, -1, version_x, version_y, name)
        writetext("This gui is mostly WIP. There may be a few weird bugs here and there lol.", -1, 20, version_y - f_tilesize*2, name)
        writetext("$4,4---------------------------------------------------------------------------------$0,3", -1, 20, version_y - f_tilesize, name)

        createbutton("pflol",726,409,2,0.5,0.7,"",name,3,2,buttonid)
        
        local item_x = screenw * 0.1
        local item_y = f_tilesize
        createbutton("pfreturn",screenw * 0.15,item_y,2,8,1,langtext("return"),name,3,2,buttonid)

        table.insert(dynamic_structure, {{"pfreturn"}})
        
        item_y = item_y + f_tilesize * 3
        
        for _, setting_name in ipairs(settings_order) do
            local data = plasma_modpack_settings[setting_name]
            local toggle = MF_read("world","Plasma Mods",setting_name)
            if #toggle == 0 then
                toggle = 0
            end
            local togglevalue, color = gettoggle(toggle)
            make_plasma_button(data.buttonfunc, name, buttonid, data.display, item_x, item_y, togglevalue, data.tooltip)
            table.insert(dynamic_structure, {{data.buttonfunc}})
            
            item_y = item_y + f_tilesize * 2
        end

        item_y = screenh - f_tilesize

        make_plasma_button("revert_plasma_settings", name, buttonid, "Restore default settings", 20, item_y)
        table.insert(dynamic_structure, {{"revert_plasma_settings"}})

        gd = MF_specialcreate("customsprite")
        MF_loadsprite(gd,"text_gd",27,true)
        local testunit = mmf.newObject(gd)
        testunit.layer = 2
        testunit.visible = false
        testunit.direction = 27
        testunit.values[XPOS] = 619
        testunit.values[YPOS] = 294
        testunit.values[ONLINE] = 1

        buildmenustructure(dynamic_structure)
    end,
    leave = function(parent,name)
        MF_cleanremove(gd)
        gd = nil
    end
}

buttonclick_list["pfsettings"] = function()
    MF_menubackground(true)
    changemenu("pfsettings")
end
buttonclick_list["pfreturn"] = function()
    MF_menubackground(false)
    changemenu("level")
end
buttonclick_list["revert_plasma_settings"] = function()
    for setting_name, data in pairs(plasma_modpack_settings) do
        local value = data.value
        MF_store("world", "Plasma Mods", setting_name, value)
        local buttons = MF_getbutton(data.buttonfunc)
        for i,unitid in ipairs(buttons) do
            updatebuttoncolour(unitid, value)
        end
    end
end
buttonclick_list["pflol"] = function()
    if gd then
        local testunit = mmf.newObject(gd)
        testunit.visible = true

        local constant = {
            71, 79, 68, 32, 68, 65, 77, 77, 73, 84, 32, 75, 82, 73, 83, 32, 87, 72, 69, 82, 69, 32, 84, 72, 69, 32, 70, 38, 37, 75, 32, 65, 82, 69, 32, 87, 69, 33, 33, 63, 63
        }
        for i, num in ipairs(constant) do
            constant[i] = string.char(num)
        end
        writetext(string.format("$4,1%s$0,3", table.concat(constant)), -1, 416, 240, "pfsettings")
        MF_playsound("gd")
    end
end

for setting_name, data in pairs(plasma_modpack_settings) do
    buttonclick_list[data.buttonfunc] = function()
        local toggle = MF_read("world","Plasma Mods",setting_name)
        if #toggle == 0 then
            toggle = 0
        end
        local togglevalue, color = gettoggle(toggle)

        if togglevalue == 1 then
            togglevalue = 0
        else
            togglevalue = 1
        end
        MF_store("world", "Plasma Mods", setting_name, togglevalue)

        local buttons = MF_getbutton(data.buttonfunc)
        for i,unitid in ipairs(buttons) do
            updatebuttoncolour(unitid, togglevalue)
        end
    end
end

-- table.insert(mod_hook_functions["always"], 
--     function()
--         local mouse_x, mouse_y = MF_mouse()
--         MF_letterclear("mousecoordstest")
--         local display = string.format("(%d,%d)", mouse_x, mouse_y)
--         writetext(display, -1, mouse_x, mouse_y, "mousecoordstest", true, 1, true)
--     end
-- )

return PlasmaSettings