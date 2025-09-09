local mega_modpack_version = "1.3.4"
local mega_modpack_name = string.format("Mega Modpack V%s - by Plasmaflare", mega_modpack_version)
local mega_modpack_name_with_color = string.format("Mega Modpack V%s - by $1,4Plasma$3,4flare$0,3", mega_modpack_version)

local mod_list = {
    {name = "Plasma's Modpack",     author = "Plasmaflare",             color={4,4}},
    {name = "Patashu's Modpack",    author = "Patashu",                 color={3,1}},
    {name = "Persist",              author = "Randomizer",              color={0,3}},
    {name = "Past",                 author = "EmilyEmmi",               color={3,1}},
    {name = "Stringwords",          author = "Wrecking Games",          color={3,4}},
    {name = "Word Salad",           author = "Huebird",                 color={2,1}},
    {name = "Visit",                author = "Btd456Creeper",           color={0,3}},
    {name = "Glitch",               author = "Btd456Creeper",           color={0,3}},
    {name = "Nuhuh",                author = "Btd456Creeper",           color={0,3}},
    {name = "Offset",               author = "Btd456Creeper",           color={0,3}},
    {name = "Local",                author = "Mathguy",                 color={1,3}},
}

local mod_setting_data = {
    persist = {
        key = "persist_settings",
        button_label = "Persist Settings",
        page_title = "$0,3Persist Settings",
        cfg_section = "Persist",
        color = {0,3},
        settings_apply_func = apply_persist_settings,
        settings = {
            allow_persist_in_editor = {
                name = "allow_persist_in_editor",
                display = "Allow persist in editor",
                default = 0,
                buttonfunc = "ws_persistInEditor",
                tooltip = "While in editor, allows the effects of PERSIST to carry over to the next level you play. Good for testing but can easily get chaotic."
            },
        },
        settings_order = {"allow_persist_in_editor"}
    },
    patashu = {
        key = "patashu_settings",
        button_label = "Patashu's Modpack Settings",
        page_title = "$3,1Patashu's$0,3 Modpack Settings",
        cfg_section = "Patashu's Mods",
        color = {3,1},
        settings_apply_func = apply_patashu_settings,
        settings = {
            very_drunk = {
                name = "very_drunk",
                display = "Very drunk",
                default = 1,
                buttonfunc = "patashudrunk",
                tooltip = "If true, MOONWALK and related properties (DRUNK, DRUNKER, SKIP) apply to PUSH, PULL, SHIFT, LAUNCH and more. Defaults to true"
            },
            float_breaks_sticky = {
                name = "float_breaks_sticky",
                display = "Float breaks sticky",
                default = 1,
                buttonfunc = "patashufloatsticky",
                tooltip = "If true, two things at different float values can't stick together. Defaults to true"
            },
            very_sticky = {
                name = "very_sticky",
                display = "Very sticky",
                default = 0,
                buttonfunc = "patashusticky",
                tooltip = "If true, two things with different names can stick together. Defaults to false"
            },
        },
        settings_order = {"very_drunk", "float_breaks_sticky", "very_sticky"},
    },
    word_salad = {
        key = "word_salad_settings",
        button_label = "Word Salad Settings",
        page_title = "$2,1Word Salad$0,3 Settings",
        cfg_section = "Word Salad",
        color = {2,1},
        settings_apply_func = apply_word_salad_settings,
        settings = {
            music_when_only_vessels = {
                name = "music_when_only_vessels",
                display = "Music when only vessels",
                default = 0,
                buttonfunc = "ws_musicOnOnlyVessels",
                tooltip = "If true, the music will still play when there are only vessels on the level"
            },
            do_hop_particles = {
                name = "do_hop_particles",
                display = "Do hop particles",
                default = 1,
                buttonfunc = "ws_hopParticles",
                tooltip = "If true, objects that are HOP will spawn some particles after a successful jump"
            },
            keep_level_karma = {
                name = "keep_level_karma",
                display = "Keep level karma",
                default = 1,
                buttonfunc = "ws_keepLevelKarma",
                tooltip = "If true, keeps the sinful status of a level upon entering"
            },
            can_enter_any = {
                name = "can_enter_any",
                display = "Can Enter Any",
                default = 1,
                buttonfunc = "ws_canEnterAny",
                tooltip = "If true, LEVEL IS ENTER behavior also applies to any X IS ENTER if X has valid level data"
            },
            echo_word_units = {
                name = "echo_word_units",
                display = "Echo Word Units",
                default = 1,
                buttonfunc = "ws_echoWordUnits",
                tooltip = "If true, ECHO units can echo WORD units"
            },
        },
        settings_order = {"music_when_only_vessels", "do_hop_particles", "keep_level_karma", "can_enter_any", "echo_word_units"}
    },
    btd456creeper = {
        key = "btd456creeper_settings",
        button_label = "Btd456creeper Settings",
        page_title = "$0,3Btd456creeper$0,3 Settings",
        cfg_section = "btd456creeper",
        color = {0,3},
        settings_apply_func = apply_btdcreeper_glitch_mod_settings, --@TODO: if there are multiple seperate mods, probably add support for calling multiple functions
        settings = {
            infloop_level_glitch = {
                name = "infloop_level_glitch",
                display = "Infinite loop level glitch",
                default = 0,
                buttonfunc = "ws_infloopLevelGlitch",
                tooltip = "Infinite loop-ed levels transform into glitch objects. Only works if glitch object is in the world's pallete (like with normal transforms)"
            },
        },
        settings_order = {"infloop_level_glitch"}
    },
    -- NIMI MOD EDIT: add setting for throw
    shrugsimontiger = {
        key = "shrugsimontiger_settings",
        button_label = "Shrugsimontiger Settings",
        page_title = "$1,4Shrugsimontiger$0,3 Settings",
        cfg_section = "shrugsimontiger",
        color = {1,4},
        settings_apply_func = apply_shrugsimontiger_settings,
        settings = {
            throw_push = {
                name = "throw_push",
                display = "Throw-push",
                default = 1,
                buttonfunc = "sh_throw_push",
                tooltip = "If true, and a Throw object tries to push, Throw effects will apply to that too."
            }
        },
        settings_order = {"throw_push"}
    }
}
local mod_setting_order = {"patashu", "word_salad", "persist", "btd456creeper", "shrugsimontiger"}

local function setfenv(fn, env)
    local i = 1
    while true do
    local name = debug.getupvalue(fn, i)
    if name == "_ENV" then
        debug.upvaluejoin(fn, i, (function()
        return env
        end), 1)
        break
    elseif not name then
        break
    end

    i = i + 1
    end

    return fn
end

local function restore_original_levelpack_menufunc()
    local new_env = {}
    setmetatable(new_env, {__index = _G})

    local f, err = loadfile("Data/Editor/editor_menudata.lua")
    assert(f, err)
    setfenv(f, new_env)
    local status, err = pcall(f)
    assert(status, err)

    for k,v in pairs(new_env) do
        print(k)
    end

    menufuncs.level.enter = new_env.menufuncs.level.enter
end
restore_original_levelpack_menufunc()

local function make_plasma_button(buttonfunc, name, buttonid, label, x, y, selected, tooltip)
    local width = (#label + 4) * baba_font_consts.total_letter_w
    local scale = width / baba_font_consts.button_w
    local final_x = x + width / 2
    createbutton(buttonfunc,final_x,y,2,scale,1,label,name,3,2,buttonid, nil, selected, tooltip)
end

local function write_settings_header(title, c1, c2, name)
    local c1_str = tostring(c1)
    local c2_str = tostring(c2)
    writetext(title, -1, screenw * 0.5, f_tilesize, name, true)
    writetext("$"..c1_str..","..c2_str.."---------------------------------------------------------------------------------$0,3", -1, 20, f_tilesize * 2, name)
    writetext("$"..c1_str..","..c2_str.."---------------------------------------------------------------------------------$0,3", -1, 20, screenh - f_tilesize * 2, name)
end

local function read_setting(section, setting_data)
    local value = MF_read("world",section,setting_data.name)
    if #value == 0 then
        value = setting_data.default
    end
    return value
end

local function display_modpack_setting_button()
    local buttonstring = "Mega Modpack Settings"
    local x = screenw-( (#buttonstring + 5) * baba_font_consts.total_letter_w)
    local y = f_tilesize * 1.5
    make_plasma_button("mega_modpack_settings", "level", menufuncs.level.button, buttonstring, x, y, false)
end
display_modpack_setting_button()

local old_level_func = menufuncs.level.enter
menufuncs.level.enter = function(...)
    old_level_func(...)
    display_modpack_setting_button()
end

menufuncs.mega_modpack_settings = {
    button = "MegaModpackSettings",
    escbutton = "mega_mod_return",
    slide = {1,0},
    enter = function(parent,name,buttonid,extra)
        MF_letterclear("leveltext")
        MF_cursorvisible(0) -- Letting this be off until Hempuli fixes "escbutton" field not working

        local dynamic_structure = {}

        write_settings_header("Mega Modpack Settings", 0, 3, name)

        local item_x = screenw * 0.1
        local item_y = f_tilesize
        createbutton("mega_mod_return",screenw * 0.15,item_y,2,8,1,langtext("return"),name,3,2,buttonid)
        table.insert(dynamic_structure, {{"mega_mod_return"}})

        -- Render sub-settings buttons
        item_y = item_y + f_tilesize * 3        
        make_plasma_button("pfsettings", name, buttonid, "Plasma Modpack Settings", item_x,item_y, false)
        item_y = item_y + f_tilesize * 2
        table.insert(dynamic_structure, {{"pfsettings"}})

        for _, mod_name in ipairs(mod_setting_order) do
            local mod_data = mod_setting_data[mod_name]
            make_plasma_button(mod_data.key, name, buttonid, mod_data.button_label, item_x,item_y, false)
            item_y = item_y + f_tilesize * 2
            table.insert(dynamic_structure, {{mod_data.key}})
        end

        item_x = screenw * 0.5
        item_y = f_tilesize * 4
        writetext("Full list of mods and their authors:", -1, item_x, item_y, name)

        for _, mod_info in ipairs(mod_list) do
            item_y = item_y + f_tilesize
            writetext(string.format("$0,2♄ %s - $%d,%d%s$0,3", mod_info.name, mod_info.color[1], mod_info.color[2], mod_info.author), -1, item_x, item_y, name)
        end
        item_y = item_y + f_tilesize * 2
        writetext("...Damn, that's a lot of mods.", -1, item_x, item_y, name)

        -- make_plasma_button("metatextsettings", name, buttonid, "Metatext Settings", screenw * 0.15,item_y, false)

        local version_x = screenw-(#mega_modpack_name * 10) - 20
        local version_y = screenh - f_tilesize
        writetext(mega_modpack_name_with_color, -1, version_x, version_y, name)

        buildmenustructure(dynamic_structure)
    end
}
buttonclick_list["mega_modpack_settings"] = function()
    MF_menubackground(true)
    changemenu("mega_modpack_settings")
end
buttonclick_list["mega_mod_return"] = function()
    MF_menubackground(false)
    changemenu("level")
end
buttonclick_list["pfreturn"] = function()
    changemenu("mega_modpack_settings")
end

local gd = nil

for name, data in pairs(mod_setting_data) do
    local mod_name, mod_data = name, data
    local revert_settings_key = "revert_"..mod_data.key

    menufuncs[mod_data.key] = {
        button = "test",
        escbutton = "mega_modpack_settings",
        slide = {1,0},
        enter = function(parent,name,buttonid,extra)
            MF_letterclear("leveltext")
            MF_cursorvisible(0) -- Letting this be off until Hempuli fixes "escbutton" field not working

            local dynamic_structure = {}
    
            write_settings_header(mod_data.page_title, mod_data.color[1], mod_data.color[2], name)
    
            local item_x = screenw * 0.1
            local item_y = f_tilesize
            createbutton("mega_modpack_settings",screenw * 0.15,item_y,2,8,1,langtext("return"),name,3,2,buttonid)
            table.insert(dynamic_structure, {{"mega_modpack_settings"}})
            item_y = item_y + f_tilesize * 3 
    
            for _, setting_name in ipairs(mod_data.settings_order) do
                local data = mod_data.settings[setting_name]
                local toggle = read_setting(mod_data.cfg_section, data)
                local togglevalue, color = gettoggle(toggle)
                make_plasma_button(data.buttonfunc, name, buttonid, data.display, item_x, item_y, togglevalue, data.tooltip)
                table.insert(dynamic_structure, {{data.buttonfunc}})
                
                item_y = item_y + f_tilesize * 2
            end

            item_y = screenh - f_tilesize
            make_plasma_button(revert_settings_key, name, buttonid, "Restore default settings", 20, item_y)
            table.insert(dynamic_structure, {{revert_settings_key}})

            gd = MF_create("object001")
            MF_loadsprite(gd,"text_persist_0",27,true)
            local testunit = mmf.newObject(gd)
            testunit.layer = 2
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

    for setting_name, data in pairs(mod_data.settings) do
        buttonclick_list[data.buttonfunc] = function()
            local toggle = read_setting(mod_data.cfg_section, data)
            local togglevalue, color = gettoggle(toggle)
    
            if togglevalue == 1 then
                togglevalue = 0
            else
                togglevalue = 1
            end
            MF_store("world", mod_data.cfg_section, setting_name, togglevalue)
    
            local buttons = MF_getbutton(data.buttonfunc)
            for i,unitid in ipairs(buttons) do
                updatebuttoncolour(unitid, togglevalue)
            end
        end
    end

    buttonclick_list[mod_data.key] = function()
        changemenu(mod_data.key)
    end

    buttonclick_list[revert_settings_key] = function()
        for setting_name, data in pairs(mod_data.settings) do
            local value = data.default
            MF_store("world", mod_data.cfg_section, setting_name, value)
            local buttons = MF_getbutton(data.buttonfunc)
            for i,unitid in ipairs(buttons) do
                updatebuttoncolour(unitid, value)
            end
        end
    end
end

table.insert(mod_hook_functions["level_start"], 
    function()
        for mod_name, mod_data in pairs(mod_setting_data) do
            local settings_dict = {}
            for setting_name, data in pairs(mod_data.settings) do
                local value = read_setting(mod_data.cfg_section, data)
                local togglevalue = gettoggle(value) == 1

                settings_dict[setting_name] = togglevalue
            end

            mod_data.settings_apply_func(settings_dict)
        end        
    end
)