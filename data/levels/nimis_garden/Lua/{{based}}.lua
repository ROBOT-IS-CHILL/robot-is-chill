local version = "1.7"
timedmessage(string.format("You are using the $0,2($0,3($2,2B$2,4A$5,4S$4,4E$3,0D$0,3)$0,2)$0,3 mod v%s!", version))

-- Get this lua file's script path to be able to use loadfile()
local function script_path()
    local str = debug.getinfo(1).source:sub(2)
    return str:match("(.*/)")
end


local function get_object_ref(name)
    return editor_objlist[editor_objlist_reference[name]]
end

local function deep_copy_table(table)
    local copy = {}
    for k,v in pairs(table) do
        if type(v) == "table" then
            v = deep_copy_table(v)
        end
        copy[k] = v
    end

    return copy
end

local State_Status = {
    RUNNING = 0,
    STOPPED_VALID = 1,
    STOPPED_INVALID = 2,
}

-- Words that can be in the form of <word><string of text> (Ex: group123123)
local variable_suffix_words = {"group", "powered", "power"}

local function update_syntax_state(state, parsing_data, tiletype, wordid, word, sent)
    local stage = state.stage
    local prevstage = state.prevstage
    local prevtiletype = state.prevtiletype
    local stage2reached = state.stage2reached
    local stage3reached = state.stage3reached
    local doingcond = state.doingcond
    local nocondsafterthis = state.nocondsafterthis
    
    local stop = false
    local prevword = state.prevword

    -- This is a copy-paste from rules.lua itself. Original credit goes to Hempuli.
    if (tiletype ~= 5) then
        if (stage == 0) then
            if (tiletype == 0) then
                prevstage = stage
                stage = 2
            elseif (tiletype == 3) then
                prevstage = stage
                stage = 1
            elseif (tiletype ~= 4) then
                prevstage = stage
                stage = -1
                stop = true
            end
        elseif (stage == 1) then
            if (tiletype == 0) then
                prevstage = stage
                stage = 2
            elseif (tiletype == 6) then
                prevstage = stage
                stage = 6
            elseif (tiletype ~= 4) then
                prevstage = stage
                stage = -1
                stop = true
            end
        elseif (stage == 2) then
            if (wordid ~= #sent) then
                if (tiletype == 1) and (prevtiletype ~= 4) and ((prevstage ~= 4) or doingcond or (stage3reached == false)) then
                    stage2reached = true
                    doingcond = false
                    prevstage = stage
                    nocondsafterthis = true
                    stage = 3
                elseif (tiletype == 7) and (stage2reached == false) and (nocondsafterthis == false) and ((doingcond == false) or (prevstage ~= 4)) then
                    doingcond = true
                    prevstage = stage
                    stage = 3
                elseif (tiletype == 6) and (prevtiletype ~= 4) then
                    prevstage = stage
                    stage = 4
                elseif (tiletype ~= 4) then
                    prevstage = stage
                    stage = -1
                    stop = true
                end
            else
                stage = -1
                stop = true
            end
        elseif (stage == 3) then
            stage3reached = true
            
            if (tiletype == 0) or (tiletype == 2) or (tiletype == 8) then
                prevstage = stage
                stage = 5
            elseif (tiletype ~= 4) then
                stage = -1
                stop = true
            end
        elseif (stage == 4) then
            if (wordid <= #sent) then
                if (tiletype == 0) or ((tiletype == 2) and stage3reached) or ((tiletype == 8) and stage3reached) then
                    prevstage = stage
                    stage = 2
                elseif ((tiletype == 1) and stage3reached) and (doingcond == false) and (prevtiletype ~= 4) then
                    stage2reached = true
                    nocondsafterthis = true
                    prevstage = stage
                    stage = 3
                elseif (tiletype == 7) and (nocondsafterthis == false) and ((prevtiletype ~= 6) or ((prevtiletype == 6) and doingcond)) then
                    doingcond = true
                    stage2reached = true
                    prevstage = stage
                    stage = 3
                elseif (tiletype ~= 4) then
                    prevstage = stage
                    stage = -1
                    stop = true
                end
            else
                stage = -1
                stop = true
            end
        elseif (stage == 5) then
            if (wordid ~= #sent) then
                if (tiletype == 1) and doingcond and (prevtiletype ~= 4) then
                    stage2reached = true
                    doingcond = false
                    prevstage = stage
                    nocondsafterthis = true
                    stage = 3
                elseif (tiletype == 6) and (prevtiletype ~= 4) then
                    prevstage = stage
                    stage = 4
                elseif (tiletype ~= 4) then
                    prevstage = stage
                    stage = -1
                    stop = true
                end
            else
                stage = -1
                stop = true
            end
        elseif (stage == 6) then
            if (tiletype == 3) then
                prevstage = stage
                stage = 1
            elseif (tiletype ~= 4) then
                prevstage = stage
                stage = -1
                stop = true
            end
        end
    end


    -- This section figures out how to interpret the currently parsed word to be formatted in the overall rule. 
    -- Most of this logic is based on the parsing diagram in my guide here https://github.com/PlasmaFlare/baba-modding-guide/blob/master/images/Baba%20syntax%20diagram.png
    -- Key reference:
    --      state.stage = previous stage
    --      stage = current stage

    if tiletype == 4 then
        -- handling "not"
        if parsing_data.not_modifier == "not " then
            parsing_data.not_modifier = ""
        else
            parsing_data.not_modifier = "not "
        end
    else
        if (tiletype == 5) then
            if prevword == "play" then
                parsing_data.curr_play_note = parsing_data.curr_play_note..word
                if wordid == #sent then
                    table.insert(parsing_data.effects, {parsing_data.curr_verb, parsing_data.not_modifier..parsing_data.curr_play_note})
                end    
            end
        else
            if nocondsafterthis then
                if #parsing_data.curr_params > 0 then
                    table.insert(parsing_data.conditions, {parsing_data.curr_infix, parsing_data.curr_params} )
                end

                parsing_data.curr_infix = "none"
                parsing_data.curr_params = {}
            end

            if (state.stage == 0 and stage == 1) or (state.stage == 6 and stage == 1) then
                -- handling prefix conditions
                table.insert(parsing_data.conditions, {parsing_data.not_modifier..word, {}})
            elseif (state.stage == 0 or state.stage == 1 or (state.stage == 4 and tiletype == 0 and not doingcond and not stage3reached)) and stage == 2 then
                -- handling targets
                table.insert(parsing_data.targets, parsing_data.not_modifier..word)
            elseif tiletype == 1 then
                -- handling verbs
                parsing_data.curr_verb = word
            elseif ((state.stage == 3 and stage == 5) or (state.stage == 4 and stage == 2)) and not doingcond then
                -- handling effects
                table.insert(parsing_data.effects, {parsing_data.curr_verb, parsing_data.not_modifier..word})
            elseif tiletype == 7 then
                if #parsing_data.curr_params > 0 then
                    table.insert(parsing_data.conditions, {parsing_data.curr_infix, parsing_data.curr_params} )
                end

                -- handling inflix conditions
                parsing_data.curr_infix = parsing_data.not_modifier..word
                parsing_data.curr_params = {}
            elseif ((state.stage == 3 and stage == 5) or (state.stage == 4 and stage == 2)) and doingcond then
                -- handling "baba on keke and ghost is you"
                table.insert(parsing_data.curr_params, parsing_data.not_modifier..word)
            end
        end

        parsing_data.not_modifier = ""
    end

    local status = State_Status.RUNNING
    if stop then
        status = State_Status.STOPPED_INVALID
    elseif wordid == #sent then
        if #parsing_data.targets > 0 and #parsing_data.effects > 0 then
            status = State_Status.STOPPED_VALID
        end
    end

    -- Detect if THIS mod is installed. If so, don't allow any sentences that involve THIS
    if this_mod_globals ~= nil then
        if is_name_text_this(word) then
            status = State_Status.STOPPED_INVALID
        end
    end
    
    state.stage = stage
    state.prevword = word
    state.prevstage = prevstage
    state.prevtiletype = tiletype
    state.stage2reached = stage2reached
    state.stage3reached = stage3reached
    state.doingcond = doingcond
    state.nocondsafterthis = nocondsafterthis

    return status
end

local function convert_sentence_to_rule(sentence)
    local outrules = {}

    local syntax_state = {
        stage = 0,
        prevstage = 0,
        prevword = "",
        prevtiletype = 0,
        stage2reached = false,
        stage3reached = false,
        doingcond = false,
        nocondsafterthis = false,
    }
    local rule_parsing_data = {
        not_modifier = "",
        curr_verb = "none",
        curr_infix = "none",
        curr_params = {},
        curr_play_note = "",
        targets = {},
        effects = {},
        conditions = {},
    }

    local sent = {}
    for word in sentence:gmatch("%S+") do 
        table.insert(sent, word)
    end

    for wordid=1,#sent do 
        local word = string.lower(sent[wordid])
        local obj_ref = get_object_ref("text_"..word)
        local text_type = 0
        if obj_ref ~= nil then
            text_type = obj_ref.type
        else
            for _, var_suffix_word in ipairs(variable_suffix_words) do
                if string.sub(word, 1, #var_suffix_word) == var_suffix_word then
                    local var_suffix_obj_ref = get_object_ref("text_"..var_suffix_word)
                    if var_suffix_obj_ref ~= nil then
                        text_type = var_suffix_obj_ref.type
                        break
                    end
                end
            end
        end

        local status = update_syntax_state(syntax_state, rule_parsing_data, text_type, wordid, word, sent)

        if status == State_Status.STOPPED_VALID then
            -- add the sentence
            for _, target in ipairs(rule_parsing_data.targets) do
                for _, effects in ipairs(rule_parsing_data.effects) do
                    table.insert(outrules, {
                        {target, effects[1], effects[2]},
                        deep_copy_table(rule_parsing_data.conditions)
                    })
                end
            end
            return outrules
        elseif status == State_Status.STOPPED_INVALID then
            return nil
        else
            -- continue the loop
        end
    end

    return nil
end

local function convert_baserule_sets(baserulesets)
    local outrulesets = {}
    local invalid_sents = {}
    for textname, sentences in pairs(baserulesets) do
        for _, sent in ipairs(sentences) do
            local rules = convert_sentence_to_rule(sent)
            if rules then
                if outrulesets[textname] == nil then
                    outrulesets[textname] = {}
                end

                for _, rule in ipairs(rules) do
                    table.insert(outrulesets[textname], rule)

                    -- @nocommit - we are modifying a global here. Make sure this doesn't affect anything else
                    -- To make sure transformations work, we need to insert into objectlist, which the transformation code uses
                    local basic_rule = rule[1]
                    if basic_rule[3] then
                        local targetnot = string.sub(basic_rule[3], 1, 4)
		                local target = string.sub(basic_rule[3], 5)
                        
                        if targetnot ~= "not " then
                            target = basic_rule[3]
                        end

                        local obj_ref = get_object_ref("text_"..target)
                        if obj_ref then
                            if obj_ref.type == 0 then
                                objectlist[target] = 1
                            end
                        else
                            objectlist[target] = 1
                        end
                    end
                    
                    if basic_rule[1] then
                        local targetnot = string.sub(basic_rule[1], 1, 4)
		                local target = string.sub(basic_rule[1], 5)
                        
                        if targetnot ~= "not " then
                            target = basic_rule[1]
                        end

                        local obj_ref = get_object_ref("text_"..target)
                        if obj_ref then
                            if obj_ref.type == 0 then
                                objectlist[target] = 1
                            end
                        else
                            objectlist[target] = 1
                        end
                    end
                end
            else
                table.insert(invalid_sents, sent)
            end
        end
    end

    return outrulesets, invalid_sents 
end

local final_baserules = {}
local function init_baserules(on_load)
    final_baserules = {}
    local invalid_sents = {}

    -- Use loadfile() instead of require() since require() does not reload the module if it already exists 
    local get_baserules, err = loadfile(script_path().."/basedconfig/baserules.lua")

    if err ~= nil then
        if on_load then
            error("[BASED MOD] Loading baserules.lua returned the error below. Please resolve the error and reload the levelpack to load baserules. \n\nError:\n"..err)
        else
            error("[BASED MOD] Loading baserules.lua returned the error below. Please resolve the error and restart the level to load baserules. \n\nError:\n"..err)
        end
    end

    local mod_config, global_sents, level_sents, persist_sents = get_baserules()
    mod_config = mod_config or {}
    global_sents = global_sents or {}
    level_sents = level_sents or {}
    persist_sents = persist_sents or {}

    local final_sents = {}
    if global_sents then
        for text_name, sentences in pairs(global_sents) do
            final_sents[text_name] = sentences
        end
    end

    if persist_sents then
        final_sents["@always"] = persist_sents
    end

    if level_sents then
        local level_sentences = nil
        if level_sents[generaldata.strings[LEVELNAME]] then
            level_sentences = level_sents[generaldata.strings[LEVELNAME]]
        elseif level_sents[generaldata.strings[CURRLEVEL]] then
            level_sentences = level_sents[generaldata.strings[CURRLEVEL]]
        end
        if level_sentences then
            for text_name, sentences in pairs(level_sentences) do
                final_sents[text_name] = sentences
            end
        end
    end

    final_baserules, invalid_sents = convert_baserule_sets(final_sents)

    if mod_config.report_invalid_sentences and #invalid_sents > 0 and not on_load then
        local err_str = "[Based Mod] Found invalid sentences. These will be excluded from baserules:"
        for i, sent in ipairs(invalid_sents) do
            err_str = err_str.."\n    "..tostring(sent)
        end
        error(err_str)
    end


    baserulelist = {}
    if not mod_config.disable_normal_baserules then
        setupbaserules()
    end

    updatecode = 1
    code()
end

--[[ 
    Weird case:
    - we need to init_baserules here because when just starting a level with baserules, "level_start" is called after code()
    - we set on_load = true to avoid the error message on levelpack load
 ]]
-- init_baserules(true)


table.insert(mod_hook_functions["level_start"], 
    function()
        print("level_start")
        init_baserules()
    end
)
table.insert(mod_hook_functions["level_restart"], 
    function()
        print("level_restart")
        init_baserules()
    end
)

local old_docode = docode
function docode(firstwords)
    local ret = old_docode(firstwords)

    local baserules = final_baserules

    for textname, rules in pairs(baserules) do
        if textname ~= "@always" and hasfeature("level", "is", textname, 1) then
            for _, rule in ipairs(rules) do
                addoption(
                    deep_copy_table(rule[1]),
                    deep_copy_table(rule[2]),
                    {},
                    false,
                    nil,
                    {"base"}
                )
            end 
        end
    end

    if baserules["@always"] ~= nil then
        for _, rule in ipairs(baserules["@always"]) do
            addoption(
                deep_copy_table(rule[1]),
                deep_copy_table(rule[2]),
                {},
                false,
                nil,
                {"base"}
            )
        end 
    end

    return ret
end

PLASMA_BASED_MOD_INITIALIZED = true


-- TESTING AREA
-- local testsents = {
--     "baba is you",
--     "baba is lonely you",
--     "keke on flag is push",
--     "lonely flag is win",
--     "baba is group",
--     "baba is group2",
--     "not rock is not push",
--     "rock is not not not push",
--     "sdlfn",
--     "baba",
--     "baba is",
--     "level is level",
--     "all is empty",
--     "baba and keke is you and push",
--     "me on ghost and bird is rock",
--     "me on ghost and near bird is rock",
--     "lonely idle rock is push",
--     "lonely and idle rock is push",
--     "not me not on not ghost is not rock",
--     "not not me not not on not not ghost is not not rock",
--     "rock not is me",
--     "rock is me not",
--     "baba play a",
--     "baba facing right is you",
--     "baba is group is you",
--     "baba not facing not right is you",
--     "baba is pull and has keke and me",
--     "baba is pull and has keke and make me and wall",
--     "baba is pull and make not me and not wall",
--     "baba is pull and make not me and not wall",
-- }

-- local function print_rule(rule)
--     local outstr = rule[1][1].." "..rule[1][2].." "..rule[1][3].." | Conditions: "

--     for _, cond in ipairs(rule[2]) do
--         local condtype = cond[1]
--         local params = cond[2]

--         local paramstr = condtype
--         if params then
--             for _, param in ipairs(params) do
--                 paramstr = paramstr.." "..tostring(param)
--             end
--         end
--         outstr = outstr.."("..paramstr..") "
--     end

--     return outstr
-- end

-- for _, sent in ipairs(testsents) do
--     local outstr = ""
--     local rules = convert_sentence_to_rule(sent)
--     if rules then
--         for _, rule in ipairs(rules) do
--             outstr = outstr.."["..print_rule(rule).."] "
--         end
--     else
--         outstr = outstr.."Rule is invalid"
--     end

--     print(sent.." -> \t\t"..outstr)
-- end