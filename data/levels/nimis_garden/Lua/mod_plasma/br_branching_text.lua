local utils = PlasmaModules.load_module("general/utils")
BRANCHING_TEXT_LOGGING = false

function is_branching_text_defined(name)
    return branching_text_names[name]
end

function name_is_branching_text(name, check_omni, check_pivot)
	return parse_branching_text(name, check_omni, check_pivot) ~= name
end

function name_is_branching_text_with_special_unitreference(name)
    local special_texts = {
        br_prefix.."is",
        br_prefix.."play",
        pivot_prefix.."is",
        pivot_prefix.."play",
    }
    for _, text in ipairs(special_texts) do
        if name == text then return true end
    end
    return false
end

function get_perp_direction(dir)
    if dir == 1 then 
        return 2 
    elseif dir == 2 then
        return 1
    end
    return dir
end

-- @TODO: is this needed?
function create_empty_sentence_metadata()
    return {
        branching_points_bitfield = {}
    }
end

function name_is_branching_and(name, check_omni, check_pivot)
	if check_omni == nil then check_omni = true end
	if check_pivot == nil then check_pivot = true end

	if check_omni then return name == br_prefix.."and" end
	if check_pivot then return name == pivot_prefix.."and" end

	return false
end


function parse_branching_text(name, check_omni, check_pivot)
	local main_text = ""
	if check_omni == nil then check_omni = true end
	if check_pivot == nil then check_pivot = true end

	if check_omni and string.sub(name, 1, br_prefix_len) == br_prefix then 
		main_text = string.sub(name, br_prefix_len + 1)
	elseif check_pivot and string.sub(name, 1, pivot_prefix_len) == pivot_prefix then
		main_text = string.sub(name, pivot_prefix_len + 1)
    end
	if is_branching_text_defined(main_text) then
		return main_text
	else
		return name
	end
end

function get_branching_sent_id_char(c)
    local asciicode = string.byte(c)
    if asciicode < 58 and asciicode >= 48 then
        -- The character is a number. Shift the ascii code so that it starts at 58
        asciicode = asciicode + 17
    end
    if asciicode > 126 then
        asciicode = 126
    end
    if asciicode >= 65 and asciicode <= 90 then
        asciicode = asciicode + 32
    elseif asciicode >= 97 and asciicode <= 122 then
        asciicode = asciicode - 32
    end
    return string.char(asciicode)
end    

-- local bases = {
--     parallel = 48,
--     perp = 65
-- }
-- local range = 10

-- local function ascii_in_sent_id_range(ascii, perp)
--     local base = bases.parallel
--     if perp then base = bases.perp end

--     return ascii >= base and ascii < base + range
-- end

-- function convert_sent_id(sent_id, perp)
--     local final_sent_id = ""
--     for c in sent_id:gmatch"." do
--         local ascii = string.byte(c)

--         if perp then
--             if ascii_in_sent_id_range(ascii, false) then
--                 ascii = ascii + (bases.perp - bases.parallel)
--             end
--         else
--             if ascii_in_sent_id_range(ascii, true) then
--                 ascii = ascii + (bases.parallel - bases.perp)
--             end
--         end

--         final_sent_id = final_sent_id..string.char(ascii)
--     end

--     return final_sent_id
-- end

-- function toggle_sent_id(sent_id)
--     local final_sent_id = ""
--     for c in sent_id:gmatch"." do
--         local ascii = string.byte(c)

--         if ascii_in_sent_id_range(ascii, false) then
--             ascii = ascii + (bases.perp - bases.parallel)
--         elseif ascii_in_sent_id_range(ascii, true) then
--             ascii = ascii + (bases.parallel - bases.perp)
--         end

--         final_sent_id = final_sent_id..string.char(ascii)
--     end
--     return final_sent_id
-- end

function convert_to_old_sent_id(sent_id)
    local normal_sent_id = ""
    for c in sent_id:gmatch"." do
        local asciicode = string.byte(c)
        local index = 0
        if asciicode >= 65 and asciicode <= 90 then
            index = asciicode - 65
        elseif asciicode >= 97 and asciicode <= 122 then
            index = asciicode - 97
        end
        normal_sent_id = normal_sent_id..tostring(index)
    end
    return normal_sent_id
end

function br_process_branches(branches, br_dir, found_branch_on_last_word, limiter)
    local totalvariants = 0
    local sentences = {}
    local sentence_ids = {}
    local maxpos = 0
    local br_and_text_with_split_parsing = {}
    local sentence_metadata = {}

    for _, branch in ipairs(branches) do
        local br_sentences,br_finals,br_maxpos,br_totalvariants,br_sent_ids, firstwords, perp_br_and_texts_with_split_parsing, perp_sentence_metadata = calculatesentences(branch.firstwords[1], branch.x, branch.y, br_dir, nil,nil,nil, true)
		-- maxpos = math.max(maxpos, br_maxpos + branch.step_index)

		if (br_sentences == nil or br_totalvariants >= limiter) then
            MF_alert("Level destroyed - too many variants D")
			destroylevel("toocomplex")
			return nil
        end

        utils.debug_assert(#br_sentences == #perp_sentence_metadata, "#Br sentences:"..tostring(#br_sentences).." != #perp_sentence_metadata:"..tostring(#perp_sentence_metadata))
        
        local lhs_totalcombos = 0 -- Note: this isn't the total number of lhs sentences, but the total number of ways to make a combination of words from lhs_word_slots
        local lhs_combo_tracker = {}
        for i, slot in ipairs(branch.lhs_word_slots) do
            if #slot > 0 then
                if lhs_totalcombos == 0 then
                    lhs_totalcombos = 1
                end
                lhs_totalcombos = lhs_totalcombos * #slot
            end
            lhs_combo_tracker[i] = 1
        end

        for unitid, _ in pairs(perp_br_and_texts_with_split_parsing) do
			br_and_text_with_split_parsing[unitid] = true
		end
        
        -- Create all lhs sentences by finding all combinations of words in slot. Also build the sentence ids of the lhs sentences
        local lhs_sentences = {}
        for variant_num = 1, lhs_totalcombos do
            local lhs_sent_id_base = ""
            local lhs_sentence = {}
            local slotindex = 1
            while slotindex <= #branch.lhs_word_slots do
                local slot = branch.lhs_word_slots[slotindex]
                if #slot > 0 then
                    local word_index = lhs_combo_tracker[slotindex]
                    local word = slot[word_index]
                    local width = word[2]

                    local branching_text_name = parse_branching_text(word[3])

                    -- @Note: we might've needed this for eliminating duplicate sentences with branching ands?
                    if branching_text_name == "and" then
                        branching_text_name = word[3]
                    end

                    table.insert(lhs_sentence, {branching_text_name, word[4], word[1], word[2]})
                    lhs_sent_id_base = lhs_sent_id_base..tostring(word_index-1)

                    slotindex = slotindex + width
                else
                    break
                end

            end
            if slotindex == branch.step_index then
                table.insert(lhs_sentences, {lhs_sentence = lhs_sentence, lhs_sent_id_base = lhs_sent_id_base})
            end

            --[[ 
            Update combo tracker so that we get every single sentences in the lhs.
            
            Context: in order to iterate through all possible sentence combinations, the game keeps track of a "combo" data structure. 
            Each index represents a word position, or slot. And a slot can contain multiple words/texts. The content of this "combo" data structure
            implements something similar to continuously adding 1 to a number of say base 2 (though it could be any base). 
            However in this case, each "digit" of number has its own base thats equal to the number of words in that slot.

            Ex: baba&keke is you&push&pull. The iteration through all combinations can be represented by the numbers, 111 112 113 211 212 213.
            The leftmost digit is base 2 because the leftmost slot has two words and the rightmost digit is base 3 because the rightmost slot has 3 words.
            And same thing with the middle digit. Iteration involves adding 1 to the previous result while taking into account each of 
            the digit's individal bases. And the resulting number is used to get the indexes of the words by matching the indexes to the digits.
            Note: the digit analogy doesnt hold if the index is >= 10, but it is still the same idea.
            ]]
            if variant_num ~= lhs_totalcombos then
                local curr_slot = 1
                lhs_combo_tracker[curr_slot] = lhs_combo_tracker[curr_slot] + 1 -- "add one"
                while lhs_combo_tracker[curr_slot] and lhs_combo_tracker[curr_slot] > #branch.lhs_word_slots[curr_slot] do
                    lhs_combo_tracker[curr_slot] = 1 -- reset the base of the lesser significant digit
                    lhs_combo_tracker[curr_slot + 1] = lhs_combo_tracker[curr_slot + 1] + 1 -- carry over the add
                    curr_slot = curr_slot + 1 -- go to the next more significant digit
                end
            end
        end

        if BRANCHING_TEXT_LOGGING then
            print("totalvariants = "..tostring(totalvariants))
            print("#lhs_sentences = "..tostring(#lhs_sentences))
            print("#branch.branching_texts = "..tostring(#branch.branching_texts))
            print("#br_sentences = "..tostring(#br_sentences))
            print("br_totalvariants = "..tostring(br_totalvariants))
        end

        if #lhs_sentences == 0 then
            -- This case specifically handles when there are no non branching texts before the current branching point. 
            totalvariants = totalvariants + #branch.branching_texts * br_totalvariants -- Exclude #lhs_sentences from totalvariants since it doesn't contribute any new words
            table.insert(lhs_sentences, {lhs_sentence = {}, lhs_sent_id_base = ""}) -- Add a dummy entry so that the below for loop can run once without adding extra words from lhs
        else
            totalvariants = totalvariants + #lhs_sentences * #branch.branching_texts * br_totalvariants
        end

        if (totalvariants >= limiter) then
            MF_alert("Level destroyed - too many variants E")
			destroylevel("toocomplex")
			return nil
        end

        -- Combine lhs sentence + omni text + rhs or branching sentence into one final sentence
        for _, lhs_sentence_data in ipairs(lhs_sentences) do
            for br_word_index, br_word in ipairs(branch.branching_texts) do
                for br_sent_index, br_sentence in ipairs(br_sentences) do
                    local perp_metadata = perp_sentence_metadata[br_sent_index]
                    local branching_points_bitfield = {} -- boolean list where for each index, the boolean is true if the word at that index was parsed from moving perpendicularly
                    local final_sentence = {}

                    -- Insert LHS sentence
                    local lhs_sentence = lhs_sentence_data.lhs_sentence
                    for _, word in ipairs(lhs_sentence) do
                        table.insert(final_sentence, word)

                        table.insert(branching_points_bitfield, false) -- We define whether or not if we are on perp parsing relative to the LHS sentence. So output false for each lhs sentence index
                    end
                    -------------------

                    -- Insert branching text with normal variant's name
                    local branching_text_name = parse_branching_text(br_word[3])

                    -- @Note: we might've needed this for eliminating duplicate sentences with branching ands?
                    if branching_text_name == "and" then
                        branching_text_name = br_word[3]
                    end
                    table.insert(final_sentence, {branching_text_name, br_word[4], br_word[1], br_word[2]})
                    table.insert(branching_points_bitfield, true) -- This could either be true or false depending on how we define when we start perp parsing. I'll say true for now.
                    -------------------

                    -- Insert the branched sentence into the final sentence
                    local s_display = ""
                    for i, word in ipairs(br_sentence) do
                        s_display = s_display.. " "..word[1]
                    end
                    utils.debug_assert(#br_sentence == #perp_metadata.branching_points_bitfield, tostring(#br_sentence).." "..tostring(#perp_metadata.branching_points_bitfield).." ".. s_display)
                    for i, word in ipairs(br_sentence) do
                        table.insert(final_sentence, word)

                        -- Since the br_sentence comes from moving perpendicularly from the lhs sentence, flip all the bits of the br_sentence branching_points
                        table.insert(branching_points_bitfield, perp_metadata.branching_points_bitfield[i])
                    end
                    -------------------

                    table.insert(sentences, final_sentence)
					maxpos = math.max(maxpos, #final_sentence)

                    local metadata = create_empty_sentence_metadata()
                    metadata.branching_points_bitfield = branching_points_bitfield
                    table.insert(sentence_metadata, metadata)


                    --[[ 
                    Omni text does sentence ids a bit differently than the main game. For context a "sentence id" is a unique id 
                    within the scope of a single calculatesentences() call that identifies the sentence by a concatenation of indexes 
                    of each word within its slot. For example, if the game has Baba/Keke is you/push, and we parse the sentence "Baba is push", 
                    the sentence id would look like "112" where the two 1s represent the first word of the first slot (Baba) and the first word 
                    of the second slot (is), while the "2" represents the second word of the third slot (push).

                    The problem with this id scheme is that if the index is at least two digits, then you store more characters to represent a single slot. 
                    If in the previous example, the index of "push" was 10, then the sentence id of "baba is push" would be "1110", where the last two characters 
                    represent the third slot. However, these sentence ids also get spliced to represent sub sentences and the splicing assumes that each character 
                    in a sentence id = 1 slot (look for "string.sub(sent_id,...)"). This could lead to id collisions since splicing "1112" and "1113" after the third "1" will yield the same sub sentence id, even if "1113" actually represents 3 slots while "1112" represents 4 slots.

                    As of 5/19/21, we don't know how Hempuli will resolve this, if at all. So in omni text, we do our own implementation of this. 
                    A BIG assumption is that the game does not interpret the index information directly from the sentence id. It only uses 
                    the combination of indexes to uniquely identify a sentence. So knowing this, we can put in any character to represent 
                    an index within each slot, which includes letters. With this implementation, ascii values from 58-126 are supported, 
                    which significantly increases the max num of stacked text it could handle without losing support of detecting stacked text bugs.

                    One thing to note is that the lhs sentences still uses the old algorithm while the branched sentences uses the new algorithm. 
                    This is so that splicing the lhs part of the sent id will match other sentencesthat share the same slots.
                    ]]
                    local final_sentid = lhs_sentence_data.lhs_sent_id_base
                    final_sentid = final_sentid..tostring(br_word_index - 1) --@TODO: not sure branching text char should be a hard number

                    for c in br_sent_ids[br_sent_index]:gmatch"." do
                        final_sentid = final_sentid..get_branching_sent_id_char(c)
                    end
                    table.insert(sentence_ids, final_sentid)
                    
                end
            end
        end
    end
    


    return sentences, sentence_ids, totalvariants, maxpos, br_and_text_with_split_parsing, sentence_metadata
end