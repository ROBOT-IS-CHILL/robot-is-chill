--[[ 
    This module's purpose is to record seperately extra metadata related to rules submitted to the featureindex from docode().
    Primarily, it stores the unitids of the texts that were used to form a rule. RuleMetadata is currently used to get unitids of pointer nouns
    since a rule with one depends on the position and direction of the pointer noun.

    - Currently only supports rules 
 ]]
local RuleMetadata = {
    target_unitid = nil,
    verb_unitid = nil,
    property_unitid = nil,
}
RuleMetadata.__index = RuleMetadata

function RuleMetadata:new()
    local o = {}
    setmetatable(o, self) 
    return o
end

local RuleMetadataIndex = {
    ruleindex = {}
}

function RuleMetadataIndex:reset()
    self.ruleindex = {}
end

-- Note: "rule" is a table containing {target, verb, property}. It's used as the key.
-- It's slightly iffy in using tables as keys though...
function RuleMetadataIndex:register_rule(rule, target_unitid, verb_unitid, property_unitid)
    local rule_metadata = RuleMetadata.new()
    rule_metadata.target_unitid = concatenate(target_unitid)
    rule_metadata.verb_unitid = concatenate(verb_unitid)
    rule_metadata.property_unitid = concatenate(property_unitid)
    
    self.ruleindex[rule] = rule_metadata
end

function RuleMetadataIndex:get_rule_metadata(rule)
    return self.ruleindex[rule]
end

return RuleMetadataIndex