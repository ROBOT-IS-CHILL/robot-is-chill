local PNoun = {}

PNoun.Groups = {
    THIS_IS_BLOCK =     1, -- All "THIS is block" rules
    THIS_IS_RELAY =     2, -- All "THIS is relay" rules
    THIS_IS_PASS =      3, -- All "THIS is pass" rules
    VARIABLE =          4,
    OTHER_ACTIVE =      5, -- This should have no features. Only pnouns not part of an active rule. This is populated only when calling populate_inactive_pnouns()
    OTHER_INACTIVE =    6, -- This should have no features. Only pnouns not part of an active rule. This is populated only when calling populate_inactive_pnouns()
}

PNoun.Ops = {
    block = {
        filter_func = function(name) return name == "block" end,
    },
    relay = {
        filter_func = function(name) return name == "relay" end,
    },
    pass = {
        filter_func = function(name) return name == "pass" end,
    },
    other = {
        filter_func = function(name) return name ~= "block" and name ~= "pass" and name ~= "relay" end,
    },
}

PNoun.Pnoun_Group_Lookup = {
    [PNoun.Groups.THIS_IS_BLOCK] = {
        ops = {"block"}
    },
    [PNoun.Groups.THIS_IS_RELAY] = {
        ops = {"relay"}
    },
    [PNoun.Groups.THIS_IS_PASS] = {
        ops = {"pass"}
    },
    [PNoun.Groups.VARIABLE] = {
        ops = {"block", "relay", "pass"},
        repeat_until_no_more_processing = true,
        redirect_pnoun_group = PNoun.Groups.OTHER_ACTIVE,
    },
    [PNoun.Groups.OTHER_ACTIVE] = {
        ops = {"other"},
        redirect_pnoun_group = PNoun.Groups.OTHER_INACTIVE,
    },
    [PNoun.Groups.OTHER_INACTIVE] = {
        ops = {"other"}
    },
}


return PNoun