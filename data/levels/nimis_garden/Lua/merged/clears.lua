function clearunits(restore_)
	units = {}
	tiledunits = {}
	codeunits = {}
	animunits = {}
	unitlists = {}
	undobuffer = {}
	unitmap = {}
	unittypeshere = {}
	prevunitmap = {}
	ruleids = {}
	objectlist = {}
	objectpalette = {}
	updatelist = {}
	objectcolours = {}
	wordunits = {}
	wordrelatedunits = {}
	letterunits = {}
	letterunits_map = {}
	paths = {}
	paradox = {}
	movelist = {}
	deleted = {}
	effecthistory = {}
	notfeatures = {}
	groupfeatures = {}
	groupmembers = {}
	pushedunits = {}
	customobjects = {}
	cobjects = {}
	condstatus = {}
	emptydata = {}
	leveldata = {}
	leveldata.colours = {}
	leveldata.currcolour = 0
	poweredstatus = {}
	specialtiling = {}
	
	visiontargets = {}
	vision_rendered = {}
	
	generaldata.values[CURRID] = 1
	updateundo = true
	hiddenmap = nil
	levelconversions = {}
	last_key = 0
	auto_dir = {}
	destroylevel_check = false
	destroylevel_style = ""

	doreset = false
	resetting = false
	resetcount = 0
	resetmoves = 0
	
	HACK_MOVES = 0
	HACK_INFINITY = 0
	movemap = {}
	
	local restore = true
	if (restore_ ~= nil) then
		restore = norestore_
	end
	
	if restore then
		newundo()
		
		print("clearunits")
		
		restoredefaults()
	end
end

function clear()
	features = {}
	featureindex = {}
	condfeatureindex = {}
	visualfeatures = {}
	objectdata = {}
	deleted = {}
	ruleids = {}
	updatelist = {}
	wordunits = {}
	wordrelatedunits = {}
	letterunits_map = {}
	paradox = {}
	movelist = {}
	effecthistory = {}
	notfeatures = {}
	groupfeatures = {}
	groupmembers = {}
	pushedunits = {}
	condstatus = {}
	emptydata = {}
	leveldata = {}
	leveldata.colours = {}
	leveldata.currcolour = 0
	poweredstatus = {}
	
	visiontargets = {}
	vision_rendered = {}
	
	updatecode = 1
	updateundo = false
	hiddenmap = nil
	levelconversions = {}
	maprotation = 0
	mapdir = 3
	last_key = 0
	auto_dir = {}
	destroylevel_check = false
	destroylevel_style = ""

	doreset = false
	resetting = false
	resetcount = 0
	resetmoves = 0
	
	HACK_MOVES = 0
	HACK_INFINITY = 0
	movemap = {}
	
	--print("clear")
	collectgarbage()
end