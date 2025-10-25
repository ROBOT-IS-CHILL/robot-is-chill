
--[[ @Merge: effectblock() was merged ]]



--[[ @Merge: brokenblock() was merged ]]



--[[ @Merge: statusblock() was merged ]]



--[[ @Merge: moveblock() was merged ]]



--[[ @Merge: fallblock() was merged ]]



--[[ @Merge: block() was merged ]]



--[[ @Merge: handledels() was merged ]]



--[[ @Merge: startblock() was merged ]]



--[[ @Merge: visionblock() was merged ]]



--[[ @Merge: levelblock() was merged ]]



--[[ @Merge: findplayer() was merged ]]



--[[ @Merge: diceblock() was merged ]]


function resetlevel()
	if (hasfeature("level","is","noreset",1) ~= nil) then
		doreset = false
		return
	end
	MF_playsound("restart")
	resetting = true
	while #undobuffer > 1 do
		undo()
	end
	resetting = false
	undobuffer = {}
	newundo()
	doreset = false
	resetcount = resetcount + 1
	resetmoves = resetcount
end