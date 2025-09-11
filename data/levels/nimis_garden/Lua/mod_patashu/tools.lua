if (fixed_to_str == nil) then
	fixed_to_str = tostring
end


--[[ @Merge: rotate() was merged ]]



--[[ @Merge: isthis() was merged ]]



--[[ @Merge: xthis() was merged ]]



--[[ @Merge: findall() was merged ]]



--[[ @Merge: delunit() was merged ]]



--[[ @Merge: findtype() was merged ]]



--[[ @Merge: findobstacle() was merged ]]



--[[ @Merge: update() was merged ]]



--[[ @Merge: updatedir() was merged ]]



--[[ @Merge: findtext() was merged ]]



--[[ @Merge: findallhere() was merged ]]



--[[ @Merge: findempty() was merged ]]



--[[ @Merge: handleinside() was merged ]]



--[[ @Merge: delete() was merged ]]



--[[ @Merge: writerules() was merged ]]



--[[ @Merge: mapcells() was merged ]]



--[[ @Merge: copy() was merged ]]



--[[ @Merge: create() was merged ]]



--[[ @Merge: getunitid() was merged ]]



--[[ @Merge: newid() was merged ]]



--[[ @Merge: addunitmap() was merged ]]



--[[ @Merge: updateunitmap() was merged ]]



--[[ @Merge: inside() was merged ]]



--[[ @Merge: animate() was merged ]]



--[[ @Merge: updateanimations() was merged ]]
	


--[[ @Merge: issolid() was merged ]]



--[[ @Merge: isgone() was merged ]]



--[[ @Merge: floating() was merged ]]



--[[ @Merge: floating_level() was merged ]]



--[[ @Merge: emptydir() was merged ]]



--[[ @Merge: issafe() was merged ]]



--[[ @Merge: issleep() was merged ]]



--[[ @Merge: isstill() was merged ]]



--[[ @Merge: isstill_or_locked() was merged ]]



--[[ @Merge: getmat() was merged ]]



--[[ @Merge: getmat_text() was merged ]]



--[[ @Merge: destroylevel() was merged ]]

	


--[[ @Merge: destroylevel_do() was merged ]]



--[[ @Merge: findunitat() was merged ]]



--[[ @Merge: checkwordchanges() was merged ]]



--[[ @Merge: getpath() was merged ]]



--[[ @Merge: append() was merged ]]



--[[ @Merge: getname() was merged ]]



--[[ @Merge: getemptytiles() was merged ]]



--[[ @Merge: setsoundname() was merged ]]



--[[ @Merge: checkturnsound() was merged ]]



--[[ @Merge: getlevelsurrounds() was merged ]]



--[[ @Merge: parsesurrounds() was merged ]]



--[[ @Merge: copytable() was merged ]]



--[[ @Merge: copysubtable() was merged ]]



--[[ @Merge: copyconds() was merged ]]



--[[ @Merge: concatenate() was merged ]]



--[[ @Merge: checkeffecthistory() was merged ]]



--[[ @Merge: updateeffecthistory() was merged ]]



--[[ @Merge: reseteffecthistory() was merged ]]



--[[ @Merge: genflowercolour() was merged ]]



--[[ @Merge: gettiledata() was merged ]]



--[[ @Merge: displaybigtext() was merged ]]



--[[ @Merge: gettablestring() was merged ]]



--[[ @Merge: pickoption() was merged ]]



--[[ @Merge: getpathdetails() was merged ]]



--[[ @Merge: getnamegivingtitle() was merged ]]



--[[ @Merge: findunits() was merged ]]



--[[ @Merge: flipnot() was merged ]]



--[[ @Merge: inbounds() was merged ]]



--[[ @Merge: findfears() was merged ]]



--[[ @Merge: getlettermultiplier() was merged ]]



--[[ @Merge: isitbroken() was merged ]]



--[[ @Merge: getinputcount() was merged ]]



--[[ @Merge: cantmove() was merged ]]



--[[ @Merge: findnoun() was merged ]]



--[[ @Merge: findgroup() was merged ]]



--[[ @Merge: hasconds() was merged ]]



--[[ @Merge: groupcheck() was merged ]]



--[[ @Merge: reversedir() was merged ]]



--[[ @Merge: reversecheck() was merged ]]



--[[ @Merge: simplecheck() was merged ]]


function table.has_value(tab, val)
	for index, value in ipairs(tab) do
			if value == val then
					return true
			end
	end

	return false
end

function mergeTable(t, other)
	if other ~= nil then
		for k,v in pairs(other) do
			if type(k) == "number" then
				if not table.has_value(t, v) then
					table.insert(t, v)
				end
			else
				if t[k] ~= nil then
					if type(t[k]) == "table" and type(v) == "table" then
						mergeTable(t[k], v)
					end
				else
					t[k] = v
				end
			end
		end
	end
	return t
end