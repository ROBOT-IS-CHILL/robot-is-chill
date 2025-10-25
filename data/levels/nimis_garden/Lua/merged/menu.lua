

function changemenu(menuitem,extra)
	local currmenu = menu[1]
	MF_letterclear(currmenu,0)
	menu[1] = menuitem
	editor.strings[MENU] = menu[1]
	MF_clearcontrolicons(0)
	
	if (menufuncs[currmenu] ~= nil) then
		local func = menufuncs[currmenu]
		local buttonid = func.button or nil
		
		if (buttonid ~= nil) then
			MF_delete(buttonid)
			MF_clearthumbnails(buttonid)
		end
		
		if (func.leave ~= nil) then
			func.leave(menu[2],currmenu,buttonid,extra)
		end
	end
	
	editor2.values[ALLOWSCROLL] = 0
	generaldata2.values[INMENU] = 0
	editor3.strings[ESCBUTTON] = ""
	
	if (menufuncs[menuitem] ~= nil) then
		local func = menufuncs[menuitem]
		local buttonid = func.button or nil
		
		if (func.enter ~= nil) then
			func.enter(currmenu,menuitem,buttonid,extra)
		end
		
		if (func.structure ~= nil) then
			generaldata2.values[INMENU] = 1
		end
		
		if (func.scrolling ~= nil) then
			editor2.values[ALLOWSCROLL] = func.scrolling
		end
		
		if (func.escbutton ~= nil) then
			editor3.strings[ESCBUTTON] = func.escbutton
		end
		
		if (func.slide ~= nil) then
			local slide = func.slide
			editor2.values[MENU_XOFFSET] = slide[1] * screenw or 0
			editor2.values[MENU_YOFFSET] = slide[2] * screenh or 0
		end
	end
	
	--MF_alert("Changed to menu " .. menuitem)
	
	editor.values[SCROLLAMOUNT] = 0

	--these are the only four lines i added to this function for visit
	if menuitem == "main" then
		visit_innerlevelid = ""
		visit_fullsurrounds = ""
	end
end