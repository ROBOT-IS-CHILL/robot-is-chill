-- Global variable to keep track of texts overlapping a level - including the level itself if it was converted to text! (probably not the best way to handle this)
ws_overlapping_texts = {}

-- Global variable to keep track of whether the entered level was sinful
ws_wasLevelSinful = false

-- Global variable to keep track of what this level is
ws_ambientObject = "level"

-- Global variables to keep track of whether the outer level was aligned
ws_levelAlignedRow = false
ws_levelAlignedColumn = false