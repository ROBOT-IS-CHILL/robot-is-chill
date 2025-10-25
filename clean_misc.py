import tomllib
from pathlib import Path
import re
import shutil

with open("data/custom/misc.toml", "rb") as f:
	data = tomllib.load(f)

for tile in data.values():
	sprite = tile["sprite"]
	source = tile.get("source", "misc")
	for path in (Path("data/sprites") / source).glob(f"{sprite}_*.png"):
		if not re.fullmatch(rf"{re.sub("([^0-9A-Za-z])", r"\\\1", sprite)}_\d+_\d+", path.stem):
			continue
		shutil.copyfile(path, Path("miscsprites") / path.name)
