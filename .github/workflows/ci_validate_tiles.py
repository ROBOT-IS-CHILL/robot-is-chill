import tomllib
from pathlib import Path
import sys
from typing import Optional


def isstr(val):
    if val is None:
        return False, "missing"
    return type(val) is str, "not a string"


def isint(val):
    if val is None:
        return False, "missing"
    return type(val) is int, "not an integer"


def iscol(val):
    if val is None:
        return False, "missing"
    if type(val) is not list:
        return False, "not a list"
    if len(val) != 2:
        return False, "not length 2"
    if type(val[0]) is not int:
        return False, "color x is not integer"
    if type(val[1]) is not int:
        return False, "color y is not integer"
    return True, None


def istiling(val):
    if val is None:
        return False, "missing"
    if type(val) is not str:
        return False, "tiling is not a string"
    return val in {
        "icon",
        "custom",
        "none",
        "directional",
        "tiling",
        "character",
        "animated_directional",
        "animated",
        "static_character",
        "diagonal_tiling"
    }, "invalid tiling mode"


def opt(fn):
    def wrap(val):
        if val is None:
            return True, None
        return fn(val)
    return wrap


def islist(fn):
    def wrap(val):
        if val is None:
            return False, "missing"
        if type(val) is not list:
            return False, "was not list"
        for i, el in enumerate(val):
            res, err = fn(el)
            if not res:
                return False, f"element {i} failed: {err}"
        return True, None
    return wrap


entry_model = {
    "sprite": isstr,
    "color": iscol,
    "tiling": istiling,
    "extra_frames": opt(islist(isint)),
    "author": opt(isstr),
    "tags": opt(islist(isstr)),
    "active_color": opt(iscol),
    "source": opt(isstr),
    "text_type": opt(isint),
    "text_direction": opt(isint),
    "object_id": opt(isint)
}


def check_entries(entry: dict) -> list[str]:
    failures = []

    for key, fn in entry_model.items():
        val = entry.get(key)
        res, err = fn(val)
        if not res:
            failures.append(f"Entry `{key}` failed: {err}")
    for key in entry.keys():
        if key not in entry_model:
            failures.append(f"Extraneous entry `{key}`")
    return failures


def main():
    failures = []
    for file in Path("data/custom").glob("*.toml"):
        try:
            print(f"Checking {file}...")
            with open(file, "rb") as f:
                data = tomllib.load(f)
            failed = False
            for tile, entry in data.items():
                fails = check_entries(entry)
                if len(fails):
                    if not failed:
                        failed = True
                        failures.append(f"File `{file}` failed to validate:")
                    failures.append(f"\tTile `{tile}`:")
                for fail in fails:
                    failures.append(f"\t\t{fail}")
        except Exception as err:
            failures.append(f"File `{file}` failed to parse: {err}")
    for err in failures:
        print(err)
    return 1 if len(failures) else 0


if __name__ == "__main__":
    sys.exit(main())
