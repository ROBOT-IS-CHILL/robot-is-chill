import math
import re
from random import random, seed
from cmath import log
from functools import reduce
from typing import Optional, Callable
import json
import time
import base64
import zlib
import textwrap
import asyncio
import cython
import macrosia_glue

from .. import constants, errors
from ..types import Bot, TilingMode
from ..stringview import StringView

from typing import Tuple

class MacroCog:

    def __init__(self, bot: Bot):
        self.bot = bot
        self.builtins = macrosia_glue.get_builtins() | {
            "tiles":
                "Performs a search on the tile database, and returns the names of all tiles that match, separated by `/`.\n"\
                "The arguments are expected to be an arbitrarily long list of search queries.\n"\
                "Valid search queries are:\n"\
                "- `name:<pattern>` Fully matches the tile name using a regex pattern (e.g. `baba` will match baba only, but `baba.*` will match all tiles starting with baba)\n"\
                "- `tiling:<tiling mode>` Matches the tiling mode\n"\
                "- `source:<string>` Matches the source the tile came from\n"\
                "- `tag:<string>` Matches if the tile has this tag\n"\
                "Note that database operations are slow, and using this too many times may time out your execution.\n"\
                "It's recommended to store the output of this to a variable.\n"\
                "# Arguments\n"\
                "1... The queries to search the tile database with."
        }

    def update_macros(self):
        macrosia_glue.update_macros()

    async def parse_macros(self, objects: str, cmd="x", debug = None) -> str:
        res, val, tb = await macrosia_glue.evaluate(objects, ord(cmd), constants.TIMEOUT_DURATION, debug)
        if not res:
            raise errors.MacroError(val, tb)
        return val


async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)
