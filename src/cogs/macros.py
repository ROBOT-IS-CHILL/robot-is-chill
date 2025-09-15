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
from ..types import Bot, BuiltinMacro, TilingMode
from ..stringview import StringView

from typing import Tuple

class MacroCog:

    def __init__(self, bot: Bot):
        self.bot = bot
        self.builtins = macrosia_glue.get_builtins()

    def update_macros(self):
        macrosia_glue.update_macros()

    async def parse_macros(self, objects: str, debug = None, cmd="x") -> str:
        res, val, tb = await macrosia_glue.evaluate(objects, ord(cmd), constants.MACRO_STEP_LIMIT, debug)
        if not res:
            raise errors.MacroError(val, tb)
        return val


async def setup(bot: Bot):
    bot.macro_handler = MacroCog(bot)
