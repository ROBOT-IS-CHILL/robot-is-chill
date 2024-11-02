from __future__ import annotations

import abc
import datetime
from enum import IntEnum, Enum, auto
import inspect
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Literal, BinaryIO, Callable, Self
import io

from attr import define

from . import errors, constants
import re

import discord
from discord.ext import commands
from PIL import Image

if TYPE_CHECKING:
    from .cogs.render import Renderer
    from .cogs.variants import VariantHandlers
    from .cogs.macros import MacroCog


class Context(commands.Context):
    async def error(self, msg: str, **kwargs) -> discord.Message: ...

    async def send(self,
                   content: str = "",
                   embed: Optional[discord.Embed] = None,
                   **kwargs) -> discord.Message: ...

    async def warn(self, msg: str, **kwargs) -> discord.Message: ...


class Database:
    ...

class Bot(commands.Bot):
    db: Database
    cogs: list[str]
    embed_color: discord.Color
    webhook_id: int
    prefixes: list[str]
    exit_code: int
    loading: bool
    started: datetime.datetime
    renderer: Renderer
    handlers: VariantHandlers

    def __init__(
            self,
            *args,
            cogs: list[str],
            embed_color: discord.Color,
            webhook_id: int,
            prefixes: list[str],
            exit_code: int = 0,
            **kwargs):
        ...

    async def get_context(self,
                          message: discord.Message,
                          **kwargs) -> Coroutine[Any,
    Any,
    Context]: ...


class Variant:
    args: list = None
    type: str = None
    pattern: str = None
    signature = None
    syntax: str = None

    def apply(self, obj, **kwargs):
        raise NotImplementedError("Tried to apply an abstract variant!")

    def __repr__(self):
        return f"{self.__class__.__name__}{self.args}"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash((self.__class__.__name__, *self.args, self.type))


class VaryingArgs:
    """A thin wrapper to tell the argument parser that this is a varying length list."""

    def __init__(self, held_type):
        self.type = held_type

    def __call__(self, value):
        return self.type(value)


# https://github.com/LumenTheFairy/regexdict
# Put here to prevent Python from yelling at me

class RegexDict:

    # d should be a list of pairs if precedence of matches should be enforced
    # (precedence is in the order of the given list; earlier in the list means higher precedence)
    # if the precedence is not important, and the user has a dict d, d.items() can be passed
    # flags are passed on the underlying re functions called when using this object
    def __init__(self, d, flags=0):

        # give indices to the (key, value) pairs
        self._patterns = []
        self._values = []
        self._compiled_patterns = []
        # maps what will be a full match's lastindex
        # to the index of the associated key, value pair
        # this is relevant in case the keys themselves have groups
        self._group_indices = {}

        cur_key = 0
        cur_group_index = 1
        for key, value in d:
            # compile the given expression
            compiled = re.compile(key, flags)

            # store the info for this pair
            self._patterns.append(key)
            self._values.append(value)
            self._compiled_patterns.append(compiled)
            self._group_indices[cur_group_index] = cur_key

            # increment counters
            cur_key += 1
            cur_group_index += compiled.groups + 1

        # this creates a single compiled regex to test all keys 'simultaneously'
        full_regex = '|'.join(('({})'.format(k)) for k in self._patterns)
        self._full_regex = re.compile(full_regex, flags)

    # takes a string key, and returns the index of the looked up value
    # if there is more than one match, the one with highest precedence is used
    # if there are no matches, KeyError is raised
    def _get_index(self, key):

        # try to match the key
        match = self._full_regex.fullmatch(key)

        if match:
            # return the appropriate index
            return self._group_indices[match.lastindex]
        else:
            # there was no match, so raise an error
            raise KeyError(key)

    # takes a string key, and returns the value that was looked up
    # if there is more than one match, the one with highest precedence is used
    # if there are no matches, KeyError is raised
    def get(self, key):
        # get the index of the value
        index = self._get_index(key)
        # return the value
        return self._values[index]

    # takes a string key, and returns a pair (value, match)
    # where value is the value that was looked up
    # and match is the Match object that matched the given key
    # if there is more than one match, the one with highest precedence is used
    # if there are no matches, KeyError is raised
    def get_with_match(self, key):
        # get the index of the value
        index = self._get_index(key)
        # match the original key regex
        match = self._compiled_patterns[index].fullmatch(key)
        # return the value and match
        return self._values[index], match

    # for dicts with functions as values, this can be used as a shortcut
    # to call the looked up function with the captured groups as parameters
    def apply(self, key):
        # get the function and match for the key
        f, match = self.get_with_match(key)
        # call the function with the match groups as parameters
        return f(*match.groups())

    # updates the value for the pattern that matches the given key
    # raises KeyError is the key does not match (unlike a normal dictionary, which would add the value)
    def update(self, key, value):
        # get the index of the value
        index = self._get_index(key)
        # update it
        self._values[index] = value

    # returns the underlying list of (pattern, value) pairs
    def get_underlying_dict(self):
        return list(zip(self._patterns, self._values))

    # regdictojb[key] can be used as shortcut for get
    __getitem__ = get

    # regdictojb[key] = val can be used as shortcut for update
    __setitem__ = update

    # regdictojb(key) can be used as shortcut for apply
    __call__ = apply


class Color(tuple):
    """Helper class for colors in variants."""

    def __new__(cls, value: str):
        try:
            assert value[0] == "#"
            value = value[1:]
            if len(value) < 6:
                value = ''.join(c * 2 for c in value)
            color_int = int(value, base=16)
            if len(value) < 8:
                color_int <<= 8
                color_int |= 0xFF
            return super(Color, cls).__new__(cls, ((color_int & 0xFF000000) >> 24, (color_int & 0xFF0000) >> 16,
                                                   (color_int & 0xFF00) >> 8, color_int & 0xFF))
        except (ValueError, AssertionError):
            if value in constants.COLOR_NAMES:
                return super(Color, cls).__new__(cls, constants.COLOR_NAMES[value])
            try:
                x, y = value.split("/")
                x = x.lstrip("(")
                y = y.rstrip(")")
                return super(Color, cls).__new__(cls, (int(x), int(y)))
            except ValueError:
                traceback.print_exc()
                raise AssertionError("Failed to parse seemingly valid color! This should never happen.")

    @staticmethod
    def parse(tile, palette_cache, color=None):
        if color is None:
            color = tile.color
        if type(color) == str:
            color = tuple(Color(color))
        if len(color) == 4:
            return color
        else:
            try:
                pal = palette_cache[tile.palette].convert("RGBA")
                return pal.getpixel(color)
            except IndexError:
                raise AssertionError(f"The palette index `{color}` is outside of the palette.")

# NOTE: Due to the inner workings of CPython, the slice class cannot be subclassed.
class Slice:
    """Custom slice class for variant parsing."""
    def __init__(self, *args):
        self.slice = slice(*args)

# Macro stuff

@dataclass
class Unpack:
    inner: list

    def __repr__(self):
        return f"Unpack({self.inner})"


class VariableRegistry:
    variables: dict[str, Any] 
    parent: Self | None
    mutable: bool

    def __init__(self, vars: dict = None, *, parent: Self | None = None, mutable: bool = True):
        self.variables = {} if vars is None else vars
        self.parent = parent
        self.mutable = mutable
    
    def __getitem__(self, name: str):
        if name in self.variables:
            return self.variables[name]
        if self.parent is not None:
            return self.parent[name]
        return None

    def __setitem__(self, name: str, value: str):
        assert self.mutable, "cannot set value in immutable variable scope (if you're seeing this, it's probably a bug)"
        # We can skip the traversal if it's in here
        if name not in self.variables:
            # Check any parent scopes to see if they have the value
            curr = self.parent
            while curr is not None:
                if curr.mutable and name in curr.variables:
                    curr[name] = value
                    return
                curr = curr.parent
        # Make a new value
        if value is None:
            if name in self.variables:
                del self.variables[name]
        else:
            self.variables[name] = value

    def branch(self) -> Self:
        return VariableRegistry(parent = self)
    
    def shallow_copy(self) -> Self:
        return VariableRegistry(vars = self.variables, parent = self.parent, mutable = self.mutable)

class AbstractMacro(abc.ABC):
    name: str | None
    """The macro's name, if any."""

    description: str
    """A description of what the macro does."""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"<macro: {self.name}>"

    async def eval(self, _ctx, _vars, _args):
        raise NotImplementedError("cannot evaluate abstract base class")

    def expansion_count(self):
        return 1

class TextMacro(AbstractMacro):
    source: str
    inputs: list[str]
    variable_args: bool
    author: int
    forest: list[MacroTree | Any]
    captured_scope: VariableRegistry | None = None

    def __init__(self, name, description, source, inputs, variable_args, author, forest):
        super().__init__(name, description)
        self.source = source
        self.inputs = inputs
        self.variable_args = variable_args
        self.author = author
        self.forest = forest

    async def eval(self, ctx: MacroCog, vars: VariableRegistry, args: list[Any]):
        if self.variable_args:
            if len(args) >= len(self.inputs):
                variable = args[len(self.inputs) - 1:]
                args[len(self.inputs) - 1] = variable
        if self.captured_scope is not None:
            self.captured_scope.parent = vars
            vars = self.captured_scope
        scope = vars.branch()
        for name, val in zip(self.inputs, args):
            if isinstance(val, MacroTree):
                val = await ctx.evaluate_tree(val, vars)
            scope.variables[name] = val
        return await ctx.evaluate_forest(self.forest, vars = scope)

class BuiltinMacro(AbstractMacro):
    """A built-in macro."""

    function: Callable
    """The function to call to run the macro."""

    arg_count: int
    """The amount of arguments the macro takes."""

    allows_many: bool
    """Whether the macro allows a variable number of arguments."""

    greedy: bool
    """Whether the macro evaluates its arguments before execution."""

    def __init__(self, name, description, function, arg_count, allows_many, greedy):
        self.name = name
        self.description = description
        self.function = function
        self.arg_count = arg_count
        self.allows_many = allows_many
        self.greedy = greedy

    async def eval(self, ctx: MacroCog, vars: VariableRegistry, args: list[Any]):
        # HACK: There's probably a better way to do this
        if len(args) > self.arg_count and not self.allows_many:
            args = args[:self.arg_count]
        if self.greedy:
            args = [await ctx.evaluate_tree(val, vars.branch()) for val in args]
        elif len(args) < self.arg_count:
            args = [*args, *((None, ) * (self.arg_count - len(args)))]
        new_args = []
        for arg in args:
            if isinstance(arg, Unpack):
                new_args.extend(arg.inner)
            else:
                new_args.append(arg)
        return await (self.function)(vars, *new_args)

    def expansion_count(self):
        return 0.05

class MacroTree:
    arguments: list[Self | Any]
    start: int
    end: int
    source: str
    
    def __init__(self, start: int):
        self.arguments = []
        self.start = start
        self.end = start
    
    def __str__(self):
        if not len(self.arguments): return "[]"
        buf = io.StringIO()
        buf.write("[")
        buf.write(str(self.arguments[0]))
        for arg in self.arguments[1:]:
            buf.write("/")
            buf.write(str(arg))
        buf.write("]")
        return buf.getvalue()

    def __repr__(self):
        return str(self)


class MacroBreak(BaseException):
    value: Any

    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return "break while outside loop"


class MacroContinue(BaseException):
    def __str__(self):
        return "continue while outside loop"

@dataclass
class SignText:
    time_start: int = 0
    time_end: int = 0
    x: int = 0
    y: int = 0
    text: str = "null"
    size: float = 1.0
    xo: int = 0
    yo: int = 0
    color: tuple[int, int, int, int] = (255, 255, 255, 255)
    font: Optional[str] = None
    alignment: str = "center"
    anchor: str = "md"
    stroke: tuple[tuple[int, int, int, int], int] = (0, 0, 0, 0), 0

@dataclass
class RenderContext:
    """A holder class for all the attributes of a render."""
    ctx: Context = None
    before_images: list[Image] = field(default_factory=lambda: [])
    palette: str = "default"
    background_images: list[str] | list[Image] | None = None
    out: str | BinaryIO = "target/renders/render.gif"
    background: tuple[int, int] | None = None
    upscale: int = 2
    extra_out: str | BinaryIO | None = None
    extra_name: str | None = None
    frames: list[int] = (1, 2, 3)
    animation: tuple[int, int] = None
    cropped: bool = False
    speed: int = 200
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)
    image_format: str = 'gif'
    loop: bool = True
    spacing: int = constants.DEFAULT_SPRITE_SIZE
    boomerang: bool = False
    random_animations: bool = True
    expand: bool = False
    sign_texts: list = field(default_factory=lambda: [])
    _disable_limit: bool = False
    _no_sign_limit: bool = False
    raw_output: bool = False
    do_embed: bool = False
    global_variant: str = ""
    macros: dict = field(default_factory=lambda: {})
    tileborder: bool = False
    gscale: int = 1
    sprite_cache: dict = field(default_factory=lambda: {})
    tile_cache: dict = field(default_factory=lambda: {})
    letters: bool = False


class TilingMode(IntEnum):
    CUSTOM = -2
    NONE = -1
    DIRECTIONAL = 0
    TILING = 1
    CHARACTER = 2
    ANIMATED_DIRECTIONAL = 3
    ANIMATED = 4
    STATIC_CHARACTER = 5
    DIAGONAL_TILING = 6

    def __str__(self) -> str:
        if self == TilingMode.CUSTOM: return "custom"
        if self == TilingMode.NONE: return "none"
        if self == TilingMode.DIRECTIONAL: return "directional"
        if self == TilingMode.TILING: return "tiling"
        if self == TilingMode.CHARACTER: return "character"
        if self == TilingMode.ANIMATED_DIRECTIONAL: return "animated_directional"
        if self == TilingMode.ANIMATED: return "animated"
        if self == TilingMode.STATIC_CHARACTER: return "static_character"
        if self == TilingMode.DIAGONAL_TILING: return "diagonal_tiling"

    def parse(string: str) -> TilingMode | None:
        return {
            "custom": TilingMode.CUSTOM,
            "none": TilingMode.NONE,
            "directional": TilingMode.DIRECTIONAL,
            "tiling": TilingMode.TILING, # lol
            "character": TilingMode.CHARACTER,
            "animated_directional": TilingMode.ANIMATED_DIRECTIONAL,
            "animated": TilingMode.ANIMATED,
            "static_character": TilingMode.STATIC_CHARACTER,
            "diagonal_tiling": TilingMode.DIAGONAL_TILING
        }.get(string, None)

    def expected(self) -> set[int]:
        if self == TilingMode.CUSTOM:
            return set()
        if self == TilingMode.DIAGONAL_TILING:
            return set(range(47))
        if self == TilingMode.NONE:
            return {0}
        if self == TilingMode.DIRECTIONAL:
            return {0, 8, 16, 24}
        if self == TilingMode.TILING:
            return set(range(16))
        if self == TilingMode.CHARACTER:
            return {0, 1, 2, 3, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 31}
        if self == TilingMode.ANIMATED_DIRECTIONAL:
            return {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27}
        if self == TilingMode.ANIMATED:
            return {0, 1, 2, 3}
        if self == TilingMode.STATIC_CHARACTER:
            return {0, 1, 2, 3, 31}