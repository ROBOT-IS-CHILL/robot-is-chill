from __future__ import annotations

import datetime
from enum import IntEnum
import inspect
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Coroutine, Optional, Literal, BinaryIO, Callable
from abc import ABC, abstractmethod

from attr import define

from . import errors, constants
import re
import numpy as np

import discord
from discord.ext import commands
from PIL import Image

if TYPE_CHECKING:
    from .cogs.render import Renderer
    from .db import Database
else:
    class Database:
        ...


class Context(commands.Context):
    async def error(self, msg: str, **kwargs) -> discord.Message: ...

    async def send(self,
                   content: str = "",
                   embed: Optional[discord.Embed] = None,
                   **kwargs) -> discord.Message: ...

    async def warn(self, msg: str, **kwargs) -> discord.Message: ...


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


@dataclass
class AbstractVariantFactory(ABC):
    name: str
    applicator: Callable[[Any, Any, ...], None]
    hashable: bool
    type: str

    def parser(self, bot: Bot):
        pass


@dataclass
class SkeletonVariantFactory(AbstractVariantFactory):
    type: str = "skel"
    applicator: Callable[[TileSkeleton, Any, ...], None]


@dataclass
class SignVariantFactory(AbstractVariantFactory):
    type: str = "sign"
    applicator: Callable[[SignText, Any, ...], None]


@dataclass
class TileVariantFactory(AbstractVariantFactory):
    type: str = "tile"
    applicator: Callable[[Tile, Any, ...], None]


@dataclass
class SpriteVariantFactory(AbstractVariantFactory):
    type: str = "sprite"
    applicator: Callable[[np.ndarray[tuple, np.dtype[np.uint8]], Any, ...], None]


@dataclass
class PostVariantFactory(AbstractVariantFactory):
    type: str = "post"
    applicator: Callable[[ProcessedTile, Any, ...], None]


@dataclass
class AbstractVariantContext(ABC):
    pass


@dataclass
class SkeletonVariantContext(AbstractVariantContext):
    pass


@dataclass
class SignVariantContext(AbstractVariantContext):
    bot: Bot
    ctx: RenderContext
    renderer: Renderer


@dataclass
class TileVariantContext(AbstractVariantContext):
    tile_data_cache: dict[str, TileData]


@dataclass
class SpriteVariantContext(AbstractVariantContext):
    tile: Tile
    wobble: int
    renderer: Renderer


@dataclass
class PostVariantContext(AbstractVariantContext): pass


@dataclass
class Variant:
    args: tuple
    factory: AbstractVariantFactory
    persistent: bool
    type: str

    def __hash__(self): pass

    def apply(self, target: Any, context: AbstractVariantContext): pass


@dataclass
class Macro:
    value: str
    description: str
    author: int


@dataclass
class ExternalMacro:
    description: str


@dataclass
class SignText:
    t: int = 0
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

    name: str = "<st>"
    prefix: str = ""
    postfix: str = "<st>"
    variants: list = field(default_factory=list)

    def clone(self):
        clone = SignText(**self.__dict__)
        clone.variants = [var for var in self.variants]
        return clone


@dataclass
class RenderContext:
    """A holder class for all the attributes of a render."""
    ctx: Context = None
    prefix: str | None = None
    before_images: list[Image] = field(default_factory=lambda: [])
    palette: tuple[str, str | None] = ("default", "vanilla")
    background_images: list[str] | list[Image] | None = None
    out: str | BinaryIO = "target/renders/render.webp"
    background: tuple[int, int] | None = None
    upscale: int = 2
    extra_name: str | None = None
    frames: list[int] = (1, 2, 3)
    animation: tuple[int, int] = None
    cropped: bool = False
    speed: int = 200
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)
    image_format: str = 'webp'
    loop: bool = True
    spacing: int = constants.DEFAULT_SPRITE_SIZE
    boomerang: bool = False
    random_animations: bool = True
    expand: bool = False
    sign_texts: list = field(default_factory=lambda: [])
    do_embed: bool = False
    global_variant: str = ""
    macros: dict = field(default_factory=lambda: {})
    tileborder: bool = False
    gscale: int = 1
    sprite_cache: dict = field(default_factory=lambda: {})
    tile_cache: dict = field(default_factory=lambda: {})
    letters: bool = False
    limited_palette: bool = False
    bypass_limits: bool = False
    custom_filename: str | None = None


class TilingMode(IntEnum):
    ICON = -3
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
        if self == TilingMode.ICON:
            return "icon"
        if self == TilingMode.CUSTOM:
            return "custom"
        if self == TilingMode.NONE:
            return "none"
        if self == TilingMode.DIRECTIONAL:
            return "directional"
        if self == TilingMode.TILING:
            return "tiling"
        if self == TilingMode.CHARACTER:
            return "character"
        if self == TilingMode.ANIMATED_DIRECTIONAL:
            return "animated_directional"
        if self == TilingMode.ANIMATED:
            return "animated"
        if self == TilingMode.STATIC_CHARACTER:
            return "static_character"
        if self == TilingMode.DIAGONAL_TILING:
            return "diagonal_tiling"

    def parse(string: str) -> TilingMode | None:
        return {
            "icon": TilingMode.ICON,
            "custom": TilingMode.CUSTOM,
            "none": TilingMode.NONE,
            "directional": TilingMode.DIRECTIONAL,
            "tiling": TilingMode.TILING,  # lol
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
        if self == TilingMode.ICON:
            return {0}
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

COLOR_PAL_REGEX = re.compile(
    r'([1-9][0-9]*|0)/([1-9][0-9]*|0)',
    re.IGNORECASE
)
COLOR_HEX_REGEX = re.compile(
    r'#([0-9A-F]+)',
    re.IGNORECASE
)

@dataclass
class Color:
    """Helper class for colors in variants."""
    r: int
    g: int
    b: int
    a: int

    def __hash__(self):
        return hash((self.r, self.g, self.b, self.a))

    @staticmethod
    def parse(string: str, palette: tuple[str, str], db: Database) -> tuple[str, Self | None]:
        for name, color in constants.COLOR_NAMES.items():
            if string.startswith(name):
                return string.removeprefix(name), Color.from_index(color, palette, db)
        if (match := COLOR_PAL_REGEX.match(string)):
            return string[match.end():], \
                Color.from_index(
                    (int(match.group(1)), int(match.group(2))),
                    palette, db
                )
        if (match := COLOR_HEX_REGEX.match(string)):
            value = match.group(1)
            assert len(value) in (3, 4, 6, 8), f"Invalid color `#{value}`! Length must be 3, 4, 6, or 8."
            if len(value) < 6:
                value = ''.join(c * 2 for c in value)
            if len(value) != 8:
                value = value + "FF"
            return string[match.end():], \
                Color(
                    int(value[0:2], base=16),
                    int(value[2:4], base=16),
                    int(value[4:6], base=16),
                    int(value[6:8], base=16)
                )
        return string, None

    @staticmethod
    def from_index(index: tuple[int, int], palette: tuple[str, str], db: Database) -> Self:
        pal = db.palette(*palette)
        assert pal is not None, f"Palette `{palette[0]}.{palette[1]}` doesn't seem to exist."
        pal: Image.Image = pal
        try:
            col = pal.getpixel(index)
            return Color(*col)
        except IndexError:
            raise AssertionError(f"The palette index `{color}` is outside of the palette `{palette[0]}.{palette[1]}`.")

    def as_array(self) -> tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

class Renderer:
    pass


type NumpySprite = np.ndarray[tuple, np.dtype[np.uint8]]
