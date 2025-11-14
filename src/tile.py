from __future__ import annotations

from dataclasses import dataclass, field

from typing import Literal, Optional
import re
import numpy as np

from src.types import TilingMode
from . import errors, constants, utils
from .db import TileData
from .types import Variant, Context, Bot
from .variant_types import SkeletonVariantContext, TileVariantContext

@dataclass
class TileSkeleton:
    """A tile that hasn't been assigned a sprite yet."""
    name: str = "<empty tile>"
    prefix: str = None
    postfix: str = ""
    raw_string: str = ""
    variants: list = field(default_factory = list)
    palette: tuple[str, str | None] = ("default", "vanilla")
    beta: bool = False
    custom: bool = False
    force_color: tuple[int, int] | None = None

    def clone(self):
        clone = TileSkeleton(**self.__dict__)
        clone.variants = [var for var in self.variants]
        return clone

    @classmethod
    async def parse(cls, bot, string: str, default_prefix: str | None = None,
                    palette: tuple[str, str | None] = ("default", "vanilla"),
                    global_variant="", *, prefix: str = None):
        out = cls()
        if not default_prefix:
            default_prefix = None
        if match := re.fullmatch(r"^(\$|\w+_)(.*)$", string):
            out.prefix = match.group(1)
            string = match.group(2)
            if out.prefix != "$":
                out.prefix = out.prefix[:-1]
            if default_prefix is not None and out.prefix == "tile":
                out.prefix = None
            elif out.prefix == "$":
                if default_prefix is not None:
                    out.prefix = ""
                else:
                    out.prefix = "text"
            elif prefix == "":
                pass
            elif default_prefix is not None:
                out.prefix = default_prefix + "_" + out.prefix
        elif prefix or (prefix == "" and default_prefix is not None):
            out.prefix = prefix
        elif default_prefix is not None:
            out.prefix = default_prefix

        out.postfix = string
        out.raw_string = f"{out.prefix}_{string}" if out.prefix else string
        out.palette = palette
        last_escaped = False

        is_persistent = []
        variants = []
        split_vars: list[(str, str)] = utils.split_escaped(string, (":", ';'), True, True)
        out.name, split_vars = split_vars[0][0], split_vars[1:]
        out.name = f"{out.prefix}_{out.name}" if out.prefix and out.name else out.name
        if out.name == "2":
            async with bot.db.conn.cursor() as cur:
                await cur.execute("""
                    SELECT DISTINCT name FROM tiles WHERE
                        tiling = 'character' AND
                        name NOT LIKE 'text_anni'
                        ORDER BY RANDOM() LIMIT 1
                    """)
                out.name = (await cur.fetchall())[0][0]
            out.force_color = (4, 2)
        split_vars_ex = []
        if global_variant:
            split_global = utils.split_escaped(global_variant, (":", ';'), True, True)
            split_vars_ex.extend(split_global)
        for i, (raw_var, split) in enumerate(split_vars):
            split_vars_ex.append((raw_var, split))
        for raw_var, split in split_vars_ex:
            var = bot.parse_variant(raw_var, out.palette)
            if var is None:
                raise errors.UnknownVariant(raw_var)
            if split == ";":
                var.persistent = True
            out.variants.append(var)
            await var.apply(out, SkeletonVariantContext(bot))
        return out


def is_adjacent(pos, tile, grid, width, height, tile_borders=False) -> bool:
    """Tile is next to a joining tile."""
    w, x, y, z = pos
    joining_tiles = (tile.name, "level", "edge")
    if x < 0 or y < 0 or \
            y >= height or x >= width:
        return tile_borders
    tile = grid.get((y, x, z, w))
    return tile is not None and tile.name in joining_tiles


def get_bitfield(*arr: bool):
    return sum(b << a for a, b in enumerate(list(arr)[::-1]))


def handle_tiling(tile: Tile, grid, width, height, pos, tile_borders=False):
    w, z, y, x = pos
    adj_r = is_adjacent((w, x + 1, y, z), tile, grid, width, height, tile_borders)
    adj_u = is_adjacent((w, x, y - 1, z), tile, grid, width, height, tile_borders)
    adj_l = is_adjacent((w, x - 1, y, z), tile, grid, width, height, tile_borders)
    adj_d = is_adjacent((w, x, y + 1, z), tile, grid, width, height, tile_borders)
    adj_ru = adj_lu = adj_ld = adj_rd = False
    if tile.tiling == TilingMode.DIAGONAL_TILING:
        adj_ru = adj_r and adj_u and is_adjacent(
            (w, x + 1, y - 1, z), tile, grid, width, height, tile_borders)
        adj_lu = adj_u and adj_l and is_adjacent(
            (w, x - 1, y - 1, z), tile, grid, width, height, tile_borders)
        adj_ld = adj_l and adj_d and is_adjacent(
            (w, x - 1, y + 1, z), tile, grid, width, height, tile_borders)
        adj_rd = adj_d and adj_r and is_adjacent(
            (w, x + 1, y + 1, z), tile, grid, width, height, tile_borders)
    tile.frame = constants.TILING_VARIANTS.get(get_bitfield(adj_r, adj_u, adj_l, adj_d, adj_ru, adj_lu, adj_ld, adj_rd))


@dataclass
class Tile:
    """A tile that's ready for processing."""
    name: str = None
    sprite: tuple[str, str] | np.ndarray | None = None
    tiling: TilingMode = TilingMode.NONE
    surrounding: int = 0b00000000  # RULDEQZC
    frame: int = 0
    wobble_frames: tuple[int] | None = None
    custom_color: bool = False
    color: tuple[int, int] = (0, 3)
    custom: bool = False
    oneline: bool = False
    style: Literal["noun", "property", "letter"] = "noun"
    palette: tuple[str, str] = ("default", "vanilla")
    overlay: str | None = None
    hue: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    filterimage: str | None = None
    palette_snapping: bool = False
    normalize_gamma: bool = False
    variants: list = field(default_factory = list)
    altered_frame: bool = False
    text_squish_width: int = 24
    undef: bool = False

    def __hash__(self):
        print("hashing tile")
        return hash((self.name, self.sprite if type(self.sprite) is tuple else 0, self.frame,
                     self.custom, self.color,
                     self.style, self.palette, self.overlay, self.hue,
                     self.gamma, self.saturation, self.filterimage,
                     self.palette_snapping, self.normalize_gamma, self.altered_frame,
                     hash(tuple(var for var in self.variants if var.factory.hashed)),
                     self.custom_color, self.palette, self.text_squish_width))

    @classmethod
    async def prepare(
        cls, bot: Bot, tile: TileSkeleton, tile_data_cache: dict[str, TileData], grid,
        width: int, height: int,
        position: tuple[int, int, int, int], tile_borders: bool = False, ctx: Context = None
    ):
        esc_name = name = utils.split_escaped(tile.name, [])[0]
        value = cls(custom = tile.custom)
        metadata = tile_data_cache.get(name)
        if tile.beta:
            value.style = "beta"
        if metadata is not None:
            value.name = tile.name
            value.sprite = (metadata.source, metadata.sprite)
            value.tiling = metadata.tiling
            value.color = color = metadata.active_color
            value.variants = variants=tile.variants
            value.palette = palette=tile.palette
            if metadata.tiling == TilingMode.TILING or metadata.tiling == TilingMode.DIAGONAL_TILING:
                handle_tiling(value, grid, width, height, position, tile_borders=tile_borders)
        else:
            name = tile.name
            if name[:5] == "text_":
                value.name = name
                value.tiling = TilingMode.NONE
                value.variants = tile.variants
                value.custom = True
                value.palette = tile.palette
            elif name[:5] == "char_" and ctx is not None:  # allow external calling for potential future things?
                seed = int(name[5:]) if re.fullmatch(r'-?\d+', name[5:]) else name[5:]
                character = ctx.bot.generator.generate(seed=seed)
                color = character[1]["color"]
                value.name=name
                value.tiling=TilingMode.CHARACTER
                value.variants=tile.variants
                value.custom=True
                value.sprite=character[0]
                value.color=color
                value.palette=tile.palette
            elif name[:6] == "cchar_" and ctx is not None:  # allow external calling for potential future things? again?
                customid = int(name[6:]) if re.fullmatch(r'-?\d+', name[6:]) else name[6:]
                character = ctx.bot.generator.generate(customid=customid)
                color = character[1]["color"]
                value.name=name
                value.tiling=TilingMode.CHARACTER
                value.variants=tile.variants
                value.custom=True
                value.sprite=character[0]
                value.color=color
                value.palette=tile.palette
            else:
                raise errors.TileNotFound(esc_name)
        if tile.force_color is not None:
            value.color = tile.force_color
        for variant in value.variants:
            await variant.apply(
                value, TileVariantContext(tile_data_cache)
            )
            if value.surrounding != 0:
                if metadata.tiling == TilingMode.TILING:
                    value.surrounding &= 0b11110000
                value.frame = constants.TILING_VARIANTS[value.surrounding]
        if not (metadata is None or value.frame in metadata.extra_frames or value.frame in metadata.tiling.expected()):
            value.frame = 0
        return value

    def clone(self):
        clone = Tile(**self.__dict__)
        clone.variants = [var for var in self.variants]
        return clone


@dataclass
class ProcessedTile:
    """A tile that's been processed, and is ready to render."""
    name: str = "?"
    wobble_frames: tuple[int] | None = None
    frames: list[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = field(
        default_factory=lambda: [None, None, None], repr=False)
    blending: Literal[*tuple(constants.BLENDING_MODES.keys())] = "normal"
    displacement: list[int, int] = field(default_factory=lambda: [0, 0])
    keep_alpha: bool = True

    def copy(self):
        return ProcessedTile(self.empty, self.name, self.wobble_frames, self.frames, self.blending, self.displacement,
                             self.keep_alpha)
