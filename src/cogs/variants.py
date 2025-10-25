import glob
import inspect
from inspect import Parameter
import math
import types
import typing
from typing import Any, Literal, Optional, Union, get_origin, \
                    get_args, Callable, Self, Type
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import cv2
import numpy as np
import visual_center

from . import liquify
from ..utils import recolor, composite
from .. import constants, errors
from ..tile import Tile, TileSkeleton, TileData, ProcessedTile
from ..types import Bot, RenderContext, Renderer, SignText, NumpySprite
from ..variant_types import \
    SkeletonVariantContext, SkeletonVariantFactory, \
    TileVariantContext, TileVariantFactory, \
    SpriteVariantContext, SpriteVariantFactory, \
    SignVariantContext, SignVariantFactory, \
    PostVariantContext, PostVariantFactory, \
    AbstractVariantContext, AbstractVariantFactory, \
    ALL_VARIANTS, Variant


async def setup(bot: Bot):
    ALL_VARIANTS.clear()

#region Variants

    @SkeletonVariantFactory.define_variant(names="porp")
    async def porp(
        skel: TileSkeleton, ctx: SkeletonVariantContext
    ):
        """Does nothing. Nothing useful, anyways."""
        raise errors.Porp()

    @TileVariantFactory.define_variant(names=None)
    async def direction(
        tile: Tile, ctx: TileVariantContext,
        direction: Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]
    ):
        """Sets the direction of a tile."""
        tile.altered_frame = True
        tile.frame = constants.DIRECTION_VARIANTS[direction]

    @TileVariantFactory.define_variant(names=None)
    async def frame(
        tile: Tile, ctx: TileVariantContext,
        frame: int
    ):
        """Sets the animation frame of a sprite."""
        tile.altered_frame = True
        tile.frame = frame
        tile.surrounding = 0

    @TileVariantFactory.define_variant(names=None)
    async def tiling(
        tile: Tile, ctx: TileVariantContext,
        tiling: Literal[*tuple(constants.AUTO_VARIANTS.keys())]
    ):
        """Alters the tiling of a tile. Only works on tiles that tile."""
        tile.altered_frame = True
        tile.surrounding |= constants.AUTO_VARIANTS[tiling]

    @TileVariantFactory.define_variant(names=["a"])
    async def animation_frame(
        tile: Tile, ctx: TileVariantContext,
        frame: int
    ):
        """Sets the animation frame of a tile."""
        tile.altered_frame = True
        tile.frame += a_frame

    @TileVariantFactory.define_variant(names=["s", "sleep"])
    async def animation_frame(
        tile: Tile, ctx: TileVariantContext,
    ):
        """Makes the tile fall asleep. Only functions correctly on character tiles."""
        tile.altered_frame = True
        tile.frame = (tile.frame - 1) % 32

    @TileVariantFactory.define_variant(names=["tw", "textwidth"])
    async def textwidth(
        tile: Tile, ctx: TileVariantContext,
        width: int
    ):
        """Sets the width of the custom text the text generator tries to expand to."""
        tile.text_squish_width = width


    @SpriteVariantFactory.define_variant(names=["posterize"])
    async def posterize(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        bands: int
    ):
        """Posterizes a sprite."""
        return np.dstack([
            np.digitize(
                sprite[..., i],
                np.linspace(0, 255, bands)
            ) * (255 / bands) for i in range(4)
        ]).astype(np.uint8)



    @PostVariantFactory.define_variant(names=None)
    async def blending(
        post: ProcessedTile, ctx: PostVariantContext,
        mode: Literal[*constants.BLENDING_MODES],
        keep_alpha: bool = True
    ):
        """Sets the blending mode for a tile."""
        post.blending = mode
        post.keep_alpha = keep_alpha and mode != "mask"

    @PostVariantFactory.define_variant(names=["displace", "disp", "d"])
    async def displace(
        post: ProcessedTile, ctx: PostVariantContext,
        x: int, y: int
    ):
        """Displaces the tile by the specified coordinates."""
        post.displacement = [post.displacement[0] + x, post.displacement[1] + y]

#endregion

    all_vars = [(key, value) for (key, value) in ALL_VARIANTS.items()]

    def sort_variants(a: tuple[str, AbstractVariantFactory]) -> int:
        if a[1].nameless:
            return -1
        return len(a[0]) * 1000 + hash(a[0]) % 1000

    all_vars = sorted(all_vars, key=sort_variants)
    ALL_VARIANTS.clear()
    for key, val in all_vars:
        ALL_VARIANTS[key] = val

    def parse_variant(string: str) -> tuple[str, Variant | None]:
        for var in ALL_VARIANTS.values():
            string, parsed = var.parser(string)
            if parsed is not None and string == "":
                return parsed
        return None

    bot.variants = ALL_VARIANTS
    bot.parse_variant = parse_variant
