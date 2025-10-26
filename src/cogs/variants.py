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
from .. import constants, errors, utils
from ..tile import Tile, TileSkeleton, TileData, ProcessedTile
from ..types import Bot, RenderContext, Renderer, SignText, NumpySprite, Color
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

    @SkeletonVariantFactory.define_variant(names=None)
    async def noop(
        skel: TileSkeleton, ctx: SkeletonVariantContext
    ):
        """Does nothing. Useful for resetting persistent variants."""

    @SkeletonVariantFactory.define_variant(names="porp")
    async def porp(
        skel: TileSkeleton, ctx: SkeletonVariantContext
    ):
        """Does nothing. Nothing useful, anyways."""
        raise errors.Porp()

    @SkeletonVariantFactory.define_variant(names=["p!", "pal", "palette"])
    async def palette(
        skel: TileSkeleton, ctx: SkeletonVariantContext,
        palette: str
    ):
        """Sets a tile's palette."""
        source = None
        if "." in palette:
            source, palette = palette.split(".", 1)
        skel.palette = (palette, source)

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
    async def sleep(
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

    @TileVariantFactory.define_variant(names=["inactive", "in"])
    async def inactive(
        tile: Tile, ctx: TileVariantContext,
    ):
        """Applies the color that an inactive text of a tile's color would have. This only operates on the default color!"""
        tile.color = constants.INACTIVE_COLORS[tile.color]

    @SpriteVariantFactory.define_variant(names=None)
    async def color(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color
    ):
        """Sets a sprite's color."""
        ctx.tile.custom_color = True
        ctx.tile.color = color
        return utils.recolor(sprite, color)

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

    @SpriteVariantFactory.define_variant(names=["gradient", "grad"])
    async def gradient(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color, angle: float = 0.0, width: float = 1.0,
        offset: float = 0, steps: int = 0, raw: bool = False,
        extrapolate: bool = False, dither: bool = False
    ):
        """
        Applies a gradient to a tile.
        Interpolates color through CIELUV color space by default. This can be toggled with the `raw` argument.
        If `extrapolate` is enabled, then colors outside the gradient will be extrapolated, as opposed to clamping from 0% to 100%.
        Enabling `dither` does nothing with `steps` set to 0.
        """
        ctx.tile.custom_color = True
        src = Color.from_index(ctx.tile.color, ctx.tile.palette, ctx.renderer.bot.db).as_array()
        dst = color.as_array()
        if not raw:
            src = np.hstack((cv2.cvtColor(np.array([[src[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], src[3]))
            dst = np.hstack((cv2.cvtColor(np.array([[dst[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], dst[3]))
        # thank you hutthutthutt#3295 you are a lifesaver
        scale = math.cos(math.radians(angle % 90)) + math.sin(math.radians(angle % 90))
        maxside = max(*sprite.shape[:2]) + 1
        grad = np.mgrid[offset:width + offset:maxside * 1j]
        grad = np.tile(grad[..., np.newaxis], (maxside, 1, 4))
        if not extrapolate:
            grad = np.clip(grad, 0, 1)
        grad_center = maxside // 2, maxside // 2
        rot_mat = cv2.getRotationMatrix2D(grad_center, angle, scale)
        warped_grad = cv2.warpAffine(grad, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_LINEAR)
        if steps:
            if dither:
                needed_size = np.ceil(np.array(warped_grad.shape) / 8).astype(int)
                image_matrix = np.tile(bayer_matrix, needed_size[:2])[:warped_grad.shape[0], :warped_grad.shape[1]]
                mod_warped_grad = warped_grad[:, :, 0]
                mod_warped_grad *= steps
                mod_warped_grad %= 1.0
                mod_warped_grad = (mod_warped_grad > image_matrix).astype(int)
                warped_grad = (np.floor(warped_grad[:, :, 1] * steps) + mod_warped_grad) / steps
                warped_grad = np.array((warped_grad.T, warped_grad.T, warped_grad.T, warped_grad.T)).T
            else:
                warped_grad = np.round(warped_grad * steps) / steps
        mult_grad = np.clip(((1 - warped_grad) * src + warped_grad * dst), 0, 255)
        if not raw:
            mult_grad[:, :, :3] = cv2.cvtColor(mult_grad[:, :, :3].astype(np.uint8), cv2.COLOR_Luv2RGB).astype(
                np.float64)
        mult_grad /= 255
        return (sprite * mult_grad).astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["overlay", "o!"])
    async def overlay(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        overlay: str, x: int = 0, y: int = 0,
    ):
        """Applies an overlay to a sprite. X and Y can be given to offset the overlay."""
        ctx.tile.custom_color = True
        assert overlay in ctx.renderer.overlay_cache, f"`{overlay}` isn't a valid overlay!"
        overlay_image = ctx.renderer.overlay_cache[overlay]
        tile_amount = np.ceil(np.array(sprite.shape[:2]) / overlay_image.shape[:2]).astype(int)
        overlay_image = np.roll(overlay_image, (x, y), (0, 1))
        overlay_image = np.tile(overlay_image, (*tile_amount, 1))[:sprite.shape[0], :sprite.shape[1]].astype(float)
        return np.multiply(sprite, overlay_image / 255, casting="unsafe").astype(np.uint8)

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

    def parse_variant(string: str, palette: tuple[str, str]) -> tuple[str, Variant | None]:
        for var in ALL_VARIANTS.values():
            string, parsed = var.parser(string, bot = bot, palette = palette)
            if parsed is not None and string == "":
                return parsed
        return None

    bot.variants = ALL_VARIANTS
    bot.parse_variant = parse_variant
