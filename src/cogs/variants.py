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

from . import liquify as lq
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

#this is a C# thing but my ide supports it in python so why not
# region Variants

    @AbstractVariantFactory.define_variant(names=None)
    async def noop(_target: type(None), _ctx: type(None)):
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

    @SkeletonVariantFactory.define_variant(names=["beta"])
    async def beta(
        skel: TileSkeleton, ctx: SkeletonVariantContext,
    ):
        """Makes custom words appear as beta text."""
        skel.custom = True
        skel.beta = True

    async def sign_color(
        sign: SignText,
        color: Color
    ):
        sign.color = color.as_array()

    async def sign_displace(
        sign: SignText,
        x: int, y: int
    ):
        sign.xo += x
        sign.yo += y

    async def sign_scale(
        sign: SignText,
        scale: float, _unused = None
    ):
        sign.size *= scale


    @SignVariantFactory.define_variant(names=["anchor!", "a!"])
    async def anchor(
        sign: SignText, ctx: SignVariantContext,
    ):
        """Sets the anchor of a sign text. https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html"""
        assert (
            len(anchor) == 2 and
            anchor[0] in ('l', 'm', 'r') and
            anchor[1] in ('a', 'm', 's', 'd')
        ), f"Anchor of `{anchor}` is invalid!"
        sign.anchor = anchor

    @SignVariantFactory.define_variant(names=["stroke", ])
    async def stroke(
        sign: SignText, ctx: SignVariantContext,
        color: Color, size: int,
    ):
        """Sets the sign text's stroke."""
        sign.stroke = color.as_array(), size

    @SignVariantFactory.define_variant(names=["f!", "font!"])
    async def font(
        sign: SignText, ctx: SignVariantContext,
        name: Literal[*tuple(Path(f).stem for f in glob.glob('data/fonts/*.ttf'))]
    ):
        """Applies a font to a sign text object."""
        sign.font = name

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

    @TileVariantFactory.define_variant(names=["custom", "ct"])
    async def custom(
        tile: Tile, ctx: TileVariantContext,
    ):
        """Forces custom generation of the text."""
        tile.custom = True
        tile.style = "noun"

    @TileVariantFactory.define_variant(names=["let", "letter"])
    async def letter(
        tile: Tile, ctx: TileVariantContext,
    ):
        """Makes custom words appear as letter groups."""
        tile.style = "letter"

    @TileVariantFactory.define_variant(names=["oneline", "1l"])
    async def oneline(
        tile: Tile, ctx: TileVariantContext,
    ):
        """Makes custom words appear in one line."""
        tile.oneline = True

    @TileVariantFactory.define_variant(names=["frames", "f"])
    async def frames(
        tile: Tile, ctx: TileVariantContext,
        frame: list[int]
    ):
        """Sets the wobble of the tile to the specified frame(s). 1 or 3 can be specified."""
        assert all(f in range(1, 4) for f in frame), f"One or more wobble frames is outside of the supported range of [1, 3]!"
        assert len(frame) <= 3 and len(frame) != 2, "Only 1 or 3 frames can be specified."
        tile.wobble_frames = [f - 1 for f in frame]


    @SignVariantFactory.define_variant(names=["anchor!", "a!"])
    async def anchor(
        sign: SignText, ctx: SignVariantContext,
    ):
        """Sets the anchor of a sign text. https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html"""
        assert (
            len(anchor) == 2 and
            anchor[0] in ('l', 'm', 'r') and
            anchor[1] in ('a', 'm', 's', 'd')
        ), f"Anchor of `{anchor}` is invalid!"
        sign.anchor = anchor

    @SignVariantFactory.define_variant(names=["stroke", ])
    async def stroke(
        sign: SignText, ctx: SignVariantContext,
        color: Color, size: int,
    ):
        """Sets the sign text's stroke."""
        sign.stroke = color.as_array(), size

    @SignVariantFactory.define_variant(names=["f!", "font!"])
    async def font(
        sign: SignText, ctx: SignVariantContext,
        name: Literal[*tuple(Path(f).stem for f in glob.glob('data/fonts/*.ttf'))]
    ):
        """Applies a font to a sign text object."""
        sign.font = name

    @SpriteVariantFactory.define_variant(names=["%", "dcol"])
    async def default_color(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color
    ):
        """Sets the default color of a sprite."""
        ctx.color = color
        return sprite

    @SpriteVariantFactory.define_variant(names=["apply", "ac", "~"])
    async def apply(
        sprite: NumpySprite, ctx: SpriteVariantContext,
    ):
        """Immediately applies a sprite's default color."""
        col = ctx.color
        ctx.color = Color(255, 255, 255, 255)
        sprite = utils.recolor(sprite, col)
        return sprite

    @SpriteVariantFactory.define_variant(names=None, sign_alt = sign_color)
    async def color(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color
    ):
        """Sets a sprite's color."""
        ctx.color = Color(255, 255, 255, 255)
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
        src = ctx.color.as_array()
        ctx.color = Color(255, 255, 255, 255)
        dst = color.as_array()
        if not raw:
            src = np.hstack((cv2.cvtColor(
                np.array([[src[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], src[3]))
            dst = np.hstack((cv2.cvtColor(
                np.array([[dst[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], dst[3]))
        # thank you hutthutthutt#3295 you are a lifesaver
        scale = math.cos(math.radians(angle % 90)) + \
                         math.sin(math.radians(angle % 90))
        maxside = max(*sprite.shape[:2]) + 1
        grad = np.mgrid[offset:width + offset:maxside * 1j]
        grad = np.tile(grad[..., np.newaxis], (maxside, 1, 4))
        if not extrapolate:
            grad = np.clip(grad, 0, 1)
        grad_center = maxside // 2, maxside // 2
        rot_mat = cv2.getRotationMatrix2D(grad_center, angle, scale)
        warped_grad = cv2.warpAffine(
            grad, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_LINEAR)
        if steps:
            if dither:
                needed_size = np.ceil(
                    np.array(warped_grad.shape) / 8).astype(int)
                image_matrix = np.tile(bayer_matrix, needed_size[:2])[
                                       :warped_grad.shape[0], :warped_grad.shape[1]]
                mod_warped_grad = warped_grad[:, :, 0]
                mod_warped_grad *= steps
                mod_warped_grad %= 1.0
                mod_warped_grad = (mod_warped_grad > image_matrix).astype(int)
                warped_grad = (
                    np.floor(warped_grad[:, :, 1] * steps) + mod_warped_grad) / steps
                warped_grad = np.array(
                    (warped_grad.T, warped_grad.T, warped_grad.T, warped_grad.T)).T
            else:
                warped_grad = np.round(warped_grad * steps) / steps
        mult_grad = np.clip(
            ((1 - warped_grad) * src + warped_grad * dst), 0, 255)
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
        ctx.color = Color(255, 255, 255, 255)
        assert overlay in ctx.renderer.overlay_cache, f"`{utils.sanitize(overlay)}` isn't a valid overlay!"
        overlay_image = ctx.renderer.overlay_cache[overlay]
        tile_amount = np.ceil(
            np.array(sprite.shape[:2]) / overlay_image.shape[:2]).astype(int)
        overlay_image = np.roll(overlay_image, (x, y), (0, 1))
        overlay_image = np.tile(overlay_image, (*tile_amount, 1)
                                )[:sprite.shape[0], :sprite.shape[1]].astype(float)
        return np.multiply(sprite, overlay_image / 255, casting="unsafe").astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["noun", "prop", "quality"])
    async def property(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        plate: Literal["blank", "left", "up", "right",
            "down", "turn", "deturn", "soft", ""] = "",
    ):
        """Applies a property plate to a sprite."""
        if plate == "":
            plate = ctx.tile.frame if ctx.tile.altered_frame else None
        else:
            plate = {v: k for k, v in constants.DIRECTIONS.items()}[plate]
        sprite = sprite[:, :, 3] > 0
        plate, _ = ctx.renderer.bot.db.plate(plate, ctx.wobble)
        plate = np.array(plate)[..., 3] > 0
        size = tuple(max(a, b) for a, b in zip(sprite.shape[:2], plate.shape))
        dummy = np.zeros(size, dtype=bool)
        delta = ((plate.shape[0] - sprite.shape[0]) // 2,
                 (plate.shape[1] - sprite.shape[1]) // 2)
        p_delta = max(-delta[0], 0), max(-delta[1], 0)
        delta = max(delta[0], 0), max(delta[1], 0)
        dummy[p_delta[0]:p_delta[0] + plate.shape[0],
        p_delta[1]:p_delta[1] + plate.shape[1]] = plate
        dummy[delta[0]:delta[0] + sprite.shape[0],
        delta[1]:delta[1] + sprite.shape[1]] &= ~sprite
        return np.dstack([dummy[..., np.newaxis].astype(np.uint8) * 255] * 4)

    @SpriteVariantFactory.define_variant(names=["hide", "-"])
    async def hide(
        sprite: NumpySprite, ctx: SpriteVariantContext,
    ):
        """Sets the tile's opacity to 0."""
        sprite[..., 3] = 0
        return sprite

    @SpriteVariantFactory.define_variant(names=["rotate", "rot"])
    async def rotate(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        angle: float, expand: bool = False
    ):
        """Rotates a sprite."""
        if expand:
            scale = math.cos(math.radians(-angle % 90)) + math.sin(math.radians(-angle % 90))
            padding = int(sprite.shape[0] * ((scale - 1) / 2)), int(sprite.shape[1] * ((scale - 1) / 2))
            dst_size = sprite.shape[0] + padding[0], sprite.shape[1] + padding[1]
            utils.check_size(*dst_size)
            sprite = np.pad(sprite,
                            (padding,
                             padding,
                             (0, 0)))
        image_center = tuple(np.array(sprite.shape[1::-1]) / 2 - 0.5)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        return cv2.warpAffine(sprite, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_NEAREST)

    @SpriteVariantFactory.define_variant(names=["rotate3d", "rot3d", "r3d"])
    async def rotate3d(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        phi: float, theta: float, gamma: float
    ):
        """Rotates a sprite in 3D space."""
        phi, theta, gamma = math.radians(phi), math.radians(theta), math.radians(gamma)
        d = np.sqrt(sprite.shape[1] ** 2 + sprite.shape[0] ** 2)
        f = d / (2 * math.sin(gamma) if math.sin(gamma) != 0 else 1)
        w, h = sprite.shape[1::-1]
        proj_23 = np.array([[1, 0, -w / 2],
                            [0, 1, -h / 2],
                            [0, 0, 1],
                            [0, 0, 1]])
        rot_mat = np.dot(np.dot(
            np.array([[1, 0, 0, 0],
                      [0, math.cos(theta), -math.sin(theta), 0],
                      [0, math.sin(theta), math.cos(theta), 0],
                      [0, 0, 0, 1]]),
            np.array([[math.cos(phi), 0, -math.sin(phi), 0],
                      [0, 1, 0, 0],
                      [np.sin(phi), 0, math.cos(phi), 0],
                      [0, 0, 0, 1]])),
            np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                      [math.sin(gamma), math.cos(gamma), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]))
        trans_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, f],
            [0, 0, 0, 1]])
        proj_32 = np.array([
            [f, 0, w / 2, 0],
            [0, f, h / 2, 0],
            [0, 0, 1, 0]
        ])
        final_matrix = np.dot(proj_32, np.dot(trans_mat, np.dot(rot_mat, proj_23)))
        return cv2.warpPerspective(sprite, final_matrix, sprite.shape[1::-1], flags=cv2.INTER_NEAREST)

    @SpriteVariantFactory.define_variant(names=["scale", "s"], sign_alt = sign_scale)
    async def scale(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        w: float, h: float = None,
        interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"] = "nearest"
    ):
        """Scales a sprite by the given multipliers."""
        if h is None:
            h = w
        dst_size = (int(w * sprite.shape[0]), int(h * sprite.shape[1]))
        if dst_size[0] <= 0 or dst_size[1] <= 0:
            raise AssertionError(
                f"Can't scale a tile to `{int(w * sprite.shape[0])}x{int(h * sprite.shape[1])}`, as it has a non-positive target area.")
        utils.check_size(*dst_size)
        dim = sprite.shape[:2] * np.array((h, w))
        dim = dim.astype(int)
        return cv2.resize(sprite[:, ::-1], dim[::-1], interpolation={
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }[interpolation])[:, ::-1]

    @SpriteVariantFactory.define_variant(names=["pad", "p"])
    async def pad(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        left: int, top: int, right: int, bottom: int
    ):
        """Pads a sprite by the specified values."""
        utils.check_size(sprite.shape[1] + max(left, 0) + max(right, 0), sprite.shape[0] + max(top, 0) + max(bottom, 0))
        return np.pad(sprite, ((top, bottom), (left, right), (0, 0)))

    @SpriteVariantFactory.define_variant(names=["pixelate", "px"])
    async def pixelate(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x: int, y: int
    ):
        """Pixelates a sprite."""
        if y is None:
            y = x
        return sprite[y - 1::y, x - 1::x].repeat(y, axis=0).repeat(x, axis=1)

    def get_kernel(size: int, kernel: Literal["full", "edge", "unit"]):
        ksize = 2*size+1
        ker = np.ones((ksize, ksize))
        if kernel == 'full':
            ker[size, size] = - ksize**2 + 1
        elif kernel == 'edge':
            ker[size, size] = - ksize**2 + 5
            ker[0,0] = 0
            ker[0,ksize-1] = 0
            ker[ksize-1,ksize-1] = 0
            ker[ksize-1,0] = 0
        elif kernel == 'unit':
            ker[size, size] = - ksize**2 + 5
            ker[0,0] = 0
            ker[0,ksize-1] = 0
        return ker

    @SpriteVariantFactory.define_variant(names=["meta", "m"])
    async def meta(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        level: int = 1, kernel: Literal["full", "edge", "unit"] = "full", size: int = 1
    ):
        """Applies a meta filter to an sprite."""
        if level is None: level = 1
        if size is None: size = 1
        assert size > 0, f"The given meta size of {size} is too small!"
        assert size <= constants.MAX_META_SIZE, f"The given meta size of {size} is too large! Try something lower than `{constants.MAX_META_SIZE}`."
        assert abs(level) <= constants.MAX_META_DEPTH, f"Meta depth of {level} too large! Try something lower than `{constants.MAX_META_DEPTH}`."
        # Not padding at negative values is intentional
        padding = max(level*size, 0)
        orig = np.pad(sprite, ((padding, padding), (padding, padding), (0, 0)))
        utils.check_size(*orig.shape[size::-1])
        base = orig[..., 3]
        if level < 0:
            base = 255 - base
        ker = get_kernel(size, kernel)
        for _ in range(abs(level)):
            base = cv2.filter2D(src=base, ddepth=-1, kernel=ker)
        base = np.dstack((base, base, base, base))
        mask = orig[..., 3] > 0
        if not (level % 2) and level > 0:
            base[mask, ...] = orig[mask, ...]
        else:
            base[mask ^ (level < 0), ...] = 0
        return base

    @SpriteVariantFactory.define_variant(names=["outline", "o"])
    async def outline(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color = None, size: int = 1, kernel: Literal["full", "edge", "unit"] = "full"
    ):
        """Applies an outline to a sprite."""
        assert size > 0, f"The given outline size of {size} is too small!"
        assert size <= constants.MAX_META_SIZE, f"The given outline size of {size} is too large! Try something lower than `{constants.MAX_META_SIZE}`."
        col = ctx.color
        ctx.color = Color(255, 255, 255, 255)
        sprite = utils.recolor(sprite, col)
        if color is None:
            color = Color.from_index((0, 4), ctx.tile.palette, ctx.renderer.bot.db)
        orig = np.pad(sprite, ((size, size), (size, size), (0, 0)))
        utils.check_size(*orig.shape[size::-1])
        base = (orig[..., 3] > 0).astype(np.uint8) * 255
        ksize = 2*size + 1
        ker = get_kernel(size, kernel)
        outline = cv2.filter2D(src=base, ddepth=-1, kernel=ker)
        outline = np.dstack((outline, outline, outline, outline))
        outline = utils.recolor(outline, color)
        return orig + outline

    @SpriteVariantFactory.define_variant(names=None)
    async def omni(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        type: Literal["pivot", "branching"] = "branching",
    ):
        """Gives the tile an overlay, like the omni text."""
        opvalue = [0xcb, 0xab, 0x8b][ctx.wobble]
        num = 3
        if type == "pivot":
            num = 1
        nsprite = await meta(sprite, ctx, num)
        sprite = await pad(sprite, ctx, num, num, num, num)
        for i in range(nsprite.shape[0]):
            for j in range(nsprite.shape[1]):
                if nsprite[i, j, 3] == 0:
                    try:
                        nsprite[i, j] = sprite[i, j]
                    except:
                        pass
                else:
                    nsprite[i, j, 3] = opvalue
        return nsprite

    @SpriteVariantFactory.define_variant(names=["land"])
    async def land(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        direction: Literal["left", "top", "right", "bottom"] = "bottom"
    ):
        """Removes all space between the sprite and its bounding box on the specified side."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        left, right = np.where(cols)[0][[0, -1]]
        top, bottom = np.where(rows)[0][[0, -1]]
        displacement = {"left": left, "top": top, "right": right+1-sprite.shape[1], "bottom": bottom+1-sprite.shape[0]}[direction]
        index = {"left": 0, "top": 1, "right": 0, "bottom": 1}[direction]
        return await wrap(sprite, ctx, ((1 - index) * displacement), index * displacement)

    @SpriteVariantFactory.define_variant(names=["bbox", ])
    async def bbox(
        sprite: NumpySprite, ctx: SpriteVariantContext
    ):
        """Puts the sprite's bounding box behind it. Useful for debugging."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        try:
            left, right = np.where(cols)[0][[0, -1]]
            top, bottom = np.where(rows)[0][[0, -1]]
        except IndexError:
            return sprite
        out = np.zeros_like(sprite).astype(float)
        out[top:bottom,   left:right] = (0xFF, 0xFF, 0xFF, 0x80)
        out[top,          left:right] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[bottom,       left:right] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[top:bottom,   left      ] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[top:bottom+1, right     ] = (0xFF, 0xFF, 0xFF, 0xc0)
        sprite = sprite.astype(float)
        mult = sprite[..., 3, np.newaxis] / 255
        sprite[..., :3] = (1 - mult) * out[..., :3] + mult * sprite[..., :3]
        sprite[...,  3] = (sprite[..., 3] + out[..., 3] * (1 - mult[..., 0]))
        return sprite.astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["warp", ])
    async def warp(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x1: int, y1: int,
        x2: int, y2: int,
        x3: int, y3: int,
        x4: int, y4: int
    ):
        """Warps the sprite by displacing the bounding box's corners.
    Point 1 is top-left, point 2 is top-right, point 3 is bottom-right, and point 4 is bottom-left.
    If the sprite grows past its original bounding box, it will need to be recentered manually."""
        x1_y1 = x1, y1
        x2_y2 = x2, y2
        x3_y3 = x3, y3
        x4_y4 = x4, y4
        src_shape = np.array(sprite.shape[-2::-1])
        src = (np.array([
            [[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
            [[1.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            [[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.0, 1.0], [0.5, 0.5], [0.0, 0.0]],
        ]) * (src_shape - 1)).astype(np.int32)
        pts = np.array((x1_y1, x2_y2, x3_y3, x4_y4))
        # This package is only 70kb and I'm lazy
        center = visual_center.find_pole(pts, precision=1)[0]
        dst = src + np.array([
            [x1_y1, center, x2_y2],
            [x2_y2, center, x3_y3],
            [x3_y3, center, x4_y4],
            [x4_y4, center, x1_y1],
        ], dtype=np.int32)
        # Set padding values
        before_padding = np.array([
            max(-x1, -x4, 0),  # Added padding for left
            max(-y1, -y2, 0)  # Added padding for top
        ])
        after_padding = np.array([
            (max(x2, x3, 0)),  # Added padding for right
            (max(y3, y4, 0))  # Added padding for bottom
        ])
        dst += before_padding
        new_shape = (src_shape + before_padding + after_padding).astype(np.uint32)[::-1]
        utils.check_size(*new_shape)
        final_arr = np.zeros((*new_shape, 4), dtype=np.uint8)
        for source, destination in zip(src, dst):  # Iterate through the four triangles
            clip = cv2.fillConvexPoly(np.zeros(new_shape, dtype=np.uint8), destination, 1).astype(bool)
            M = cv2.getAffineTransform(source.astype(np.float32), destination.astype(np.float32))
            warped_arr = cv2.warpAffine(sprite, M, new_shape[::-1], flags=cv2.INTER_NEAREST)
            final_arr[clip] = warped_arr[clip]
        return final_arr

    @SpriteVariantFactory.define_variant(names=["matrix", "mm", "matmul"])
    async def matmul(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        aa: float, ab: float, ac: float, ad: float,
        ba: float, bb: float, bc: float, bd: float,
        ca: float, cb: float, cc: float, cd: float,
        da: float, db: float, dc: float, dd: float,
        ea: float = None, eb: float = None, ec: float = None, ed: float = None,
    ):
        """Multiplies the sprite by the given RGBA matrix."""
        offset = (0, 0, 0, 0)
        if ea is not None or eb is not None or ec is not None or ed is not None:
            assert ea is not None and eb is not None and ec is not None and ed is not None, \
                "Must specify all 4 values for extra matmul row."
            offset = (ea, eb, ec, ed)
        matrix = np.array((
            (aa, ba, ca, da),
            (ab, bb, cb, db),
            (ac, bc, cc, dc),
            (ad, bd, cd, dd)
        ))
        img = sprite.astype(np.float64) / 255.0
        immul = img.reshape(-1, 4) @ matrix  # @ <== matmul
        immul[..., 0] += offset[0]
        immul[..., 1] += offset[1]
        immul[..., 2] += offset[2]
        immul[..., 3] += offset[3]
        immul = (np.clip(immul, 0.0, 1.0) * 255).astype(np.uint8)
        return immul.reshape(img.shape)

    @SpriteVariantFactory.define_variant(names=["clip", ])
    async def clip(
        sprite: NumpySprite, ctx: SpriteVariantContext,
    ):
        """Crops the sprite to within its grid space."""
        width = sprite.shape[1]
        height = sprite.shape[0]
        left = (width - 24) // 2
        up = (height - 24) // 2
        right = (width + 24) // 2
        down = (height + 24) // 2
        return await crop(sprite, ctx, left, up, right, down, True)

    @SpriteVariantFactory.define_variant(names=["neon", ])
    async def neon(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        strength: float = 0.714
    ):
        """Darkens the inside of each region of color."""
        # This is approximately 2.14x faster than Charlotte's neon, profiling at strength 0.5 with 2500 iterations on baba/frog_0_1.png.
        unique_colors = lq.get_colors(sprite)
        final_mask = np.ones(sprite.shape[:2], dtype=np.float64)
        for color in unique_colors:
            mask = (sprite == color).all(axis=2)
            float_mask = mask.astype(np.float64)
            card_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=CARD_KERNEL)
            oblq_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=OBLQ_KERNEL)
            final_mask[card_mask == 4] -= strength / 2
            final_mask[oblq_mask == 4] -= strength / 2
        if strength < 0:
            final_mask = np.abs(1 - final_mask)
        sprite[:, :, 3] = np.multiply(sprite[:, :, 3], np.clip(final_mask, 0, 1), casting="unsafe")
        return sprite.astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["scan", ])
    async def scan(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        axis: Literal["x", "y"], on: int = 1, off: int = 1, offset: int = 0
    ):
        """Removes rows or columns of pixels to create a scan line effect."""
        assert on >= 0 and off >= 0 and on + off > 0, f"Scan mask of `{on}` on and `{off}` off is invalid!"
        axis = ("y", "x").index(axis)
        mask = np.roll(np.array([1] * on + [0] * off, dtype=np.uint8), offset)
        mask = np.tile(mask, (
            sprite.shape[1 - axis],
            int(math.ceil(sprite.shape[axis] / mask.shape[0]))
        ))[:, :sprite.shape[axis]]
        if not axis:
            mask = mask.T
        return np.dstack((sprite[:, :, :3], sprite[:, :, 3] * mask))

    @SpriteVariantFactory.define_variant(names=["flip", ])
    async def flip(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        axis: Literal["x", "y"]
    ):
        """Flips the sprite along the specified axis."""
        if a == "x":
            return sprite[:, ::-1, :]
        else:
            return sprite[::-1, :, :]

    @SpriteVariantFactory.define_variant(names=["mirror", ])
    async def mirror(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        axis: Literal["x", "y"], half: Literal["back", "front"]
    ):
        """Mirrors the sprite along the specified direction."""
        if axis == "x":
            sprite = np.rot90(sprite)
        if half == "front":
            sprite = np.flipud(sprite)
        sprite[:sprite.shape[0] // 2] = sprite[:sprite.shape[0] // 2 - 1:-1]
        if half == "front":
            sprite = np.flipud(sprite)
        if axis == "x":
            sprite = np.rot90(sprite, -1)
        return sprite

    @SpriteVariantFactory.define_variant(names=["normalize", "norm"])
    async def normalize(
        sprite: NumpySprite, ctx: SpriteVariantContext
    ):
        """Centers the sprite on its visual bounding box."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        if not len(row_check := np.where(rows)[0]) or not len(col_check := np.where(cols)[0]):
            return sprite
        left, right = col_check[[0, -1]]
        top, bottom = row_check[[0, -1]]
        sprite_center = sprite.shape[0] // 2 - 1, sprite.shape[1] // 2 - 1
        center = int((top + bottom) // 2), int((left + right) // 2)
        displacement = np.array((sprite_center[0] - center[0], sprite_center[1] - center[1]))
        return np.roll(sprite, displacement, axis=(0, 1))

    # Original code by Charlotte (CenTdemeern1)
    @SpriteVariantFactory.define_variant(names=["floodfill", "flood"])
    async def floodfill(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color, inside: bool = True,
    ):
        """Floodfills either inside or outside a sprite with a given brightness value."""
        sprite[sprite[:, :, 3] == 0] = 0  # Optimal
        sprite_alpha = sprite[:, :, 3]  # Stores the alpha channel separately
        sprite_alpha[sprite_alpha > 0] = -1  # Sets all nonzero numbers to a number that's neither 0 nor 255.
        # Pads the alpha channel by 1 on each side to allow flowing past
        # where the sprite touches the edge of the bounding box.
        sprite_alpha = np.pad(sprite_alpha, ((1, 1), (1, 1)))
        sprite_flooded = cv2.floodFill(
            image=sprite_alpha,
            mask=None,
            seedPoint=(0, 0),
            newVal=255
        )[1]
        mask = sprite_flooded != (inside * 255)
        sprite_flooded[mask] = ((not inside) * 255)
        mask = mask[1:-1, 1:-1]
        if inside:
            sprite_flooded = 255 - sprite_flooded
        # Crops the alpha channel back to the original size and positioning
        sprite[:, :, 3][mask] = sprite_flooded[1:-1, 1:-1][mask].astype(np.uint8)
        sprite[(sprite[:, :] == [0, 0, 0, 255]).all(2)] = color.as_array()
        return sprite

    @SpriteVariantFactory.define_variant(names=["pointfill", "pf"])
    async def pointfill(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color, x: int, y: int,
    ):
        """Floodfills a sprite starting at a given point."""
        assert x >= 0 and y >= 0 and y < sprite.shape[0] and x < sprite.shape[1], f"Target point `{x},{y}` must be inside the sprite!"
        target_color = sprite[y,x]
        sprite[sprite[:, :, 3] == 0] = 0  # Optimal
        sprite_alpha = sprite[:, :, :].copy()  # Stores the alpha channel separately
        not_color_mask = (sprite[:, :, 0] != target_color[0]) | (sprite[:, :, 1] != target_color[1]) | (sprite[:, :, 2] != target_color[2])
        color_mask = (sprite[:, :, 0] == target_color[0]) & (sprite[:, :, 1] == target_color[1]) & (sprite[:, :, 2] == target_color[2])
        sprite_alpha[not_color_mask] = 255
        sprite_alpha[color_mask] = 0 # and now to override it
        sprite_alpha = sprite_alpha[:, :, 3].copy() #???
        sprite_flooded = cv2.floodFill(
            image=sprite_alpha,
            mask=None,
            seedPoint=(x, y),
            newVal=100
        )[1]
        mask = sprite_flooded == 100
        sprite[mask] = color.as_array()
        return sprite

    @SpriteVariantFactory.define_variant(names=["remove", "rm"])
    async def remove(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color: Color, invert: bool = False,
    ):
        """Removes a certain color from the sprite. If `invert` is on, then it removes all but that color."""
        color = color.as_array()
        if invert:
            sprite[(sprite[:, :, 0] != color[0]) | (sprite[:, :, 1] != color[1]) | (sprite[:, :, 2] != color[2])] = 0
        else:
            sprite[(sprite[:, :, 0] == color[0]) & (sprite[:, :, 1] == color[1]) & (sprite[:, :, 2] == color[2])] = 0
        return sprite

    @SpriteVariantFactory.define_variant(names=["replace", "rp"])
    async def replace(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        color1: Color, color2: Color, invert: bool = False,
    ):
        """Replaces a certain color with a different color. If `invert` is on, then it replaces all but that color."""
        color1 = color1.as_array()
        color2 = color2.as_array()
        if invert:
            sprite[(sprite[:, :, 0] != color1[0]) | (sprite[:, :, 1] != color1[1]) | (sprite[:, :, 2] != color1[2])] = color2
        else:
            sprite[(sprite[:, :, 0] == color1[0]) & (sprite[:, :, 1] == color1[1]) & (sprite[:, :, 2] == color1[2])] = color2
        return sprite

    @SpriteVariantFactory.define_variant(names=["pad", ])
    async def pad(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        left: int, top: int, right: int, bottom: int
    ):
        """Pads the sprite by the specified values."""
        utils.check_size(sprite.shape[1] + max(left, 0) + max(right, 0), sprite.shape[0] + max(top, 0) + max(bottom, 0))
        return np.pad(sprite, ((top, bottom), (left, right), (0, 0)))

    def slice_image(sprite, color_slice: slice):
        colors = lq.get_colors(sprite)
        if len(colors) > 1:
            colors = list(sorted(
                colors,
                key=lambda color: lq.count_instances_of_color(sprite, color),
                reverse=True
            ))
            try:
                selection = np.arange(len(colors))[color_slice]
            except IndexError:
                raise AssertionError(f'The color slice `{color_slice}` is invalid.')
            if isinstance(selection, np.ndarray):
                selection = selection.flatten().tolist()
            else:
                selection = [selection]
            # Modulo the value field
            positivevalue = [(color % len(colors)) for color in selection]
            # Remove most used color
            for color_index, color in enumerate(colors):
                if color_index not in positivevalue:
                    sprite = lq.remove_instances_of_color(sprite, color)
        return sprite

    @SpriteVariantFactory.define_variant(names=["color_select", "csel", "c"])
    async def color_select(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        index: list[int]
    ):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, index)

    @SpriteVariantFactory.define_variant(names=["color_slice", "cslice", "cs"])
    async def color_slice(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        s: list[int]
    ):
        """Keeps only the slice of colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        slc = slice(s[0] if len(s) > 0 else None, s[1] if len(s) > 1 else None, s[2] if len(s) > 2 else None)
        return slice_image(sprite, slc)

    @SpriteVariantFactory.define_variant(names=["color_shift", "cshift", "csh"])
    async def color_shift(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        s: list[int]
    ):
        """Shifts the colors of a sprite around, by index of occurence."""
        slc = slice(s[0] if len(s) > 0 else None, s[1] if len(s) > 1 else None, s[2] if len(s) > 2 else None)
        unique_colors = lq.get_colors(sprite)
        unique_colors = np.array(
            sorted(unique_colors, key=lambda color: lq.count_instances_of_color(sprite, color), reverse=True))
        final_sprite = np.tile(sprite, (len(unique_colors), 1, 1, 1))
        mask = np.equal(final_sprite[:, :, :, :], unique_colors.reshape((-1, 1, 1, 4))).all(axis=3)
        out = np.zeros(sprite.shape)
        for i, color in enumerate(unique_colors[slc]):
            out += np.tile(mask[i].T, (4, 1, 1)).T * color
        return out.astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["croppoly", ])
    async def croppoly(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        point_coords: list[int]
    ):
        """Crops the sprite to the specified polygon."""
        assert len(point_coords) % 2 == 0, "Must have an even number of numbers for a point list!"
        assert len(point_coords) >= 6, "Must have at least 3 points to define a polygon!"
        pts = np.array(point_coords, dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[1::-1], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        return np.multiply(sprite, clip_poly, casting="unsafe").astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["croppoly", ])
    async def snippoly(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        point_coords: list[int]
    ):
        """Snips the sprite to the specified polygon."""
        assert len(point_coords) % 2 == 0, "Must have an even number of numbers for a point list!"
        assert len(point_coords) >= 6, "Must have at least 3 points to define a polygon!"
        pts = np.array(point_coords, dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[1::-1], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        return np.multiply(sprite, 1 - clip_poly, casting="unsafe").astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["aberrate", "abberate"])
    async def aberrate(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x: int = 1, y: int = 0
    ):
        """Abberates the colors of a sprite."""
        utils.check_size(sprite.shape[0] + abs(x) * 2, sprite.shape[1] + abs(y) * 2)
        sprite = np.pad(sprite, ((abs(y), abs(y)), (abs(x), abs(x)), (0, 0)))
        sprite[:, :, 0] = np.roll(sprite[:, :, 0], -x, 1)
        sprite[:, :, 2] = np.roll(sprite[:, :, 2], x, 1)
        sprite[:, :, 0] = np.roll(sprite[:, :, 0], -y, 0)
        sprite[:, :, 2] = np.roll(sprite[:, :, 2], y, 0)
        sprite = sprite.astype(np.uint16)
        sprite[:, :, 3] += np.roll(np.roll(sprite[:, :, 3], -x, 1), -y, 0)
        sprite[:, :, 3] += np.roll(np.roll(sprite[:, :, 3], x, 1), y, 0)
        sprite[sprite > 255] = 255
        return sprite.astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["opacity", "alpha", "op"])
    async def opacity(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        amount: float
    ):
        """Sets the opacity of the sprite, from 0 to 1."""
        sprite[:, :, 3] = np.multiply(sprite[:, :, 3], np.clip(amount, 0, 1), casting="unsafe")
        return sprite

    @SpriteVariantFactory.define_variant(names=["negative", "neg"])
    async def negative(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        alpha: bool = False
    ):
        """Inverts the sprite's RGB or RGBA values."""
        sl = slice(None, None if alpha else 3)
        sprite[..., sl] = 255 - sprite[..., sl]
        return sprite

    @SpriteVariantFactory.define_variant(names=["wrap", ])
    async def wrap(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x: int, y: int
    ):
        """Wraps the sprite around its image box."""
        return np.roll(sprite, (y, x), (0, 1))

    @SpriteVariantFactory.define_variant(names=["melt", ])
    async def melt(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        side: Literal["left", "top", "right", "bottom"] = "bottom"
    ):
        """Removes transparent pixels from each row/column and shifts the remaining ones to the end."""
        is_vertical = side in ("top", "bottom")
        at_end = side in ("right", "bottom")
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        # NOTE: I couldn't find a way to do this without at least one Python loop :/
        for i in range(sprite.shape[0]):
            sprite_slice = sprite[i, sprite[i, :, 3] != 0]
            sprite[i] = np.pad(sprite_slice,
                               ((sprite[i].shape[0] - sprite_slice.shape[0], 0)[::2 * at_end - 1], (0, 0)))
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        return sprite

    @SpriteVariantFactory.define_variant(names=["bend", ])
    async def bend(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        axis: Literal["x", "y"], amplitude: int, offset: float, frequency: float
    ):
        """Displaces the sprite by a wave. Frequency is a percentage of the sprite's size along the axis."""
        if axis == "y":
            sprite = np.rot90(sprite)
        offset = ((np.sin(
            np.linspace(offset, np.pi * 2 * (frequency + offset), sprite.shape[0])) / 2) * amplitude).astype(
            int)
        # NOTE: np.roll can't be element wise :/
        sprite[:] = sprite[np.mod(np.arange(sprite.shape[0]) + offset, sprite.shape[1])]
        if axis == "y":
            sprite = np.rot90(sprite, -1)
        return sprite

    @SpriteVariantFactory.define_variant(names=["hueshift", "hs"])
    async def hueshift(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        angle: int
    ):
        """Shifts the hue of the sprite. 0 to 360."""
        hsv = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + int(angle // 2), 180)
        sprite[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sprite

    @SpriteVariantFactory.define_variant(names=["brightness", "gamma", "g"])
    async def brightness(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        brightness: float
    ):
        """Sets the brightness of the sprite."""
        sprite = sprite.astype(float)
        sprite[:, :, :3] *= brightness
        sprite = sprite.clip(-256.0, 255.0) % 256
        return sprite.astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["palette_snap", "ps"])
    async def palette_snap(
        sprite: NumpySprite, ctx: SpriteVariantContext,
    ):
        """Snaps all the colors in the tile to the specified palette."""
        pal = ctx.renderer.bot.db.palette(ctx.tile.palette)
        if pal is None:
            raise errors.NoPaletteError(ctx.tile.palette)
        palette_colors = np.array(pal.convert("RGB")).reshape(-1, 3)
        sprite_lab = cv2.cvtColor(sprite.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        diff_matrix = np.full((palette_colors.shape[0], *sprite.shape[:-1]), 999)
        for i, color in enumerate(palette_colors):
            filled_color_array = np.array([[color]]).repeat(
                sprite.shape[0], 0).repeat(sprite.shape[1], 1)
            filled_color_array = cv2.cvtColor(
                filled_color_array.astype(
                    np.float32) / 255, cv2.COLOR_RGB2Lab)
            sprite_delta_e = np.sqrt(np.sum((sprite_lab - filled_color_array) ** 2, axis=-1))
            diff_matrix[i] = sprite_delta_e
        min_indexes = np.argmin(diff_matrix, 0).reshape(
            diff_matrix.shape[1:])
        result = np.full(sprite.shape, 0, dtype=np.uint8)
        for i, color in enumerate(palette_colors):
            result[:, :, :3][min_indexes == i] = color
        result[:, :, 3] = sprite[:, :, 3]
        return result

    @SpriteVariantFactory.define_variant(names=["wave", ])
    async def wave(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        axis: Literal["x", "y"], amplitude: int, offset: float, frequency: float
    ):
        """Displaces the sprite per-slice by a wave. Frequency is a percentage of the sprite's size along the axis."""
        if axis == "y":
            sprite = np.rot90(sprite)
        offset = ((np.sin(
            np.linspace(offset, np.pi * 2 * (frequency + offset), sprite.shape[0])) / 2) * amplitude).astype(
            int)
        # NOTE: np.roll can't be element wise :/
        for row in range(sprite.shape[0]):
            sprite[row] = np.roll(sprite[row], offset[row], axis=0)
        if axis == "y":
            sprite = np.rot90(sprite, -1)
        return sprite

    @SpriteVariantFactory.define_variant(names=["saturation", "sat", "grayscale", "gscale"])
    async def saturation(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        saturation: float = 0
    ):
        """Saturates or desaturates a sprite."""
        gray_sprite = sprite.copy()
        gray_sprite[..., :3] = (sprite[..., 0] * 0.299 + sprite[..., 1] * 0.587 + sprite[..., 2] * 0.114)[..., np.newaxis]
        return composite(gray_sprite, sprite, saturation).astype(np.uint8)

    @SpriteVariantFactory.define_variant(names=["blank", ])
    async def blank(
        sprite: NumpySprite, ctx: SpriteVariantContext,

    ):
        """Sets a sprite to pure white."""
        sprite[:, :, :3] = 255
        return sprite

    @SpriteVariantFactory.define_variant(names=["liquify", ])
    async def liquify(
        sprite: NumpySprite, ctx: SpriteVariantContext,

    ):
        """"Liquifies" the tile by melting every color except the main color and distributing the main color downwards."""
        return lq.liquify(sprite)

    @SpriteVariantFactory.define_variant(names=["planet", ])
    async def planet(
        sprite: NumpySprite, ctx: SpriteVariantContext,

    ):
        """Turns the tile into a planet by melting every color except the main color and distributing the main color in a circle."""
        return lq.planet(sprite)

    @SpriteVariantFactory.define_variant(names=["normalize_lightness", "nl"])
    async def normalize_lightness(
        sprite: NumpySprite, ctx: SpriteVariantContext,

    ):
        """Normalizes a sprite's HSL lightness, bringing the lightest value up to full brightness."""
        arr_hls = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HLS).astype(
            np.float64)  # since WHEN was it HLS???? huh?????
        max_l = np.max(arr_hls[:, :, 1])
        arr_hls[:, :, 1] *= (255 / max_l)
        sprite[:, :, :3] = cv2.cvtColor(arr_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)  # my question still stands
        return sprite

    @SpriteVariantFactory.define_variant(names=["crop", ])
    async def crop(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x: int, y: int, u: int, v: int, change_bbox: bool = False
    ):
        """Crops the sprite to the specified bounding box.
    If the `change_bbox` toggle is on, then the sprite's bounding box is altered, as opposed to removing pixels."""
        if change_bbox:
            return sprite[y:v, x:u]
        else:
            dummy = np.zeros_like(sprite)
            dummy[y:v, x:u] = sprite[y:v, x:u]
            return dummy

    @SpriteVariantFactory.define_variant(names=["snip", ])
    async def snip(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        x: int, y: int, u: int, v: int
    ):
        """Snips the specified box out of the sprite."""
        sprite[y:v, x:u] = 0
        return sprite

    @SpriteVariantFactory.define_variant(names=["convert", "cvt"])
    async def convert(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        direction: Literal["to", "from"], space: Literal["BGR", "HSV", "HLS", "YUV", "YCrCb", "XYZ", "Lab", "Luv"]
    ):
        """Converts the sprite's color space to or from RGB. Mostly for use with :matrix."""
        space_conversion = {
            "to": {
                "BGR": cv2.COLOR_RGB2BGR,
                "HSV": cv2.COLOR_RGB2HSV,
                "HLS": cv2.COLOR_RGB2HLS,
                "YUV": cv2.COLOR_RGB2YUV,
                "YCrCb": cv2.COLOR_RGB2YCrCb,
                "XYZ": cv2.COLOR_RGB2XYZ,
                "Lab": cv2.COLOR_RGB2Lab,
                "Luv": cv2.COLOR_RGB2Luv,
            },
            "from": {
                "BGR": cv2.COLOR_BGR2RGB,
                "HSV": cv2.COLOR_HSV2RGB,
                "HLS": cv2.COLOR_HLS2RGB,
                "YUV": cv2.COLOR_YUV2RGB,
                "YCrCb": cv2.COLOR_YCrCb2RGB,
                "XYZ": cv2.COLOR_XYZ2RGB,
                "Lab": cv2.COLOR_Lab2RGB,
                "Luv": cv2.COLOR_Luv2RGB,
            }
        }
        sprite[:, :, :3] = cv2.cvtColor(sprite[:, :, :3], space_conversion[direction][space])
        return sprite

    @SpriteVariantFactory.define_variant(names=["threshold", ])
    async def threshold(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        r: float, g: float, b: float, a: float
    ):
        """Removes all pixels below a threshold.
This can be used in conjunction with blur, opacity, and additive blending to create a bloom effect!
If a value is negative, it removes pixels above the threshold instead."""
        im_r, im_g, im_b, im_a = np.split(sprite, 4, axis=2)
        im_a[np.copysign(im_r, r) < r * 255] = 0
        im_a[np.copysign(im_g, g) < g * 255] = 0
        im_a[np.copysign(im_b, b) < b * 255] = 0
        im_a[np.copysign(im_a, a) < a * 255] = 0
        return np.dstack((im_r, im_g, im_b, im_a))

    @SpriteVariantFactory.define_variant(names=["blur", ])
    async def blur(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        radius: int, gaussian: bool = False
    ):
        """Blurs a sprite. Uses box blur by default, though gaussian blur can be used with the boolean toggle."""
        utils.check_size(sprite.shape[0] + radius * 2, sprite.shape[1] + radius * 2)
        arr = np.pad(sprite, ((radius, radius), (radius, radius), (0, 0)))
        assert radius > 0, f"Blur radius of {radius} is too small!"
        if gaussian:
            arr = cv2.GaussianBlur(arr, (radius * 2 + 1, radius * 2 + 1), 0)
        else:
            arr = cv2.boxFilter(arr, -1, (radius * 2 + 1, radius * 2 + 1))
        return arr

    @SpriteVariantFactory.define_variant(names=["fisheye", "fish"])
    async def fisheye(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        strength: float
    ):
        """Applies a fisheye effect."""
        size = np.array(sprite.shape[:2])
        filt = np.indices(sprite.shape[:2], dtype=np.float32) / size[:, np.newaxis, np.newaxis]
        filt = (2 * filt) - 1
        abs_filt = np.linalg.norm(filt, axis=0)
        filt /= (1 - (strength / 2) * (abs_filt[np.newaxis, ...]))
        filt += 1
        filt /= 2
        filt = filt * np.array(sprite.shape)[:2, np.newaxis, np.newaxis]
        filt = np.float32(filt)
        mapped = cv2.remap(sprite, filt[1], filt[0],
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0).astype(float)
        return np.uint8(mapped)

    @SpriteVariantFactory.define_variant(names=["filterimage", "filter", "fi!"])
    async def filterimage(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        name: str,
    ):
        """Applies a filter image to a sprite. For information about filter images, look at the filterimage command."""
        res = await ctx.renderer.bot.db.get_filter(name)
        assert res is not None, f"Filter `{utils.sanitize(name)}` does not exist!"
        absolute, _, _, filter = res
        filt = np.array(filter.convert("RGBA"))
        utils.check_size(*filt.shape[:2])
        filt = np.float32(filt)
        filt[..., :2] -= 0x80
        if not absolute:
            filt[..., :2] += np.indices(filt.shape[:2]).T
        mapped = cv2.remap(sprite, filt[..., 0], filt[..., 1],
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_WRAP).astype(float)
        filt /= 255
        mapped[..., :3] *= filt[..., 2, np.newaxis]
        mapped[..., 3] *= filt[..., 3]
        return np.uint8(mapped)

    @SpriteVariantFactory.define_variant(names=["glitch"], hashable=False)
    async def glitch(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        distance: int, chance: float = 1.0, seed: int = None,
    ):
        """Randomly displaces a sprite's pixels. An RNG seed is created using the tile's attributes if not specified."""
        if seed is None:
            seed = abs(hash(ctx.tile))
        dst = np.indices(sprite.shape[:2], dtype=np.float32)
        rng = np.random.default_rng(seed * 3 + ctx.wobble)
        displacement = rng.uniform(-distance, distance, dst.shape)
        mask = rng.uniform(0, 1, dst.shape)
        displacement[mask > chance] = 0
        dst += displacement
        return cv2.remap(sprite, dst[1], dst[0],
                         interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_WRAP)

    @SpriteVariantFactory.define_variant(names=["convolve", "cv"])
    async def convolve(
        sprite: NumpySprite, ctx: SpriteVariantContext,
        width: int, height: int, cell: list[float]
    ):
        """Convolves the sprite with the given 2D convolution matrix. Information on these can be found at https://en.wikipedia.org/wiki/Kernel_(image_processing)"""
        assert width * height == len(cell), f"Can't fit {len(cell)} values into a matrix that's {width}x{height}!"
        kernel = np.array(cell).reshape((height, width))
        return cv2.filter2D(src=sprite, ddepth=-1, kernel=kernel)

    @PostVariantFactory.define_variant(names=None)
    async def blending(
        post: ProcessedTile, ctx: PostVariantContext,
        mode: Literal[*constants.BLENDING_MODES],
        keep_alpha: bool = True
    ):
        """Sets the blending mode for a tile."""
        post.blending = mode
        post.keep_alpha = keep_alpha and mode != "mask"

    @PostVariantFactory.define_variant(names=["displace", "disp", "d"], sign_alt = sign_displace)
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
            if parsed is not None:
                if string == "":
                    return parsed
                else:
                    print(f"Not accepted due to remaining string: `{string}`")

        return None

    bot.variants = ALL_VARIANTS
    bot.parse_variant = parse_variant
