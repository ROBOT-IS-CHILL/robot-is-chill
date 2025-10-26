from __future__ import annotations

import asyncio
import functools
import glob
import math
import random
import re
import struct
import sys
import time
import traceback
import warnings
import zipfile
import io
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
import PIL.ImageFont as ImageFont

from src.tile import ProcessedTile, Tile
from .. import constants, errors
from ..types import Color, RenderContext, TilingMode
from ..variant_types import SpriteVariantContext, PostVariantContext, SignVariantContext
from src import utils

try:
    FONT = ImageFont.truetype("data/fonts/default.ttf")
except OSError:
    pass

if TYPE_CHECKING:
    from ...ROBOT import Bot


def shift_hue(arr, hueshift):
    arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + int(hueshift // 2), 180)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.dstack((rgb, arr_a))


def lock(t, arr, lock, nonzero: bool = False):
    arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    if nonzero:
        hsv[..., t][hsv[..., t] != 0] = lock
    else:
        hsv[..., t] = lock
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.dstack((rgb, arr_a))


def grayscale(arr, influence):
    arr = arr.astype(np.float64)
    arr[:, :, 0:3] = (((arr[:, :, 0:3].sum(2) / 3).repeat(3).reshape(
        arr.shape[:2] + (3,))) * influence) + (arr[:, :, 0:3] * (1 - influence))
    return arr.astype(np.uint8)


def alpha_paste(img1, img2, coords, func=None):
    if func is None:
        func = Image.alpha_composite
    imgtemp = Image.new('RGBA', img1.size, (0, 0, 0, 0))
    imgtemp.paste(
        img2,
        coords
    )
    return func(img1, imgtemp)


def delta_e(img1, img2):
    # compute the Euclidean distance with pixels of two images
    return np.sqrt(np.sum((img1 - img2) ** 2, axis=-1))


def get_first_frame(tile):
    for tile_frame in tile.frames:
        if tile_frame is not None:
            return np.array(tile_frame.shape[:2])  # Done for convenience on math operations
    else:
        return np.array((0, 0))  # Empty tile


class Renderer:
    """This class exposes various image rendering methods.

    Some of them require metadata from the bot to function properly.
    """

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.palette_cache = {}
        for path in glob.glob("data/palettes/*.png"):
            with Image.open(path) as im:
                self.palette_cache[Path(path).stem] = im.convert("RGBA").copy()
        self.overlay_cache = {}
        self.font_cache = {}
        for path in glob.glob("data/overlays/*.png"):
            with Image.open(path) as im:
                self.overlay_cache[Path(path).stem] = np.array(im.convert("RGBA"))

    def get_cached_font(self, font, size):
        if (font, size) not in self.font_cache:
            if font is None:
                self.font_cache[font, size] = FONT.font_variant(size=size)
            else:
                self.font_cache[font, size] = ImageFont.truetype(f"data/fonts/{font}.ttf", size=size)
        return self.font_cache[font, size]

    async def render(
        self,
        grid: dict[(int, int, int, int), ProcessedTile], shape: (int, int, int, int),
        ctx: RenderContext
    ):
        """Takes a list of tile objects and generates a gif with the associated sprites."""
        start_time = time.perf_counter()
        if ctx.animation is not None:
            animation_wobble, animation_timestep = ctx.animation
        else:
            animation_wobble, animation_timestep = 1, len(
                ctx.frames)  # number of frames per wobble frame, number of frames per timestep
        height, width, stack, stime = shape
        durations = [ctx.speed for _ in range(animation_timestep * stime + len(ctx.before_images))]
        frames = np.repeat(ctx.frames, animation_wobble).tolist()
        frames = (frames * (math.ceil(len(durations) / animation_timestep)))
        if len(ctx.sign_texts):
            for i, sign_text in enumerate(ctx.sign_texts):
                for var in sign_text.variants:
                    if variant.factory.type == "sign":
                        await variant.apply(
                            sign_text, SpriteVariantContext(bot, ctx, self)
                        )
                size = int(
                    ctx.spacing * (ctx.upscale / 2) * sign_text.size * constants.FONT_MULTIPLIERS.get(sign_text.font,
                                                                                                      1))
                assert size <= constants.DEFAULT_SPRITE_SIZE * 2 or ctx.bypass_limits, f"Font size of `{size}` is too large! The maximum is `{constants.DEFAULT_SPRITE_SIZE * 2}`."
                ctx.sign_texts[i].font = self.get_cached_font(sign_text.font, size)
        left_offset = top_offset = right_offset = bottom_offset = 0
        left = top = right = bottom = 0
        actual_width = actual_height = 0
        expected_width, expected_height = width * ctx.spacing, height * ctx.spacing
        if ctx.cropped:
            actual_width, actual_height = expected_width, expected_height
        else:
            for (y, x, _z, _t), tile in grid.items():
                if not isinstance(tile, ProcessedTile):
                    continue
                if all(frame is None for frame in tile.frames):
                    continue
                max_height, max_width = (
                    max(frame.shape[0] for frame in tile.frames if frame is not None),
                    max(frame.shape[1] for frame in tile.frames if frame is not None)
                )
                px, py = tile.displacement if ctx.expand else (0, 0)
                left_offset = max(left_offset, -px)
                right_offset = max(right_offset, px)
                top_offset = max(top_offset, -py)
                bottom_offset = max(bottom_offset, py)
                px += x * ctx.spacing
                py += y * ctx.spacing
                left_boundary = int(math.ceil(px - max(0, (max_width - ctx.spacing) / 2)))
                right_boundary = int(math.ceil(px + ctx.spacing + max(0, (max_width - ctx.spacing) / 2)))
                top_boundary = int(math.ceil(py - max(0, (max_height - ctx.spacing) / 2)))
                bottom_boundary = int(math.ceil(py + ctx.spacing + max(0, (max_height - ctx.spacing) / 2)))
                left = max(left, -left_boundary)
                top = max(top, -top_boundary)
                actual_width = max(actual_width, right_boundary)
                actual_height = max(actual_height, bottom_boundary)
            right = max(0, actual_width - expected_width)
            bottom = max(0, actual_height - expected_height)

        final_size = np.array((int(height * ctx.spacing + top + bottom),
                                 int(width * ctx.spacing + left + right)))
        print(final_size)
        true_size = final_size * ctx.upscale
        if not ctx.bypass_limits:
            assert all(
                true_size[::-1] <= constants.MAX_IMAGE_SIZE) or ctx.bypass_limits, f"Image of size `{true_size[::-1]}` is larger than the maximum allowed size of `{constants.MAX_IMAGE_SIZE}`!"
        steps = np.zeros(
            (((animation_timestep if animation_wobble else len(frames)) * stime), *final_size, 4),
            dtype=np.uint8)

        if ctx.background_images:
            for f, frame in enumerate(frames):
                img = Image.new("RGBA", tuple(final_size[::-1]))
                # for loop in case multiple background images are used
                # (i.e. baba's world map)
                bg_img: Image.Image = ctx.background_images[(frame - 1) % len(ctx.background_images)].convert("RGBA")
                bg_img = bg_img.resize((bg_img.width // ctx.upscale, bg_img.height // ctx.upscale), Image.NEAREST)
                img.paste(bg_img, (0, 0), mask=bg_img)
                for i in range(animation_wobble):
                    q = i + animation_wobble * f
                    steps[q] = np.array(img)
        for (y, x, z, t), tile in grid.items():
            print(y, x, z, t, tile.name)
            if tile is None:
                continue
            await asyncio.sleep(0)
            first_frame = get_first_frame(tile)
            displacement = (
                y * ctx.spacing - int((first_frame[0] - ctx.spacing) / 2) + top + tile.displacement[1],
                x * ctx.spacing - int((first_frame[1] - ctx.spacing) / 2) + left + tile.displacement[0]
            )
            for i, frame in enumerate(frames[animation_timestep * t:animation_timestep * (t + 1)]):
                image_index = i + animation_timestep * t
                wobble = tile.wobble_frames[
                    min(len(tile.wobble_frames) - 1, frame - 1)] if tile.wobble_frames is not None \
                    else (11 * x + 13 * y + frame - 1) % 3 if ctx.random_animations \
                    else frame - 1
                final_wobble = functools.reduce(lambda a, b: a if b is None else b, tile.frames)
                image = tile.frames[wobble] if tile.frames[wobble] is not None else final_wobble
                if image is None:
                    continue
                """
                # Image ranges
                y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
                x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

                # Overlay ranges
                y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
                x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
                """
                dy, dx = displacement
                y1, y2 = max(0, dy), min(final_size[0], dy + image.shape[0])
                x1, x2 = max(0, dx), min(final_size[1], dx + image.shape[1])
                y1o, y2o = max(0, -dy), min(image.shape[0], final_size[0] - dy)
                x1o, x2o = max(0, -dx), min(image.shape[1], final_size[1] - dx)
                if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
                    continue
                cropped_step = steps[int(image_index)][y1:y2, x1:x2]
                cropped_tile = image[y1o:y2o, x1o:x2o]

                assert cropped_step.shape == cropped_tile.shape, \
                    f"Failed to composite tile {tile.name} onto scene: `{cropped_step.shape}` != `{cropped_tile.shape}`\n"\
                    "This is a bug. Please send the command you just ran to a developer."
                cropped_step[:] = self.blend(tile.blending, cropped_step, cropped_tile, tile.keep_alpha)
        background_images = [np.array(image.convert("RGBA")) for image in ctx.before_images]
        l, u, r, d = ctx.crop
        r = r
        d = d
        wobble_range = np.arange(steps.shape[0]) // animation_timestep
        for i, step in enumerate(steps):
            step = step[u:-d if d > 0 else None, l:-r if r > 0 else None]
            if ctx.background is not None:
                if len(ctx.background) < 4:
                    ctx.background = Color.parse(Tile(palette=ctx.palette), self.bot.db, ctx.background)
                ctx.background = np.array(ctx.background).astype(np.float32)
                step_f = step.astype(np.float32) / 255
                step_f[..., :3] = step_f[..., 3, np.newaxis]
                c = ((1 - step_f) * ctx.background + step_f * step.astype(np.float32))
                step = c.astype(np.uint8)
            step = cv2.resize(
                step,
                (int(step.shape[1] * ctx.upscale), int(step.shape[0] * ctx.upscale)),
                interpolation=cv2.INTER_NEAREST
            )
            if len(ctx.sign_texts):
                anchor_disps = {
                    "l": 0.0,
                    "t": 0.0,
                    "m": 0.5,
                    "r": 1.0,
                    "s": 1.0,
                    "d": 1.0
                }
                im = Image.new("RGBA", step.shape[1::-1])
                draw = ImageDraw(im)
                if ctx.image_format == "gif" and ctx.background is None:
                    draw.fontmode = "1"
                for sign_text in ctx.sign_texts:
                    if wobble_range[i] == sign_text.t:
                        text = sign_text.text
                        text = re.sub(r"(?<!\\)\\n", "\n", text)
                        text = re.sub(r"\\(.)", r"\1", text)
                        assert len(
                            text) <= constants.MAX_SIGN_TEXT_LENGTH or ctx.bypass_limits, f"Sign text of length {len(text)} is too long! The maximum is `{constants.MAX_SIGN_TEXT_LENGTH}`."
                        try:
                            pos = (left + sign_text.xo + (
                                        ctx.spacing * ctx.upscale * (sign_text.x + anchor_disps[sign_text.anchor[0]])),
                                   top + sign_text.yo + (ctx.spacing * ctx.upscale * (
                                               sign_text.y + anchor_disps[sign_text.anchor[1]])))
                            draw.multiline_text(pos, text, font=sign_text.font,
                                                align=sign_text.alignment, anchor=sign_text.anchor,
                                                fill=sign_text.color, features=("liga", "dlig", "clig"),
                                                stroke_fill=sign_text.stroke[0], stroke_width=sign_text.stroke[1])
                        except KeyError:
                            raise AssertionError(f"Sign text anchor `{sign_text.anchor}` is unsupported!")
                sign_arr = np.array(im)
                step = self.blend("normal", step, sign_arr, True)
            if ctx.image_format == "gif":
                step_a = step[..., 3]
                step = np.multiply(step[..., :3], np.dstack([step_a] * 3).astype(float) / 255,
                                   casting="unsafe").astype(np.uint8)
                true_rgb = step.astype(float) * (step_a.astype(float) / 255).reshape(*step.shape[:2], 1)
                too_dark_mask = np.logical_and(np.all(true_rgb < 8, axis=2), step_a != 0)
                step[too_dark_mask, :3] = 4
                step = np.dstack((step, step_a))
            background_images.append(step)
        comp_ovh = time.perf_counter() - start_time
        start_time = time.perf_counter()

        self.save_frames(background_images,
                         ctx.out,
                         durations,
                         extra_name=ctx.extra_name,
                         image_format=ctx.image_format,
                         loop=ctx.loop,
                         boomerang=ctx.boomerang,
                         background=ctx.background is not None,
                         ctx=ctx)
        return comp_ovh, time.perf_counter() - start_time, background_images[0].shape[1::-1]

    def blend(self, mode, src, dst, keep_alpha: bool = True) -> np.ndarray:
        keep_alpha &= mode not in ("mask", "cut", "xora")
        if keep_alpha:
            out_a = (src[..., 3] + dst[..., 3] * (1 - src[..., 3] / 255)).astype(np.uint8)
            a, b = src[..., :3].astype(float) / 255, dst[..., :3].astype(
                float) / 255  # This is super convenient actually
        else:
            a, b = src.astype(float) / 255, dst.astype(float) / 255
        if mode == "add":
            c = a + b
        elif mode in ("subtract", "sub"):
            c = a - b
        elif mode == "multiply":
            c = a * b
        elif mode == "divide":
            c = np.clip(a / b, 0.0, 1.0)  # catch divide by 0
        elif mode == "max":
            c = np.maximum(a, b)
        elif mode == "min":
            c = a
            c[dst[..., 3] > 0] = np.minimum(a, b)[dst[..., 3] > 0]
        elif mode == "screen":
            c = 1 - (1 - a) * (1 - b)
        elif mode in ("overlay", "hardlight"):
            if mode == "hardlight":
                a, b = b, a
            c = 1 - (2 * (1 - a) * (1 - b))
            c[np.where(a < 0.5)] = (2 * a * b)[np.where(a < 0.5)]
        elif mode == "softlight":
            c = (1 - a) * a * b + a * (1 - (1 - a) * (1 - b))
        elif mode == "burn":
            c = 1 - ((1 - a) / b)
        elif mode == "dodge":
            c = a / (1 - b)
        elif mode == "normal":
            c = b
        elif mode in ("mask", "cut"):
            c = a
            if mode == "cut":
                b[..., 3] = 1 - b[..., 3]
            c[..., 3] *= b[..., 3]
            c[c[..., 3] == 0] = 0
        elif mode == "xor":
            c = (src[..., :3] ^ dst[..., :3]).astype(float) / 255
        elif mode == "xora":
            c = np.zeros_like(b)
            c[..., :3] = b[..., :3] * b[..., 3, np.newaxis] + a[..., :3] * (1 - b[..., 3, np.newaxis])
            c[..., 3] = np.abs(a[..., 3] - b[..., 3])
        else:
            raise AssertionError(f"Blending mode `{mode}` isn't implemented yet.")
        if keep_alpha:
            dst_alpha = dst[..., 3].astype(float) / 255
            dst_alpha = dst_alpha[:, :, np.newaxis]
            c = ((1 - dst_alpha) * a + dst_alpha * c)
            c[out_a == 0] = 0
            return np.dstack((np.clip(c * 255, 0, 255).astype(np.uint8), out_a[..., np.newaxis]))
        return np.clip(c * 255, 0, 255).astype(np.uint8)

    async def render_full_frame(self,
                                tile: Tile,
                                frame: int,
                                raw_sprite_cache: dict[str, Image.Image],
                                x: int,
                                y: int,
                                ctx: RenderContext
                                ) -> Image.Image:
        sprite = None
        if tile.custom:
            if isinstance(tile.sprite, np.ndarray):
                sprite = tile.sprite[(tile.frame * 3) + frame]
            else:
                sprite = await self.generate_sprite(
                    tile,
                    style=tile.style or "noun",
                    wobble=frame,
                    position=(x, y),
                    ctx=ctx
                )
        else:
            source, sprite_name = tile.sprite
            path = f"data/sprites/{source}/{sprite_name}_{tile.frame}_{frame + 1}.png"
            if source in constants.VANILLA_WORLDS:
                if tile.name == "icon":
                    path = f"data/sprites/{source}/{sprite_name}.png"
                elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                    path = f"data/sprites/{source}/{sprite_name}_1.png"
                elif tile.name == "default":
                    path = f"data/sprites/{source}/default_{frame + 1}.png"
            elif tile.tiling == TilingMode.ICON:
                path = f"data/sprites/{source}/{sprite_name}_1.png"
            try:
                sprite = utils.cached_open(
                    path, cache=raw_sprite_cache, fn=Image.open
                ).convert("RGBA")
            except (FileNotFoundError, AssertionError):
                raise AssertionError(f'The tile `{tile.name}:{tile.frame}` was found, but the files '
                                         f'don\'t exist for it.\nThis is a bug - please notify the author of the tile.\nSearched path: `{path}`')
            sprite = np.array(sprite)
        sprite = cv2.resize(sprite, (int(sprite.shape[1] * ctx.gscale), int(sprite.shape[0] * ctx.gscale)),
                            interpolation=cv2.INTER_NEAREST)
        return await self.apply_options_name(
            tile,
            sprite,
            frame
        )

    async def render_full_tile(self,
                               tile: Tile,
                               *,
                               position: tuple[int, int],
                               ctx: RenderContext) -> tuple[ProcessedTile, list[int], bool]:
        """woohoo."""
        final_tile = ProcessedTile()
        final_tile.name = tile.name
        x, y = position

        rendered_frames = []
        tile_hash = hash(tile)
        cached = tile_hash in ctx.tile_cache.keys()
        if cached:
            final_tile.frames = ctx.tile_cache[tile_hash]
        final_tile.wobble_frames = tile.wobble_frames
        done_frames = [frame is not None for frame in final_tile.frames]
        frame_range = tuple(set(tile.wobble_frames)) if tile.wobble_frames is not None \
            else tuple(set(ctx.frames))
        for frame in frame_range:
            frame -= 1
            wobble = final_tile.wobble_frames[
                min(len(final_tile.wobble_frames) - 1, frame)] if final_tile.wobble_frames is not None \
                else (11 * x + 13 * y + frame) % 3 if ctx.random_animations \
                else frame
            if not done_frames[wobble]:
                final_tile.frames[wobble] = await self.render_full_frame(tile, wobble, ctx.sprite_cache, x, y, ctx)
                rendered_frames.append(wobble)
        if not cached:
            ctx.tile_cache[tile_hash] = final_tile.frames.copy()
        return final_tile, rendered_frames, cached

    async def render_full_tiles(
        self, grid: dict[(int, int, int, int), Tile],
        shape: (int, int, int, int),
        ctx: RenderContext
    ) -> tuple[dict[(int, int, int, int), ProcessedTile], int, int, float]:
        """Final individual tile processing step."""
        rendered_frames = 0
        render_overhead = time.perf_counter()
        d = {}
        for (y, x, z, t), tile in grid.items():
            processed_tile, new_frames, cached = await self.render_full_tile(
                tile,
                position=(x, y),
                ctx=ctx
            )
            rendered_frames += len(new_frames)
            for variant in tile.variants:
                if variant.factory.type == "post":
                    await variant.apply(
                        processed_tile, PostVariantContext(self, new_frames)
                    )
            d[y, x, z, t] = processed_tile
        return d, len(ctx.tile_cache), rendered_frames, time.perf_counter() - render_overhead

    async def get_cached_letter_widths(self, text: str, char: str, mode: str) -> list[int]:
        if not hasattr(self, "letter_width_cache"):
            self.letter_width_cache = {}
        if char in (" ", "~"):
            if mode == "beta": return [1, 2, 3, 4, 5]
            elif mode == "big": return [8, 9, 10, 11, 12]
            else: return [4, 5, 6, 7, 8, 9, 10, 11, 12]
        if (char, mode) not in self.letter_width_cache:
            async with self.bot.db.conn.cursor() as cur:
                res = await cur.execute(
                    "SELECT DISTINCT width FROM letters WHERE char = ? AND mode = ?;",
                    char, mode
                )
                rows = await cur.fetchall()
            self.letter_width_cache[char, mode] = [row[0] for row in rows]
        return self.letter_width_cache[char, mode]

    async def get_cached_letter(self, text: str, char: str, mode: str, width: int) -> np.ndarray:
        if not hasattr(self, "letter_cache"):
            self.letter_cache = {}
        if (char, mode, width) not in self.letter_cache:
            async with self.bot.db.conn.cursor() as cur:
                res = await cur.execute(
                    "SELECT frames FROM letters WHERE char = ? AND mode = ? AND width = ? ORDER BY frames;",
                    char, mode, width
                )
                rows = await cur.fetchall()
            versions = []
            exists = False
            for (frame, ) in rows:
                exists = True
                buf = io.BytesIO(frame)
                buf.seek(0)
                versions.append((*(Image.fromarray(frame, "L") for frame in np.load(buf).astype(np.uint8) * 255),))
            if not exists:
                raise errors.BadCharacter(text, mode, char)
            self.letter_cache[char, mode, width] = versions
        return self.letter_cache[char, mode, width]

    async def generate_sprite(
            self,
            tile: Tile,
            *,
            style: str,
            wobble: int,
            seed: int | None = None,
            position: tuple[int, int],
            ctx: RenderContext
    ) -> np.ndarray:
        """Generates a custom text sprite."""
        if seed is None:
            seed = position[0] * 7 + position[1] * 13
        text = tile.name.removeprefix("text_")
        text = "".join(c.lower() if c.isascii() else c for c in text)
        lines = utils.split_escaped(text, ["/"])
        raw = "/".join(lines)
        target_width = tile.text_squish_width

        if tile.style == "letter":
            mode = "letter"
        else:
            if tile.style == "beta":
                target_width //= 2
                mode = "beta"
                wobble = 0
            else:
                mode = "normal"
            if len(text) >= (tile.text_squish_width // 6) and not tile.oneline and len(lines) == 1:
                text = lines[0]
                lines = [text[:len(text) // 2], text[len(text) // 2:]]

        rng = random.Random()
        rng.seed(seed)
        if mode == "letter":
            maxlen = max(len(line) for line in lines)
            width = maxlen * 12
            height = 24 * len(lines)
            sprite = Image.new("L", (width, height))
            for y, line in enumerate(lines):
                line = line.replace("~", " ")
                y *= 24
                offset = (maxlen - len(line)) * 6
                for x, c in enumerate(line):
                    x *= 12
                    x += offset
                    if c == " ": continue
                    letter_sprite = rng.choice(await self.get_cached_letter(
                        text = tile.name,
                        char = c,
                        mode = "letter",
                        width = 12
                    ))[wobble]
                    sprite.paste(letter_sprite, (x, y))
        else:
            max_line_width = 0
            line_chars = []
            line_widths = []
            line_spacings = []
            for line in lines:
                if not len(line):
                    line_chars.append([])
                    line_widths.append(0)
                    line_spacings.append(0)
                    continue
                char_space = 1 if tile.style == "beta" else 0
                char_widths = []
                # Step 1. Calculate all widths of line
                mode = "big" if len(lines) == 1 else "small"
                if tile.style == "beta":
                    mode = "beta"
                for char in line:
                    widths = [(w, mode) for w in await self.get_cached_letter_widths(text, char, mode)]
                    if mode == "big":
                        widths.extend((w, "small") for w in await self.get_cached_letter_widths(text, char, "small"))
                    assert len(widths) > 0, (
                        sane_text := text.replace('`', '\''),
                        sane_char := char.replace('`', '\''),
                        f"The text `{sane_text}` could not be generated, since no sprites for `{sane_char}` exist."
                    )[2]
                    widths = sorted(widths, key = lambda width_mode: width_mode[0])
                    char_widths.append((char, widths))
                # Step 2. See where minimum is in relation to tile size, space out characters to match
                char_width = sum(c[1][0][0] for c in char_widths) + char_space * (len(char_widths) - 1)
                i = 0
                found_any = False
                seq = [i for i in range(len(char_widths))]
                rng.shuffle(seq)
                while char_width < target_width:
                    index = seq[i]
                    if len(char_widths[index][1]) > 1:
                        found_any = True
                        smaller = char_widths[index][1].pop(0)[0]
                        char_width = char_width - smaller + char_widths[index][1][0][0]
                    i += 1
                    if i >= len(char_widths):
                        if not found_any: break
                        i = 0
                        rng.shuffle(seq)
                        found_any = False
                if char_width < target_width:
                    added_padding = min((target_width - char_width) / max(1, len(line) - 1), 2)
                    if mode == "beta":
                        added_padding = (added_padding // 2) * 2
                    char_space += added_padding
                char_width = sum(c[1][0][0] for c in char_widths) + char_space * (len(char_widths) - 1)
                line_chars.append(char_widths)
                line_widths.append(char_width)
                line_spacings.append(char_space)
            max_line_width = max(line_widths)
            print("Max line width: ", max_line_width)
            if max_line_width == 0:
                sprite = Image.new("L", (24, 24))
            else:
                # Step 3: Generate each line
                line_height = 6 if tile.style == "beta" else 12
                sprite = Image.new("L", (int(max_line_width), int(max(line_height * 2, len(lines) * line_height))))
                dy = (max(line_height * 2, len(lines) * line_height) - (len(lines) * line_height)) / 2
                char_index = 0
                for y, (line, line_width, char_spacing) in enumerate(zip(line_chars, line_widths, line_spacings)):
                    x = int((max_line_width - line_width) / 2)
                    for char, widths in line:
                        width, mode = widths[0]
                        if char in (" ", "~"):
                            x += width + char_spacing
                            continue
                        letter_sprite = rng.choice(await self.get_cached_letter(
                            text = tile.name,
                            char = char,
                            mode = mode,
                            width = int(width)
                        ))[(wobble + char_index) % 3]
                        oy = (line_height - letter_sprite.height) / 2 + dy
                        sprite.paste(letter_sprite, (int(x), int(y * line_height + oy)))
                        x += width + char_spacing
                        char_index += 1
                if sprite.height < line_height * 2:
                    spr = Image.new("L", (sprite.width, line_height))
                    spr.paste(sprite, (0, int((line_height - sprite.height) / 2)))
                    sprite = spr
        sprite = Image.merge("RGBA", (sprite, sprite, sprite, sprite))
        if tile.style == "beta":
            sprite = sprite.resize((sprite.width * 2, sprite.height * 2), Image.NEAREST)
        sprite = sprite.resize(
            (int(sprite.width * ctx.gscale), int(sprite.height * ctx.gscale)), Image.NEAREST)
        return np.array(sprite)

    async def apply_options_name(
            self,
            tile: Tile,
            sprite: np.ndarray,
            wobble: int
    ) -> Image.Image:
        """Takes an image, taking tile data from its name, and applies the
        given options to it."""
        try:
            return await self.apply_options(
                tile,
                sprite,
                wobble
            )
        except ValueError as e:
            size = e.args[0]
            raise errors.BadTileProperty(tile.name, size)

    async def apply_options(
        self,
        tile: Tile,
        sprite: np.ndarray,
        wobble: int,
        seed: int | None = None
    ):
        random.seed(seed)
        # HACK: At least it's better than the other solution.
        for variant in tile.variants:
            if variant.factory.identifier == "color":
                tile.custom_color = True

        for variant in tile.variants:
            if variant.factory.type == "sprite":
                sprite = await variant.apply(
                    sprite, SpriteVariantContext(tile, wobble, self)
                )
                if not all(np.array(sprite.shape[:2]) <= constants.MAX_TILE_SIZE):
                    raise errors.TooLargeTile(sprite.shape[1::-1], tile.name)

        if not tile.custom_color:
            color = Color.from_index(tile.color, tile.palette, self.bot.db)
            sprite = utils.recolor(sprite, color)

        return sprite

    def save_frames(
            self,
            images: list[np.ndarray],
            out: str | BinaryIO,
            durations: list[int],
            extra_name: str = 'render',
            image_format: str = 'gif',
            loop: bool = True,
            boomerang: bool = False,
            background: bool = False,
            ctx: RenderContext | None = None
    ) -> None:
        """Saves the images as a gif to the given file or buffer.

        If a buffer, this also conveniently seeks to the start of the
        buffer. If extra_out is provided, the frames are also saved as a
        zip file there.
        """
        if boomerang and len(images) > 2:
            images += images[-2:0:-1]
            durations += durations[-2:0:-1]
        if image_format == 'gif':
            if background:
                save_images = [Image.fromarray(im) for im in images]
            else:
                save_images = []
                if ctx is not None and ctx.limited_palette:
                    total_colors, total_counts = [], []
                    for frames in ctx.tile_cache.values():
                        for frame in frames:
                            colors, counts = np.unique(frame.reshape(-1, 4), axis=0, return_counts=True)
                            total_colors.extend(colors)
                            total_counts.extend(counts)
                    for bg_frame in ctx.background_images.values():
                        colors, counts = np.unique(bg_frames.reshape(-1, 4), axis=0, return_counts=True)
                        total_colors.extend(colors)
                        total_counts.extend(counts)
                    total_colors, total_counts = np.array(total_colors), np.array(total_counts)
                    colors, inverse_indices = np.unique(total_colors, axis=0, return_inverse=True)
                    final_counts = np.bincount(inverse_indices, weights=total_counts)
                    sort_indices = np.argsort(final_counts)
                    colors = colors[sort_indices[::-1]]
                    palette_colors = [0, 0, 0]
                    formatted_colors = colors[colors[:, 3] != 0][..., :3]
                    formatted_colors = formatted_colors[:255].flatten()
                    palette_colors.extend(formatted_colors)
                    dummy = Image.new('P', (16, 16))
                    dummy.putpalette(palette_colors)
                    for i, im in enumerate(images):
                        save_images.append(Image.fromarray(im).convert('RGB').quantize(
                            palette=dummy, dither=0))
                else:
                    for i, im in enumerate(images):
                        colors, counts = np.unique(im.reshape(-1, 4), axis=0, return_counts=True)
                        sort_indices = np.argsort(counts)
                        colors = colors[sort_indices[::-1]] # Sort in descending order
                        palette_colors = [0, 0, 0]
                        formatted_colors = colors[colors[:, 3] != 0][..., :3]
                        formatted_colors = formatted_colors[:255].flatten()
                        palette_colors.extend(formatted_colors)
                        dummy = Image.new('P', (16, 16))
                        dummy.putpalette(palette_colors)
                        save_images.append(Image.fromarray(im).convert('RGB').quantize(
                            palette=dummy, dither=0))
            kwargs = {
                'format': "GIF",
                'interlace': True,
                'save_all': True,
                'append_images': save_images[1:],
                'loop': 0,
                'duration': durations,
                'disposal': 2,  # Frames don't overlap
                'background': 0,
                'transparency': 0,
                'optimize': False
            }
            if not loop:
                del kwargs['loop']
            if background:
                del kwargs['transparency']
                del kwargs['background']
                del kwargs['disposal']
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'png':
            save_images = [Image.fromarray(im) for im in images]
            kwargs = {
                'format': "PNG",
                'save_all': True,
                'append_images': save_images[1:],
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'tiff':
            save_images = [Image.fromarray(im) for im in images]
            kwargs = {
                'format': "TIFF",
                'save_all': True,
                'append_images': save_images[1:],
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'webp':
            save_images = [Image.fromarray(im) for im in images]
            kwargs = {
                'format': "WEBP",
                'save_all': True,
                'append_images': save_images[1:],
                'default_image': True,
                'loop': 0,
                'duration': durations,
                "lossless": True
            }
            if not loop:
                kwargs['loop'] = 1
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'pdf':
            save_images = [
                Image.fromarray(im).convert("RGB") for im in images
            ]
            kwargs = {
                'format': "PDF",
                'save_all': True,
                'append_images': save_images[1:],
            }
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'zip':
            file = zipfile.PyZipFile(out, "x")
            for i, img in enumerate(images):
                buffer = io.BytesIO()
                Image.fromarray(img).save(buffer, "PNG")
                if ctx.custom_filename:
                    filename = f"{ctx.custom_filename}_{i // 3}_{(i % 3) + 1}.png"
                else:
                    filename = f"{i + 1}.png"
                file.writestr(
                    filename,
                    buffer.getvalue())
            file.close()
        else:
            raise AssertionError(f"Filetype {image_format} not supported!")
        if type(out) is not str:
            out.seek(0)


async def setup(bot: Bot):
    bot.renderer = Renderer(bot)
