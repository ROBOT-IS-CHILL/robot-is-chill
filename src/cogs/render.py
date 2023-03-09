from __future__ import annotations

import glob
import math
import random
import re
import sys
import time
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
import PIL.ImageFont as ImageFont

from src.tile import ProcessedTile, Tile
from .. import constants, errors
from ..types import Color
from ..utils import cached_open

FONT = ImageFont.truetype("data/fonts/default.ttf")

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
                self.palette_cache[Path(path).stem] = im.copy()
        self.overlay_cache = {}
        for path in glob.glob("data/overlays/*.png"):
            with Image.open(path) as im:
                self.overlay_cache[Path(path).stem] = np.array(im.convert("RGBA"))

    async def render(
            self,
            grid: list[list[list[list[ProcessedTile]]]],
            *,
            before_images=None,
            palette: str = "default",
            images: list[str] | list[Image] | None = None,
            image_source: str = constants.BABA_WORLD,
            out: str | BinaryIO = "target/renders/render.gif",
            background: tuple[int] | None = None,
            upscale: int = 2,
            extra_out: str | BinaryIO | None = None,
            extra_name: str | None = None,
            frames: list[int] = (1, 2, 3),
            animation: tuple[int, int] = None,
            speed: int = 200,
            crop: tuple[int, int, int, int] = (0, 0, 0, 0),
            pad: tuple[int, int, int, int] = (0, 0, 0, 0),
            image_format: str = 'gif',
            loop: bool = True,
            spacing: int = constants.DEFAULT_SPRITE_SIZE,
            boomerang: bool = False,
            random_animations: bool = True,
            expand: bool = False,
            _disable_limit: bool = False,
            sign_texts: list,
            **_
    ):
        """Takes a list of tile objects and generates a gif with the associated sprites."""
        if before_images is None:
            before_images = []
        start_time = time.perf_counter()
        if animation is not None:
            animation_wobble, animation_timestep = animation
        else:
            animation_wobble, animation_timestep = 1, len(
                frames)  # number of frames per wobble frame, number of frames per timestep
        grid = np.array(grid, dtype=object)
        durations = [speed for _ in range(animation_timestep * grid.shape[0] + len(before_images))]
        frames = np.repeat(frames, animation_wobble).tolist()
        frames = (frames * (math.ceil(len(durations) / animation_timestep)))
        sizes = np.array(
            [get_first_frame(tile)[:2] if not (tile is None or tile.empty) else (0, 0) for tile in grid.flatten()])
        sizes = sizes.reshape((*grid.shape, 2))
        assert sizes.size > 0, "The render must have at least one tile in it."
        left_influence = np.arange(0, -spacing * sizes.shape[3], -spacing) * 2
        top_influence = np.arange(0, -spacing * sizes.shape[2], -spacing) * 2
        right_influence = np.arange(-spacing * (sizes.shape[3] - 1), spacing, spacing) * 2
        bottom_influence = np.arange(-spacing * (sizes.shape[2] - 1), spacing, spacing) * 2
        left_sizes = sizes[..., 1] + left_influence.reshape((1, 1, 1, -1))
        top_sizes = sizes[..., 0] + top_influence.reshape((1, 1, -1, 1))
        right_sizes = sizes[..., 1] + right_influence.reshape((1, 1, 1, -1))
        bottom_sizes = sizes[..., 0] + bottom_influence.reshape((1, 1, -1, 1))
        left = max((np.max(left_sizes - spacing) // 2) + pad[0], 0)
        top = max((np.max(top_sizes - spacing) // 2) + pad[1], 0)
        right = max((np.max(right_sizes - spacing) // 2) + pad[2], 0)
        bottom = max((np.max(bottom_sizes - spacing) // 2) + pad[3], 0)
        if expand:
            displacements = np.array(
                [tile.displacement if not (tile is None or tile.empty) else (0, 0)
                 for tile in grid.flatten()])
            displacements = (displacements.reshape((*grid.shape, 2)))
            left_disp = displacements[:, :, :, 0, 0]
            top_disp = displacements[:, :, 0, :, 1]
            right_disp = displacements[:, :, :, -1, 0]
            bottom_disp = displacements[:, :, -1, :, 1]
            left += max(max(left_disp.max((0, 1, 2)), left_disp.min((0, 1, 2)), key=abs), 0)
            top += max(max(top_disp.max((0, 1, 2)), top_disp.min((0, 1, 2)), key=abs), 0)
            right -= min(max(right_disp.max((0, 1, 2)), right_disp.min((0, 1, 2)), key=abs), 0)
            bottom -= min(max(bottom_disp.max((0, 1, 2)), bottom_disp.min((0, 1, 2)), key=abs), 0)
        default_size = np.array((int(sizes.shape[2] * spacing + top + bottom),
                                 int(sizes.shape[3] * spacing + left + right)))
        true_size = default_size * upscale
        if not _disable_limit:
            assert all(
                true_size <= constants.MAX_IMAGE_SIZE), f"Image of size {true_size} is larger than the maximum allowed size of {constants.MAX_IMAGE_SIZE}!"
        steps = np.zeros(
            (((animation_timestep if animation_wobble else len(frames)) * grid.shape[0]), *default_size, 4),
            dtype=np.uint8)

        if images and ((bg_images := isinstance(images[0], Image.Image)) or image_source is not None):
            for step in range(steps.shape[0] // len(frames)):
                for frame in frames:
                    img = Image.new("RGBA", tuple(default_size[::-1]))
                    # for loop in case multiple background images are used
                    # (i.e. baba's world map)
                    if bg_images:
                        bg_img: Image.Image = images[frame - 1].convert("RGBA")
                        bg_img = bg_img.resize((bg_img.width // upscale, bg_img.height // upscale), Image.NEAREST)
                        img.paste(bg_img, (0, 0), mask=bg_img)
                    else:
                        # Legacy behavior, too lazy to remove
                        for image in images:
                            try:
                                overlap = Image.open(
                                    f"data/images/{image_source}/{image}_{frame}.png").convert("RGBA")
                            except FileNotFoundError:
                                overlap = Image.open(
                                    f"data/images/{image_source}/{image}_1.png").convert("RGBA")
                        img.paste(overlap, (0, 0), mask=overlap)
                    for i in range(animation_wobble):
                        steps[i + animation_timestep * step] = np.array(img)
        for t, step in enumerate(grid):
            for z, layer in enumerate(step):
                for y, row in enumerate(layer):
                    for x, tile in enumerate(row):
                        first_frame = get_first_frame(tile)
                        if tile.empty:
                            continue
                        displacement = (y * spacing - int((first_frame[0] - spacing) / 2) + top - tile.displacement[1],
                                        x * spacing - int((first_frame[1] - spacing) / 2) + left - tile.displacement[0])
                        for i, frame in enumerate(frames[animation_timestep * t:animation_timestep * (t + 1)]):
                            image_index = i + animation_timestep * t
                            wobble = tile.wobble if tile.wobble is not None else (
                                                                                         11 * x + 13 * y + frame - 1) % 3 if random_animations else frame - 1
                            image = tile.frames[wobble]
                            dslice = (default_size - (first_frame + displacement))
                            dst_slice = (
                                slice(max(-displacement[0], 0), dslice[0] if dslice[0] < 0 else None),
                                slice(max(-displacement[1], 0), dslice[1] if dslice[1] < 0 else None)
                            )
                            image = image[*dst_slice]
                            if image.size < 1:
                                continue
                            # Get the part of the image to paste on
                            src_slice = (
                                slice(max(displacement[0], 0), image.shape[0] + max(displacement[0], 0)),
                                slice(max(displacement[1], 0), image.shape[1] + max(displacement[1], 0))
                            )
                            index = image_index, *src_slice
                            try:
                                steps[index] = self.blend(tile.blending, steps[index], image, tile.keep_alpha)
                            except IndexError:
                                pass  # warnings.warn(f"Couldn't place {tile} at {x}, {y}, {index}")
        images = [np.array(image.convert("RGBA")) for image in before_images]
        l, u, r, d = crop
        r = r
        d = d
        wobble_range = np.arange(steps.shape[0]) // animation_timestep
        for i, step in enumerate(steps):
            step = step[u:-d if d > 0 else None, l:-r if r > 0 else None]
            if background is not None:
                if len(background) < 4:
                    background = Color.parse(Tile(palette=palette), self.palette_cache, background)
                background = np.array(background).astype(np.float32)
                step_f = step.astype(np.float32) / 255
                step_f[..., :3] = step_f[..., 3, np.newaxis]
                c = ((1 - step_f) * background + step_f * step.astype(np.float32))
                step = c.astype(np.uint8)
            step = cv2.resize(
                step,
                (int(step.shape[1] * upscale), int(step.shape[0] * upscale)),
                interpolation=cv2.INTER_NEAREST
            )
            assert len(sign_texts) <= constants.MAX_SIGN_TEXTS, \
                f"Too many sign texts! The limit is `{constants.MAX_SIGN_TEXTS}`, while you have `{len(sign_texts)}`."
            if len(sign_texts):
                im = Image.new("RGBA", step.shape[1::-1])
                draw = ImageDraw(im)
                if image_format == "gif" and background is None:
                    draw.fontmode = "1"
                for sign_text in sign_texts:
                    if wobble_range[i] in range(sign_text.time_start, sign_text.time_end):
                        size = int(spacing * (upscale // 2) * sign_text.size)
                        assert size <= constants.DEFAULT_SPRITE_SIZE * 2, f"Font size of `{size}` is too large! The maximum is `{constants.DEFAULT_SPRITE_SIZE * 2}`."
                        if sign_text.font is not None:
                            font = ImageFont.truetype(f"data/fonts/{sign_text.font}.ttf", size=size)
                        else:
                            font = FONT.font_variant(size=size)
                        text = sign_text.text
                        text = re.sub(r"(?<!\\)\/n", "\n", text)
                        text = re.sub(r"\\(.)", r"\1", text)
                        assert len(text) < constants.MAX_SIGN_TEXT_LENGTH, f"Sign text of length {len(text)} is too long! The maximum is `{constants.MAX_SIGN_TEXT_LENGTH}`."
                        pos = (left + sign_text.xo + (spacing * upscale * (sign_text.x + 0.5)),
                                top + sign_text.yo + (spacing * upscale * (sign_text.y + 1.0)))
                        draw.multiline_text(pos, text, font=font,
                                            align=sign_text.alignment if sign_text.alignment is not None else "center",
                                            anchor="md",
                                            fill=sign_text.color, features=("liga", "dlig", "clig"),
                                            stroke_fill=sign_text.stroke[0], stroke_width=sign_text.stroke[1])
                sign_arr = np.array(im)
                if image_format == "gif" and background is None:
                    sign_arr[..., 3][sign_arr[..., 3] > 0] = 255
                step = self.blend("normal", step, sign_arr, True)
            if image_format == "gif":
                step_a = step[..., 3]
                step = np.multiply(step[..., :3], np.dstack([step_a] * 3).astype(float) / 255,
                                   casting="unsafe").astype(np.uint8)
                true_rgb = step.astype(float) * (step_a.astype(float) / 255).reshape(*step.shape[:2], 1)
                too_dark_mask = np.logical_and(np.all(true_rgb < 8, axis=2), step_a != 0)
                step[too_dark_mask, :3] = 4
                step = np.dstack((step, step_a))
            images.append(step)
        comp_ovh = time.perf_counter() - start_time
        start_time = time.perf_counter()
        self.save_frames(images,
                         out,
                         durations,
                         extra_out=extra_out,
                         extra_name=extra_name,
                         image_format=image_format,
                         loop=loop,
                         boomerang=boomerang,
                         background=background is not None)
        return comp_ovh, time.perf_counter() - start_time, images[0].shape[1::-1]

    def blend(self, mode, src, dst, keep_alpha: bool = True) -> np.ndarray:
        keep_alpha &= mode not in ("mask", "cut")
        if keep_alpha:
            out_a = (src[..., 3] + dst[..., 3] * (1 - src[..., 3] / 255)).astype(np.uint8)
            a, b = src[..., :3].astype(float) / 255, dst[..., :3].astype(
                float) / 255  # This is super convenient actually
        else:
            a, b = src.astype(float) / 255, dst.astype(float) / 255
        if mode == "add":
            c = a + b
        elif mode == "subtract":
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
                                raw_sprite_cache: dict[str, Image],
                                gscale: float,
                                x: int,
                                y: int,
                                ) -> Image.Image:
        if tile.custom:
            if type(tile.sprite) == tuple:
                sprite = await self.generate_sprite(
                    tile,
                    style=tile.style or (
                        "noun" if len(tile.name) < 1 else "letter"),
                    wobble=frame,
                    position=(x, y),
                    gscale=gscale
                )
        else:
            if isinstance(tile.sprite, np.ndarray):
                sprite = tile.sprite[(tile.frame * 3) + frame]
            else:
                path_fallback = None
                if tile.name == "icon":
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}.png"
                elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}_1.png"
                elif tile.name == "default":
                    path = f"data/sprites/{constants.BABA_WORLD}/default_{frame + 1}.png"
                else:
                    source, sprite_name = tile.sprite
                    if tile.frame == -1:
                        path = f"data/sprites/vanilla/error_0_{frame + 1}.png"
                    else:
                        path = f"data/sprites/{source}/{sprite_name}_{tile.frame}_{frame + 1}.png"
                    try:
                        path_fallback = f"data/sprites/{source}/{sprite_name}_{tile.fallback_frame}_{frame + 1}.png"
                    except BaseException:
                        path_fallback = None
                try:
                    sprite = cached_open(
                        path, cache=raw_sprite_cache, fn=Image.open).convert("RGBA")
                except FileNotFoundError:
                    try:
                        assert path_fallback is not None
                        sprite = cached_open(
                            path_fallback,
                            cache=raw_sprite_cache,
                            fn=Image.open).convert("RGBA")
                    except (FileNotFoundError, AssertionError):
                        raise AssertionError(f'The tile `{tile.name}:{tile.frame}` was found, but the files '
                                             f'don\'t exist for it.')
                sprite = np.array(sprite)
            sprite = cv2.resize(sprite, (int(sprite.shape[1] * gscale), int(sprite.shape[0] * gscale)),
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
                               raw_sprite_cache: dict[str, Image.Image],
                               gscale: float = 1,
                               tile_cache=None,
                               frames: tuple[int] = (1, 2, 3),
                               random_animations: bool = True
                               ) -> tuple[ProcessedTile, list[int], bool]:
        """woohoo."""
        if tile_cache is None:
            tile_cache = {}
        final_tile = ProcessedTile()
        if tile.empty:
            return final_tile, [], True
        final_tile.empty = False
        final_tile.name = tile.name
        x, y = position

        rendered_frames = []
        tile_hash = hash((tile, gscale))
        cached = tile_hash in tile_cache.keys()
        if cached:
            final_tile.frames = tile_cache[tile_hash]
        final_tile.wobble = tile.wobble
        done_frames = [frame is not None for frame in final_tile.frames]
        frame_range = tuple(set(frames)) if tile.wobble is None else (tile.wobble,)
        for frame in frame_range:
            frame -= 1
            wobble = tile.wobble if tile.wobble is not None else (
                                                                         11 * x + 13 * y + frame) % 3 if random_animations else frame
            if not done_frames[wobble]:
                final_tile.frames[wobble] = await self.render_full_frame(tile, wobble, raw_sprite_cache, gscale, x, y)
                rendered_frames.append(wobble)
        if not cached:
            tile_cache[tile_hash] = final_tile.frames.copy()
        return final_tile, rendered_frames, cached

    async def render_full_tiles(self, grid: list[list[list[list[Tile]]]], *, random_animations: bool = False,
                                gscale: float = 2, frames: tuple[int] = (1, 2, 3)) -> tuple[
        list[list[list[list[ProcessedTile]]]], int, int, float]:
        """Final individual tile processing step."""
        sprite_cache = {}
        tile_cache = {}
        rendered_frames = 0
        d = []
        render_overhead = time.perf_counter()
        for timestep in grid:
            a = []
            for layer in timestep:
                b = []
                for y, row in enumerate(layer):
                    c = []
                    for x, tile in enumerate(row):
                        processed_tile, new_frames, cached = await self.render_full_tile(
                            tile,
                            position=(x, y),
                            random_animations=random_animations,
                            gscale=gscale,
                            raw_sprite_cache=sprite_cache,
                            tile_cache=tile_cache,
                            frames=frames
                        )
                        rendered_frames += len(new_frames)
                        for variant in tile.variants["post"]:
                            await variant.apply(processed_tile, renderer=self, new_frames=new_frames)
                        c.append(
                            processed_tile
                        )
                    b.append(c)
                a.append(b)
            d.append(a)
        return d, len(tile_cache), rendered_frames, time.perf_counter() - render_overhead

    async def generate_sprite(
            self,
            tile: Tile,
            *,
            style: str,
            wobble: int,
            seed: int | None = None,
            gscale: float,
            position: tuple[int, int]
    ) -> np.ndarray:
        """Generates a custom text sprite."""
        text = tile.name[5:].lower().replace(" ","~")
        raw = text.replace("/", "")
        newline_count = text.count("/")
        assert len(text) <= 64, 'Text has a maximum length of `64` characters.'
        if seed is None:
            seed = int((7 + position[0]) / (3 + position[1]) * 100000000)
        seed_digits = [(seed >> 8 * i) | 0b11111111 for i in range(len(raw))]
        # Get mode and split status
        if newline_count >= 1:
            fixed = True
            mode = "small"
            indices = []
            offset = 0
            for match in re.finditer("/", text):
                indices.append(match.start() + offset)
                offset -= 1
        else:
            fixed = False
            mode = "big"
            indices = []
            if len(raw) >= 4:
                mode = "small"
                if tile.style != "oneline":
                    indices = [len(raw) - math.ceil(len(raw) / 2)]
        print(mode)
        indices.insert(0, 0)
        indices.append(len(raw))  # can't use -1 here because of a range() later on

        if style == "letter":
            if mode == "big":
                mode = "letter"

        width_cache: dict[str, list[int]] = {}
        for c in raw:
            rows = await self.bot.db.conn.fetchall(
                '''
                SELECT char, width FROM letters
                WHERE char == ? AND mode == ?;
                ''',
                c, mode
            )

            for row in rows:
                char, width = row
                width_cache.setdefault(char, []).append(width)

        def width_greater_than(c: str, w: int = 0) -> int:
            try:
                return min(width for width in width_cache[c] if width > w)
            except ValueError:
                raise KeyError

        # fetch the minimum possible widths first
        try:
            widths: list[int] = [width_greater_than(c) for c in raw]
        except KeyError as e:
            raise errors.BadCharacter(text, mode, e.args[0])

        max_width = constants.DEFAULT_SPRITE_SIZE
        old_index = 0
        for index in indices:
            max_width = max(max_width, sum(widths[old_index:index]))
            old_index = index

        def check_or_adjust(widths: list[int], indices: list[int]) -> list[int]:
            """Is the arrangement valid?"""
            if mode == "small":
                if not fixed:
                    for i in range(1, len(indices) - 1):
                        width_distance = (
                                sum(widths[indices[i]:indices[i + 1]]) - sum(widths[indices[i - 1]:indices[i]]))
                        old_wd = width_distance
                        debug_flag = 0
                        while old_wd > width_distance:
                            debug_flag += 1
                            indices[i] -= 1
                            new_width_distance = (
                                    sum(widths[indices[i]:indices[i + 1]]) - sum(widths[indices[i - 1]:indices[i]]))
                            if new_width_distance > width_distance:
                                indices[i] += 2
                            assert debug_flag > 200, "Ran into an infinite loop while trying to create a text! " \
                                                     "This shouldn't happen."
            return indices

        # Check if the arrangement is valid with minimum sizes
        # If allowed, shift the index to make the arrangement valid
        indices = check_or_adjust(widths, indices)

        # Expand widths where possible
        stable = [False for _ in range(len(widths))]
        while not all(stable):
            old_width, i = min((w, i)
                               for i, w in enumerate(widths) if not stable[i])
            try:
                new_width = width_greater_than(raw[i], old_width)
                a = max([0] if not len(max_list := [j for j in indices if i >= j]) else max_list)
                b = min([len(indices)] if not len(min_list := [j for j in indices if i < j]) else min_list)
                temp_widths = widths[:i] + [old_width + (new_width - old_width) * (b - a)] + widths[i:]  # BAD
                assert sum(temp_widths[a:b]) <= constants.DEFAULT_SPRITE_SIZE
            except (KeyError, AssertionError):
                stable[i] = True
                continue
            widths[i] = new_width
            try:
                indices = check_or_adjust(widths, indices)
            except errors.CustomTextTooLong:
                widths[i] = old_width
                stable[i] = True

        # Arrangement is now the widest it can be
        # Kerning: try for 1 pixel between sprites, and rest to the edges
        gaps: list[int] = []
        bounds: list[tuple(int, int)] = list(zip(indices[:-1], indices[1:]))
        if mode == "small":
            rows = [widths[a:b] for a, b in bounds]
        else:
            rows = [widths[:]]
        for row in rows:
            space = max_width - sum(row)
            # Extra -1 is here to not give kerning space outside the
            # left/rightmost char
            chars = len(row) - 1
            if space >= chars:
                # left edge
                gaps.append((space - chars) // 2)
                # char gap
                gaps.extend([1] * chars)
                # right edge gap is implied
            else:
                # left edge
                gaps.append(0)
                # as many char gaps as possible, starting from the left
                gaps.extend([1] * space)
                gaps.extend([0] * (chars - space))

        letters: list[Image.Image] = []
        for c, seed_digit, width in zip(raw, seed_digits, widths):
            l_rows = await self.bot.db.conn.fetchall(
                # fstring use safe
                f'''
                SELECT sprite_{int(wobble)} FROM letters
                WHERE char == ? AND mode == ? AND width == ?
                ''',
                c, mode, width
            )
            options = list(l_rows)
            letter_sprite, *_ = *options[seed_digit % len(options)],
            buf = BytesIO(letter_sprite)
            letters.append(Image.open(buf))

        sprite = Image.new("L",
                           (max(max(sum(row) for row in rows),
                                constants.DEFAULT_SPRITE_SIZE),
                            (max(len(rows), 2) * constants.DEFAULT_SPRITE_SIZE) // 2))
        if mode == "small":
            for j, (a, b) in enumerate(bounds):
                x = gaps[a]
                y_center = ((constants.DEFAULT_SPRITE_SIZE // 2) * j) + (constants.DEFAULT_SPRITE_SIZE // 4) \
                    if style != "oneline" else 12
                for i in range(a, b):
                    letter = letters[i]
                    y_top = y_center - letter.height // 2
                    sprite.paste(letter, (x, y_top), mask=letter)
                    x += widths[i]
                    if i != b - 1:
                        x += gaps[i + 1]
        else:
            x = gaps[0]
            y_center = 12
            for i in range(len(raw)):
                letter = letters[i]
                y_top = y_center - letter.height // 2
                sprite.paste(letter, (x, y_top), mask=letter)
                x += widths[i]
                if i != len(raw) - 1:
                    x += gaps[i + 1]

        sprite = Image.merge("RGBA", (sprite, sprite, sprite, sprite))
        sprite = sprite.resize(
            (int(sprite.width * gscale), int(sprite.height * gscale)), Image.NEAREST)
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
        for variant in tile.variants["sprite"]:
            sprite = await variant.apply(sprite, tile=tile, wobble=wobble, renderer=self)  # NOUN/PROP ARE ANNOYING
            if not all(np.array(sprite.shape[:2]) <= constants.MAX_TILE_SIZE):
                raise errors.TooLargeTile(sprite.shape[1::-1])
        return sprite

    def save_frames(
            self,
            images: list[np.ndarray],
            out: str | BinaryIO,
            durations: list[int],
            extra_out: str | BinaryIO | None = None,
            extra_name: str = 'render',
            image_format: str = 'gif',
            loop: bool = True,
            boomerang: bool = False,
            background: bool = False
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
                gif_images = [Image.fromarray(im) for im in images]
            else:
                gif_images = []
                for i, im in enumerate(images):
                    # TODO: THIS IS EXTREMELY SLOW. BETTER WAY IS NEEDED.
                    colors = np.unique(im.reshape(-1, 4), axis=0)
                    palette_colors = [0, 0, 0]
                    palette_colors.extend(colors[colors[:, 3] != 0][:254, :3].flatten())
                    dummy = Image.new('P', (16, 16))
                    dummy.putpalette(palette_colors)
                    gif_images.append(Image.fromarray(im).convert('RGB').quantize(
                        palette=dummy, dither=0))
            kwargs = {
                'format': "GIF",
                'interlace': True,
                'save_all': True,
                'append_images': gif_images[1:],
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
            gif_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'png':
            images = [Image.fromarray(im) for im in images]
            kwargs = {
                'format': "PNG",
                'save_all': True,
                'append_images': images,
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            images[0].save(
                out,
                **kwargs
            )
        if not isinstance(out, str):
            out.seek(0)
        if extra_name is None:
            extra_name = 'render'
        if extra_out is not None:
            file = zipfile.PyZipFile(extra_out, "x")
            for i, img in enumerate(images):
                buffer = BytesIO()
                Image.fromarray(img).save(buffer, "PNG")
                file.writestr(
                    f"{extra_name}_{i // 3}_{(i % 3) + 1}.png",
                    buffer.getvalue())
            file.close()


async def setup(bot: Bot):
    bot.renderer = Renderer(bot)
