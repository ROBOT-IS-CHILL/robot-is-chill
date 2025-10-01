from __future__ import annotations

import collections
import os
import signal
import time
import traceback
import warnings
from pathlib import Path

import requests

import math
from PIL import Image
import re
from datetime import datetime
from io import BytesIO
from json import load
from typing import Any, OrderedDict, Literal

import numpy as np
import emoji
from charset_normalizer import from_bytes

import aiohttp
import discord
from discord import ui, Interaction, app_commands
from discord.ext import commands, menus

import config
import webhooks
from src.types import SignText, RenderContext
from src.utils import ButtonPages
from src import utils
from ..tile import Tile, TileSkeleton, parse_variant

from .. import constants, errors
from ..db import CustomLevelData, LevelData
from ..types import Bot, Context, RegexDict


def try_index(string: str, value: str) -> int:
    """Returns the index of a substring within a string.

    Returns -1 if not found.
    """
    index = -1
    try:
        index = string.index(value)
    except BaseException:
        pass
    return index


# Splits the "text_x,y,z..." shortcuts into "text_x", "text_y", ...
def split_commas(grid: list[list[str]], prefix: str):
    for row in grid:
        to_add = []
        for i, word in enumerate(row):
            if "," in word:
                if word.startswith(prefix):
                    each = re.split(r'(?<!\\),', word)
                    expanded = [each[0]]
                    expanded.extend([prefix + segment for segment in each[1:]])
                    to_add.append((i, expanded))
                else:
                    pass
        for change in reversed(to_add):
            row[change[0]:change[0] + 1] = change[1]
    return grid


async def warn_dangermode(ctx: Context):
    warning_embed = discord.Embed(
        title="Warning: Danger Mode",
        color=discord.Color(16711680),
        description="Danger Mode has been enabled by the developer.\nOutput may not be reliable or may break entirely.\nProceed at your own risk.")
    await ctx.send(embed=warning_embed, delete_after=5)


async def coro_part(func, *args, **kwargs):
    async def wrapper():
        result = func(*args, **kwargs)
        return await result

    return wrapper


class RenderBox(ui.Modal, title='Render Body'):
    global_cog: GlobalCog
    text: bool
    bot: Bot

    def __init__(self, cog: GlobalCog, text: bool):
        super().__init__()
        self.global_cog = cog
        self.text = text
        self.bot = cog.bot

    scene = ui.TextInput(label='Scene Contents', style=discord.TextStyle.paragraph,
                         placeholder="-b\n$baba $is $you\nbaba . flag\n$flag $is $win")

    async def on_submit(self, intr: discord.Interaction):
        await intr.response.defer(ephemeral=False, thinking=True)

        try:
            webhook = await self.bot.fetch_webhook(webhooks.logging_id)
            ctx: Context
            embed = discord.Embed(
                description=f"/render {self.scene.value}",
                color=config.logging_color)
            embed.set_author(
                name=f'{intr.user.name}'[:32],
                icon_url=intr.user.avatar.url if intr.user.avatar else None
            )
            await webhook.send(embed=embed)
        except Exception as e:
            warnings.warn("\n".join(traceback.format_exception(e)))

        wrapper = RenderBoxWrapper(intr)
        wrapper.message.content = self.scene.value
        wrapper.bot = self.bot
        wrapper.fake = True
        await self.global_cog.start_timeout(
            wrapper,
            objects=self.scene.value,
            rule=self.text
        )


class FakeFlags:
    def __getattr__(self, item):
        return False

class FakeMessage:
    flags = FakeFlags()
    ...

class RenderBoxWrapper:
    intr: Interaction
    bot: Bot
    message: FakeMessage

    def __init__(self, intr: Interaction):
        self.intr = intr
        self.message = FakeMessage()

    async def send(self, *args, **kwargs):
        await self.intr.followup.send(*args, **kwargs)

    async def reply(self, *args, **kwargs):
        await self.send(*args, **kwargs)

    async def error(self, msg: str, **kwargs):
        msg = f"```\n{self.message.content.replace('`', '')}\n```\n\n:warning: {msg}"
        await self.send(msg, **kwargs)

    async def typing(self):
        pass


class GlobalCog(commands.Cog, name="Baba Is You"):
    def __init__(self, bot: Bot):
        self.bot = bot

    # Check if the bot is loading
    async def cog_check(self, ctx):
        """Only if the bot is not loading assets."""
        return not self.bot.loading

    async def start_timeout(self, ctx, *args, timeout_multiplier: float = 1.0, **kwargs):
        def handler(_signum, _frame):
            raise errors.TimeoutError()

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(constants.TIMEOUT_DURATION * timeout_multiplier))
        await self.render_tiles(ctx, *args, **kwargs)

    async def handle_variant_errors(self, ctx: Context, err: errors.VariantError):
        """Handle errors raised in a command context by variant handlers."""
        try:
            word, variant, *rest = err.args
        except ValueError:
            word, *rest = err.args
            variant = '(Unspecified in error)'
        # easter egg
        if variant == "porp":
            return await ctx.error(":porp", file=discord.File("data/misc/porp.jpg"))
        msg = f"The variant `{variant}` for `{word}` is invalid"
        if isinstance(err, errors.BadTilingVariant):
            tiling = rest[0]
            return await ctx.error(
                f"{msg}, since it can't be applied to tiles with tiling type `{tiling}`."
            )
        elif isinstance(err, errors.TileNotText):
            return await ctx.error(
                f"{msg}, since the tile is not text."
            )
        elif isinstance(err, errors.BadPaletteIndex):
            return await ctx.error(
                f"{msg}, since the color is outside the palette."
            )
        elif isinstance(err, errors.BadLetterVariant):
            return await ctx.error(
                f"{msg}, since letter-style text can only be 1-3 letters wide."
            )
        elif isinstance(err, errors.BadMetaVariant):
            depth = rest[0]
            return await ctx.error(
                f"{msg}. `abs({depth})` is greater than the maximum meta depth, which is `{constants.MAX_META_DEPTH}`."
            )
        elif isinstance(err, errors.UnknownVariant):
            return await ctx.error(
                f"The variant for `{word}:{variant}` doesn't exist."
            )
        elif isinstance(err, errors.BadVariant):
            matched_variant = rest[0]
            return await ctx.error(
                f"The arguments for `{word}:{variant}` (`{matched_variant}`) are not valid."
            )
        else:
            return await ctx.error(f"{msg}.")

    async def handle_custom_text_errors(self, ctx: Context, err: errors.TextGenerationError):
        """Handle errors raised in a command context by variant handlers."""
        text, *rest = err.args
        msg = f"The text {text} couldn't be generated automatically"
        if isinstance(err, errors.BadLetterStyle):
            return await ctx.error(
                f"{msg}, since letter style can only applied to a single row of text."
            )
        elif isinstance(err, errors.TooManyLines):
            return await ctx.error(
                f"{msg}, since it has too many lines."
            )
        elif isinstance(err, errors.LeadingTrailingLineBreaks):
            return await ctx.error(
                f"{msg}, since there's `/` characters at the start or end of the text."
            )
        elif isinstance(err, errors.BadCharacter):
            mode, char = rest
            return await ctx.error(
                f"{msg}, since the letter {char} doesn't exist in '{mode}' mode."
            )
        elif isinstance(err, errors.CustomTextTooLong):
            return await ctx.error(
                f"{msg}, since it's too long ({len(text)})."
            )
        else:
            return await ctx.error(f"{msg}.")

    async def handle_grid(
            self, ctx: Context, grid, possible_variants, shape, render_ctx: RenderContext
        ):
        """Parses a TileSkeleton array into a Tile grid."""
        tile_data_cache = {
            data.name: data async for data in self.bot.db.tiles(
                {
                    tile.name for tile in grid.values() if isinstance(tile, TileSkeleton)
                }
            )
        }
        tilegrid = {
            (y, x, z, t): (
                (await Tile.prepare(
                    possible_variants, tile, tile_data_cache,
                    grid, shape[1], shape[0], (t, z, y, x), render_ctx.tileborder, ctx
                ))
                if isinstance(tile, TileSkeleton)
                else tile
            ) for (y, x, z, t), tile in grid.items()
        }
        new_items = {}
        for (y, x, z, t), tile in tilegrid.items():
            if (y, x, z, t + 1) in tilegrid:
                continue
            if tile is None:
                continue
            t += 1
            while t < shape[3] and (y, x, z, t) not in tilegrid:
                new_items[y, x, z, t] = tile.clone()
                t += 1
        tilegrid |= new_items
        # Sort tilegrid
        tgrid = {}
        print(shape)
        for t in range(shape[3]):
            for z in range(shape[2]):
                for y in range(shape[0]):
                    for x in range(shape[1]):
                        tile = tilegrid.get((y, x, z, t), None)
                        if isinstance(tile, SignText):
                            tile = tile.clone()
                            tile.t = t
                            render_ctx.sign_texts.append(tile)
                        elif tile is not None:
                            tgrid[y, x, z, t] = tile
        return tgrid

    async def parse_tile_grid(self, render_ctx: RenderContext, scenestr: str) -> dict[(int, int, int, int), TileSkeleton]:
        scenestr = scenestr.strip()
        initial_len = len(scenestr)
        tiles = {}
        y = x = z = t = 0
        y_size = x_size = z_size = t_size = 1
        comma_prefix = None
        last_tile = None
        do_comma = False
        possible_variants = RegexDict(
            [(variant.pattern, variant) for variant in self.bot.variants._values if variant.type != "sign"])
        font_variants = RegexDict(
            [(variant.pattern, variant) for variant in self.bot.variants._values if variant.type == "sign"])

        """
            scene := (row? "\n")* EOF;
            row := (stack? ("," | " "))*;
            stack := (timeline? "&")*;
            timeline := (atom? ">")*;
            atom := ("" | signtext | tile) vars;
            vars := var*;
            var := persistent | ephemeral;
            persistent := ";" Variant.parse(UNTIL_NEXT_UNESCAPED_OR_EOF(break));
            ephemeral := ":" Variant.parse(UNTIL_NEXT_UNESCAPED_OR_EOF(break));
            signtext := "{" UNTIL_NEXT_UNESCAPED("}");
            tile := UNTIL_NEXT_UNESCAPED_OR_EOF(break);
            break := " " | "," | "\n" | ":" | ";" | "&" | ">";
        """

        EL_BREAK = (" ", ",", "\n", "&", ">")

        async def parse_signtext() -> bool:
            nonlocal scenestr, initial_len, font_variants, render_ctx, comma_prefix, do_comma, self, x, y, z, t
            text_end = utils.find_unescaped(scenestr, "}")
            assert text_end >= 0, f"Sign text started at index {initial_len - len(scenestr)} was never closed!"
            string, scenestr = scenestr[1:text_end], scenestr[text_end + 1:]
            el_end = utils.find_unescaped(scenestr, EL_BREAK)
            if el_end < 0:
                el_end = len(scenestr)
            vars, scenestr = scenestr[:el_end], scenestr[el_end:]
            el = await TileSkeleton.parse(
                self.bot, font_variants, vars,
                render_ctx.prefix, render_ctx.palette,
                render_ctx.global_variant,
                prefix = comma_prefix if do_comma else None
            )
            st = SignText(t, x, y, string)
            st.variants = el.variants
            return st

        async def parse_tile() -> TileSkeleton | None:
            nonlocal scenestr, possible_variants, render_ctx, comma_prefix, self
            el_end = utils.find_unescaped(scenestr, EL_BREAK)
            if el_end < 0:
                el_end = len(scenestr)
            tilestr, scenestr = scenestr[:el_end], scenestr[el_end:]
            return await TileSkeleton.parse(
                self.bot, possible_variants, tilestr,
                render_ctx.prefix, render_ctx.palette,
                render_ctx.global_variant,
                prefix = comma_prefix if do_comma else None
            )

        async def parse_element() -> bool:
            nonlocal scenestr, last_tile, x, y, z, t
            element_str = None
            is_signtext = False
            if scenestr.startswith("{"):
                tile = await parse_signtext()
                tile.prefix = ""
            else:
                tile = await parse_tile()
            if tile.name == "":
                if last_tile is None:
                    tile = None
                else:
                    ltile = last_tile.clone()
                    ltile.variants = [var for var in tile.variants]
                    tile = ltile
                    if len(tile.variants):
                        if any(var.persistent for var in tile.variants):
                            last_tile = last_tile.clone()
                            last_tile.variants = tile.variants
                        else:
                            tile.variants = [var for var in last_tile.variants if var.persistent] + tile.variants
                    else:
                        tile = "<EMPTY>"
            elif tile.postfix in (".", "-"):
                tile = last_tile = None
            else:
                last_tile = tile
            if tile != "<EMPTY>":
                tiles[y, x, z, t] = tile
            return True

        async def parse_timeline() -> bool:
            nonlocal scenestr, initial_len, comma_prefix, last_tile, do_comma, x, y, z, t, t_size
            cur = len(scenestr)
            while True:
                await parse_element()
                t += 1
                t_size = max(t, t_size)
                comma_prefix = last_tile.prefix if last_tile is not None else None
                if not scenestr.startswith(">"):
                    return True
                scenestr = scenestr.removeprefix(">")
                do_comma = False
                comma_prefix = None

        async def parse_stack() -> bool:
            nonlocal scenestr, do_comma, x, y, z, t, z_size
            while True:
                await parse_timeline()
                last_tile = None
                t = 0
                z += 1
                z_size = max(z, z_size)
                if not scenestr.startswith("&"):
                    return True
                do_comma = False
                scenestr = scenestr.removeprefix("&")

        async def parse_row() -> bool:
            nonlocal scenestr, comma_prefix, do_comma, x, y, z, t, x_size
            while True:
                await parse_stack()
                z = t = 0
                x += 1
                x_size = max(x, x_size)
                if scenestr.startswith(","):
                    # Comma
                    do_comma = True
                    scenestr = scenestr.removeprefix(",")
                elif scenestr.startswith(" "):
                    do_comma = False
                    comma_prefix = None
                    scenestr = scenestr.lstrip(" ")
                else:
                    return True
            return True

        while True:
            await parse_row()
            last_tile = None
            x = z = t = 0
            y += 1
            y_size = max(y, y_size)
            if not len(scenestr): break
            assert scenestr.startswith("\n"), f"The scene parser erroneously failed at this part of the string:\n```\n{scenestr}\n```"
            scenestr = scenestr.removeprefix("\n")

        return tiles, (y_size, x_size, z_size, t_size)


    async def render_tiles(self, ctx: Context, *, objects: str, rule: bool):
        """Performs the bulk work for both `tile` and `rule` commands."""
        try:
            await ctx.typing()
            ctx.silent = ctx.message is not None and ctx.message.flags.silent
            tiles = emoji.demojize(objects.strip(), language='alias')\
                .replace(":hearts:", "♥")  # keep the heart, for the people
            tiles = re.sub(r'<a?(:.+?:)\d+?>', r'\1', tiles)
            tiles = tiles.removeprefix("`").removesuffix("`")
            # Replace some phrases
            replace_list = [
                ['а', 'a'],
                ['е', 'e'],
                ['о', 'o'],
                ['р', 'p'],
                ['с', 'c'],
                ['х', 'x'],
                ['ⓜ', ':m:'],
                [':thumbsdown:', ':-1:']
            ]
            for src, dst in replace_list:
                tiles = tiles.replace(src, dst)

            # Determines if this should be a spoiler
            spoiler = "||" in tiles
            tiles = tiles.replace("||", "")

            # Check flags
            old_tiles = tiles

            parsing_overhead = time.perf_counter()

            render_ctx = RenderContext(ctx=ctx)
            render_ctx.prefix = 'text' if rule else None
            while match := re.match(r"^\s*(--?((?:(?!=)\S)+)(?:=(?:(?!(?<!\\)\s).)+)?)", tiles):
                potential_flag = match.group(1)
                for flag in self.bot.flags.list:
                    if await flag.match(potential_flag, render_ctx):
                        tiles = tiles[match.end():]
                        break
                else:
                    interp = match.group().strip().replace('`', "'")
                    raise AssertionError(f"Flag `{interp}` isn't valid.")

            if render_ctx.bypass_limits:
                signal.alarm(0)

            print(tiles)

            tiles = await ctx.bot.macro_handler.parse_macros(tiles, "r" if rule else "t")
            print(tiles)
            tiles = tiles.strip()

            tile_grid, shape = await self.parse_tile_grid(render_ctx, tiles)

            possible_variants = RegexDict(
                [(variant.pattern, variant) for variant in self.bot.variants._values if variant.type != "sign"])
            font_variants = RegexDict(
                [(variant.pattern, variant) for variant in self.bot.variants._values if variant.type == "sign"])

            maxdelta = 1
            try:
                render_ctx.out = BytesIO()
                full_grid = await self.handle_grid(ctx, tile_grid, possible_variants, shape, render_ctx)
                parsing_overhead = time.perf_counter() - parsing_overhead
                full_tiles, unique_tiles, rendered_frames, render_overhead = await self.bot.renderer.render_full_tiles(
                    full_grid, shape,
                    ctx=render_ctx
                )
                composite_overhead, saving_overhead, im_size = await self.bot.renderer.render(
                    full_tiles, shape,
                    render_ctx
                )
            except errors.TileNotFound as e:
                word = e.args[0]
                if word.startswith("tile_") and await self.bot.db.tile(word[5:]) is not None:
                    return await ctx.error(f"The tile `{word}` could not be found. Perhaps you meant `{word[5:]}`?")
                if await self.bot.db.tile("text_" + word) is not None:
                    return await ctx.error(
                        f"The tile `{word}` could not be found. Perhaps you meant `{'text_' + word}`?")
                return await ctx.error(f"The tile `{word}` could not be found.")
            except errors.BadTileProperty as e:
                traceback.print_exc()
                return await ctx.error(f"Error! `{e.args[1]}`")
            except errors.EmptyVariant as e:
                word = e.args[0]
                return await ctx.error(
                    f"You provided an empty variant for `{word}`."
                )
            except errors.TooLargeTile as e:
                return await ctx.error(
                    f"A tile of size `{e.args[0]}` (`{e.args[1]}`) is larger than the maximum allowed size of `{constants.MAX_TILE_SIZE}`.")
            except errors.VariantError as e:
                return await self.handle_variant_errors(ctx, e)
            except errors.TextGenerationError as e:
                return await self.handle_custom_text_errors(ctx, e)

            filename = render_ctx.custom_filename
            if filename is None:
                filename = datetime.utcnow().strftime(f"render_%Y-%m-%d_%H.%M.%S")
            filename = f"{filename}.{render_ctx.image_format}"
            image = discord.File(render_ctx.out, filename=filename, spoiler=spoiler)
            if hasattr(ctx, "fake") or hasattr(ctx, "is_from_file"):
                prefix = ""
            else:
                prefix = ctx.message.content.split(' ', 1)[0] + " "
            if hasattr(ctx, "is_from_file"):
                description = ""
            else:
                description = f"{'||' if spoiler else ''}```\n{prefix}{old_tiles}\n```{'||' if spoiler else ''}"
            if render_ctx.do_embed:
                embed = discord.Embed(color=self.bot.embed_color)

                def rendertime(v):
                    v *= 1000
                    nice = False
                    if math.ceil(v) == 69:
                        nice = True
                    if objects == "lag":
                        v *= 100000
                    return f'{v:.4f}' + ("(nice)" if nice else "")

                stats = f'''
- Response time: {rendertime(parsing_overhead + render_overhead + composite_overhead + saving_overhead)} ms
- Parsing overhead: {rendertime(parsing_overhead)} ms
- Rendering overhead: {rendertime(render_overhead)} ms
- Compositing overhead: {rendertime(composite_overhead)} ms
- Saving overhead: {rendertime(saving_overhead)} ms
- Tiles rendered: {unique_tiles}
- Frames rendered: {rendered_frames}
- Image size: {im_size}
    '''

                embed.add_field(name="Render statistics", value=stats)
            else:
                embed = None
            await ctx.reply(description[:2000], embed=embed, file=image)
        finally:
            signal.alarm(0)

    @app_commands.command()
    @app_commands.allowed_installs(guilds=False, users=True)
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    async def render(self, intr: Interaction, render_mode: Literal["tile", "text"] = "tile"):
        """Renders the tiles provided using a modal."""
        box = RenderBox(self, render_mode == "text")
        await intr.response.send_modal(box)
        await box.wait()

    @commands.command(aliases=["t"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def tile(self, ctx: Context, *, objects: str = ""):
        """Renders the tiles provided.

        **Flags**
        * See the `flags` commands for all the valid flags.

        **Variants**
        * `:variant`: Append `:variant` to a tile to change different attributes of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile.
        * `&` : Stacks tiles on top of each other. Tiles are rendered in stack order, so in `=rule baba&cursor me`, Baba and Me would be rendered below Cursor.
        * `$` : `$object` renders text objects.
        * `,` : `$x,y,...` is expanded into `$x $y ...`
        * `||` : Marks the output gif as a spoiler.
        * `""`: `"x y ..."` is expanded into `$x $y $...`

        **Example commands:**
        `tile baba - keke`
        `tile --palette=marshmallow keke:d baba:s`
        `tile text_baba,is,you`
        `tile baba&flag ||cake||`
        `tile -P=mountain -B baba bird:l`
        """
        if self.bot.config['danger_mode']:
            await warn_dangermode(ctx)
        await self.start_timeout(
            ctx,
            objects=objects,
            rule=False)

    @commands.command(aliases=["text", "r"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def rule(self, ctx: Context, *, objects: str = ""):
        """Renders the text tiles provided.

        If not found, the bot tries to auto-generate them!

        **Flags**
        * See the `flags` commands for all the valid flags.

        **Variants**
        * `:variant`: Append `:variant` to a tile to change different attributes of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile.
        * `&` : Stacks tiles on top of each other. Tiles are rendered in stack order, so in `=rule baba&cursor me`, Baba and Me would be rendered below Cursor.
        * `$` : `$object` renders tile objects.
        * `,` : `$x,y,...` is expanded into `$x $y ...`
        * `||` : Marks the output gif as a spoiler.
        * `""`: `"x y ..."` is expanded into `$x $y $...`

        **Example commands:**
        `rule baba is you`
        `rule -b rock is ||push||`
        `rule -p=test tile_baba on baba is word`
        `rule baba eat baba - tile_baba tile_baba:l`
        """
        if self.bot.config['danger_mode']:
            await warn_dangermode(ctx)
        await self.start_timeout(
            ctx,
            objects=objects,
            rule=True)

    # Generates tiles from a text file.
    @commands.command(aliases=["f"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def file(self, ctx: Context, rule: str = ''):
        """Renders the text from a file attatchment.

        Add -r, --rule, -rule, -t, --text, or -text to render as text.
        """
        try:
            objects = str(from_bytes((await ctx.message.attachments[0].read())).best())
            ctx.is_from_file = True
            await self.start_timeout(
                ctx,
                objects=objects,
                rule=rule in [
                    '-r',
                    '--rule',
                    '-rule',
                    '-t',
                    '--text',
                    '-text'], timeout_multiplier=1.5)
        except IndexError:
            await ctx.error('You forgot to attach a file.')

    async def search_levels(self, query: str, **flags: Any) -> OrderedDict[tuple[str, str], LevelData]:
        """Finds levels by query.

        Flags:
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        """
        levels: OrderedDict[tuple[str, str], LevelData] = collections.OrderedDict()
        f_map = flags.get("map")
        f_world = flags.get("world")
        async with self.bot.db.conn.cursor() as cur:
            # [world]/[levelid]
            parts = query.split("/", 1)
            if len(parts) == 2:
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE
						world == :world AND
						id == :id AND (
							:f_map IS NULL OR map_id == :f_map
						) AND (
							:f_world IS NULL OR world == :f_world
						);
					''',
                    dict(
                        world=parts[0],
                        id=parts[1],
                        f_map=f_map,
                        f_world=f_world)
                )
                row = await cur.fetchone()
                if row is not None:
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

            maybe_parts = query.split(" ", 1)
            if len(maybe_parts) == 2:
                maps_queries = [
                    (maybe_parts[0], maybe_parts[1]),
                    (f_world, query)
                ]
            else:
                maps_queries = [
                    (f_world, query)
                ]

            for f_world, query in maps_queries:
                # someworld/[levelid]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE id == :id AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN 'baba'
						THEN 0
						ELSE world
					END ASC;
					''',
                    dict(
                        id=query,
                        f_map=f_map,
                        f_world=f_world)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [parent]-[map_id]
                segments = query.split("-")
                if len(segments) == 2:
                    await cur.execute(
                        '''
						SELECT * FROM levels
						WHERE parent == :parent AND (
							UNLIKELY(map_id == :map_id) OR (
								style == 0 AND
								CAST(number AS TEXT) == :map_id
							) OR (
								style == 1 AND
								LENGTH(:map_id) == 1 AND
								number == UNICODE(:map_id) - UNICODE("a")
							) OR (
								style == 2 AND
								SUBSTR(:map_id, 1, 5) == "extra" AND
								number == CAST(TRIM(SUBSTR(:map_id, 6)) AS INTEGER) - 1
							)
						) AND (
							:f_map IS NULL OR map_id == :f_map
						) AND (
							:f_world IS NULL OR world == :f_world
						) ORDER BY CASE world
							WHEN 'baba'
							THEN 0
							ELSE world
						END ASC;
						''',
                        dict(parent=segments[0], map_id=segments[1], f_map=f_map, f_world=f_world)
                    )
                    for row in await cur.fetchall():
                        data = LevelData.from_row(row)
                        levels[data.world, data.id] = data

                # [name]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE name == :name AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN 'baba'
						THEN 0
						ELSE world
					END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [name-ish]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE INSTR(name, :name) AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
                    ORDER BY CASE world
                        WHEN 'baba'
                        THEN 0
                        ELSE world
                    END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [map_id]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE map_id == :map AND parent IS NULL AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN 'baba'
						THEN 0
						ELSE world
					END ASC;
					''',
                    dict(
                        map=query,
                        f_map=f_map,
                        f_world=f_world)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

        return levels

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @commands.group(name="level", invoke_without_command=True)
    async def level_command(self, ctx: Context, *, query: str):
        """Renders the Baba Is You level from a search term.

        Levels are searched for in the following order:
        * Custom level code (e.g. "1234-ABCD")
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        """
        await self.perform_level_command(ctx, query, mobile=False)

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @level_command.command()
    async def mobile(self, ctx: Context, *, query: str):
        """Renders the mobile Baba Is You level from a search term.

        Levels are searched for in the following order:
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        """
        await self.perform_level_command(ctx, query, mobile=True)

    async def perform_level_command(self, ctx: Context, query: str, *, mobile: bool):
        # User feedback
        await ctx.typing()

        custom_level: CustomLevelData | None = None

        spoiler = query.count("||") >= 2
        fine_query = query.strip().replace("|", "")

        # [abcd-0123]
        if re.match(r"^[A-Za-z\d]{4}-[A-Za-z\d]{4}$", fine_query) and not mobile:
            row = await self.bot.db.conn.fetchone(
                'SELECT * FROM custom_levels WHERE code == ?;',
                fine_query
            )
            if row is not None:
                custom_level = CustomLevelData.from_row(row)
            else:
                # Expensive operation
                await ctx.reply("Searching for custom level... this might take a while", mention_author=False,
                                delete_after=10)
                await ctx.typing()
                async with aiohttp.request("GET",
                                           f"https://baba-is-bookmark.herokuapp.com/api/level/exists?code={fine_query.upper()}") as resp:
                    if resp.status in (200, 304):
                        data = await resp.json()
                        if data["data"]["exists"]:
                            try:
                                custom_level = await self.bot.get_cog("Reader").render_custom_level(fine_query)
                            except ValueError as e:
                                return await ctx.error(
                                    f"The level code is valid, but the level's {e.args[1]} is too big to fit in a GIF. ({e.args[0] * 24} > 65535)")
                            except aiohttp.ClientResponseError:
                                return await ctx.error(
                                    f"The Baba Is Bookmark site returned a bad response. Try again later.")
        if custom_level is None:
            levels = await self.search_levels(fine_query)
            first = None
            for ((pack, name), l) in levels.items():
                if first is None:
                    first = l
                if pack in constants.VANILLA_WORLDS:
                    level = l
                    break
            else:
                level = first
            if level is None:
                return await ctx.error("A level could not be found with that query.")
        else:
            level = custom_level

        footer = None
        if isinstance(level, LevelData):
            path = level.unique()
            display = level.display().upper()
            rows = [
                f"`{path}`",
            ]
            if level.subtitle:
                rows.append(
                    f"_{level.subtitle}_"
                )
            mobile_exists = os.path.exists(
                f"target/renders/{level.world}_m/{level.id}.gif")

            if not mobile and mobile_exists:
                footer = f"This level is also on mobile, see [level mobile {level.unique()}]"
            elif mobile and mobile_exists:
                footer = f"This is the mobile version. For others, see [level {level.unique()}]"

            if mobile and mobile_exists:
                filepath = f"target/renders/{level.world}_m/{level.id}.gif"
                gif = discord.File(
                    filepath,
                    filename=level.world + '_m_' + level.id + '.gif',
                    spoiler=spoiler)
            else:
                if mobile and not mobile_exists:
                    footer = "This level doesn't have a mobile version. Using the normal gif instead..."
                filepath = f"target/renders/{level.world}/{level.id}.gif"
                gif = discord.File(
                    filepath,
                    filename=level.world + '_' + level.id + '.gif',
                    spoiler=spoiler)
        else:
            try:
                filepath = f"target/renders/levels/{level.code}.gif"
                gif = discord.File(
                    filepath,
                    filename=level.code + '.webp',
                    spoiler=spoiler)
            except FileNotFoundError:
                await self.bot.get_cog("Reader").render_custom_level(fine_query)
                filepath = f"target/renders/levels/{level.code}.webp"
                gif = discord.File(
                    filepath,
                    filename=level.code + '.gif',
                    spoiler=spoiler)
            path = level.unique()
            display = f"{level.name.upper()} (by {level.author})"
            rows = [
                f"`{path}`",
            ]
            if level.subtitle:
                rows.append(f"_{level.subtitle}_")

        formatted = "\n".join(rows)

        # Only the author should be mentioned
        mentions = discord.AllowedMentions(
            everyone=False, users=[
                ctx.author], roles=False)

        gif.spoiler = True

        emb = discord.Embed(
            color=self.bot.embed_color,
            title=display,
            description=formatted,
        )
        emb.set_footer(text=footer)
        # Send the result
        await ctx.reply(embed=emb, file=gif, allowed_mentions=mentions)


async def setup(bot: Bot):
    await bot.add_cog(GlobalCog(bot))
