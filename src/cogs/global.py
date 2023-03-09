from __future__ import annotations

import collections
import os
import signal

import requests

import math
from PIL import Image
import re
from datetime import datetime
from io import BytesIO
from json import load
from time import perf_counter
from typing import Any, OrderedDict, TYPE_CHECKING

import numpy as np
import emoji
from charset_normalizer import from_bytes

import aiohttp
import discord
from discord.ext import commands

from ..tile import Tile, TileSkeleton

from .. import constants, errors
from ..db import CustomLevelData, LevelData
from ..types import Bot, Context

from .errorhandler import CommandErrorHandler


async def start_timeout(function, *args, timeout_multiplier=1, **kwargs):
    def handler(_signum, _frame):
        raise AssertionError("The command took too long and was timed out.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(constants.TIMEOUT_DURATION * timeout_multiplier)
    await function(*args, **kwargs)
    signal.alarm(0)


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
                    each = word.split(",")
                    expanded = [each[0]]
                    expanded.extend([prefix + segment for segment in each[1:]])
                    to_add.append((i, expanded))
                else:
                    raise errors.SplittingException(word)
        for change in reversed(to_add):
            row[change[0]:change[0] + 1] = change[1]
    return grid


class GlobalCog(commands.Cog, name="Baba Is You"):
    def __init__(self, bot: Bot):
        self.bot = bot
        with open("config/leveltileoverride.json") as f:
            j = load(f)
            self.level_tile_override = j

    # Check if the bot is loading
    async def cog_check(self, ctx):
        """Only if the bot is not loading assets."""
        return not self.bot.loading

    async def handle_variant_errors(self, ctx: Context, err: errors.VariantError):
        """Handle errors raised in a command context by variant handlers."""
        try:
            word, variant, *rest = err.args
        except ValueError:
            word, *rest = err.args
            variant = '(Unspecified in error)'
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
                f"{msg}, since letter-style text can only be 1 or 2 letters wide."
            )
        elif isinstance(err, errors.BadMetaVariant):
            depth = rest[0]
            return await ctx.error(
                f"{msg}. `abs({depth})` is greater than the maximum meta depth, which is `{constants.MAX_META_DEPTH}`."
            )
        elif isinstance(err, errors.UnknownVariant):
            return await ctx.error(
                f"There isn't a variant called `{variant}`."
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
            self, grid: list[list[list[list[TileSkeleton]]]]):
        """Parses a TileSkeleton grid into a Tile grid."""
        tile_data_cache = {
            data.name: data async for data in self.bot.db.tiles(
                {
                    tile.name for steps in grid for row in steps for stack in row for tile in stack
                }
            )
        }
        return [
            [
                [
                    [
                        Tile.prepare(tile_data_cache, tile)
                        for tile in row
                    ]
                    for row in layer
                ]
                for layer in timestep
            ]
            for timestep in grid
        ]

    async def render_tiles(self, ctx: Context, *, objects: str, rule: bool):
        """Performs the bulk work for both `tile` and `rule` commands."""
        await ctx.typing()
        start = perf_counter()
        tiles = objects.lower().strip().replace("\\", "").replace("`", "")
        # Replace some phrases
        tiles = re.sub(r'<(:.+?:)\d+?>', r'\1', tiles)
        tiles = emoji.demojize(tiles, use_aliases=True)
        replace_list = [
            ['а', 'a'],
            ['в', 'b'],
            ['е', 'e'],
            ['з', '3'],
            ['к', 'k'],
            ['м', 'm'],
            ['н', 'h'],
            ['о', 'o'],
            ['р', 'p'],
            ['с', 'c'],
            ['т', 't'],
            ['х', 'x'],
            ['ⓜ', ':m:']
        ]
        for src, dst in replace_list:
            tiles = tiles.replace(src, dst)

        # Determines if this should be a spoiler
        spoiler = "|" in tiles
        tiles = tiles.replace("|", "")

        # Check for empty input
        if not tiles:
            return await ctx.error("Input cannot be blank.")

        if rule:
            tiles = tiles.replace('$', 'tile_')
        else:
            tiles = tiles.replace('$', 'text_')

        # Split input into lines
        word_rows = tiles.splitlines()

        # Split each row into words
        word_grid = [row.split() for row in word_rows]

        # Check flags
        potential_flags = filter(
            lambda i: i[0].startswith("-"),
            [(word, x, y)
             for y, row in enumerate(word_grid)
             for x, word in enumerate(row)]
        )

        kwargs = {}
        to_delete = []
        for potential_flag, x, y in potential_flags:
            for flag in self.bot.flags.list:
                to_delete, kwargs = await flag.match(ctx, potential_flag, x, y, kwargs, to_delete)
        raw_output = kwargs.get("raw_output", False)
        default_to_letters = kwargs.get("letters", False)
        tborders = kwargs.get("tborders", False)
        file_format = kwargs.get('file_format', 'gif')
        do_embed = kwargs.get('do_embed', False)
        for x, y in reversed(to_delete):
            del word_grid[y][x]
        try:
            if rule:
                comma_grid = split_commas(word_grid, "tile_")
            else:
                comma_grid = split_commas(word_grid, "text_")
        except errors.SplittingException as e:
            cause = e.args[0]
            return await ctx.error(f"I couldn't split the following input into separate objects: \"{cause}\".")

        tilecount = 0
        maxstack = 1
        maxdelta = 1
        try:
            for row in comma_grid:
                for stack in row:
                    maxstack = max(maxstack, len(stack.split("&")))
                    for timeline in stack.split("&"):
                        maxdelta = max(maxdelta, len(timeline.split(">")))
            w, h, d, t = max([len(comma_grid[n]) for n in range(len(comma_grid))]), len(
                comma_grid), maxstack, maxdelta  # width, height, depth, time
            layer_grid = np.full((t, d, h, w), '-', dtype=object)
            if maxstack > constants.MAX_STACK and ctx.author.id != self.bot.owner_id:
                return await ctx.error(
                    f"Stack too high ({maxstack}).\nYou may only stack up to {constants.MAX_STACK} tiles on one space.")
            # Splits "&"-joined words into stacks
            for y, row in enumerate(comma_grid):
                for x, stack in enumerate(row):
                    for layer, timeline in enumerate(stack.split('&')):
                        for d, tile in enumerate(timeline.split('>')):
                            if len(tile):
                                assert not len(tile.split(':', 1)) - 1 or not tile.split(':', 1)[1].count(
                                    ';'), 'Error! Persistent variants (`;`) can\'t come after ephemeral ones (`:`).'
                                if len(tile.split(':', 1)[0].split(';', 1)[0]):
                                    tilecount += 1
                                    tile = tile.replace('rule_', 'text_')
                                    layer_grid[d:, layer, y, x] = TileSkeleton.parse(ctx, tile)
                                else:
                                    if len(tile.split(';', 1)) == 2:
                                        layer_grid[d:, layer, y, x] = layer_grid[d, layer, y, x].split(';', 1)[
                                            0] + tile
                                    else:
                                        layer_grid[d:, layer, y, x] = layer_grid[d, layer, y, x].split(':', 1)[
                                                0] + tile
            # Get the dimensions of the grid
            grid_shape = layer_grid.shape
            layer_grid = layer_grid.tolist()
            # Don't proceed if the request is too large.
            # (It shouldn't be that long to begin with because of Discord's 2000-character limit)
            if tilecount > constants.MAX_TILES and not (
                    ctx.author.id in [self.bot.owner_id, 280756504674566144]):
                return await ctx.error(
                    f"Too many tiles ({tilecount}). You may only render up to {constants.MAX_TILES} tiles at once, including empty tiles.")
            # Handles variants based on `:` affixes
            buffer = BytesIO()
            extra_buffer = BytesIO() if raw_output else None
            extra_names = [] if raw_output else None
            full_grid = await self.handle_grid(layer_grid)
            avgdelta, maxdelta, tiledelta, unique_sprites = await self.bot.renderer.render(
                await self.bot.renderer.render_full_tiles(
                    full_grid,
                    palette=kwargs.get("palette", None),
                    random_animations=kwargs.get("random_animations", True),
                    gscale=kwargs.get("gscale", 1)
                ),
                out=buffer,
                extra_out=extra_buffer,
                **kwargs
            )
        except errors.TileNotFound as e:
            word = e.args[0]
            if word.name.startswith("tile_") and await self.bot.db.tile(word.name[5:]) is not None:
                return await ctx.error(f"The tile `{word}` could not be found. Perhaps you meant `{word.name[5:]}`?")
            if await self.bot.db.tile("text_" + word.name) is not None:
                return await ctx.error(
                    f"The tile `{word}` could not be found. Perhaps you meant `{'text_' + word.name}`?")
            return await ctx.error(f"The tile `{word}` could not be found.")
        except errors.BadTileProperty as e:
            return await ctx.error(f"Error! `{e.args[1]}`")
        except errors.EmptyVariant as e:
            word = e.args[0]
            return await ctx.error(
                f"You provided an empty variant for `{word}`."
            )
        except errors.VariantError as e:
            return await self.handle_variant_errors(ctx, e)
        except errors.TextGenerationError as e:
            return await self.handle_custom_text_errors(ctx, e)

        filename = datetime.utcnow().strftime(
            f"render_%Y-%m-%d_%H.%M.%S.{file_format}")
        delta = perf_counter() - start
        image = discord.File(buffer, filename=filename, spoiler=spoiler)
        description = f"{'||' if spoiler else ''}`{ctx.message.content.replace('||', '').replace('`', '')}`{'||' if spoiler else ''}"
        if do_embed:
            embed = discord.Embed(color=self.bot.embed_color)

            def rendertime(v):
                v *= 1000
                nice = False
                if math.ceil(v) == 69:
                    nice = True
                if objects == "lag":
                    v *= 100000
                return f'{v:.4f}' + ("(nice)" if nice else "")

            totalrendertime = rendertime(delta)
            activerendertime = rendertime(tiledelta)
            averagerendertime = rendertime(avgdelta)
            maxrendertime = rendertime(maxdelta)
            stats = f'''
			Total render time: {totalrendertime} ms
			Active render time: {activerendertime} ms
			Tiles rendered: {tilecount}
			Average render time of all tiles: {averagerendertime} ms
			Maximum render time of any tile: {maxrendertime} ms
			Cached frames: {unique_sprites}
            Tile matrix shape: {grid_shape}
			'''

            embed.add_field(name="Render statistics", value=stats)
        else:
            embed = None
        if extra_buffer is not None and extra_names is not None:
            extra_buffer.seek(0)
            await ctx.reply(description[:2000], embed=embed,
                            files=[discord.File(extra_buffer, filename=f"{extra_names[0]}_raw.zip"), image])
        else:
            await ctx.reply(description[:2000], embed=embed, file=image)

    # hack
    async def log_exceptions(self, ctx, awaitable):
        try:
            return await awaitable
        except Exception as e:
            await CommandErrorHandler(self.bot).on_command_error(ctx, e)

    @commands.command(aliases=["text"])
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
        * `tile_` : `tile_object` renders regular objects.
        * `||` : Marks the output gif as a spoiler.

        **Example commands:**
        `rule baba is you`
        `rule -b rock is ||push||`
        `rule -p=test tile_baba on baba is word`
        `rule baba eat baba - tile_baba tile_baba:l`
        """
        if self.bot.config['danger_mode']:
            await self.warn_dangermode(ctx)
        await start_timeout(self.render_tiles,
                            ctx,
                            objects=objects,
                            rule=True)

    # Generates tiles from a text file.
    @commands.command()
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def file(self, ctx: Context, rule: str = ''):
        """Renders the text from a file attatchment.

        Add -r, --rule, -rule, -t, --text, or -text to render as text.
        """
        try:
            objects = str(from_bytes((await ctx.message.attachments[0].read())).best())
            await start_timeout(self.render_tiles,
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

    # Generates an animated gif of the tiles provided, using the default
    # palette
    @commands.command()
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
        * `tile_` : `tile_object` renders regular objects.
        * `,` : `tile_x,y,...` is expanded into `tile_x tile_y ...`
        * `||` : Marks the output gif as a spoiler.

        **Example commands:**
        `tile baba - keke`
        `tile --palette=marshmallow keke:d baba:s`
        `tile text_baba,is,you`
        `tile baba&flag ||cake||`
        `tile -P=mountain -B baba bird:l`
        """
        if self.bot.config['danger_mode']:
            await self.warn_dangermode(ctx)
        await start_timeout(self.render_tiles,
                            ctx,
                            objects=objects,
                            rule=False)

    async def warn_dangermode(self, ctx: Context):
        warning_embed = discord.Embed(
            title="Warning: Danger Mode",
            color=discord.Color(16711680),
            description="Danger Mode has been enabled by the developer.\nOutput may not be reliable or may break entirely.\nProceed at your own risk.")
        await ctx.send(embed=warning_embed, delete_after=5)

    async def search_levels(self, query: str, **flags: Any) -> OrderedDict[tuple[str, str], LevelData]:
        """Finds levels by query.

        Flags:
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        """
        levels: OrderedDict[tuple[str, str],
                            LevelData] = collections.OrderedDict()
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
						WHEN :default
						THEN NULL
						ELSE world
					END ASC;
					''',
                    dict(
                        id=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
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
							WHEN :default
							THEN NULL
							ELSE world
						END ASC;
						''',
                        dict(parent=segments[0], map_id=segments[1], f_map=f_map, f_world=f_world,
                             default=constants.BABA_WORLD)
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
						WHEN :default
						THEN NULL
						ELSE world
					END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
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
						WHEN :default
						THEN NULL
						ELSE world
					END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
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
						WHEN :default
						THEN NULL
						ELSE world
					END ASC;
					''',
                    dict(
                        map=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
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
        fine_query = query.lower().strip().replace("|", "")

        # [abcd-0123]
        if re.match(r"^[a-z\d]{4}-[a-z\d]{4}$", fine_query) and not mobile:
            row = await self.bot.db.conn.fetchone(
                '''
				SELECT * FROM custom_levels WHERE code == ?;
				''',
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
                                    f"The level code is valid, but the level's {e.args[1]} is too big to fit in a GIF. ({e.args[0] * 48} > 65535)")
                            except aiohttp.ClientResponseError:
                                return await ctx.error(
                                    f"The Baba Is Bookmark site returned a bad response. Try again later.")
        if custom_level is None:
            levels = await self.search_levels(fine_query)
            try:
                _, level = levels.popitem(last=False)
            except KeyError:
                return await ctx.error("A level could not be found with that query.")
        else:
            levels = {}
            level = custom_level

        if isinstance(level, LevelData):
            path = level.unique()
            display = level.display()
            rows = [
                f"Name: ||{display}||" if spoiler else f"Name: {display}",
                f"ID: {path}",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: {level.subtitle}"
                )
            mobile_exists = os.path.exists(
                f"target/renders/{level.world}_m/{level.id}.gif")

            if not mobile and mobile_exists:
                rows.append(
                    f"*This level is also on mobile, see `+level mobile {level.unique()}`*"
                )
            elif mobile and mobile_exists:
                rows.append(
                    f"*This is the mobile version. For others, see `+level {level.unique()}`*"
                )

            if mobile and mobile_exists:
                gif = discord.File(
                    f"target/renders/{level.world}_m/{level.id}.gif",
                    filename=level.world + '_m_' + level.id + '.gif',
                    spoiler=spoiler)
            else:
                if mobile and not mobile_exists:
                    rows.append(
                        "*This level doesn't have a mobile version. Using the normal gif instead...*")
                gif = discord.File(
                    f"target/renders/{level.world}/{level.id}.gif",
                    filename=level.world + '_' + level.id + '.gif',
                    spoiler=spoiler)
        else:
            try:
                gif = discord.File(
                    f"target/renders/levels/{level.code}.gif",
                    filename=level.code + '.gif',
                    spoiler=spoiler)
            except FileNotFoundError:
                await self.bot.get_cog("Reader").render_custom_level(fine_query)
                gif = discord.File(
                    f"target/renders/levels/{level.code}.gif",
                    filename=level.code + '.gif',
                    spoiler=spoiler)
            path = level.unique()
            display = level.name
            rows = [
                f"Name: ||{display}|| (by {level.author})"
                if spoiler else f"Name: {display} (by {level.author})",
                f"Level code: {path}",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: {level.subtitle}"
                )

        if len(levels) > 0:
            extras = [level.unique() for level in levels.values()]
            if len(levels) > constants.OTHER_LEVELS_CUTOFF:
                extras = extras[:constants.OTHER_LEVELS_CUTOFF]
            paths = ", ".join(f"{extra}" for extra in extras)
            plural = "result" if len(extras) == 1 else "results"
            suffix = ", ..." if len(
                levels) > constants.OTHER_LEVELS_CUTOFF else ""
            rows.append(
                f"*Found {len(levels)} other {plural}: {paths}{suffix}*"
            )

        formatted = "\n".join(rows)

        # Only the author should be mentioned
        mentions = discord.AllowedMentions(
            everyone=False, users=[
                ctx.author], roles=False)

        # Send the result
        await ctx.reply(formatted, file=gif, allowed_mentions=mentions)

    @commands.command(aliases=["filterimages", "fi"])
    @commands.cooldown(5, 8, commands.BucketType.channel)
    async def filterimage(self, ctx: Context, *, query: str = ""):
        """Performs filterimage-related actions like template creation,
        conversion and accessing the database."""
        name = None
        if query.startswith("convert "):
            query = query.split(" ")
            url = query[2]
            if url.startswith("http://"):
                url = url[7:]
            if not url.startswith("https://"):
                url = "https://" + url
            relative = (query[1] in ("relative", "rel"))
            ifilterimage = Image.open(requests.get(
                url, stream=True).raw).convert("RGBA")
            fil = np.array(ifilterimage)
            if relative:
                fil[:, :, 0] -= np.arange(fil.shape[0], dtype="uint8")
                fil[:, :, 1] = (fil[:, :, 1].T -
                                np.arange(fil.shape[1], dtype="uint8")).T
            else:
                fil[:, :, 0] += np.arange(fil.shape[0], dtype="uint8")
                fil[:, :, 1] = (fil[:, :, 1].T +
                                np.arange(fil.shape[1], dtype="uint8")).T
            ifilterimage = Image.fromarray(fil)
            out = BytesIO()
            ifilterimage.save(out, format="png", optimize=False)
            out.seek(0)
            file = discord.File(out, filename="filterimage.png")
            await ctx.reply(
                f'Converted filterimage from {"absolute" if relative else "relative"} to {"relative" if relative else "absolute"}:',
                file=file
            )
        elif query == "convert":
            await ctx.reply("""Converts a filterimage to relative or absolute from the other type.
Usage:
```filterimage convert [<relative|rel|absolute|abs> <URL>]```
URL can be supplied with or without http(s) in this command, since it's not limited by colon separation.""")
        elif query.startswith("create "):
            query = query.split(" ")
            size = query[2].split(",")
            size = int(size[0]), int(size[1])
            relative = (query[1] in ("relative", "rel"))
            fil = np.zeros(size + (4,), dtype="uint8")
            fil[:, :] = (128, 128, 255, 255)
            if not relative:
                fil[:, :, 0] += np.arange(fil.shape[0], dtype="uint8")
                fil[:, :, 1] = (fil[:, :, 1].T +
                                np.arange(fil.shape[1], dtype="uint8")).T
            ifilterimage = Image.fromarray(fil)
            out = BytesIO()
            ifilterimage.save(out, format="png", optimize=False)
            out.seek(0)
            file = discord.File(out, filename="filterimage.png")
            await ctx.reply(
                f'Created filterimage template of size {size} in mode {"relative" if relative else "absolute"}:',
                file=file
            )
        elif query == "create":
            await ctx.reply("""Creates a filterimage template.
Usage:
```filterimage create [<relative|rel|absolute|abs> <sizeX>,<sizeY>]```""")
        elif query == "db" or query == "database":
            embed = discord.Embed(
                title=f"Sub-commands",
                color=discord.Color(8421631)).set_author(
                name="Filterimage Database",
                icon_url="https://cdn.discordapp.com/attachments/580445334661234692/896745220757155840/filterimageicon.png")
            embed.add_field(
                name="Add a new filterimage to the database",
                value="filterimage database register <name> <relative> <absolute> <url>",
                inline=False)
            embed.add_field(
                name="Find a filterimage in the database",
                value="filterimage database get <name>",
                inline=False)
            embed.add_field(
                name="Delete an entry from the database",
                value="filterimage database delete <name>",
                inline=False)
            embed.add_field(
                name="Search the database",
                value="filterimage database search <query>",
                inline=False)
            embed.add_field(
                name="Count filterimages from the database",
                value="filterimage database count",
                inline=False)
            await ctx.reply(embed=embed)
        elif query.startswith("db") or query.startswith("database"):
            query = query.split(" ")
            if query[1] == "register":
                if len(query) > 6:
                    await ctx.reply("ERROR: Too many arguments (wrong command / syntax / accidental space somewhere?)")
                    return
                if len(query) < 6:
                    await ctx.reply("ERROR: Not enough arguments (wrong command / syntax / forgot some arguments?)")
                    return
                name = query[2].lower()
                truthy = ("yes", "true", "1")
                relative = query[3].lower() in truthy
                absolute = query[4].lower() in truthy
                url = query[5]
                if url.startswith("http://"):
                    url = url[7:]
                if not url.startswith("https://"):
                    url = "https://" + url
                async with self.bot.db.conn.cursor() as cursor:
                    command = "SELECT name FROM filterimages WHERE url == ?;"
                    args = (url,)
                    await cursor.execute(command, args)
                    dname = await cursor.fetchone()
                    if dname:
                        await ctx.reply(f"Filterimage already exists in the filterimage database with name `{dname}`!")
                        return
                command = "INSERT INTO filterimages VALUES (?, ?, ?, ?, ?);"
                args = (name, relative, absolute, url, ctx.author.id)
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute(command, args)
                await ctx.reply(f"Success! Registered filterimage `{name}` in the filterimage database!")
            elif query[1] == "get":
                if len(query) > 3:
                    await ctx.reply("ERROR: A name can't have spaces.")
                    return
                if len(query) < 3:
                    await ctx.reply("ERROR: No name provided.")
                    return
                name = query[2].lower()
                command = "SELECT * FROM filterimages WHERE name == ?;"
                args = (name,)
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute(command, args)
                    results = await cursor.fetchone()
                    if results is None:
                        await ctx.reply(f"Could not find filterimage `{name}` in the database!")
                        return
                    name, relative, absolute, url, userid = results
                if url.startswith("http://"):
                    url = url[7:]
                if not url.startswith("https://"):
                    url = "https://" + url
                truefalseemoji = (
                    ":negative_squared_cross_mark:",
                    ":white_check_mark:")
                description = f"""(Right click to copy url!)
Relative: {truefalseemoji[int(relative)]}
Absolute: {truefalseemoji[int(absolute)]}"""
                user = await self.bot.fetch_user(userid)
                embed = discord.Embed(
                    title=f"Name: {name}",
                    color=discord.Color(8421631),
                    description=description,
                    url=url).set_image(
                    url=url).set_footer(
                    text="Filterimage Database",
                    icon_url="https://cdn.discordapp.com/attachments/580445334661234692/896745220757155840/filterimageicon.png")
                try:
                    embed.set_author(name=user.name, icon_url=user.avatar.url)
                except AttributeError:
                    embed.set_author(name="[Icon unavailable] " + user.name)
                await ctx.reply(embed=embed)
            elif query[1] == "delete":
                if len(query) > 3:
                    await ctx.reply("ERROR: A name can't have spaces.")
                    return
                if len(query) < 3:
                    await ctx.reply("ERROR: No name provided.")
                    return
                name = query[2].lower()
                command = "SELECT * FROM filterimages WHERE name == ? AND creator == ?;"
                args = (name, ctx.author.id)
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute(command, args)
                    results = await cursor.fetchone()
                    if results is None:
                        await ctx.reply(
                            f"Could not find filterimage `{name}` in the database! Does the entry exist, and did you create it?")
                        return
                command = "DELETE FROM filterimages WHERE name == ? AND creator == ?;"
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute(command, args)
                await ctx.reply("Success!")
            elif query[1] == "search":
                if len(query) > 3:
                    await ctx.reply("ERROR: A name can't have spaces.")
                    return
                if len(query) < 3:
                    command = "SELECT name FROM filterimages"
                    args = tuple()
                else:
                    command = "SELECT name FROM filterimages WHERE INSTR(name,?)<>0;"
                    name = query[2].lower()
                    args = (name,)
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute(command, args)
                    results = await cursor.fetchall()
                    if results is None:
                        await ctx.reply(f"Could not find filterimage `{name}` in the database!")
                        return
                description = '\n'.join(''.join(str(value)
                                                for value in row) for row in results)
                embed = discord.Embed(
                    title=f"Filterimage Database search results",
                    color=discord.Color(8421631),
                    description=description).set_footer(
                    text="Filterimage Database",
                    icon_url="https://cdn.discordapp.com/attachments/580445334661234692/896745220757155840/filterimageicon.png")
                await ctx.reply(embed=embed)
            elif query[1] == "count":
                async with self.bot.db.conn.cursor() as cursor:
                    await cursor.execute("SELECT COUNT(*) FROM filterimages;")
                    countall = (await cursor.fetchone())[0]
                    await cursor.execute("SELECT COUNT(*) FROM filterimages WHERE relative==1;")
                    countrelative = (await cursor.fetchone())[0]
                    await cursor.execute("SELECT COUNT(*) FROM filterimages WHERE absolute==1;")
                    countabsolute = (await cursor.fetchone())[0]
                embed = discord.Embed(
                    title=f"Filterimage Database numbers",
                    color=discord.Color(8421631)).set_footer(
                    text="Filterimage Database",
                    icon_url="https://cdn.discordapp.com/attachments/580445334661234692/896745220757155840/filterimageicon.png")
                embed.add_field(name="Total filterimages", value=int(countall))
                embed.add_field(
                    name="Relative filterimages",
                    value=int(countrelative))
                embed.add_field(
                    name="Absolute filterimages",
                    value=int(countabsolute))
                await ctx.reply(embed=embed)
        else:
            await ctx.reply("""Sub-commands:
```convert [<relative|rel|absolute|abs> <URL>]
create [<relative|rel|absolute|abs> <sizeX>,<sizeY>]
database [...]```""")


async def setup(bot: Bot):
    await bot.add_cog(GlobalCog(bot))
