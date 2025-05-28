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
import datetime
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
from ..tile import Tile, TileSkeleton, parse_variants

from .. import constants, errors
from ..db import CustomLevelData, LevelData
from ..types import Bot, Context, RegexDict


class FilterQuerySource(menus.ListPageSource):
    def __init__(
            self, data: list[str]):
        super().__init__(data, per_page=45)

    async def format_page(self, menu: menus.Menu, entries: list[str]) -> discord.Embed:
        embed = discord.Embed(
            title=f"{menu.current_page + 1}/{self.get_max_pages()}",
            color=menu.bot.embed_color
        ).set_footer(
            text="Filters by Charlotte",
            icon_url="https://sno.mba/assets/filter_icon.png"
        )
        while len(entries) > 0:
            field = ""
            for entry in entries[:15]:
                field += f"{entry}\n"
            embed.add_field(
                name="",
                value=field,
                inline=True
            )
            del entries[:15]
        return embed

class FilterCog(commands.Cog, name="Filters"):
    def __init__(self, bot: Bot):
        self.bot = bot

    # Check if the bot is loading
    async def cog_check(self, ctx):
        """Only if the bot is not loading assets."""
        return not self.bot.loading

    @commands.group(aliases=["fi"], pass_context=True, invoke_without_command=True)
    async def filter(self, ctx: Context):
        """Performs filter-related actions like template creation, conversion and accessing the database.
Filters are formatted as follows:
- R: X offset of pixel's UV (-128 to 127)
- G: Y offset of pixel's UV (-128 to 127)
- B: Brightness of pixel (0 to 255)
- A: Alpha of pixel (0 to 255)"""
        await ctx.invoke(ctx.bot.get_command("cmds"), "filter")

    @filter.command(aliases=["cv"])
    async def convert(self, ctx: Context, target_mode: Literal["abs", "absolute", "rel", "relative"]):
        """Converts a filter to its opposing mode. An attachment with the filter is required."""
        # Get the attached image, or throw an error
        try:
            filter_url = ctx.message.attachments[0].url
        except IndexError:
            return await ctx.error("The filter to be converted wasn't attached.")
        filter_headers = requests.head(filter_url, timeout=3).headers
        assert int(filter_headers.get("content-length", 0)) < constants.FILTER_MAX_SIZE, f"Filter is too big!"
        with Image.open(requests.get(filter_url, stream=True).raw) as im:
            assert im.width <= 256 and im.height <= 256, "Can't create a filter greater than 256 pixels on either side!"
            fil = np.array(im.convert("RGBA"), dtype=np.uint8)
        fil[..., :2] += np.indices(fil.shape[1::-1]).astype(np.uint8).T * np.uint8(
            1 if target_mode.startswith("abs") else -1)
        out = BytesIO()
        Image.fromarray(fil).save(out, format="png", optimize=False)
        out.seek(0)
        filename = f"{Path(ctx.message.attachments[0].filename).stem}-{target_mode}.png"
        file = discord.File(out, filename=filename)
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Converted!",
            description=f'Converted filter to {target_mode}.'
        ).set_footer(
            text="Filters by Charlotte",
            icon_url="https://sno.mba/assets/filter_icon.png"
        ).set_image(url=f"attachment://{filename}")
        await ctx.reply(embed=emb, file=file)

    @filter.command(aliases=["t"])
    async def template(self, ctx: Context, target_mode: Literal["abs", "absolute", "rel", "relative"], width: int,
                     height: int):
        """Creates a template filter."""
        assert width > 0 and height > 0, "Can't create a filter with a non-positive area!"
        assert width <= 256 and height <= 256, "Can't create a filter greater than 256 pixels on either side!"
        size = (height, width)
        fil = np.ones((*size, 4), dtype=np.uint8) * 0xFF
        fil[..., :2] -= 0x7F
        fil[..., :2] += np.indices(fil.shape[1::-1], dtype=np.uint8).T * target_mode.startswith("abs")
        out = BytesIO()
        Image.fromarray(fil).save(out, format="png", optimize=False)
        out.seek(0)
        filename = f"filter-{size[1]}x{size[0]}-{target_mode}.png"
        file = discord.File(out, filename=filename)
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Created!",
            description=f'Created filter template of size {size} in mode {target_mode}.'
        ).set_footer(
            text="Filters by Charlotte",
            icon_url="https://sno.mba/assets/filter_icon.png"
        ).set_image(url=f"attachment://{filename}")
        await ctx.reply(embed=emb, file=file)

    @filter.command(aliases=["mk", "make"])
    async def create(
        self,
        ctx: Context,
        name: str,
        target_mode: Literal["abs", "absolute", "rel", "relative"]
    ):
        """Adds a filter to the database from an attached image."""
        try:
            filter_url = ctx.message.attachments[0].url
        except IndexError:
            return await ctx.error("The image to add as a filter wasn't attached.")

        filter_headers = requests.head(filter_url, timeout=3).headers
        assert int(filter_headers.get("content-length", 0)) < constants.FILTER_MAX_SIZE, f"Filter is too big! Filters must be at most `{constants.FILTER_MAX_SIZE}` bytes."
        im = Image.open(requests.get(filter_url, stream=True).raw)
        width, height = im.size
        if width > constants.MAX_TILE_SIZE or height > constants.MAX_TILE_SIZE:
            return await ctx.error(f"The given filter is too large! Images must be at max `{constants.MAX_TILE_SIZE}` pixels on either side.")
        buf = BytesIO()
        im.save(buf, format = "PNG")
        buf.seek(0)

        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM filters WHERE name like ?", name)
            dname = await cursor.fetchone()
            if dname is not None:
                return await ctx.error(f"Filter of name `{name}` already exists in the database!")
            command = "INSERT INTO filters VALUES (?, ?, ?, ?, ?);"
            timestamp = int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)
            args = (name, target_mode.startswith("abs"), ctx.author.id, timestamp, buf.getvalue())
            await cursor.execute(command, args)
            emb = discord.Embed(
                color=ctx.bot.embed_color,
                title="Registered!",
                description=f'Registered filter `{name}` in the filter database!\n\n-# Do not upload illegal content. You *will* be blacklisted.'
            ).set_footer(
                text="Filters by Charlotte",
                icon_url="https://sno.mba/assets/filter_icon.png"
            )
            await ctx.reply(embed=emb)

    @filter.command(aliases=["i"])
    async def info(self, ctx: Context, name: str):
        """Gets information about a filter."""
        attrs = await self.bot.db.get_filter(name)
        if attrs is None:
            return await ctx.error(f"Filter of name `{name}` isn't in the database!")
        absolute, author, upload_time, data = attrs
        buf = BytesIO()
        data.save(buf, format = "PNG")
        buf.seek(0)
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title=name,
            description=f'Mode: `{"absolute" if absolute else "relative"}`\nUpload date: ' + ("???" if upload_time is None else f"<t:{int(upload_time)}>")
        ).set_footer(
            text="Filters by Charlotte",
            icon_url="https://sno.mba/assets/filter_icon.png"
        ).set_image(url="attachment://filter.png")
        user = await ctx.bot.fetch_user(author)
        emb.set_author(
            name=f"{user.name}",
            icon_url=user.avatar.url if user.avatar is not None else
            f"https://cdn.discordapp.com/embed/avatars/0.png"
        )
        await ctx.reply(embed=emb, file = discord.File(buf, filename="filter.png"))

    @filter.command(aliases=["del", "remove", "rm"])
    async def delete(self, ctx: Context, name: str):
        """Removes a filter from the database. You must have made it to do this."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute(
                f"SELECT name FROM filters WHERE name == ?{'' if await ctx.bot.is_owner(ctx.author) else ' AND creator == ?'};",
                (name,) if await ctx.bot.is_owner(ctx.author) else (name, ctx.author.id))
            url = (await cursor.fetchone())
            assert url is not None, f"The filter `{name}` doesn't exist, or you don't have permission to remove it!"
            await cursor.execute(f"DELETE FROM filters WHERE name == ?;", name)
            if name in self.bot.db.filter_cache:
                del self.bot.db.filter_cache[name]
            emb = discord.Embed(
                color=ctx.bot.embed_color,
                title="Deleted!",
                description=f"Removed the filter {name} from the database."
            ).set_footer(
                text="Filters by Charlotte",
                icon_url="https://sno.mba/assets/filter_icon.png"
            )
            await ctx.reply(embed=emb)

    @filter.command(aliases=["?", "query", "find", "list"])
    async def search(self, ctx: Context, pattern: str = ".*"):
        """Lists filters that match a regular expression."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM filters WHERE name REGEXP ?", pattern)
            names = [row[0] for row in await cursor.fetchall()]
        return await ButtonPages(FilterQuerySource(sorted(names))).start(ctx)

    @filter.command(aliases=["#"])
    async def count(self, ctx: Context):
        """Gets the amount of filters in the database."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM filters;")
            count = (await cursor.fetchone())[0]
            await cursor.execute("SELECT COUNT(*) FROM filters WHERE absolute == 1;")
            count_abs = (await cursor.fetchone())[0]
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Stats",
            description=f"There are {count} filters in the database, {count_abs} absolute and {count - count_abs} relative."
        ).set_footer(
            text="Filters by Charlotte",
            icon_url="https://sno.mba/assets/filter_icon.png"
        )
        await ctx.reply(embed=emb)



async def setup(bot: Bot):
    await bot.add_cog(FilterCog(bot))
