from __future__ import annotations

import shutil
from glob import glob
from io import BytesIO
import io
import random
import time
import typing
import zipfile
import pathlib
import re
import json
import tomlkit
import urllib
import hashlib
from pathlib import Path

import requests
import itertools
import collections

import tomlkit.exceptions
from src import constants
from src.types import TilingMode
from typing import Any, Optional
import os
import numpy as np
import subprocess
import asyncio
import sqlite3

import discord
from discord.ext import commands
from PIL import Image, ImageChops, ImageDraw

from src.log import LOG
from ..db import TileData
from ..types import Bot, Context


class OwnerCog(commands.Cog, name="Admin", command_attrs=dict(hidden=True)):
    def __init__(self, bot: Bot):
        self.bot = bot
        self.identifies = []
        self.resumes = []
        # Are assets loading?
        self.bot.loading = False

    @commands.command()
    @commands.is_owner()
    async def danger(self, ctx: Context):
        """Toggles danger mode."""
        self.bot.config['danger_mode'] = not self.bot.config['danger_mode']
        await ctx.send(f'Toggled danger mode o{"n" if self.bot.config["danger_mode"] else "ff"}.')

    @commands.command()
    @commands.is_owner()
    async def lockdown(self, ctx: Context, *, reason: str = ''):
        """Toggles owner-only mode."""
        assert self.bot.config['owner_only_mode'][0] or len(
            reason), 'Specify a reason.'
        self.bot.config['owner_only_mode'] = [
            not self.bot.config['owner_only_mode'][0], reason]
        await ctx.send(f'Toggled lockdown mode o{"n" if self.bot.config["owner_only_mode"][0] else "ff"}.')

    @commands.command(aliases=["rekiad", "relaod", "reloa", "re;pad", "relad", "reolad", "rr"])
    @commands.is_owner()
    async def reload(self, ctx: Context, cog: str = ""):
        """Reloads extensions within the bot while the bot is running."""
        if not cog:
            extensions = [a for a in self.bot.extensions.keys()]
            await asyncio.gather(*((self.bot.reload_extension(extension)) for extension in extensions))
            await ctx.send("Reloaded all extensions.")
        elif "src.cogs." + cog in self.bot.extensions.keys():
            await self.bot.reload_extension("src.cogs." + cog)
            await ctx.send(f"Reloaded extension `{cog}` from `src/cogs/{cog}.py`.")
        else:
            await ctx.send("Unknown extension provided.")
            return None


    @commands.command(name="blacklist")
    @commands.is_owner()
    async def blacklist(self, ctx: Context, mode: str, user_id: int):
        """Set up a blacklist of users."""
        try:
            user = await self.bot.fetch_user(user_id)
        except discord.NotFound:
            return await ctx.error(f'User of id {user_id} was not found.')
        assert mode in [
            'add', 'remove'], 'Mode invalid! Has to be `add` or `remove`.'
        async with self.bot.db.conn.cursor() as cur:
            if mode == 'add':
                await cur.execute(f'''INSERT INTO blacklistedusers
                                VALUES ({user_id})''')
                return await ctx.reply(f'Added user `{user.name}#{user.discriminator}` to the blacklist.')
            else:
                await cur.execute(f'''DELETE FROM blacklistedusers
                                WHERE id={user_id}''')
                return await ctx.reply(f'Removed user `{user.name}#{user.discriminator}` from the blacklist.')
            

    @commands.command(aliases=["reboot", "rs"])
    @commands.is_owner()
    async def restart(self, ctx: Context):
        """Restarts the bot process."""
        await ctx.send("Restarting bot process...")
        await self.bot.change_presence(status=discord.Status.idle, activity=discord.Game(name="Rebooting..."))
        self.bot.exit_code = 1
        await self.bot.close()

    @commands.command(aliases=["kill", "yeet",
                               "defeat", "empty", "not", "kil", "k"])
    @commands.is_owner()
    async def logout(self, ctx: Context, endsentence: str = ""):
        """Kills the bot process."""
        if endsentence != "":  # Normally, logout doesn't trigger with arguments.
            if ctx.invoked_with == "not":
                if endsentence == "robot":  # Check if the argument is *actually* robot, making robot is not robot
                    await ctx.send("Poofing bot process...")
                    await self.bot.close()  # Trigger close before returning
            LOG.debug("Almost killed")
            return  # Doesn't close the bot if any of these logic statements is false
        elif ctx.invoked_with == "not":
            return  # Catch "robot is not"
        elif ctx.invoked_with == "yeet":
            await ctx.send("Yeeting bot process...")
        elif ctx.invoked_with == "defeat":
            await ctx.send("Processing robot is defeat...")
        elif ctx.invoked_with == "empty":
            await ctx.send("Voiding bot process...")
        elif ctx.invoked_with == "kil":
            await ctx.send("<:wah:950360195199041556>")
        else:
            await ctx.send("Killing bot process...")
        await self.bot.close()

    @commands.command()
    @commands.is_owner()
    async def leave(self, ctx: Context, guild: Optional[discord.Guild] = None):
        if guild is None:
            if ctx.guild is not None:
                await ctx.send("Bye!")
                await ctx.guild.leave()
            else:
                await ctx.send("Not possible in DMs.")
        else:
            await guild.leave()
            await ctx.send(f"Left {guild}.")

    @commands.command(hidden = True)
    async def woof(self, ctx: Context):
        """:3 (Dog)"""
        if self.bot.channel == "production":
            await ctx.reply(random.choice(["Woof!", "Bark!", "Arf arf!", "Ruff!"]))

    @commands.command(hidden = True)
    async def meow(self, ctx: Context):
        """:3 (Cat)"""
        if self.bot.channel == "staging":
            await ctx.reply(random.choice(["Mraow!", "Mrow!", "Meow!", "Purr...", "*Hiss*", "*bap*"]))

    @commands.command(hidden = True)
    async def squeak(self, ctx: Context):
        """:3 (Bunny)"""
        if self.bot.channel == "development":
            await ctx.reply(random.choice(["Squeak!", "Squeak squeak!"]))

    @commands.command(hidden = True)
    async def syncmacros(self, ctx: Context):
        """Syncs macros from production."""
        assert self.bot.channel != "production", "> you're already on prod, dumbass\n-balt"
        async with ctx.typing():
            r = requests.get('https://ric-api.sno.mba/macros.json')
            r.raise_for_status()
            data = r.json()
            macros = []
            for name, data in data.items():
                macros.append({"name": name} | data)
            async with ctx.bot.db.conn.transaction():
                await ctx.bot.db.conn.execute("DELETE FROM macros;")
                await ctx.bot.db.conn.executemany(
                    '''
                    INSERT INTO macros
                    VALUES (
                        :name,
                        :value,
                        :description,
                        :creator
                    )
                    ON CONFLICT(name) DO NOTHING;
                    ''',
                    macros
                )
            return await ctx.reply("Synchronized macros from production.")

    @commands.command()
    @commands.is_owner()
    async def hidden(self, ctx: Context):
        """Lists all hidden commands."""
        cmds = "\n".join([cmd.name for cmd in self.bot.commands if cmd.hidden])
        await ctx.author.send(f"All hidden commands:\n{cmds}")

    @commands.command(aliases=['clear', 'cls'])
    @commands.is_owner()
    async def clearconsole(self, ctx: Context):
        os.system('cls||clear')
        await ctx.send('Console cleared.')

    @commands.command(aliases=['execute', 'exec'], rest_is_raw=True)
    @commands.is_owner()
    async def run(self, ctx: Context, *, command: str):
        """Run a command from the command prompt."""
        result = subprocess.getoutput(command)
        if len(result) + 15 > 2000:
            result = result[:1982] + '...'
        await ctx.send(f'Output:\n```\n{result}```')

    @commands.command()
    @commands.is_owner()
    async def py(self, ctx: Context, *, command: str):
        """Run some python."""
        _loc = {}
        lines = ["async def func(ctx, bot):", *("    " + line for line in command.splitlines())]
        exec("\n".join(lines), locals = _loc)
        result = str(await _loc["func"](ctx, self.bot))
        if len(result) + 15 > 2000:
            result = result[:1982] + '...'
        await ctx.send(f'Output:\n```\n{result}```')

    @commands.command()
    @commands.is_owner()
    async def sql(self, ctx: Context, *, query: str):
        """Run some sql."""
        filemode = False
        if query[:3] == '-f ':
            query = query[3:]  # hardcode but whatever
            filemode = True
        query = query.replace('```', '')
        async with self.bot.db.conn.cursor() as cur:
            try:
                result = await cur.execute(query)
            except sqlite3.OperationalError as err:
                return await ctx.error(f"SQL error: {err}")
            try:
                data_rows = await result.fetchall()
                data_column_headers = np.array(
                    [column[0] for column in result.get_cursor().description])
                data_columns = np.array(data_rows, dtype=object)
                formattable_columns = np.vstack(
                    [data_column_headers, data_columns]).T
                header = '+'
                for i, column in enumerate(formattable_columns):
                    max_length = 0
                    for j, cell in enumerate(column):
                        if type(cell) is bytes:
                            column[j] = cell = f"[blob: {len(cell)} bytes]"
                        if len(str(cell)) > max_length:
                            max_length = len(str(cell))
                    for j, cell in enumerate(column):
                        column[j] = f'{str(cell):{max_length}}'.replace(
                            '\t', ' ')
                    formattable_columns[i] = column
                    header = header + '-' * max_length + '+'
                formattable_rows = formattable_columns.T
                formatted = '|' + \
                            '|'.join(formattable_rows[0]) + f'|\n{header}'
                for row in formattable_rows[1:]:
                    formatted_row = '|' + '|'.join(row) + '|'
                    if not filemode and len(row) + len(formatted) > 1800:
                        formatted = formatted + '\n...Reached character limit!'
                        break
                    formatted = formatted + '\n' + formatted_row
            except TypeError:
                return await ctx.send(f"No output.")
        if filemode:
            await ctx.send('Writing file...', delete_after=5)
            out = BytesIO()
            out.write(bytes(formatted, 'utf-8'))
            out.seek(0)
            return await ctx.send('Output:', file=discord.File(out, filename='sql-output.txt'))
        return await ctx.send(f"Output:\n```\n{formatted}\n```")


    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        webhook = await self.bot.fetch_webhook(self.bot.webhook_id)
        embed = discord.Embed(
            color=self.bot.embed_color,
            title="Joined Guild",
            description=f"Joined {guild.name}."
        )
        embed.add_field(name="ID", value=str(guild.id))
        embed.add_field(name="Member Count", value=str(guild.member_count))
        await webhook.send(embed=embed)

    @commands.command()
    @commands.is_owner()
    async def dumpletters(self, ctx: Context):
        """cba"""
        res = await self.bot.db.conn.execute("SELECT * FROM letters")
        rows = await res.fetchall()
        tomlfile = {}
        buf = io.BytesIO()
        zfile = zipfile.ZipFile(buf, "x")
        chars = {}
        for row in rows:
            mode, char, width, s0, s1, s2 = row
            charcode = f"U+{ord(char):04x}" if (
                ord(char) not in range(32, 127) or
                char in (".", "/", "\\", ":", "|", "*", "?", "<", ">")
            ) else char
            zfile.mkdir(charcode)
            if char not in chars:
                chars[char] = ({}, charcode)
            if (mode, width) not in chars[char][0]:
                chars[char][0][(mode, width)] = []
            chars[char][0][(mode, width)].append((s0, s1, s2))
        for char, (widths, charcode) in chars.items():
            tomlfile[charcode] = {"value": char}
            for (mode, width), tuples in widths.items():
                if mode not in tomlfile[charcode]:
                    tomlfile[charcode][mode] = []
                for i, (s0, s1, s2) in enumerate(tuples):
                    s = ""
                    first = True
                    while True:
                        s = chr((i % 26) + 97) + s
                        i //= 26
                        if first:
                            i -= 1
                            first = False
                        if i <= 0: break
                    tomlfile[charcode][mode].append(f"{mode}_{width}{s}.png")
                    with zfile.open(f"{charcode}/{mode}_{width}{s}.png", "w") as imbuf:
                        def fix(arr):
                            _, w = arr.shape
                            if w > width:
                                return arr[:, :width]
                            return np.pad(arr, ((0, 0), (0, width - w)))
                        f0 = fix(np.array(Image.open(io.BytesIO(s0))))
                        f1 = fix(np.array(Image.open(io.BytesIO(s1))))
                        f2 = fix(np.array(Image.open(io.BytesIO(s2))))
                        sheet = np.zeros((f0.shape[0], width * 3), dtype=np.uint8)
                        sheet[:, :width] = f0
                        sheet[:, width: width * 2] = f1
                        sheet[:, width * 2:] = f2
                        sheet = np.dstack((np.ones_like(sheet) * 255, sheet))
                        Image.fromarray(sheet, "LA").save(imbuf)
        buf.seek(0)
        tbuf = io.StringIO()
        tomlkit.dump(tomlfile, tbuf)
        tbuf.seek(0)
        await ctx.send(files=[discord.File(buf, filename="letters.zip"), discord.File(tbuf, filename="letters.toml")])



async def setup(bot: Bot):
    await bot.add_cog(OwnerCog(bot))
