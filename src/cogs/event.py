import asyncio
import datetime
import multiprocessing
import sys

import discord
from discord.ext import commands

import time

from ..types import Bot, Context

import config



class EventCog(commands.Cog, name='Events'):
    def __init__(self, bot: Bot):
        self.bot = bot

    async def bot_check(self, ctx: Context):
        async with self.bot.db.conn.cursor() as cur:
            # this is risky but guaranteed int so i think it's fine?
            await cur.execute('SELECT blacklisted FROM users WHERE user_id = ?;', ctx.author.id)
            if len(await cur.fetchall()):
                dm_channel = await self.bot.create_dm(ctx.author)
                await dm_channel.send('''You can\'t use this bot, as you have been blacklisted.
If you feel this was unjustified, please DM the bot owner.''')
                return False
        if self.bot.config['owner_only_mode'][0] and ctx.author.id != self.bot.owner_id:
            await ctx.error(f'The bot is currently in owner only mode. The owner specified this reason:\n`{config.owner_only_mode[1]}`')
            return False
        if self.bot.config['debug']:
            content = ctx.message.content[:100]
            content = "".join(c if (32 <= ord(c) < 127) | (ord(c) == 10) else f"\\x{ord(c):02X}" for c in content)
            print(f"{datetime.datetime.now()}\n{ctx.author.name}\n{content}", flush=True)
        return True

async def setup(bot: Bot):
    await bot.add_cog(EventCog(bot))
