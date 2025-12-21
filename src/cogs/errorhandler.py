from __future__ import annotations

import asyncio
import os
import re
import signal
import sqlite3
import sys
import traceback
from random import random
import io
from datetime import datetime

import pyo3_runtime
import numpy
import discord
from discord.ext import commands
import requests

import webhooks
from ..types import Bot, Context
from .. import errors, constants
import macrosia_glue

class DummyLogger:
    async def send(self, *args, **kwargs): pass

class CommandErrorHandler(commands.Cog):
    def __init__(self, bot: Bot):
        self.bot = bot
        self.webhook_id = bot.webhook_id
        self.logger = None

    async def setup_logger(self, webhook_id: int):
        try:
            return await self.bot.fetch_webhook(webhook_id)
        except:
            return DummyLogger()

    @commands.Cog.listener()
    async def on_command_error(self, ctx: Context, error: Exception):
        """The event triggered when an error is raised while invoking a
        command.

        ctx   : Context
        error : Exception
        """
        self.bot.loading = False
        try:
            if self.logger is None:
                self.logger = await self.setup_logger(webhooks.error_id)

            # This prevents any commands with local handlers being handled here
            # in on_command_error.
            if hasattr(ctx.command, 'on_error'):
                return

            ignored = (
                commands.CommandNotFound,
                commands.NotOwner,
                commands.CheckFailure)
            whitelist = ()

            # Allows us to check for original exceptions raised and sent to CommandInvokeError.
            # If nothing is found. We keep the exception passed to
            # on_command_error.
            error = getattr(error, 'original', error)

            # Constructs the error embed for logging
            emb = discord.Embed(title="Command Error", color=0xffff00)
            emb.description = str(error)

            # Adds embed fields
            # Bot
            if self.bot.user:  # tautology but fits the scheme
                ID = self.bot.user.id
                name = self.bot.user.display_name
            # Message
            if ctx.message:
                ID = ctx.message.id
                content = ctx.message.content
                if len(
                        content) > 1024:  # Only use the first 1000 characters, avoid 1024 char value limits
                    content = content[1000] + "`...`"
                formatted = f"ID: {ID}\nContent: `{content}`"
                emb.add_field(name="Message", value=formatted)
            # Channel
            if isinstance(ctx.channel, discord.TextChannel):
                ID = ctx.channel.id
                name = ctx.channel.name
                nsfw = "[NSFW Channel]" if ctx.channel.is_nsfw() else ""
                news = "[News Channel]" if ctx.channel.is_news() else ""
                formatted = f"ID: {ID}\nName: {name}\n{nsfw} {news}"
                emb.add_field(name="Channel", value=formatted)
            # Guild (if in a guild)
            if ctx.guild is not None:
                ID = ctx.guild.id
                name = ctx.guild.name
                member_count = ctx.guild.member_count
                formatted = f"ID: {ID}\nName: {name}\nMember count: {member_count}"
                emb.add_field(name="Guild", value=formatted)
            # Author (DM information if any)
            if ctx.author:
                ID = ctx.author.id
                name = ctx.author.name
                discriminator = ctx.author.discriminator
                nick = f"({ctx.author.nick})" if ctx.guild else ""
                DM = "Message Author" if ctx.guild else "Direct Message"
                formatted = f"ID: {ID}\nName: {name}#{discriminator} ({nick})"
                emb.add_field(name=DM, value=formatted)
            # Message link
            if all([ctx.guild is not None, ctx.channel, ctx.message]):
                guild_ID = ctx.guild.id
                channel_ID = ctx.channel.id
                message_ID = ctx.message.id
                formatted = f"[Jump to message](https://discordapp.com/channels/{guild_ID}/{channel_ID}/{message_ID})"
                emb.add_field(name="Jump", value=formatted)

            # Anything in ignored will return and prevent anything happening.
            if isinstance(error, whitelist):
                pass
            elif isinstance(error, ignored):
                return

            elif isinstance(error, errors.Porp):
                return await ctx.error(f':porp', file=discord.File("data/misc/porp.jpg"))

            if isinstance(error, commands.CommandOnCooldown):
                if ctx.author.id == self.bot.owner_id:
                    return await ctx.reinvoke()
                else:
                    await ctx.error(str(error))
                    return await self.logger.send(embed=emb)

            elif isinstance(error, commands.DisabledCommand):
                await ctx.error(f'{ctx.command} has been disabled.')
                return await self.logger.send(embed=emb)

            elif isinstance(error, commands.NoPrivateMessage):
                try:
                    await ctx.author.send(f'{ctx.command} can not be used in Private Messages.')
                except BaseException:
                    emb.add_field(
                        name="Notes",
                        value="Could not send private messages to user.")
                return await self.logger.send(embed=emb)

            elif isinstance(error, commands.ExpectedClosingQuoteError):
                return await ctx.error(f"Expected closing quotation mark `{error.close_quote}`.")

            elif isinstance(error, commands.InvalidEndOfQuotedStringError):
                return await ctx.error(f"Expected a space after a quoted string, got `{error.char}` instead.")

            elif isinstance(error, commands.UnexpectedQuoteError):
                return await ctx.error(f"Got unexpected quotation mark `{error.quote}` inside a string.")

            elif isinstance(error, commands.ConversionError):
                await self.logger.send(embed=emb)
                return await ctx.error("Invalid function arguments provided. Check the help command for the proper format.")

            elif isinstance(error, AssertionError) or isinstance(error, NotImplementedError):
                await self.logger.send(embed=emb)
                if len(error.args) == 0 or len(error.args[0]) == 0:
                    raise error
                if len(error.args[0]) == 0:
                    raise
                return await ctx.error(error.args[0])

            elif isinstance(error, errors.UnknownVariant):
                return await ctx.error(
                    f"The variant `{error.args[0]}` doesn't exist or is malformed."
                )

            elif isinstance(error, macrosia_glue.PanicException) or isinstance(error, macrosia_glue.RustPanic):
                buf = io.StringIO()
                buf.write("Error occurred at {}\n".format(datetime.utcnow().strftime(f"%Y-%m-%d %H:%M:%S UTC")))
                buf.write("Message content:\n")
                buf.write(ctx.message.content)
                buf.write("\n\nBacktrace:\n")
                buf.write(f"{error}")
                buf.seek(0)
                buf.truncate(8 * 1000 * 1000)

                await ctx.error(
                    "PyO3 binding panicked!\n"\
                    "This is a __critical bug__, and should be reported to the developers as soon as possible.\n"\
                    "Attached is a backtrace of the panic.\n"\
                    "Please send this file to the developers.\n"\
                    "The bot will be restarted to prevent data corruption.",
                    file=discord.File(buf, filename = "traceback.log")
                )
                self.bot.exit_code = 1
                return await self.bot.close()

            elif isinstance(error, commands.BadArgument):
                await self.logger.send(embed=emb)
                return await ctx.error(f"Invalid argument provided. Check the help command for the proper format.")

            elif isinstance(error, commands.ArgumentParsingError):
                await self.logger.send(embed=emb)
                return await ctx.error("Invalid function arguments provided.")

            elif isinstance(error, commands.MissingRequiredArgument):
                return await ctx.error(f"Required argument {error.param} is missing.")

            elif isinstance(error, discord.errors.HTTPException):
                if error.status == 400:
                    return await ctx.error(f"This action could not be performed.\n`{error}`")
                if error.status == 429:
                    return await ctx.error("We're being ratelimited. Try again later.")
                if error.status == 401:
                    return await ctx.error("This action cannot be performed.")
                if error.status == 503:
                    return await ctx.error("HTTP server unavailable.")
                return await ctx.error(f"There was an error while processing this action.\n> {error}")

            elif isinstance(error, numpy.linalg.LinAlgError):
                return await ctx.error("The given warp points are unsolvable.")
            elif isinstance(error, requests.exceptions.ConnectionError):
                return await ctx.error('A given link for the filterimage was invalid.')
            elif isinstance(error, errors.OverlayNotFound):
                return await ctx.error(f'The overlay `{error}` does not exist.')
            elif isinstance(error, errors.InvalidFlagError):
                return await ctx.error(f'A flag failed to parse:\n> `{error}`')
            elif isinstance(error, errors.TimeoutError):
                if random() < 0.01:
                    return await ctx.error("The command was `       TAKING TOO LONG` and was timed out.")
                return await ctx.error("The command took too long and was timed out.")
            elif isinstance(error, errors.MacroError):
                if error.args[1] is None:
                    return await ctx.error(f'Macro execution failed: {error.args[0]}')
                buf = io.StringIO()
                buf.write("-----\n")
                for traceback_frame in reversed(error.args[1]):
                    buf.write(traceback_frame)
                    buf.write("\n-----\n")
                buf.seek(0)
                buf.truncate(8 * 1000 * 1000)
                if len(buf.getvalue()) < 1024:
                    val = buf.getvalue().replace('`', '\'')
                    return await ctx.error(
                        f'Macro execution failed: {error.args[0]}\n'\
                        f"Traceback: ```\n{val}\n```"
                    )
                return await ctx.error(
                    f'Macro execution failed: {error.args[0]}\n'\
                    'Traceback:',
                    file=discord.File(buf, filename=datetime.utcnow().strftime(f"%Y-%m-%d-%H.%M.%S-macro-tb.log"))
                )
            elif isinstance(error, errors.NoPaletteError):
                palette = error.args[0]
                if palette[1] is None:
                    return await ctx.error(f"Palette `{palette[0].replace('`', '')[:32]}` does not exist!")
                return await ctx.error(f"Palette `{palette[1].replace('`', '')[:32]}/{palette[0].replace('`', '')[:32]}` does not exist!")
            elif isinstance(error, commands.BadLiteralArgument):
                return await ctx.error(f"An argument for the command wasn't in the allowed values of `{', '.join(repr(o) for o in error.literals)}`.")
            elif isinstance(error, re.error):
                return await ctx.error(f"The regular expression `{error.pattern}` is invalid. `{error}`")
            # All other Errors not returned come here... And we can just print
            # the default TraceBack + log
            if os.name == "nt":
                trace = '\n'.join(
                    traceback.format_tb(
                        error.__traceback__)).replace(
                    os.getcwd(),
                    os.path.curdir).replace(
                    os.environ["USERPROFILE"],
                    "")
            else:
                trace = '\n'.join(
                    traceback.format_tb(
                        error.__traceback__)).replace(
                    os.getcwd(),
                    os.path.curdir)
            if len(trace) > 1000:
                trace = trace[:500] + "\n\n...\n\n" + trace[-500:] 
            title = f'**Unhandled exception!**'
            err_desc = str(error)
            if len(err_desc) > 500:
                err_desc = err_desc[:250] + "..." + err_desc[-250:]
            emb = discord.Embed(
                title=title,
                description=(f"## {type(error).__name__}\n{err_desc}\n```\n{trace}\n```"),
                color=15029051
            )
            await self.logger.send(embed=emb)
            await ctx.error(msg='', embed=emb)
            print(
                f'Ignoring exception in command {ctx.command}:',
                file=sys.stderr)
            traceback.print_exception(
                type(error),
                error,
                error.__traceback__,
                file=sys.stderr)
        except Exception as err:
            try:
                title = f'**Error in error handler!!!**'
                if len(title) > 32:
                    title = title[:32]
                if os.name == "nt":
                    trace = '\n'.join(
                        traceback.format_tb(
                            err.__traceback__)).replace(
                        os.getcwd(),
                        os.path.curdir).replace(
                        os.environ["USERPROFILE"],
                        "")
                else:
                    trace = '\n'.join(
                        traceback.format_tb(
                            err.__traceback__)).replace(
                        os.getcwd(),
                        os.path.curdir)
                if len(trace) > 1000:
                    trace = trace[:500] + "\n\n...\n\n" + trace[-500:] 
                err_desc = str(error)
                if len(err_desc) > 500:
                    err_desc = err_desc[:250] + "..." + err_desc[-250:]
                emb = discord.Embed(
                    title=title,
                    description=(f"## {type(error).__name__}\n{err_desc}\n```\n{trace}\n```"),
                    color=0xFF0000
                )
                await ctx.error(msg='', embed=emb)
                print("--- ERROR HANDLER ERROR ---")
                traceback.print_exception(
                    type(error),
                    error,
                    error.__traceback__,
                    file=sys.stderr)
                print("--- ERROR HANDLER ERROR CRASH ---")
                traceback.print_exception(
                    type(err),
                    err,
                    err.__traceback__,
                    file=sys.stderr)
                print("-------------")
                return
            except:
                await ctx.error(msg="Error handler fatally errored. Contact the bot owner as soon as possible.")
        finally:
            signal.alarm(0)


async def setup(bot: Bot):
    await bot.add_cog(CommandErrorHandler(bot))
