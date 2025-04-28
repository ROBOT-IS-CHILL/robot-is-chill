import inspect
import io
import signal
from datetime import datetime
from typing import Literal
import warnings

import discord
from discord import Member, User
from discord.ext import commands, menus

from .. import constants
from ..types import Bot, Context, TextMacro, BuiltinMacro
from ..utils import ButtonPages

import re


async def coro_part(func, *args, **kwargs):
    async def wrapper():
        result = func(*args, **kwargs)
        return await result

    return wrapper


async def start_timeout(fn, *args, **kwargs):
    def handler(_signum, _frame):
        raise AssertionError("The command took too long and was timed out.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(constants.TIMEOUT_DURATION))
    return await fn(*args, **kwargs)


class MacroQuerySource(menus.ListPageSource):
    def __init__(
            self, bot, data: list[str]):
        self.count = len(data)
        self.bot = bot
        super().__init__(data, per_page=15)

    async def format_page(self, menu: menus.Menu, entries: list[str]) -> discord.Embed:
        embed = discord.Embed(
            title="Search results",
            # color=menu.bot.embed_color  I think the theme color suits it better.
        ).set_footer(
            text=f"Page {menu.current_page + 1} of {self.get_max_pages()}   ({self.count} entries)",
        )
        while len(entries) > 0:
            field = ""
            for entry in entries[:5]:
                if entry not in self.bot.macros:
                    entry = f":gear: {entry}"
                elif self.bot.macros[entry].forest is None:
                    entry = f":warning: {entry}"
                field += f"{entry[:50]}\n"
            embed.add_field(
                name="",
                value=field,
            )
            del entries[:5]
        return embed


class MacroCommandCog(commands.Cog, name='Macros'):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.group(aliases=["m", "macros"], pass_context=True, invoke_without_command=True)
    async def macro(self, ctx: Context):
        """Front-end for letting users (that means you!) create, edit, and remove macros."""
        await ctx.invoke(ctx.bot.get_command("cmds"), "macro")

    @macro.command(aliases=["r"])
    @commands.is_owner()
    async def refresh(self, ctx: Context):
        """Refreshes the macro database."""
        self.bot.macros = {}
        await self.bot.load_macros()
        return await ctx.reply("Refreshed database.")

    @macro.command()
    @commands.is_owner()
    async def chown(self, ctx: Context, source: Member | User, dest: Member | User):
        """Changes the owner of all macros by a user to another user. Use sparingly."""
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute("""
                UPDATE macros
                SET creator = ?
                WHERE creator == ?
            """, dest.id, source.id)
        return await ctx.reply(f"Done. Moved all macros from account {source} to account {dest}.")

    def parse_macro_inputs(self, inputs: str) -> tuple[list[str], bool]:
        varargs = False
        input_list = []
        if not inputs:
            return ([], False)
        for input in inputs.split(","):
            input = input.replace("`", "").replace("\n", "").strip()
            assert re.fullmatch(r"\*?[A-Za-z_][0-9A-Za-z_]*", input) is not None, f"Argument `{input}` has an invalid name!\nNames must be alphanumeric, including underscores, and cannot start with a digit."
            assert not varargs, f"Variable inputs must be the last input in the list! Try removing `{input}` and after."
            if input.startswith("*"):
                varargs = True
                input = input[1:]
            input_list.append(input)
        return (input_list, varargs)

    def validate_name(self, name: str, check_exists: bool = True):
        assert len(name) <= 50, "Macro name cannot be longer than 50 characters!"
        assert all([c not in name for c in "[]/ :;\"\'"]), "Name uses invalid characters (`[]/ :;\"\'`)!"
        if check_exists:
            assert name not in self.bot.macros, f"Macro `{name}` already exists in the database!"
            assert (name not in self.bot.macro_handler.builtins),\
                f"Macro name `{name}` is reserved for a builtin!"
    
    def clean_input(self, input: str):
        input = input.strip()
        input = re.sub("^```[A-Za-z0-9_]*", "", input, 1)
        input = input.removesuffix("```")
        input = input.strip()
        return input

    @macro.command(aliases=["mk", "make"])
    async def create(self, ctx: Context, name: str, description: str, inputs: str = "", *, source: str):
        """
        Adds a macro to the database.
        
        The `inputs` parameter contains a comma-separated list of the variables to set
        the macro's arguments to when calling it. Preceding an argument name with `*` makes it
        collect any arguments past it into a list. 

        Input variable names must be alphanumeric (including underscores).

        Note that each string argument can be surrounded by "double quotes" to allow spaces,
        and triple backticks are removed from the start and end of source code,
        meaning this is a valid usage of this command:

        =macro create sample "Lorem ipsum dolor sit amet." "arg1, arg2, *args" \`\`\` ?{arg1} ?{arg2} [unpack/?{args}] \`\`\`
        """
        self.validate_name(name)
        inputs, varargs = self.parse_macro_inputs(inputs)
        source = self.clean_input(source)
        forest = self.bot.macro_handler.parse_forest(source)

        async with self.bot.db.conn.cursor() as cursor:
            self.bot.macros[name] = TextMacro(name, description, source, inputs, varargs, ctx.author.id, forest)
            await cursor.execute(
                "INSERT INTO macros VALUES (?, ?, ?, ?, ?, ?);",
                (name, source, description, ctx.author.id, "\t".join(inputs), varargs)
            )
            return await ctx.reply(f"Successfully added `{name}` to the database!")

    async def check_macro_ownership(self, ctx: Context, name: str, action: str):
        async with self.bot.db.conn.cursor() as cursor:
            if not await ctx.bot.is_owner(ctx.author):
                await cursor.execute("SELECT creator FROM macros WHERE name == ?", name)
                row = await cursor.fetchone()
                assert row is not None, f"Macro `{name}` doesn't exist in the database!\n-# _Note: Builtin macros are not stored in the database._"
                id = row[0]
                try:
                    await self.bot.fetch_user(id)
                except discord.NotFound:
                    raise AssertionError(
                        f"The user who created this macro (`{id}`) has been deleted.\n"
                        "If you own this macro, reach out to the bot owner to have them migrate it to your account."
                    )
                assert id == ctx.author.id, f"You can't {action} a macro you don't own."

    @macro.command(aliases=["e"])
    async def edit(self, ctx: Context, name: str, attribute: Literal["source", "description", "name", "inputs"], *, new: str):
        """Edits a macro. You must own said macro to edit it."""
        await self.check_macro_ownership(ctx, name, "edit")
        if attribute == "source":
            new = self.clean_input(new)
            forest = self.bot.macro_handler.parse_forest(new)
            async with self.bot.db.conn.cursor() as cursor:
                await cursor.execute(f"UPDATE macros SET value = ? WHERE name == ?", new, name)
                self.bot.macros[name].source = new
                self.bot.macros[name].forest = forest
            return await ctx.reply(f"Successfully edited `{name}`'s source.")
        elif attribute == "inputs":
            inputs, varargs = self.parse_macro_inputs(new)
            print(inputs, varargs)
            async with self.bot.db.conn.cursor() as cursor:
                await cursor.execute(f"UPDATE macros SET inputs = ?, varargs = ? WHERE name == ?", "\t".join(inputs), varargs, name)
                self.bot.macros[name].inputs = inputs
                self.bot.macros[name].varargs = varargs
            return await ctx.reply(f"Successfully edited `{name}`'s inputs.")
        elif attribute == "description":
            async with self.bot.db.conn.cursor() as cursor:
                await cursor.execute(f"UPDATE macros SET description = ? WHERE name == ?", new, name)
                self.bot.macros[name].source = new
                self.bot.macros[name].forest = forest
        elif attribute == "name":
            self.validate_name(new)
            async with self.bot.db.conn.cursor() as cursor:
                await cursor.execute(f"UPDATE macros SET name = ? WHERE name == ?", new, name)
                macro = self.bot.macros[name]
                del self.bot.macros[name]
                self.bot.macros[new] = macro
            
        return await ctx.reply(f"Successfully `{name}`'s {attribute} to be `{new}`.")

    @macro.command(aliases=["rm", "remove", "del"])
    async def delete(self, ctx: Context, name: str):
        """Deletes a macro. You must own said macro to delete it."""
        await self.check_macro_ownership(ctx, name, "delete")
        self.validate_name(name, check_exists = False)
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute(f"DELETE FROM macros WHERE name == ?", name)
        del self.bot.macros[name]
        return await ctx.reply(f"Deleted macro `{name}`.")

    @macro.command(aliases=["?", "list", "query"])
    async def macro_search(self, ctx: Context, *, pattern: str = '.*'):
        """Searches the database for macros by name."""
        author = None
        if match := re.search(r"--?a(?:uthor)?=(\S+)", pattern):
            author = match.group(1)
            # HACK: this sucks
            try:
                author = (await commands.UserConverter().convert(ctx, author)).id
            except commands.errors.UserNotFound:
                try:
                    author = (await commands.MemberConverter().convert(ctx, author)).id
                except:
                    try:
                        author = int(author)
                    except:
                        return await ctx.error(f"Could not convert `{author.replace('`', '')[:32]}` to a user or user ID!")
            pattern = pattern[:match.start()] + pattern[match.end():]
        
        only_broken = False
        if match := re.search(r"--broken|-!", pattern):
            only_broken = True
            pattern = pattern[:match.start()] + pattern[match.end():]
        
        only_builtins = False
        names = []
        if match := re.search(r"--builtin|-b", pattern):
            only_builtins = True
            pattern = pattern[:match.start()] + pattern[match.end():]

        pattern = pattern.strip()
        if not only_builtins:
            async with self.bot.db.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT name FROM macros 
                    WHERE name REGEXP :name
                    AND (
                        :author IS NULL OR creator == :author
                    )
                    """,
                    {
                        "name": pattern,
                        "author": author
                    }
                )
                names = [name for (name,) in await cursor.fetchall()]
            if only_broken:
                names = [name for name in names if self.bot.macros[name].forest is None]
        names.extend(name for name in self.bot.macro_handler.builtins if re.fullmatch(pattern, name) is not None)
        return await ButtonPages(MacroQuerySource(self.bot, sorted(names))).start(ctx)


    @macro.command(aliases=["x", "run"])
    async def execute(self, ctx: Context, *, macro: str):
        """Executes some given macro tree and outputs its return value."""
        try:
            async def exec():
                nonlocal macro
                macro = self.clean_input(macro)
                parsed = ctx.bot.macro_handler.parse_forest(macro.strip())
                return await ctx.bot.macro_handler.evaluate_forest(parsed, vars = {"_CONTEXT": "x"})

            output = await start_timeout(exec)
            output = "" if output is None else str(output)
            output = output.strip()

            message, files = "", []

            if len(output) > 1900:
                out = io.BytesIO()
                out.write(bytes(output, 'utf-8'))
                out.seek(0)
                files.append(discord.File(out, filename=f'output-{datetime.now().isoformat()}.txt'))
                message = 'Output:'
            else:
                sanitized_output = output.strip().replace("```", "'''")
                message = f'Output: ```\n{sanitized_output}\n```'
            return await ctx.reply(message, files=files)
        finally:
            signal.alarm(0)

    @macro.command(aliases=["i", "get"])
    async def info(self, ctx: Context, name: str):
        """Gets info about a specific macro."""
        macro = None
        source = None
        author = ctx.bot.user.id
        if name in self.bot.macro_handler.builtins:
            macro = self.bot.macro_handler.builtins[name]
            source_function = macro.function
            if hasattr(source_function, "_source"):
                source_function = source_function._source
            try:
                source = re.sub(r'""".*?"""\n\s+', '', inspect.getsource(source_function), 1, re.S)
                source = f"```py\n{source}\n```"
            except OSError as err:
                source = f"```\n<failed to get source code of builtin: {err}>\n```"
        else:
            assert name in self.bot.macros, f"Macro `{name}` isn't in the database!"
            macro: TextMacro = self.bot.macros[name]
            author = macro.author
            sanitized_source = macro.source.strip().replace('```', "'''")
            source = f"```bf\n{sanitized_source}\n```"
        emb = discord.Embed(
            title=macro.name
        )
        emb.add_field(
            name="",
            value=macro.description
        )
        if isinstance(macro, TextMacro):
            if len(macro.inputs): 
                inputs = ", ".join(macro.inputs[:-1]) + ", " f"*{macro.inputs[-1]}" if macro.variable_args else macro.inputs[-1]
                emb.add_field(
                    name="Inputs",
                    value=inputs,
                    inline=False
                )
            if macro.forest is None:
                emb.add_field(
                    name="__⚠️ This macro is broken. ⚠️__",
                    value=f"**Error:**\n```\n{macro.failure}\n```",
                    inline = False
                )
        emb.add_field(
            name="Source",
            value=source,
            inline=False
        )
        try:
            user = await ctx.bot.fetch_user(author)
            emb.set_footer(
                text=user.name,
                icon_url=user.avatar.url if user.avatar is not None else
                    f"https://cdn.discordapp.com/embed/avatars/{hash(user.name) % 5}.png"
                )
        except discord.NotFound:
            emb.set_footer(text=f"The author of this macro has been deleted.")
        try:
            await ctx.reply(embed=emb)
        except discord.errors.HTTPException:
            emb.set_field_at(
                1,
                name="Source",
                value=f"_Source too long to embed. It has been attached as a text file._",
                inline=False
            )
            buf = io.BytesIO()
            buf.write(source.encode("utf-8", "ignore"))
            buf.seek(0)
            await ctx.reply(embed=emb, file=discord.File(buf, filename=f"{name}-source.txt"))


async def setup(bot: Bot):
    await bot.add_cog(MacroCommandCog(bot))
