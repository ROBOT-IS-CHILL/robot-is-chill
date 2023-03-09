from typing import Literal

import discord
from discord.ext import commands, menus

from ..types import Bot, Context, Macro
from ..utils import ButtonPages

import re

class MacroQuerySource(menus.ListPageSource):
    def __init__(
            self, data: list[str]):
        super().__init__(data, per_page=45)

    async def format_page(self, menu: menus.Menu, entries: list[str]) -> discord.Embed:
        embed = discord.Embed(
            title="Search results",
            # color=menu.bot.embed_color  I think the theme color suits it better.
        ).set_footer(
            text=f"Page {menu.current_page + 1} of {self.get_max_pages()}",
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

class MacroCog(commands.Cog, name='Macros'):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.group(aliases=["m"], pass_context=True, invoke_without_command=True)
    async def macro(self, ctx: Context):
        """Front-end for letting users (that means you!) create, edit, and remove variant macros.
    Macros are simply a way of aliasing one or more variants to one name.
    For example, if a macro called `face` with the value `csel-1` exists,
    rendering `baba:m!face` would actually render `baba:csel-1`.
    Arguments can be specified in macros with $#. As an example,
    `transpose` aliased to `rot$1:scale$2` would mean that
    rendering `baba:m!transpose/45/2` would give you `baba:rot45:scale2`.
    Important to note, double negatives are valid inputs to variants, so
    something like `baba:scale--2` would give the same as `baba:scale2`."""
        await ctx.error("Invalid subcommand specified! Use `commands macros` to see what subcommmands there are.")

    @macro.command(aliases=["r"])
    @commands.is_owner()
    async def refresh(self, ctx: Context):
        """Refreshes the macro database."""
        self.bot.macros = {}
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute("SELECT * from macros")
            for (name, value, description, author) in await cur.fetchall():
                self.bot.macros[name] = Macro(value, description, author)
        return await ctx.reply("Refreshed database.")

    @macro.command(aliases=["mk", "make"])
    async def create(self, ctx: Context, name: str, value: str, *, description: str = None):
        """Adds a macro to the database."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT value FROM macros WHERE name == ?", name)
            dname = await cursor.fetchone()
            if dname is not None:
                return await ctx.error(
                    f"Macro of name `{name}` already exists in the database!")
            assert ";" not in value, "Sorry, but macros can't have persistent variants in them."
            assert "/" not in name, "A macro's name can't have `/` in it, as it'd clash with parsing arguments."
            # Call the user out on not adding a description, hopefully making them want to add a good one
            assert description is not None, "A description is _required_. Please describe what your macro does understandably!"
            command = "INSERT INTO macros VALUES (?, ?, ?, ?);"
            args = (name, value, description, ctx.author.id)
            self.bot.macros[name] = Macro(value, description, ctx.author.id)
            await cursor.execute(command, args)
            return await ctx.reply(f"Successfully added `{name}` to the database, aliased to `{value}`!")

    @macro.command(aliases=["e"])
    async def edit(self, ctx: Context, name: str, attribute: Literal["value", "description"], *, new: str):
        """Edits a macro. You must own said macro to edit it."""
        assert name in self.bot.macros, f"Macro `{name}` isn't in the database!"
        assert "value" != attribute or ";" not in new, "Sorry, but macros can't have persistent variants in them."
        async with self.bot.db.conn.cursor() as cursor:
            if not await ctx.bot.is_owner(ctx.author):
                await cursor.execute("SELECT name FROM macros WHERE name == ? AND id == ?", name, ctx.author.id)
                check = await cursor.fetchone()
                assert check is not None, "You can't edit a macro you don't own, silly."
            # NOTE: I know I shouldn't use fstrings with execute, but it won't allow me to specify a row name with ?.
            await cursor.execute(f"UPDATE macros SET {attribute} = ? WHERE name == ?", new, name)
        setattr(self.bot.macros[name], attribute, new)
        return await ctx.reply(f"Edited `{name}`'s {attribute} to be `{new}`.")

    @macro.command(aliases=["rm", "remove", "del"])
    async def delete(self, ctx: Context, name: str):
        """Deletes a macro. You must own said macro to delete it."""
        assert name in self.bot.macros, f"Macro `{name}` already isn't in the database!"
        async with self.bot.db.conn.cursor() as cursor:
            if not await ctx.bot.is_owner(ctx.author):
                await cursor.execute("SELECT name FROM macros WHERE name == ? AND id == ?", name, ctx.author.id)
                check = await cursor.fetchone()
                assert check is not None, "You can't delete a macro you don't own, silly."
            await cursor.execute(f"DELETE FROM macros WHERE name == ?", name)
        del self.bot.macros[name]
        return await ctx.reply(f"Deleted `{name}`.")

    @macro.command(aliases=["?", "list", "query"])
    async def search(self, ctx: Context, pattern: str = '.*'):
        """Searches the database for macros."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM macros WHERE name REGEXP ?", pattern)
            names = [name for (name,) in await cursor.fetchall()]
        return await ButtonPages(MacroQuerySource(sorted(names))).start(ctx)

    @macro.command(aliases=["i", "get"])
    async def info(self, ctx: Context, name: str):
        """Gets info about a specific macro."""
        assert name in self.bot.macros, f"Macro `{name}` isn't in the database!"
        macro = self.bot.macros[name]
        value = re.sub(r"(\$\d+)", r"[36m\1[0m", macro.value) \
            .replace(":", r"[30m:[0m")
        emb = discord.Embed(
            title=name
        )
        emb.add_field(
            name="",
            value=macro.description
        )
        emb.add_field(
            name="Value",
            value=f"```ansi\n{value}```",
            inline=False
        )
        user = await ctx.bot.fetch_user(macro.author)
        emb.set_footer(text=f"{user.name}#{user.discriminator}",
                       icon_url=user.avatar.url if user.avatar is not None else
                       f"https://cdn.discordapp.com/embed/avatars/{int(user.discriminator) % 5}.png")
        await ctx.reply(embed=emb)

async def setup(bot: Bot):
    await bot.add_cog(MacroCog(bot))
