from __future__ import annotations

import string
from dataclasses import dataclass
from io import BytesIO
from sqlite3.dbapi2 import Row
from typing import AsyncGenerator, Iterable, Any

import re
import asqlite
import numpy as np
import requests
import tldextract as tldextract
from PIL import Image

from .types import TilingMode

from . import constants
from .constants import DIRECTIONS


class Database:
    """Everything relating to persistent readable & writable data."""
    conn: asqlite.Connection
    bot: None
    filter_cache: dict[str, (Image.Image, bool)]
    palette_store: dict[(str, str), Image.Image]

    def __init__(self, bot):
        self.filter_cache = {}
        self.palette_store = {}
        self.bot = bot

    async def connect(self, db: str) -> None:
        """Startup."""
        # not checking for same thread probably is a terrible idea but
        # whateverrr
        self.conn = await asqlite.connect(db, check_same_thread=False)

        def regexp(x, y):
            return bool(re.search(x, y))

        self.conn.get_connection().create_function('regexp', 2, regexp)
        print("Initialized database connection.")
        await self.create_tables()
        print("Verified database tables.")
        await self.store_palettes()
        print("Stored palettes.")

    async def store_palettes(self):
        async with self.conn.cursor() as cur:
            res = await cur.execute("""
                SELECT name, source, data FROM palettes
            """)
            res = [(*row, ) for row in await res.fetchall()]
        for (name, source, data) in res:
            im = Image.open(BytesIO(data))
            im.load()
            self.palette_store[(name, None)] = im.copy()
            self.palette_store[(name, source)] = im.copy()

    async def close(self) -> None:
        """Teardown."""
        if hasattr(self, "conn"):
            await self.conn.close()

    async def create_tables(self) -> None:
        """Creates tables in the database according to a schema in code.

        (Useful for documentation.)
        """
        async with self.conn.cursor() as cur:
            await cur.execute(
                # `name` is not specified to be a unique field.
                # We allow multiple "versions" of a tile to exist,
                # to account for differences between "world" and "editor" tiles.
                # One example of this is with `belt` -- its color inside levels
                # (which use "world" tiles) is different from its editor color.
                # These versions are differentiated by `version`.
                #
                # For tiles where the active/inactive distinction doesn't apply
                # (i.e. all non-text tiles), only `active_color` fields are
                # guaranteed to hold a meaningful, non-null value.
                #
                # `text_direction` defines whether a property text tile is
                # "pointed towards" any direction. It is null otherwise.
                # The directions are right: 0, up: 8, left: 16, down: 24.
                #
                # `tags` is a tab-delimited sequence of strings. The empty
                # string denotes no tags.
                '''
				CREATE TABLE IF NOT EXISTS tiles (
					name TEXT NOT NULL,
					sprite TEXT NOT NULL,
					source TEXT NOT NULL,
					version INTEGER NOT NULL,
					inactive_color_x INTEGER DEFAULT 3,
					inactive_color_y INTEGER DEFAULT 0,
					active_color_x INTEGER NOT NULL DEFAULT 0,
					active_color_y INTEGER NOT NULL DEFAULT 3,
					tiling INTEGER NOT NULL DEFAULT -1,
					text_type INTEGER NOT NULL DEFAULT 0,
					text_direction INTEGER,
					tags TEXT NOT NULL DEFAULT "",
                    extra_frames TEXT,
                    object_id TEXT,
					UNIQUE(name, version)
				);
				'''
            )
            await cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS ServerActivity (
					id INTEGER NOT NULL,
					timestamp INTEGER NOT NULL
				);
				'''
            )
            await cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS blacklistedusers (
					id INTEGER NOT NULL
				);
                '''
            )
            # We create different tables for levelpacks and custom levels.
            # While both share some fields, there are mutually exclusive
            # fields which are more sensible in separate tables.
            #
            # The world/id combination is unique across levels. However,
            # a world can have multiple levels and multiple worlds can share
            # a level id. Thus neither is unique alone.
            await cur.execute(
                '''
				CREATE TABLE IF NOT EXISTS levels (
					id TEXT NOT NULL,
					world TEXT NOT NULL,
					name TEXT NOT NULL,
					subtitle TEXT,
					number INTEGER,
					style INTEGER,
					parent TEXT,
					map_id TEXT,
					UNIQUE(id, world)
				);
				'''
            )
            await cur.execute(
                # There have been multiple valid formats of level
                # codes, so we don't assume a constant-width format.
                '''
				CREATE TABLE IF NOT EXISTS custom_levels (
					code TEXT UNIQUE NOT NULL,
					name TEXT NOT NULL,
					subtitle TEXT,
					author TEXT NOT NULL
				);
				'''
            )
            await cur.execute(
                '''
				CREATE TABLE IF NOT EXISTS letters (
					mode TEXT NOT NULL,
					char TEXT NOT NULL,
					width INTEGER NOT NULL,
					sprite_0 BLOB,
					sprite_1 BLOB,
					sprite_2 BLOB
				);
				'''
            )
            await cur.execute(
                '''
				CREATE TABLE IF NOT EXISTS users (
					user_id INTEGER PRIMARY KEY,
					blacklisted INTEGER,
					silent_commands INTEGER,
					render_background INTEGER
				);
				'''
            )
            await cur.execute(
                '''
				CREATE TABLE IF NOT EXISTS filters (
                    name STRING NOT NULL,
                    absolute BOOLEAN NOT NULL,
                    author INT NOT NULL,
                    upload_time INT,
                    data BLOB NOT NULL
                );
				'''
            )
            await cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS macros (
                    name TEXT UNIQUE PRIMARY KEY,
                    value TEXT,
                    description TEXT,
                    creator INT
                );
                '''
            )
            await cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS palettes (
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data BLOB NOT NULL,
                    hash INT UNIQUE NOT NULL
                );
                '''
            )

    async def tile(self, name: str, *, maximum_version: int = 1000) -> TileData | None:
        """Convenience method to fetch a single thing of tile data.

        Returns None on failure.
        """
        row = await self.conn.fetchone(
            '''
			SELECT * FROM tiles
			WHERE name == ? AND version <= ?
			ORDER BY version DESC;
			''',
            name, maximum_version
        )
        if row is None:
            return None
        return TileData.from_row(row)

    def palette(self, name: str, source: str = None) -> Image.Image | None:
        """Convenience method to fetch a palette from the database.

        Returns None on failure.
        """
        if type(name) is tuple:
            name, source = name
        return self.palette_store.get((name, source))

    async def tiles(self, names: Iterable[str], *, maximum_version: int = 1000) -> AsyncGenerator[TileData, None]:
        """Convenience method to fetch a single thing of tile data.

        Returns None on failure.
        """
        async with self.conn.cursor() as cur:
            for name in names:
                await cur.execute(
                    '''
					SELECT * FROM tiles
					WHERE name == ? AND version < ?
					ORDER BY version DESC;
					''',
                    name, maximum_version
                )
                row = await cur.fetchone()
                if row is not None:
                    yield TileData.from_row(row)

    def plate(self, direction: int | None,
              wobble: int) -> tuple[Image.Image, tuple[int, int]]:
        """Plate sprites.

        Raises FileNotFoundError on failure.
        """
        # Strongly assure type to protect against potential security issue
        assert type(
            direction) == int or direction is None, f"Plate of type {type(direction)} wasn't allowed? This shouldn't happen."
        if direction is None:
            return (
                Image.open(
                    f"data/plates/plate_property_0_{wobble + 1}.png").convert("RGBA"),
                (0, 0)
            )
        return (
            Image.open(
                f"data/plates/plate_property{DIRECTIONS.get(direction, '')}_0_{wobble + 1}.png").convert("RGBA"),
            (3, 3)
        )

    async def get_filter(self, name: str):
        """Get a filter from the database."""
        if name not in self.filter_cache:
            async with (self.conn.cursor() as cur):
                await cur.execute("SELECT absolute, author, upload_time, data FROM filters WHERE name == ?;", name)
                res = await cur.fetchone()
                if res is None: return None
                absolute, author, upload_time, data = res
                im = Image.open(BytesIO(data))
                im.load()          
                final = absolute, author, None if upload_time is None else upload_time / 1000, im
                self.filter_cache[name] = final
                return final
        return self.filter_cache[name]


@dataclass
class TileData:
    name: str
    sprite: str
    source: str
    inactive_color: tuple[int, int]
    active_color: tuple[int, int]
    tiling: TilingMode
    text_type: int
    text_direction: int | None
    tags: list[str]
    extra_frames: list[int]

    @classmethod
    def from_row(cls, row: Row) -> TileData:
        """Create a tiledata object from a database row."""
        tags: str | None = row["tags"]
        if tags is not None:
            tags = tags.strip()
            if len(tags) == 0:
                tags = None
                
        extra_frames: str | None = row["extra_frames"]
        if extra_frames is not None:
            extra_frames = extra_frames.strip()
            if len(extra_frames) == 0:
                extra_frames = None

        return TileData(
            row["name"],
            row["sprite"],
            row["source"],
            (row["inactive_color_x"], row["inactive_color_y"]),
            (row["active_color_x"], row["active_color_y"]),
            TilingMode(row["tiling"]),
            row["text_type"],
            row["text_direction"],
            tags.split("\t") if tags is not None else [],
            [int(frame) for frame in (extra_frames).split("\t")] if extra_frames is not None else []
        )


@dataclass
class LevelData:
    id: str
    world: str
    name: str
    subtitle: str | None
    number: int | None
    style: int | None
    parent: str | None
    map_id: str | None

    @classmethod
    def from_row(cls, row: Row) -> LevelData:
        """Level from db row."""
        return LevelData(*row)

    def display(self) -> str:
        """The level display string."""
        if self.parent is None or self.parent == "<empty>":
            return self.name
        if self.map_id is not None and self.map_id != "<empty>":
            return f"{self.parent}-{self.map_id}: {self.name}"
        if self.style is not None and self.number is not None:
            if self.style == 0:
                # numbers
                return f"{self.parent}-{self.number}: {self.name}"
            if self.style == 1:
                # letters
                letter = string.ascii_lowercase[self.number]
                return f"{self.parent}-{letter}: {self.name}"
            if self.style == 2:
                # extra dots
                return f"{self.parent}-extra {self.number + 1}: {self.name}"
        return self.name  # raise RuntimeError("Level is in a bad state")

    def unique(self) -> str:
        """Uniquely identifying string."""
        return f"{self.world}/{self.id}"


@dataclass
class CustomLevelData:
    code: str
    name: str
    subtitle: str | None
    author: str

    @classmethod
    def from_row(cls, row: Row) -> CustomLevelData:
        """Level from db row."""
        return CustomLevelData(*row)

    def unique(self) -> str:
        """Uniquely identifying string."""
        return self.code
