import glob
import inspect
from inspect import Parameter
import math
import types
import typing
from typing import Any, Literal, Optional, Union, get_origin, \
                    get_args, Callable, Self, Type
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import cv2
import numpy as np
import visual_center

from . import liquify
from ..utils import recolor, composite
from .. import constants, errors
from ..tile import Tile, TileSkeleton, TileData, ProcessedTile
from ..types import Bot, RenderContext, Renderer, SignText, NumpySprite


ALL_VARIANTS: dict[str, "AbstractVariantFactory"] = {}


@dataclass
class AbstractVariantContext(ABC):
    pass


@dataclass
class SkeletonVariantContext(AbstractVariantContext):
    pass


@dataclass
class SignVariantContext(AbstractVariantContext):
    bot: Bot
    ctx: RenderContext
    renderer: Renderer


@dataclass
class TileVariantContext(AbstractVariantContext):
    tile_data_cache: dict[str, TileData]


@dataclass
class SpriteVariantContext(AbstractVariantContext):
    tile: Tile
    wobble: int
    renderer: Renderer


@dataclass
class PostVariantContext(AbstractVariantContext):
    pass


#region Parser

type_parsers = {
    int: parse_int,
    float: parse_float,
    bool: parse_bool,
    typing._LiteralGenericAlias: parse_literal,
}

#endregion


@dataclass
class AbstractVariantFactory(ABC):
    identifier: str
    description: str
    syntax_description: str
    parser: Callable[[str], Self | None]
    target: Type
    context: Type
    applicator: Callable[[Any, Any, ...], None]
    hashable: bool = True
    nameless: bool = False

    @property
    @abstractmethod
    def type(self):
        raise NotImplementedError("cannot get type of abstract variant factory")

    @classmethod
    def define_variant(
        cls, func: Callable, *,
        names: tuple[str] | None = (),
        hashable: bool = True
    ):
        """Dynamically defines a Variant subclass based on an annotated function definition."""
        global ALL_VARIANTS

        # This code is kind of weird.
        # Basically, we create an opaque subclass of the
        # class that we're calling this from only when
        # this module is loaded.

        identifier = func.__name__
        description = func.__doc__
        ty = cls.type
        assert description is not None, \
            f"Variant `{identifier}` is missing a docstring."

        signature = inspect.signature(func)
        params: tuple[tuple[str, Parameter], ...] = tuple((name, value) for name, value in signature.parameters.items())
        assert len(params) >= 2, f"Variant `{identifier}` must have at least two parameters."

        # Check all parameters are annotated
        for param in params:
            if param[1].annotation is Parameter.empty:
                raise AssertionError(f"Parameter `{param[0]}` of variant `{identifier}` has no type annotation.")
        # Sanity check the first two parameters
        assert params[0][1].annotation == cls.target, \
            f"Variant `{identifier}` has an incorrectly annotated target."
        assert params[1][1].annotation == cls.context, \
            f"Variant `{identifier}` has an incorrectly annotated context."

        params = params[2:]
        # Generate parser
        parser = cls.generate_parser(names, params)
        syntax_description = cls.generate_syntax_description(names, params)

        # Generate the subclass
        variant = type(
            identifier,
            (cls, ),
            dict(
                identifier = identifier,
                description = description,
                applicator = func,
                syntax_description = syntax_description,
                parser = parser,
                hashable = hashable,
                nameless = names is None
            )
        )

        ALL_VARIANTS[identifier] = variant

        return variant

    def generate_parser(
        names: tuple[str] | None,
        params: tuple[tuple[str, Parameter], ...]
    ) -> Callable[[str], Self | None]:
        return None

    def generate_syntax_description(
        names: tuple[str] | None,
        params: tuple[tuple[str, Parameter], ...]
    ) -> str:
        return "<todo>"


@dataclass
class SkeletonVariantFactory(AbstractVariantFactory):
    applicator: Callable[[TileSkeleton, Any, ...], None] = lambda *_: None
    target: Type = TileSkeleton
    context: Type = SkeletonVariantContext
    type: str = "skel"


@dataclass
class SignVariantFactory(AbstractVariantFactory):
    applicator: Callable[[SignText, Any, ...], None] = lambda *_: None
    target: Type = SignText
    context: Type = SignVariantContext
    type: str = "sign"


@dataclass
class TileVariantFactory(AbstractVariantFactory):
    applicator: Callable[[Tile, Any, ...], None] = lambda *_: None
    target: Type = Tile
    context: Type = TileVariantContext
    type: str = "tile"


@dataclass
class SpriteVariantFactory(AbstractVariantFactory):
    applicator: Callable[[NumpySprite, Any, ...], None] = lambda *_: None
    target: Type = NumpySprite
    context: Type = SpriteVariantContext
    type: str = "sprite"


@dataclass
class PostVariantFactory(AbstractVariantFactory):
    applicator: Callable[[ProcessedTile, Any, ...], None] = lambda *_: None
    target: Type = ProcessedTile
    context: Type = PostVariantContext
    type: str = "post"


@dataclass
class Variant:
    args: tuple
    factory: AbstractVariantFactory
    persistent: bool

    def __hash__(self):
        if not self.factory.hashable:
            return id(self)
        return hash((self.factory.name, *self.args))

    @property
    def type(self):
        return self.factory.type

    def apply(self, target: Any, context: AbstractVariantContext):
        self.factory.applicator(target, context, *self.args)


# API Example
@SpriteVariantFactory.define_variant
def posterize(
    sprite: NumpySprite, ctx: SpriteVariantContext,
    bands: int
):
    """Posterizes the sprite."""
    return np.dstack([
        np.digitize(
            sprite[..., i],
            np.linspace(0, 255, bands)
        ) * (255 / bands) for i in range(4)
    ]).astype(np.uint8)


async def setup(bot: Bot):
    bot.variants = ALL_VARIANTS
