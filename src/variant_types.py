import glob
import inspect
from inspect import Parameter
import math
import types
import typing
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, get_origin, \
                    get_args, Callable, Self, Type
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re

import cv2
import numpy as np
import visual_center

from . import constants, errors
from .types import Bot, RenderContext, Renderer, SignText, NumpySprite

if TYPE_CHECKING:
    from .tile import Tile, TileSkeleton, TileData, ProcessedTile
else:
    class Tile: pass
    class TileSkeleton: pass
    class TileData: pass
    class ProcessedTile: pass

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

INT_REGEX = re.compile(
    r'[+-]?(?:0x[0-9A-Fa-f]+|0o[0-7]+|0b[01]+|[1-9][0-9]*|0)',
    re.IGNORECASE
)

type ParseError = type("ParseError", (), {})
PARSE_ERROR = type("ParseError", (), {})()  # Unique singleton


def parse_int(string: str) -> tuple[str, int | ParseError]:
    match = INT_REGEX.match(string)
    if match is None:
        return string, PARSE_ERROR
    try:
        return string[match.end():], int(match.group(0), base=0)
    except ValueError:
        return string, PARSE_ERROR


FLOAT_REGEX = re.compile(
    r'[+-]?(((\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)|inf|nan)',
    re.IGNORECASE
)


def parse_float(string: str) -> tuple[str, float | ParseError]:
    match = FLOAT_REGEX.match(string)
    if match is None:
        return string, PARSE_ERROR
    try:
        return string[match.end():], float(match.group(0))
    except ValueError:
        return string, PARSE_ERROR


def parse_bool(string: str) -> tuple[str, bool | ParseError]:
    if string.startswith("true") or string.startswith("True"):
        return string[4:], True
    elif string.startswith("false") or string.startswith("False"):
        return string[5:], False
    else:
        return string, PARSE_ERROR


def parse_literal(ty) -> Callable[[str], tuple[str, str | ParseError]]:
    valid_values = ty.__args__

    def parse(string: str) -> tuple[str, str | ParseError]:
        for value in valid_values:
            if string.startswith(value):
                return string.removeprefix(value), value
        return string, PARSE_ERROR

    return parse


PRIMITIVE_PARSERS = {
    int: parse_int,
    float: parse_float,
    bool: parse_bool,
    typing._LiteralGenericAlias: parse_literal,
}


def get_parser(ty) -> Callable[[Type, str], tuple[str, Any | ParseError]] | None:
    if type(ty) is typing._LiteralGenericAlias:
        return parse_literal(ty)
    return PRIMITIVE_PARSERS.get(ty, None)


#endregion


@dataclass
class Variant:
    args: tuple
    factory: "AbstractVariantFactory"
    persistent: bool
    full_string: str

    def __hash__(self):
        if not self.factory.hashable:
            return id(self)
        return hash((self.factory.identifier, *self.args))

    @property
    def type(self):
        return self.factory.type

    async def apply(self, target: Any, context: "AbstractVariantContext"):
        return await self.factory.applicator(target, context, *self.args)


@dataclass
class AbstractVariantFactory(ABC):
    identifier: str
    description: str
    syntax_description: str
    parser: Callable[[str], tuple[str, Union["Variant", None]]]
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
        cls, *,
        names: tuple[str] | None = (),
        hashable: bool = True
    ):
        """Dynamically defines a Variant subclass based on an annotated function definition."""
        def decorator(func: Callable):
            global ALL_VARIANTS
            nonlocal cls, names, hashable

            # This code is kind of weird.
            # Basically, we create an opaque subclass of the
            # class that we're calling this from only when
            # this module is loaded.

            identifier = func.__name__
            description = func.__doc__
            ty = cls.type
            assert description is not None, \
                f"Variant `{identifier}` is missing a docstring."
            if type(names) == str:
                names = [names]

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

            # Generate the subclass
            variant = type(
                identifier,
                (cls, ),
                dict(
                    identifier = identifier,
                    description = description,
                    applicator = None,
                    syntax_description = None,
                    parser = None,
                    hashable = hashable,
                    nameless = names is None
                )
            )

            variant.applicator = func
            variant.parser = variant.generate_parser(names, params)
            variant.syntax_description = variant.generate_syntax_description(names, params)

            ALL_VARIANTS[identifier] = variant

            return variant

        return decorator

    @classmethod
    def generate_parser(
        cls,
        names: tuple[str] | None,
        params: tuple[tuple[str, Parameter], ...]
    ) -> Callable[[str], tuple[str, Union["Variant", None]]]:

        arg_parsers = []
        for param in params:
            parser = get_parser(param[1].annotation)
            assert parser is not None, f"Type {param[1].annotation} does not have an implemented parser"
            arg_parsers.append(parser)

        def parser(string: str) -> tuple[str, Union["Variant", None]]:
            orig_str = string
            # Check for names first
            if names is not None:
                for name in names:
                    if string.startswith(name):
                        string = string.removeprefix(name)
                        break
                else:
                    print("No names found")
                    return orig_str, None
                if string.startswith("/"):
                    string = string.removeprefix("/")

            args = []
            for i, parser in enumerate(arg_parsers):
                string, res = parser(string)
                if res is PARSE_ERROR:
                    print(f"Argument {i} failed at {string}")
                    return orig_str, None
                args.append(res)
                if string == "":
                    for _ in range(i + 1, len(arg_parsers)):
                        args.append(None)
                    break
                if i + 1 == len(arg_parsers):
                    if not string.startswith("/"):
                        return orig_str, None
                    string = string.removeprefix("/")

            return string, Variant(tuple(args), cls, False, orig_str[:len(orig_str) - len(string)])

        return parser

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
