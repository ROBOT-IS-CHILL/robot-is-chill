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
from PIL import Image
from os.path import commonprefix # why is it *THERE*
import textwrap

import cv2
import numpy as np
import visual_center

from . import constants, errors, utils
from .types import Bot, RenderContext, Renderer, SignText, NumpySprite, Color

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
    bot: Bot


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
    color: Color


@dataclass
class PostVariantContext(AbstractVariantContext):
    pass


#region Parser

INT_REGEX = re.compile(
    r'[+-]?(?:0x[0-9A-Fa-f]+|0o[0-7]+|0b[01]+|[1-9][0-9]*|0)',
    re.IGNORECASE
)

PARSE_ERROR = type("ParseError", (), {})()  # Unique singleton


def parse_int(string: str, **_) -> tuple[str, int | "ParseError"]:
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


def parse_float(string: str, **_) -> tuple[str, float | "ParseError"]:
    match = FLOAT_REGEX.match(string)
    if match is None:
        return string, PARSE_ERROR
    try:
        return string[match.end():], float(match.group(0))
    except ValueError:
        return string, PARSE_ERROR


def parse_bool(string: str, **_) -> tuple[str, bool | "ParseError"]:
    if string.startswith("true") or string.startswith("True"):
        return string[4:], True
    elif string.startswith("false") or string.startswith("False"):
        return string[5:], False
    else:
        return string, PARSE_ERROR


def parse_literal(ty) -> Callable[[str], tuple[str, str | "ParseError"]]:
    valid_values = ty.__args__

    def parse(string: str, **_) -> tuple[str, str | "ParseError"]:
        for value in valid_values:
            if string.startswith(value):
                return string.removeprefix(value), value
        return string, PARSE_ERROR

    return parse


def parse_color(string: str, *, bot: Bot, palette: tuple[str, str], **_) -> tuple[str, Color | "ParseError"]:
    pstring, res = Color.parse(string, palette, bot.db)
    if res is None:
        return string, PARSE_ERROR
    return pstring, res


def parse_str(string: str, **_) -> tuple[str, str | "ParseError"]:
    splits = string.split("/", 1)
    end = ("/" + "/".join(splits[1:])) if len(splits[1:]) else ""
    return end, splits[0]


PRIMITIVE_PARSERS = {
    int: parse_int,
    float: parse_float,
    bool: parse_bool,
    str: parse_str,
    Color: parse_color
}


def parse_list(ty) -> Callable[[str], tuple[str, list | "ParseError"]]:
    list_type = ty.__args__[0]
    parser = get_parser(list_type)

    def parse(string: str, **_) -> tuple[str, list | "ParseError"]:
        args = []
        while True:
            string, res = parser(string)
            if res is PARSE_ERROR:
                return string, PARSE_ERROR
            args.append(res)
            if not len(string):
                break
            if not string.startswith("/"):
                return string, PARSE_ERROR
            string = string.removeprefix("/")
        return string, tuple(args)

    return parse

def get_parser(ty) -> Callable[[Type, str], tuple[str, Any | "ParseError"]] | None:
    if type(ty) is typing._LiteralGenericAlias:
        return parse_literal(ty)
    if type(ty) is types.GenericAlias and typing.get_origin(ty) is list:
        return parse_list(ty)
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
        if type(target).__name__ == "SignText" and self.factory.sign_alt:
            return await self.factory.sign_alt(target, *self.args)
        #if isinstance(target, self.factory.target):
        # HACK: We need to do this instead of isinstance because of type stubbing for IDE type checking. :P
        if type(target).__name__ == self.factory.target.__name__:
            return await self.factory.applicator(target, context, *self.args)


@dataclass
class AbstractVariantFactory(ABC):
    identifier: str
    description: str
    syntax_description: str
    parser: Callable[[str], tuple[str, Union["Variant", None]]]
    applicator: Callable[[Any, Any, ...], None]
    target: Type = type(None)
    context: Type = type(None)
    hashable: bool = True
    hashed: bool = True
    nameless: bool = False
    sign_alt: Callable | None = None
    ty: str = "None"

    @classmethod
    def define_variant(
        cls, *,
        names: tuple[str] | None = (),
        hashable: bool = True, hashed: bool = True,
        sign_alt: Callable | None = None
    ):
        """Dynamically defines a Variant subclass based on an annotated function definition."""
        def decorator(func: Callable):
            global ALL_VARIANTS
            nonlocal cls, names, hashable

            identifier = func.__name__
            description = func.__doc__

            assert description is not None, \
                f"Variant `{identifier}` is missing a docstring."
            if type(names) == str:
                names = [names]

            # Clean up description
            description = textwrap.dedent("\n".join(line for line in description.splitlines() if len(line.strip()) > 0))

            signature = inspect.signature(func)
            params: tuple[tuple[str, Parameter], ...] = tuple((name, value) for name, value in signature.parameters.items())
            assert len(params) >= 2, f"Variant `{identifier}` must have at least two parameters."

            # Check all parameters are annotated
            for param in params:
                if param[1].annotation is Parameter.empty:
                    raise AssertionError(f"Parameter `{param[0]}` of variant `{identifier}` has no type annotation.")
            # Sanity check the first two parameters
            assert params[0][1].annotation.__name__ == cls.target.__name__, \
                f"Variant `{identifier}` has an incorrectly annotated target."
            assert params[1][1].annotation.__name__ == cls.context.__name__, \
                f"Variant `{identifier}` has an incorrectly annotated context."

            params = params[2:]

            variant = cls(
                identifier = identifier,
                description = description,
                applicator = None,
                syntax_description = None,
                parser = None,
                hashable = hashable, hashed = hashed,
                nameless = names is None,
                sign_alt = sign_alt
            )

            variant.applicator = func
            variant.parser = cls.generate_parser(variant, names, params)
            variant.syntax_description = cls.generate_syntax_description(names, params)

            ALL_VARIANTS[identifier] = variant

            return func

        return decorator

    @classmethod
    def generate_parser(
        cls, variant,
        names: tuple[str] | None,
        params: tuple[tuple[str, Parameter], ...]
    ) -> Callable[[str], tuple[str, Union["Variant", None]]]:

        arg_parsers = []
        required = 0
        for param in params:
            parser = get_parser(param[1].annotation)
            assert parser is not None, f"Type {param[1].annotation} does not have an implemented parser"
            arg_parsers.append(parser)
            if param[1].default is Parameter.empty:
                required += 1

        def parser(string: str, **kwargs) -> tuple[str, Union["Variant", None]]:
            orig_str = string
            # Check for names first
            if names is not None:
                for name in names:
                    if string.startswith(name):
                        string = string.removeprefix(name)
                        break
                else:
                    return orig_str, None
                if string.startswith("/"):
                    string = string.removeprefix("/")

            args = []
            for i, parser in enumerate(arg_parsers):
                if string == "" and i >= required:
                    break
                string, res = parser(string, **kwargs)
                if res is PARSE_ERROR:
                    return orig_str, None
                args.append(res)
                if i + 1 < required:
                    if not string.startswith("/"):
                        return orig_str, None
                elif i + 1 == len(arg_parsers) and string != "":
                    return orig_str, None
                string = string.removeprefix("/")

            return string, Variant(tuple(args), variant, False, orig_str[:len(orig_str) - len(string)])

        return parser

    def generate_syntax_description(
        names: tuple[str] | None,
        params: tuple[tuple[str, Parameter], ...]
    ) -> str:
        s = [("<" + "|".join(names) + ">") if names is not None else ""]
        for name, param in params:
            if param.default is not Parameter.empty:
                s.append(f"[{param}]")
            else:
                s.append(f"<{param}>")
            s.append("/")
        del s[-1]
        return "".join(s)


@dataclass
class SkeletonVariantFactory(AbstractVariantFactory):
    applicator: Callable[[TileSkeleton, Any, ...], None] = lambda *_: None
    target: Type = TileSkeleton
    context: Type = SkeletonVariantContext
    ty: str = "While parsing"


@dataclass
class SignVariantFactory(AbstractVariantFactory):
    applicator: Callable[[SignText, Any, ...], None] = lambda *_: None
    target: Type = SignText
    context: Type = SignVariantContext
    ty: str = "While placing sign texts"


@dataclass
class TileVariantFactory(AbstractVariantFactory):
    applicator: Callable[[Tile, Any, ...], None] = lambda *_: None
    target: Type = Tile
    context: Type = TileVariantContext
    ty: str = "While deciding sprite"

@dataclass
class SpriteVariantFactory(AbstractVariantFactory):
    applicator: Callable[[NumpySprite, Any, ...], None] = lambda *_: None
    target: Type = NumpySprite
    context: Type = SpriteVariantContext
    ty: str = "While applying effects to sprite image"


@dataclass
class PostVariantFactory(AbstractVariantFactory):
    applicator: Callable[[ProcessedTile, Any, ...], None] = lambda *_: None
    target: Type = ProcessedTile
    context: Type = PostVariantContext
    ty: str = "While placing sprite onto image"
