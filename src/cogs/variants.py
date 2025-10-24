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
from ..variant_types import \
    SpriteVariantFactory, SpriteVariantContext, \
    ALL_VARIANTS, Variant, AbstractVariantFactory


async def setup(bot: Bot):
    ALL_VARIANTS.clear()

    @SpriteVariantFactory.define_variant(names="posterize")
    async def posterize(
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

    all_vars = [(key, value) for (key, value) in ALL_VARIANTS.items()]

    def sort_variants(a: tuple[str, AbstractVariantFactory]) -> int:
        if a[1].nameless:
            return -1
        return len(a[0]) * 1000 + hash(a[0]) % 1000

    all_vars = sorted(all_vars, key=sort_variants)
    ALL_VARIANTS.clear()
    for key, val in all_vars:
        ALL_VARIANTS[key] = val

    def parse_variant(string: str) -> tuple[str, Variant | None]:
        for var in ALL_VARIANTS.values():
            string, parsed = var.parser(string)
            if parsed is not None and string == "":
                return parsed
        return None

    bot.variants = ALL_VARIANTS
    bot.parse_variant = parse_variant
