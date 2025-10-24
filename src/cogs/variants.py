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
from ..variant_types import SpriteVariantFactory, SpriteVariantContext

ALL_VARIANTS: dict[str, "AbstractVariantFactory"] = {}

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
