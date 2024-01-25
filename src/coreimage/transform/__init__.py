from pathlib import Path
from typing import Optional
from PIL import Image
from coreimage.utils import round8
from .crop import Cropper
from .upscale import Upscale

__all__ = [
    "normalize",
    "Cropper",
    "Upscale"
]


def normalize(
    paths: list[Path],
    max_size: Optional[int] = None,
):
    for pth in paths:
        im = Image.open(pth.as_posix())
        if not max_size:
            max_size = max(im.width, im.height)
        max_size = round8(max_size)
        im.thumbnail((max_size, max_size))
        yield im
