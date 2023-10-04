from pathlib import Path
from typing import Optional
from PIL import Image
from coreimage.utils import round8

__all__ = [
    "normalize"
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
