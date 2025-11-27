from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps
from coreimage.utils import round8
from .crop import Cropper
from .upscale import Upscale
from typing import Optional
from corefile import TempPath
from .background import RemoveBackground

__all__ = [
    "normalize",
    "Cropper",
    "Upscale",
    "convert",
    "remove_background"
]


def normalize(
    paths: list[Path]|Path,
    max_size: Optional[int] = None,
):
    if not isinstance(paths, list):
        paths = [paths]
    for pth in paths:
        im = Image.open(pth.as_posix())
        ImageOps.exif_transpose(im, in_place=True)
        if not max_size:
            max_size = max(im.width, im.height)
        max_size = round8(max_size)
        im.thumbnail((max_size, max_size))
        yield im

def convert_to(
    pth: Path,
    format: str
):
    im = Image.open(pth.as_posix())
    tmp = TempPath(f"{pth.stem}.{format}")
    im.save(tmp.as_posix())
    return tmp
        
def remove_background(
    image: Image.Image|Path
) -> Image.Image:
    remover = RemoveBackground()
    return remover.remove_background(image)