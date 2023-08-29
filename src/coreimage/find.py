from pathlib import Path
from corefile import filepath
from .utils import IMAGE_EXT


def find_images(src: list[Path]):
    suffixes = IMAGE_EXT.get_suffixes()
    for pth in src:
        pth = Path(pth)
        yield from filepath(
            root=pth,
            suffixes=suffixes
        )
