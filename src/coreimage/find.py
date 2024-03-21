from pathlib import Path
from corefile import filepath
from .utils import IMAGE_EXT


def find_images(src: list[Path|str]):
    suffixes = IMAGE_EXT.get_suffixes()
    for pth in src:
        pth = Path(pth) if isinstance(pth , str) else pth
        yield from filepath(
            root=pth,
            suffixes=suffixes
        )
