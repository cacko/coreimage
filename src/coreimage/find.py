from pathlib import Path
from glob import iglob
from .utils import IMAGE_EXT


def find_images(src: list[Path]):
    for pth in src:
        pth = Path(pth)
        if all([pth.is_file(), IMAGE_EXT.is_allowed(pth.suffix)]):
            yield pth.as_posix()
        elif pth.is_dir():
            yield from map(
                lambda f: pth / f,
                filter(
                    lambda fn: IMAGE_EXT.endwith(fn),
                    iglob("*.*", root_dir=pth.as_posix())
                )
            )
