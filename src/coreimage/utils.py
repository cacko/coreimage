from enum import StrEnum
from itertools import filterfalse
from PIL import Image
import cv2
import numpy as np
from functools import lru_cache
from typing import Optional


def resize_set(arg: str) -> Optional[tuple]:
    try:
        parts = arg.split("x",1)
        assert len(parts) == 2
        return tuple([int(x) for x in parts])
    except Exception:
        return None


class IMAGE_EXT(StrEnum):
    JPEG = "jpg"
    PNG = "png"
    WEBP = "webp"

    @lru_cache
    @staticmethod
    def supported():
        exts = Image.registered_extensions()
        return [ex for ex, f in exts.items() if f in Image.OPEN]

    @classmethod
    def get_suffixes(cls) -> list[str]:
        return cls.supported()

    @classmethod
    def is_allowed(cls, ext: str) -> bool:
        return f'.{ext.lower().lstrip(".")}' in cls.supported()

    @classmethod
    def endwith(cls, fname: str) -> bool:
        return (
            next(
                filterfalse(lambda e: fname.lower().endswith(e), cls.supported()),
                None,
            )
            is not None
        )


def pil_to_mat(pil_img: Image.Image) -> cv2.Mat:
    nimg = np.array(pil_img)
    mat = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    return mat


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def nearest_bytes(n):
    p = int(float(n).hex().split("p+")[1]) + 1
    return 2**p


def round8(a):
    return int(a) + 4 & ~7


def round2(a):
    return int(a) + int(int(a) % 2)
