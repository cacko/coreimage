from pathlib import Path
from typing import Any, Optional
from PIL import Image
from .utils import round8
import qrcode
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q
from enum import IntEnum


class ERROR_CORRECTION(IntEnum):
    LOW = ERROR_CORRECT_L
    MID = ERROR_CORRECT_M
    HIGH = ERROR_CORRECT_Q
    EXTREME = ERROR_CORRECT_H


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


def get_qrcode(
    data: Any,
    box_area: Optional[int] = 16,
    border: Optional[int] = 1,
    fill_color: Optional[str] = "black",
    back_color: Optional[str] = "white",
    **kwds
) -> Image.Image:

    qr = qrcode.QRCode(
        version=1,
        box_size=box_area,
        border=border,
        **kwds
    )
    qr.add_data(data)
    qr.make(fit=True)

    return qr.make_image(
        fill_color=fill_color,
        back_color=back_color
    ).get_image()
