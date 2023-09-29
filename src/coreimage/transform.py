from pathlib import Path
from typing import Any, Optional
from PIL import Image
from .utils import round8
import qrcode


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
    box_area: Optional[int] = 20,
    border: Optional[int] = 1,
    fill_color: Optional[str] = "black",
    back_color: Optional[str] = "white",
    **kwds
) -> Image.Image:

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_area,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    return qr.make_image(
        fill_color=fill_color,
        back_color=back_color
    ).get_image()
