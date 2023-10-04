from enum import StrEnum
import segno
from segno.helpers import (
    make_make_email_data,
    make_geo_data,
    make_vcard_data,
    make_wifi_data
)
from typing import Optional, Any
from PIL import Image
from io import BytesIO


__all__ = [
    "get_qrcode",
]


class ERROR(StrEnum):
    LOW = "L"
    MID = "M"
    HIGH = "Q"
    EXTREME = "H"


class Code:

    def __init__(
        self,
        content: Any,
        scale=16,
        border=4,
        error: Optional[ERROR] = None
    ) -> None:
        self.__content = content
        self.__scale = scale
        self.__border = border
        self.__error = error

    def __to_pil(
        self,
        qr: segno.QRCode,
    ) -> Image.Image:
        qr_data = BytesIO()
        qr.save(
            out=qr_data,
            kind='png',
            scale=self.__scale,
            border=self.__border
        )
        return Image.open(qr_data)

    def __parse_content(self) -> dict[str, str]:
        print(self.__content)
        parts = self.__content[1:]
        args = {k: v for k, v in map(lambda p: p.split("=", 1), parts)}
        return args

    def gen_wifi(self) -> Image.Image:
        data = make_wifi_data(**self.__parse_content())
        qr = segno.make_qr(data, error=self.__error)
        return self.__to_pil(qr)

    def gen_email(self) -> Image.Image:
        data = make_make_email_data(**self.__parse_content())
        qr = segno.make_qr(data, error=self.__error)
        return self.__to_pil(qr)

    def gen_geo(self) -> Image.Image:
        args = self.__parse_content()
        lat = float(args.get("lat", 0))
        lng = float(args.get("lng", 0))
        data = make_geo_data(lat=lat, lng=lng)
        qr = segno.make_qr(data, error=self.__error)
        return self.__to_pil(qr)

    def gen_vcard(self) -> Image.Image:
        data = make_vcard_data(**self.__parse_content())
        qr = segno.make_qr(data, error=self.__error)
        return self.__to_pil(qr)

    def gen(self):
        try:
            generator_name = self.__content[0].strip(":")
            assert generator_name
            generator = f"gen_{generator_name}"
            assert hasattr(self, generator)
            assert callable(getattr(self, generator, None))
            return getattr(self, generator)()
        except AssertionError:
            self.__content = " ".join(self.__content)
            qr = segno.make_qr(self.__content, error=self.__error)
            return self.__to_pil(qr)


def get_qrcode(
    data: Any,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    **kwds
) -> Image.Image:
    code = Code(
        content=data,
        scale=box_area,
        border=border,
        **kwds
    )
    return code.gen()
