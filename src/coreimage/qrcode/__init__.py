from enum import StrEnum
import segno
from segno.helpers import make_email, make_geo, make_vcard, make_wifi
from typing import Optional, Any
from PIL import Image
from io import BytesIO
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q
import qrcode

__all__ = [
    "get_qrcode",
    "get_wifi",
    "get_email",
    "get_geo",
    "get_vcard"
]


class ERROR(StrEnum):
    LOW = "L"
    MID = "M"
    HIGH = "Q"
    EXTREME = "H"


MAP_ERRORS = {
    ERROR.LOW: ERROR_CORRECT_L,
    ERROR.MID: ERROR_CORRECT_M,
    ERROR.HIGH: ERROR_CORRECT_Q,
    ERROR.EXTREME: ERROR_CORRECT_H
}


def qr_to_pil(
        qr: segno.QRCode,
        scale: Optional[int] = 16,
        border: Optional[int] = 4
) -> Image.Image:
    qr_data = BytesIO()
    qr.save(out=qr_data, kind='png', scale=scale, border=border)
    return Image.open(qr_data)


def get_qrcode_legacy(
    data: Any,
    box_area: Optional[int] = 16,
    border: Optional[int] = 1,
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

    return qr.make_image().get_image()


def get_qrcode(
    data: Any,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    **kwds
) -> Image.Image:

    qr = segno.make_qr(data, **kwds)
    return qr_to_pil(qr, scale=box_area, border=border)


def get_wifi(
    ssid: str,
    password: Optional[str] = None,
    security: Optional[str] = None,
    hidden: Optional[bool] = None,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    *kwds
) -> Image.Image:
    qr = make_wifi(
        ssid=ssid,
        password=password,
        security=security,
        hidden=hidden,
        *kwds
    )
    return qr_to_pil(qr, scale=box_area, border=border)


def get_email(
    to: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    *kwds
) -> Image.Image:
    qr = make_email(
        to=to, 
        cc=cc, 
        bcc=bcc, 
        subject=subject, 
        body=body,
        *kwds
    )
    return qr_to_pil(qr, scale=box_area, border=border)


def get_geo(
    lat: float,
    lng: float,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    *kwds
) -> Image.Image:
    qr = make_geo(lat=lat, lng=lng, *kwds)
    return qr_to_pil(qr, scale=box_area, border=border)


def get_vcard(
    name,
    displayname,
    email=None,
    phone=None,
    fax=None,
    videophone=None,
    memo=None,
    nickname=None,
    birthday=None,
    url=None,
    pobox=None,
    street=None,
    city=None,
    region=None,
    zipcode=None,
    country=None,
    org=None,
    lat=None,
    lng=None,
    source=None,
    rev=None,
    title=None,
    photo_uri=None,
    cellphone=None,
    homephone=None,
    workphone=None,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4,
    *kwds
) -> Image.Image:
    qr = make_vcard(
        name, displayname, email=email,
        phone=phone, fax=fax,
        videophone=videophone, memo=memo,
        nickname=nickname, birthday=birthday,
        url=url, pobox=pobox, street=street,
        city=city, region=region,
        zipcode=zipcode, country=country,
        org=org, lat=lat, lng=lng,
        source=source, rev=rev, title=title,
        photo_uri=photo_uri,
        cellphone=cellphone,
        homephone=homephone,
        workphone=workphone,
        *kwds
    )
    return qr_to_pil(qr, scale=box_area, border=border)
