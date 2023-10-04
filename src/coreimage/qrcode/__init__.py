from enum import StrEnum
import segno
from segno.helpers import make_email, make_geo, make_vcard, make_wifi
from typing import Optional, Any
from PIL import Image
from io import BytesIO


__all__ = [
    "get_qrcode",
    "get_wifi",
    "get_email",
    "get_geo",
    "get_vcard"
]


class ERROR_CORRECTION(StrEnum):
    LOW = "L"
    MID = "M"
    HIGH = "Q"
    EXTREME = "H"


def qr_to_pil(
        qr: segno.QRCode,
        scale: Optional[int] = 16,
        border: Optional[int] = 4
) -> Image.Image:
    qr_data = BytesIO()
    qr.save(out=qr_data, kind='png', scale=scale, border=border)
    return Image.open(qr_data)


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
    border: Optional[int] = 4
) -> Image.Image:
    qr = make_wifi(
        ssid=ssid,
        password=password,
        security=security,
        hidden=hidden
    )
    return qr_to_pil(qr, scale=box_area, border=border)


def get_email(
    to: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4
) -> Image.Image:
    qr = make_email(
        to=to, cc=cc, bcc=bcc, subject=subject, body=body
    )
    return qr_to_pil(qr, scale=box_area, border=border)


def get_geo(
    lat: float,
    lng: float,
    box_area: Optional[int] = 16,
    border: Optional[int] = 4
) -> Image.Image:
    qr = make_geo(lat=lat, lng=lng)
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
    border: Optional[int] = 4
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
        workphone=workphone
    )
    return qr_to_pil(qr, scale=box_area, border=border)
