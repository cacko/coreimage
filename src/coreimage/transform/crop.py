from typing import Optional
from uuid import uuid4
from PIL import Image
from pathlib import Path
from corefile import TempPath
from pydantic import BaseModel
from coreimage.resources import HAARCASCADE_XML
from PIL.ImageOps import exif_transpose
import itertools
from corestring import to_int, nearest_bytes
import cv2
import numpy as np
from PIL import Image
from coreimage.transform.upscale import Upscale

FIXEXP = True  # Flag to fix underexposition
MINFACE = 16  # Minimum face size ratio; too low and we get false positives
INCREMENT = 0.06
FACE_RATIO = 6  # Face / padding ratio

PILLOW_FILETYPES = [k for k in Image.registered_extensions().keys()]
INPUT_FILETYPES = PILLOW_FILETYPES + [s.upper() for s in PILLOW_FILETYPES]


class CropPosition(BaseModel):
    y1: int
    y2: int
    x1: int
    x2: int


class Cropper:
    __image: Optional[np.ndarray] = None
    __faces: Optional[list[list[int]]] = None
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 640
    DEFAULT_FACE_PERCENTAGE = 50
    DEFAULT_PADDING = 0

    def __init__(
        self,
        path: Path,
        width=640,
        height=640,
        face_percent=50,
        padding=0,
        resize=True,
        blur=True,
    ):
        self.img_path = path
        img_height, img_width = self.image.shape[:2]
        if not height:
            height = img_height
        if not width:
            width = img_width
        self.height = to_int(height, self.DEFAULT_HEIGHT)
        self.width = to_int(width, self.DEFAULT_WIDTH)
        self.padding = (max(self.width, self.height) // 100) * to_int(
            padding, self.DEFAULT_PADDING
        )
        self.face_percent = to_int(face_percent, self.DEFAULT_FACE_PERCENTAGE)
        self.aspect_ratio = self.width / self.height
        self.resize = resize
        self.blur = blur
        self.casc_path = HAARCASCADE_XML.as_posix()

    @property
    def image(self):
        if self.__image is None:
            self.__image = self.__open()
        return self.__image

    @image.setter
    def image(self, value):
        self.__image = value

    @property
    def image_width(self) -> int:
        return self.image.shape[:2][1]

    @property
    def image_height(self) -> int:
        return self.image.shape[:2][0]

    def __open(self):
        """Given a filename, returns a numpy array"""
        with Image.open(self.img_path) as img_orig:
            img_orig = exif_transpose(img_orig)
            w, h = img_orig.width, img_orig.height
            if max(img_orig.width, img_orig.height) < 2000:
                img_orig = Upscale.upscale_img(img_orig, scale=2)
                img_orig = img_orig.resize((w, h))
            return np.array(img_orig)

    @staticmethod
    def distance(pt1, pt2):
        distance = np.linalg.norm(pt2 - pt1)
        return distance

    @staticmethod
    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    @classmethod
    def intersect(cls, v1, v2):
        a1, a2 = v1
        b1, b2 = v2
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = cls.perp(da)
        denom = np.dot(dap, db).astype(float)
        num = np.dot(dap, dp)
        return (num / denom) * db + b1

    @property
    def faces(self):
        if self.__faces is None:
            try:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                gray = self.image
            minface = int(
                np.sqrt(self.image_height**2 + self.image_width**2) / MINFACE
            )
            face_cascade = cv2.CascadeClassifier(self.casc_path)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(minface, minface),
                flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
            )

            try:
                assert len(faces)
                self.__faces = sorted(faces.tolist(), key=lambda p: p[0])
            except AssertionError:
                self.__faces = []
            if len(faces) == 0:
                return None
        return self.__faces

    def show_faces(self) -> Path:
        faces = self.faces

        if not len(faces):
            return None

        faces_image = self.image.copy()

        for x, y, w, h in faces:
            cv2.rectangle(faces_image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        out = TempPath(f"{uuid4()}.png")
        cv2.imwrite(out.as_posix(), cv2.cvtColor(faces_image, cv2.COLOR_BGR2RGB))
        return out

    def crop(self, face_idx: int = -1, out: Optional[Path] = None) -> Path:
        if not out:
            out = (
                self.img_path.parent
                / f"{self.img_path.stem}_crop{self.img_path.suffix}"
            )

        faces = self.faces

        if not len(faces):
            return None

        face_idx = len(faces) - 1 if face_idx == -1 else min(face_idx, len(faces) - 1)

        x, y, w, h = faces.pop(face_idx)
        if self.blur:
            for pos in faces:
                x1, x2, y1, y2 = pos[0], pos[0] + pos[2], pos[1], pos[1] + pos[3]
                self.image[y1:y2, x1:x2] = cv2.medianBlur(self.image[y1:y2, x1:x2], 35)
        pos = self.__crop_positions(x, y, w, h)
        self.image = self.image[pos.y1 : pos.y2, pos.x1 : pos.x2]

        if self.resize:
            with Image.fromarray(self.image) as img:
                self.image = np.array(img.resize((self.width, self.height)))

        cv2.imwrite(out.as_posix(), cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return out

    def __determine_safe_zoom(self, x, y, w, h):
        # Find out what zoom factor to use given self.aspect_ratio
        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [
                (0, 0),
                (0, self.image_height),
                (self.image_width, self.image_height),
                (self.image_width, 0),
                (0, 0),
            ]
        )  # image_corners
        image_sides = [(i[n], i[n + 1]) for n in range(4)]

        corner_ratios = [self.face_percent]  # Hopefully we use this one
        for c in corners:
            corner_vector = np.array([center, c])
            a = self.__class__.distance(*corner_vector)
            intersects = list(
                self.__class__.intersect(corner_vector, side) for side in image_sides
            )
            for pt in intersects:
                if (pt >= 0).all() and (pt <= i[2]).all():  # if intersect within image
                    dist_to_pt = self.__class__.distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)

    def __crop_positions(
        self,
        x,
        y,
        w,
        h,
    ) -> CropPosition:
        if self.padding:
            w += self.padding
            h += self.padding
        zoom = self.__determine_safe_zoom(x, y, w, h)

        # Adjust output height based on percent
        if self.height >= self.width:
            height_crop = h * 100.0 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100.0 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        # Calculate padding by centering face
        xpad = (width_crop - w + self.padding) / 2
        ypad = (height_crop - h + self.padding) / 2

        # Calc. positions of crop
        h1 = max(0, x - xpad)
        h2 = x + w + xpad
        v1 = max(0, y - ypad)
        v2 = y + h + ypad

        return CropPosition(y1=int(v1), y2=int(v2), x1=int(h1), x2=int(h2))
