from typing import Optional
from PIL import Image
from pathlib import Path
from coreimage.resources import HAARCASCADE_XML
from PIL.ImageOps import exif_transpose
import itertools
from corestring import to_int
import cv2
import numpy as np
from PIL import Image

FIXEXP = True  # Flag to fix underexposition
MINFACE = 8  # Minimum face size ratio; too low and we get false positives
INCREMENT = 0.06
FACE_RATIO = 6  # Face / padding ratio

PILLOW_FILETYPES = [k for k in Image.registered_extensions().keys()]
INPUT_FILETYPES = PILLOW_FILETYPES + [s.upper() for s in PILLOW_FILETYPES]


class Cropper:
    __image: Optional[np.ndarray] = None
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 640
    DEFAULT_FACE_PERCENTAGE = 50
    DEFAULT_PADDING = 1

    def __init__(
        self,
        path: Path,
        width=640,
        height=640,
        face_percent=50,
        padding=None,
        resize=True,
        face_idx=None,
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
        self.padding = (max(self.width, self.height) // 20) * to_int(padding, self.DEFAULT_PADDING)
        self.face_percent = to_int(face_percent, self.DEFAULT_FACE_PERCENTAGE)
        self.aspect_ratio = width / height
        self.resize = resize
        self.face_idx = to_int(face_idx, -1)
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

    def __open(self):
        """Given a filename, returns a numpy array"""
        with Image.open(self.img_path) as img_orig:
            img_orig = exif_transpose(img_orig)
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

    def crop(self, out: Optional[Path] = None) -> Path:
        if not out:
            out = (
                self.img_path.parent
                / f"{self.img_path.stem}_crop{self.img_path.suffix}"
            )

        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = self.image

        img_height, img_width = self.image.shape[:2]
        minface = int(np.sqrt(img_height**2 + img_width**2) / MINFACE)

        face_cascade = cv2.CascadeClassifier(self.casc_path)

        faces: list = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        ).tolist()

        if len(faces) == 0:
            return None

        face_idx = (
            len(faces) - 1
            if self.face_idx == -1
            else min(self.face_idx, len(faces) - 1)
        )

        x, y, w, h = faces.pop(face_idx)
        if self.blur:
            for pos in faces:
                x1, x2, y1, y2 = pos[0], pos[0] + pos[2], pos[1], pos[1] + pos[3]
                self.image[y1:y2, x1:x2] = cv2.medianBlur(self.image[y1:y2, x1:x2], 35)
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )
        self.image = self.image[pos[0] : pos[1], pos[2] : pos[3]]

        if self.resize:
            with Image.fromarray(self.image) as img:
                self.image = np.array(img.resize((self.width, self.height)))

        cv2.imwrite(out.as_posix(), cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return out

    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):
        # Find out what zoom factor to use given self.aspect_ratio
        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
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

    def _crop_positions(
        self,
        imgh,
        imgw,
        x,
        y,
        w,
        h,
    ):
        if self.padding:
            w += self.padding
            h += self.padding
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

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

        return [int(v1), int(v2), int(h1), int(h2)]
