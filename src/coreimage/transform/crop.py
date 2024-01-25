from pydoc import classname
from tkinter import N
from typing import Optional
from PIL import Image
from pathlib import Path
from coreimage.resources import HAARCASCADE_XML
from PIL.ImageOps import exif_transpose
import itertools

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

    def __init__(
        self,
        path: Path,
        width=0,
        height=0,
        face_percent=50,
        padding=0,
        resize=True,
    ):
        self.img_path = path
        img_height, img_width = self.image.shape[:2]
        if not height:
            height = img_height
        if not width:
            width = img_width
        self.height = self.__class__.check_positive_scalar(height)
        self.width = self.__class__.check_positive_scalar(width)
        self.padding = padding
        self.aspect_ratio = width / height
        self.resize = resize

        # Face percent
        if face_percent > 100 or face_percent < 1:
            fp_error = "The face_percent argument must be between 1 and 100"
            raise ValueError(fp_error)
        self.face_percent = self.__class__.check_positive_scalar(face_percent)

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
        """Returns the euclidian distance in 2D between 2 pts."""
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

    def __bgr_to_rbg(self):
        if self.image.ndim == 2:
            return self.image
        return self.image[:, :, [2, 1, 0]]

    @staticmethod
    def check_positive_scalar(num):
        """Returns True if value if a positive scalar."""
        if num > 0 and not isinstance(num, str) and np.isscalar(num):
            return int(num)
        raise ValueError("A positive scalar is required")

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

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        )

        if len(faces) == 0:
            return None

        x, y, w, h = faces[-1]
        print(faces)
        # if self.padding:
        #     x = x - (self.padding // 2)
        #     y = y + (self.padding // 2)
        #     w = w + self.padding 
        #     h = h + self.padding
        print(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )

        print(pos)

        self.image = self.image[pos[0] - 200 : pos[1], pos[2] : pos[3]]

        if self.resize:
            with Image.fromarray(self.image) as img:
                self.image = np.array(img.resize((self.width, self.height)))

        cv2.imwrite(out.as_posix(), self.__bgr_to_rbg())
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
        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        # Calc. positions of crop
        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]
