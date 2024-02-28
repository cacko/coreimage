import logging
from typing import Optional
from uuid import uuid4
from PIL import Image
from pathlib import Path
from corefile import TempPath
from pydantic import BaseModel
from PIL.ImageOps import exif_transpose
from corestring import to_int
import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import pad
from facenet_pytorch import MTCNN

PILLOW_FILETYPES = [k for k in Image.registered_extensions().keys()]
INPUT_FILETYPES = PILLOW_FILETYPES + [s.upper() for s in PILLOW_FILETYPES]


class CropPosition(BaseModel):
    y1: int
    y2: int
    x1: int
    x2: int


class Cropper:
    __image: Optional[Image.Image] = None
    __faces: Optional[list[list[int]]] = None
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 640

    def __init__(
        self,
        path: Path,
        width=640,
        height=640,
        resize=True,
        blur=True,
        margin=300
    ):
        self.img_path = path
        self.height = to_int(height, self.DEFAULT_HEIGHT)
        self.width = to_int(width, self.DEFAULT_WIDTH)
        self.aspect_ratio = self.width / self.height
        self.resize = resize
        self.blur = blur
        self.margin = margin

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
        with Image.open(self.img_path) as img_orig:
            img_orig = img_orig.convert("RGB")
            img_orig = exif_transpose(img_orig)
            img_orig.thumbnail((1200, 1200))
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
                mtcnn = MTCNN(image_size=640, margin=100)
                boxes, _ = mtcnn.detect(self.image)
                assert len(boxes)
                
                def face_box(box):
                    margin = [
                        self.margin * (box[2] - box[0]) / (self.width - self.margin),
                        self.margin * (box[3] - box[1]) / (self.height - self.margin),
                    ]
                    if any([box[2] - box[0] < 20, box[3] - box[1] < 20]):
                        return None
                    box = [
                        int(max(box[0] - margin[0] / 2, 0)),
                        int(max(box[1] - margin[1] / 2, 0)),
                        int(min(box[2] + margin[0] / 2, self.image_width)),
                        int(min(box[3] + margin[1] / 2, self.image_height)),
                    ]
                    logging.debug(box)
                    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                faces = list(filter(None, [face_box(box) for box in boxes]))
                self.__faces = sorted(faces, key=lambda p: p[0])
            except AssertionError:
                raise ValueError("No faces found")
        return self.__faces

    def show_faces(self) -> Path:
        faces = self.faces

        assert faces

        faces_image = self.image.copy()

        for x, y, w, h in faces:
            cv2.rectangle(faces_image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        out = TempPath(f"{uuid4()}.png")
        cv2.imwrite(out.as_posix(), cv2.cvtColor(faces_image, cv2.COLOR_BGR2RGB))
        return out

    def crop(self, face_idx: Optional[int] = 0, out: Optional[Path] = None) -> Path:
        if not out:
            out = self.img_path.parent / f"{self.img_path.stem}_crop.jpg"

        faces = self.faces

        if not len(faces):
            return None

        if not face_idx:
            face_idx = 0
        # face_idx = min(to_int(face_idx, len(faces) - 1), len(faces) - 1)

        x, y, w, h = faces.pop(face_idx)
        if self.blur:
            for pos in faces:
                x1, x2, y1, y2 = pos[0], pos[0] + pos[2], pos[1], pos[1] + pos[3]
                self.image[y1:y2, x1:x2] = cv2.medianBlur(self.image[y1:y2, x1:x2], 35)
        self.image = self.image[y : (y+h), x : (x+w)]

        if self.resize:
            with Image.fromarray(self.image) as img:
                self.image = np.array(pad(img, (self.width, self.height), color=(255,255,255)))
        cv2.imwrite(out.as_posix(), cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return out

