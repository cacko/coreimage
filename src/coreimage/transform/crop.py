import logging
from re import ASCII
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
from coreimage.transform.upscale import Upscale
from corestring import round2

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
        self, path: Path, width=640, height=640, resize=True, blur=True, margin=200
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
    
    def __upscale(self, image: Image.Image) -> Image:
        try:
            w, h  = image.width, image.height
            min_dim = min(w, h)
            assert 1200 > min_dim
            scale =  min(round2(1200/min_dim), 4)     
            return Upscale.upscale_img(img=image, scale=scale) 
        except AssertionError:
            pass
        
        return image

    def __open(self):
        with Image.open(self.img_path) as img_orig:
            img_orig = img_orig.convert("RGB")
            img_orig = exif_transpose(img_orig)
            img_orig = self.__upscale(img_orig)
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
                mtcnn = MTCNN(image_size=640)
                boxes, _ = mtcnn.detect(self.image)
                assert boxes

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

        assert len(faces)

        faces_image = self.image.copy()
        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(faces_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img=faces_image,
                text=f"{idx}",
                org=(x + 5, y + h - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        out = TempPath(f"{uuid4()}.png")
        cv2.imwrite(out.as_posix(), cv2.cvtColor(faces_image, cv2.COLOR_BGR2RGB))
        return out
    

    def crop(self, face_idx: Optional[int] = None, out: Optional[Path] = None) -> Path:
        if not out:
            out = self.img_path.parent / f"{self.img_path.stem}_crop.jpg"

        faces = self.faces

        try:
            assert len(faces)
        except AssertionError:
            return None


        try:
            assert isinstance(face_idx, int)
            assert faces[face_idx]
        except (AssertionError, IndexError):
            idx_by_size = list(
            map(
                lambda fs: fs[0],
                sorted(enumerate(faces), key=lambda ff: ff[1][2] * ff[1][3]),
            )
            )
            face_idx = idx_by_size[-1]


        x, y, w, h = faces.pop(face_idx)
        if self.blur:
            for pos in faces:
                x1, x2, y1, y2 = pos[0], pos[0] + pos[2], pos[1], pos[1] + pos[3]
                self.image[y1:y2, x1:x2] = cv2.medianBlur(self.image[y1:y2, x1:x2], 35)
        diff = h - w
        self.image = self.image[
            y : (y + h),
            max((x - diff // 2), 0) : min((x + w + diff // 2), self.image_width),
        ]

        if self.resize:
            with Image.fromarray(self.image) as img:
                self.image = np.array(
                    pad(img, (self.width, self.height), color=(255, 255, 255))
                )
        cv2.imwrite(out.as_posix(), cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return out
