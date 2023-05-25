
import math
from typing import Optional
from .find import find_images
from .utils import pil_to_mat, chunks
import cv2
from pathlib import Path
from uuid import uuid4
from .transform import normalize


def concat_vh(list_2d):
    return cv2.vconcat([
        cv2.hconcat(list_h) for list_h in list_2d
    ])


def concat(
    paths: list[Path],
    dst: Optional[Path] = None,
    max_size: Optional[int] = None
):
    if not dst or not dst.exists():
        dst = Path.cwd()/ f"collage-{uuid4().hex}.png"
    WHITE = [255, 255, 255]
    
    print(dst)

    images = list(find_images(paths))
    n_images = len(images)
    tiles = concat_vh(
        chunks([*map(
            lambda pil_img: cv2.copyMakeBorder(
                cv2.resize(
                    pil_to_mat(pil_img),
                    dsize=(0, 0),
                    fx=1,
                    fy=1
                ), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE),
            normalize(images, max_size)
        )], round(math.sqrt(n_images)))
    )
    return cv2.imwrite(dst.as_posix(), tiles)
