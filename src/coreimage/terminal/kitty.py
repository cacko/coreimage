from term_image.image import KittyImage
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image


def get_term_image(
    image: Optional[np.ndarray | Image.Image] = None,
    image_path: Optional[Path] = None,
    height=10,
    fmt="{:1.1#}",
    **kwds
):

    def get_image(image_data: Any) -> Optional[Image.Image]:
        match(image_data):
            case Image.Image():
                return image_data
            case np.ndarray():
                return Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        return None

    kitty_image = None
    if image is not None:
        kitty_image = KittyImage(get_image(image), height=height, **kwds)
    if image_path and all([image_path.exists(), image_path.is_file()]):
        cvimage = cv2.imread(image_path.as_posix())
        kitty_image = KittyImage(get_image(cvimage), height=height, **kwds)
    if kitty_image:
        return fmt.format(kitty_image)
