from term_image.image import GraphicsImage
from pathlib import Path
from typing import  Optional
import numpy as np
from PIL import Image

def print_term_image(
    image: Optional[np.ndarray | Image.Image] = None,
    image_path: Optional[Path] = None,
    height: Optional[int]=10,
    **kwds
):
    try:
        assert GraphicsImage.is_supported()
        from coreimage.terminal.kitty import get_term_image as get_kitty_image
        from term_image.image import KittyImage
        assert KittyImage.is_supported()
        kwds.setdefault("height", height)
        with get_kitty_image(
            image=image,
            image_path=image_path,
            **kwds
        ) as term_image:
            print(term_image, end="\n\n")
    except AssertionError:
        pass

__all__ = [
    "print_term_image"
]
