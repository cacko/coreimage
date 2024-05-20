from pathlib import Path
from typing import Generator, Optional
import cv2
import numpy as np
from PIL import Image
from contextlib import contextmanager

@contextmanager
def get_image(
    image_data: Optional[np.ndarray | Image.Image | Path] = None,
    **kwds
) -> Generator[Image.Image, None, None]:
    try:
        from term_image.image import KittyImage
        assert KittyImage.is_supported()
        match(image_data):
            case Image.Image():
                yield KittyImage(image=image_data.convert("RGB"),  **kwds)
            case np.ndarray():
                yield KittyImage(image=Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)), **kwds)
            case Path():
                yield KittyImage(image=Image.open(image_data.as_posix()).convert("RGB"),  **kwds)
    except (AssertionError, ValueError):
        raise NotImplementedError("kirtty is not support on this termianl")
    finally:
        pass


@contextmanager
def get_term_image(
    image: Optional[np.ndarray | Image.Image] = None,
    image_path: Optional[Path] = None,
    **kwds
):
    try:
        with get_image(next(filter(lambda x: x is not None, [image, image_path])), **kwds) as kitty_image:
            fmt = kwds.get("fmt", "{:1.1#}")
            yield fmt.format(kitty_image)
    finally:
        pass
