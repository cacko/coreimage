
from functools import reduce
import pprint
from typing import Optional
from coreimage.find import find_images
from coreimage.utils import IMAGE_EXT, pil_to_mat
from pathlib import Path
from uuid import uuid4
from PIL import Image


class Concat:

    __output_path: Optional[Path] = None

    def __init__(
        self,
        dst: Optional[Path] = None
    ) -> None:
        self.dst = dst

    @property
    def output_path(self) -> Path:
        if not self.__output_path:
            dst = self.dst
            dst_root = Path.cwd()
            dst_name = f"collage-{uuid4().hex}.png"
            if dst:
                dst_root = dst if dst.is_dir() else dst.parent
                dst_name = dst.name if IMAGE_EXT.is_allowed(dst.suffix) else dst_name
            if not dst_root.exists():
                dst_root.mkdir(parents=True)
            self.__output_path = dst_root / dst_name
        return self.__output_path

    def concat_from_paths(self, paths: list[Path]) -> Path:
        return self.concat_from_images([Image.open(p) for p in find_images(paths)])

    def concat_from_images(self, images: list[Image.Image]) -> Path:
        maxwidth, maxheight = reduce(lambda mx, im: [max(mx[0], im.width), max(mx[1], im.height)], images, [0, 0])
        cols = 4
        rows = round(len(images) / 4)
        # Resize images to be the same size
        resized_images = []
        max_height = max([img.size[1] for img in images])
        max_width = max([img.size[0] for img in images])
        for img in images:
            resized_images.append(img.resize((max_width, max_height)))

        # Create the blank canvas
        collage_width = max_width * cols
        collage_height = max_height * rows
        collage = Image.new('RGB', (collage_width, collage_height))

        # Paste the images onto the canvas
        for i in range(rows):
            for j in range(cols):
                img_index = i * cols + j
                if img_index < len(resized_images):
                    collage.paste(resized_images[img_index], (j * max_width, i * max_height))
        collage.save(self.output_path.as_posix()) 
        return self.output_path
