
import logging
from typing import Optional
from coreimage.find import find_images
from coreimage.utils import IMAGE_EXT, pil_to_mat
from pathlib import Path
from uuid import uuid4
from PIL import Image
from cv2_collage_v2 import create_collage_v2


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
        logging.warn(self.__output_path)
        return self.__output_path

    def concat_from_paths(self, paths: list[Path]) -> Path:
        return self.concat_from_images([Image.open(p) for p in find_images(paths)])

    def concat_from_images(self, images: list[Image.Image]) -> Path:
        create_collage_v2(
            [pil_to_mat(pil) for pil in images],
            maxwidth=3000,
            heightdiv=6,
            widthdiv=2,
            background=(0, 0, 0),
            save_path=self.output_path.as_posix(),
        )
        return self.output_path
