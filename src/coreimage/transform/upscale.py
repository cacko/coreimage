import logging
import os
from PIL import Image
from pathlib import Path
from typing import Optional, Any
import torch
from PIL.ExifTags import Base as TagNames
from corefile import TempPath
from spandrel import ImageModelDescriptor, ModelLoader
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from coreimage.resources import RESOURCES_ROOT


class UpscaleMeta(type):
    __models = {
        2: "BSRGANx2.pth",
        4: "RealESRGAN_x4plus.pth",
        8: "RealESRGAN_x8.pth",
    }

    def get_upscaler(cls, scale: int) -> ImageModelDescriptor:
        model_path = RESOURCES_ROOT / cls.__models[scale]
        logging.debug(model_path)
        model = ModelLoader().load_from_file(model_path.as_posix())
        model.to(cls.device)
        model.eval()
        return model

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        return type.__call__(cls, *args, **kwds)

    @property
    def device(cls):
        return torch.device(os.environ.get("DEVICE", "mps"))

    def upscale(
        cls, src_path: Path, dst_path: Optional[Path] = None, **kwds
    ) -> Optional[Path]:
        if not dst_path:
            dst_path = TempPath(f"{src_path.stem}.png")
        if dst_path.is_dir():
            dst_path = dst_path / f"{src_path.stem}.png"
        res = cls().do_upscale(src=src_path, dst=dst_path, **kwds)
        return dst_path if res else None

    def upscale_img(cls, img: Image.Image, **kwds) -> Image.Image:
        return cls().do_upscale_img(img, **kwds)


class Upscale(object, metaclass=UpscaleMeta):
    def do_upscale_img(self, low_res_img, **kwds):
        with torch.no_grad():
            scale = kwds.get("scale", 2)
            small_tensor = to_tensor(low_res_img).unsqueeze(0).to(self.__class__.device)
            upscaler = self.__class__.get_upscaler(scale)
            upscaled_tensor = upscaler(small_tensor)
            return to_pil_image(upscaled_tensor.squeeze())

    @staticmethod
    def set_info(image: Image.Image, prompt) -> Image.Exif:
        ex = image.getexif()
        ex[TagNames.ImageDescription] = prompt
        return ex

    def do_upscale(self, src: Path, dst: Path, **kwds) -> bool:
        scale = kwds.get("scale", 4)
        low_res_img = Image.open(src.as_posix()).convert("RGB")

        upscaled = self.do_upscale_img(low_res_img, scale=scale)

        prompt = None
        try:
            ex = low_res_img.getexif()
            prompt = ex.get(TagNames.ImageDescription)
        except Exception as e:
            logging.exception(e)

        if prompt:
            upscaled.save(
                dst.as_posix(),
                optimize=False,
                exif=Upscale.set_info(image=upscaled, prompt=prompt),
            )
        else:
            upscaled.save(dst.as_posix())
        return True
