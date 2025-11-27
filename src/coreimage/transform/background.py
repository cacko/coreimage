from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image
from coreimage.utils import DEVICE
from pathlib import Path

class RemoveBackground:
    
    def __init__(self):
        torch.set_float32_matmul_precision(["high", "highest"][0])

        self.__processor = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.__processor.to(DEVICE)
        
    def __load_img(self, image: Union[Image.Image, Path], output_type: str = "pil") -> Image.Image:
        if isinstance(image, Path):
            img = Image.open(image.as_posix())
        else:
            img = image
        if output_type == "pil":
            return img
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")


    def remove_background(self, image: Union[Image.Image, Path]) -> Image.Image:
        im = self.__load_img(image, output_type="pil")
        im = im.convert("RGB")
        processed_image = self.__process(im)
        return processed_image

    def __process(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        transform_image =  transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_images = transform_image(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = self.__processor(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image