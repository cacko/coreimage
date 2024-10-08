
from genericpath import exists
import json
from typing import Optional
from coreimage.find import find_images
from coreimage.utils import IMAGE_EXT
from pathlib import Path
from uuid import uuid4
from PIL import Image
from PIL.ImageOps import exif_transpose
import math
from operator import itemgetter
from hashlib import sha1
from random import randint, shuffle as random_shuffle
from PIL.ExifTags import Base as TagNames
from pydantic import BaseModel

class ConcatImage(BaseModel):
    img_path: Path
    width: int
    height: int
    row: int
    col: int



def linear_partition(seq, k, dataList=None):
    if k <= 0:
        return []
    n = len(seq) - 1
    if k > n:
        return map(lambda x: [x], seq)
    table, solution = linear_partition_table(seq, k)
    k, ans = k-2, []
    if dataList is None or len(dataList) != len(seq):
        while k >= 0:
            ans = [[seq[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
            n, k = solution[n-1][k], k-1
        ans = [[seq[i] for i in range(0, n+1)]] + ans
    else:
        while k >= 0:
            ans = [[dataList[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
            n, k = solution[n-1][k], k-1
        ans = [[dataList[i] for i in range(0, n+1)]] + ans
    return ans


def linear_partition_table(seq, k):
    n = len(seq)
    table = [[0] * k for x in range(n)]
    solution = [[0] * (k-1) for x in range(n-1)]
    for i in range(n):
        table[i][0] = seq[i] + (table[i-1][0] if i else 0)
    for j in range(k):
        table[0][j] = seq[0]
    for i in range(1, n):
        for j in range(1, k):
            table[i][j], solution[i-1][j-1] = min(
                ((max(table[x][j-1], table[i][0]-table[x][0]), x) for x in range(i)),
                key=itemgetter(0))
    return (table, solution)

# end partition problem algorithm


def clamp(v, ll, h):
    return ll if v < ll else h if v > h else v


class Concat:

    __output_path: Optional[Path] = None
    __hash: str = ""

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

    def concat_from_paths(self, paths: list[Path], shuffle: bool = False) -> tuple[Path, str]:
        def loader():
            names = []
            for p in find_images(paths):
                with p.open("rb") as fp:
                    names.append(p.name)
                    img = Image.open(fp)
                    img.load()
                    yield img
            ids = "-".join(sorted(names))
            self.__hash = sha1(ids.encode()).hexdigest()

        return self.concat_from_images([p for p in loader()], shuffle=shuffle)

    def makeCollage(
            self,
            imgList,
            spacing=10,
            background=(0, 0, 0),
            max_width=500
    ):
        aspectratiofactor = max(1.5, 4 / len(imgList))
        fragments = [];

        [img.thumbnail((randint(300, max_width), img.height), Image.Resampling.LANCZOS)
         if img.width > max_width else img for img in imgList
         ]

        maxHeight = max([img.height for img in imgList])
        imgList = [img.resize((int(img.width / img.height * maxHeight), maxHeight))
                   if img.height < maxHeight else img for img in imgList]
        totalWidth = sum([img.width for img in imgList])
        avgWidth = totalWidth / len(imgList)
        targetWidth = avgWidth * math.sqrt(len(imgList) * aspectratiofactor)

        numRows = clamp(int(round(totalWidth / targetWidth)), 1, len(imgList))
        if numRows == 1:
            imgRows = [imgList]
        elif numRows == len(imgList):
            imgRows = [[img] for img in imgList]
        else:
            aspectRatios = [int(img.width / img.height * 100) for img in imgList]

            imgRows = linear_partition(aspectRatios, numRows, imgList)

            rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
            minRowWidth = min(rowWidths)
            rowWidthRatios = [minRowWidth / w for w in rowWidths]
            imgRows = [[img.resize((int(img.width * widthRatio), int(img.height * widthRatio)))
                        for img in row] for row, widthRatio in zip(imgRows, rowWidthRatios)]

        rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
        rowHeights = [max([img.height for img in row]) for row in imgRows]
        minRowWidth = min(rowWidths)
        w, h = (minRowWidth, sum(rowHeights) + spacing * (numRows - 1))

        if background == (0, 0, 0):
            background += tuple([0])
        else:
            background += tuple([255])
        outImg = Image.new("RGBA", (w, h), background)
        xPos, yPos = (0, 0)
        for row in imgRows:
            row_fragments = []
            for img in row:
                fragment = [xPos, yPos]
                outImg.paste(img, (xPos, yPos))
                xPos += img.width + spacing
                fragment.append(xPos)
                row_fragments.append(fragment)
                outpath = self.output_path.parent / f"{self.output_path.name}_images" / f"{xPos}-{yPos}.webp"
                if not outpath.parent.exists():
                    outpath.parent.mkdir(parents=True, exist_ok=True)
                img.save(outpath)
                continue
            yPos += max([img.height for img in row]) + spacing
            fragments += [[*rf, yPos] for rf in row_fragments]
            xPos = 0
            continue
        ex = outImg.getexif()
        ex[TagNames.ImageDescription] = json.dumps(fragments)
        return outImg, ex

    def concat_from_images(self, images: list[Image.Image], shuffle: bool = False) -> tuple[Path, str]:
        if shuffle:
            random_shuffle(images)
        collage, ex = self.makeCollage(images)
        if self.output_path.suffix.lower() in [".jpg", ".jpeg"]:
            collage = exif_transpose(collage.convert("RGB"))
        collage.save(self.output_path.as_posix(), exif=ex)
        return (self.output_path, self.__hash)
