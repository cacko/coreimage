
from typing import Optional
from coreimage.find import find_images
from coreimage.utils import IMAGE_EXT
from pathlib import Path
from uuid import uuid4
from PIL import Image
import math
from operator import itemgetter


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

    def makeCollage(
            self,
            imgList,
            spacing=0,
            antialias=False,
            background=(0, 0, 0),
            aspectratiofactor=1.0,
            max_height=500
    ):

        maxHeight = min(max([img.height for img in imgList]), max_height)
        if antialias:
            imgList = [img.resize((int(img.width / img.height * maxHeight), maxHeight),
                                  Image.ANTIALIAS) if img.height > maxHeight else img for img in imgList]
        else:
            imgList = [img.resize((int(img.width / img.height * maxHeight), maxHeight))
                       if img.height > maxHeight else img for img in imgList]
        # generate the input for the partition problem algorithm
        # need list of aspect ratios and number of r[ows (partitions)
        # imgHeights = [img.height for img in imgList]
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

            # get nested list of images (each sublist is a row in the collage)
            imgRows = linear_partition(aspectRatios, numRows, imgList)

            # scale down larger rows to match the minimum row width
            rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
            minRowWidth = min(rowWidths)
            rowWidthRatios = [minRowWidth / w for w in rowWidths]
            if antialias:
                imgRows = [[
                    img.resize(
                        (int(img.width * widthRatio), int(img.height * widthRatio)), Image.ANTIALIAS
                    )
                    for img in row] for row, widthRatio in zip(imgRows, rowWidthRatios)]
            else:
                imgRows = [[img.resize((int(img.width * widthRatio), int(img.height * widthRatio)))
                            for img in row] for row, widthRatio in zip(imgRows, rowWidthRatios)]

        # pupulate new image
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
            for img in row:
                outImg.paste(img, (xPos, yPos))
                xPos += img.width + spacing
                continue
            yPos += max([img.height for img in row]) + spacing
            xPos = 0
            continue

        return outImg

    def concat_from_images(self, images: list[Image.Image]) -> Path:
        # max_width, max_height = reduce(
        #     lambda mx, im: [max(mx[0], im.width), max(mx[1], im.height)],
        #     images,
        #     [0, 0]
        # )
        # cols = 4
        # rows = round(len(images) / 4)
        # # Resize images to be the same size
        # resized_images = []
        # for img in images:
        #     resized_images.append(img.resize((max_width, max_height)))

        # # Create the blank canvas
        # collage_width = max_width * cols
        # collage_height = max_height * rows
        # collage = Image.new('RGB', (collage_width, collage_height))

        # # Paste the images onto the canvas
        # for i in range(rows):
        #     for j in range(cols):
        #         img_index = i * cols + j
        #         if img_index < len(resized_images):
        #             collage.paste(resized_images[img_index], (j * max_width, i * max_height))
        collage = self.makeCollage(images)
        if self.output_path.suffix in [".jpg", ".jpeg"]:
            collage = collage.convert("RGB")
        collage.save(self.output_path.as_posix())
        return self.output_path
