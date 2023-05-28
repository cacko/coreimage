
import math
from typing import Optional
from coreimage.find import find_images
from coreimage.utils import pil_to_mat
import cv2
from pathlib import Path
from uuid import uuid4
from operator import itemgetter
from PIL import Image


class Concat:

    __output_path: Optional[Path] = None
    __images: list[Image.Image] = []

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
            if dst and dst.exists():
                dst_root = dst if dst.is_dir() else dst_root
                dst_name = dst.name if dst.is_file() else dst_name

            self.__output_path = dst_root / dst_name
        return self.__output_path

    def concat_vh(self, list_2d):
        try:
            list_h = []
            concat_h = []
            for list_h in list_2d:
                concat_h.append(
                    cv2.hconcat([
                        pil_to_mat(h) for h in list_h
                    ])
                )
            return cv2.vconcat(concat_h)
        except Exception:
            print(list_h)

    def linear_partition(self, seq, k):
        if k <= 0:
            return []
        n = len(seq) - 1
        if k > n:
            return map(lambda x: [x], seq)
        table, solution = self.linear_partition_table(seq, k)
        k, ans = k-2, []
        dataList = self.__images
        if len(dataList) != len(seq):
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

    def linear_partition_table(self, seq, k):
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

    @staticmethod
    def clamp(v, h):
        return 1 if v < 1 else h if v > h else v

    def normalize(
        self,
        spacing=0,
        aspectratiofactor=1.0
    ):
        imgList = self.__images
        maxHeight = max([img.height for img in imgList])
        imgList = [img.resize((int(img.width / img.height * maxHeight), maxHeight))
                   if img.height < maxHeight else img for img in imgList]

        totalWidth = sum([img.width for img in imgList])
        avgWidth = totalWidth / len(imgList)
        targetWidth = avgWidth * math.sqrt(len(imgList) * aspectratiofactor)

        numRows = self.clamp(int(round(totalWidth / targetWidth)), len(imgList))

        aspectRatios = [int(img.width / img.height * 100) for img in imgList]

        imgRows = self.linear_partition(aspectRatios, numRows)

        rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
        minRowWidth = min(rowWidths)
        rowWidthRatios = [minRowWidth / w for w in rowWidths]
        for row, widthRatio in zip(imgRows, rowWidthRatios):
            row_images = [img.resize((int(img.width * widthRatio), int(img.height * widthRatio)))
                          for img in row]
            yield row_images

    def concat_from_paths(self, paths: list[Path]):
        images = [Image.open(p) for p in find_images(paths)]
        return self.concat_from_images(images)

    def concat_from_images(self, images: list[Image.Image]):
        self.__images = images
        tiles = self.concat_vh(list(self.normalize()))
        return cv2.imwrite(self.output_path.as_posix(), tiles)
