
import logging
import math
from typing import Optional
from .find import find_images
from .utils import pil_to_mat
import cv2
from pathlib import Path
from uuid import uuid4
from operator import itemgetter
from PIL import Image


def concat_vh(list_2d):

    def convert(pil_img):
        WHITE = [255, 255, 255]
        return cv2.copyMakeBorder(
            cv2.resize(
                pil_to_mat(pil_img),
                dsize=(0, 0),
                fx=1,
                fy=1
            ), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)

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


def clamp(v, h):
    return 1 if v < 1 else h if v > h else v


def normalize(
    imgList,
    spacing=0,
    aspectratiofactor=1.0
):
    maxHeight = max([img.height for img in imgList])
    imgList = [img.resize((int(img.width / img.height * maxHeight), maxHeight))
               if img.height < maxHeight else img for img in imgList]

    totalWidth = sum([img.width for img in imgList])
    avgWidth = totalWidth / len(imgList)
    targetWidth = avgWidth * math.sqrt(len(imgList) * aspectratiofactor)

    numRows = clamp(int(round(totalWidth / targetWidth)), len(imgList))

    aspectRatios = [int(img.width / img.height * 100) for img in imgList]

    imgRows = linear_partition(aspectRatios, numRows, imgList)

    rowWidths = [sum([img.width + spacing for img in row]) - spacing for row in imgRows]
    minRowWidth = min(rowWidths)
    rowWidthRatios = [minRowWidth / w for w in rowWidths]
    for row, widthRatio in zip(imgRows, rowWidthRatios):
        row_images = [img.resize((int(img.width * widthRatio), int(img.height * widthRatio)))
                      for img in row]
        yield row_images


def concat(
    paths: list[Path],
    dst: Optional[Path] = None
):
    dst_root = Path.cwd()
    dst_name = f"collage-{uuid4().hex}.png"
    if dst and dst.exists():
        dst_root = dst if dst.is_dir() else dst_root
        dst_name = dst.name if dst.is_file() else dst_name

    output = dst_root / dst_name

    logging.info(output)

    images = [Image.open(p) for p in find_images(paths)]

    tiles = concat_vh(list(normalize(images)))
    return cv2.imwrite(output.as_posix(), tiles)
