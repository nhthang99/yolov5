import json
import argparse
from pathlib import Path

import cv2
import numpy as np


def region2xyxy(region: dict):
    points = np.array(region["points"])
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])
    return xmin, ymin, xmax, ymax


def xyxy2xywh(xyxy):
    x_center = (xyxy[0] + xyxy[2]) / 2  # x center
    y_center = (xyxy[1] + xyxy[3]) / 2  # y center
    width = xyxy[2] - xyxy[0]  # width
    height = xyxy[3] - xyxy[1]  # height
    xywh = [x_center, y_center, width, height]
    return xywh


def normalize(xywh, width, height):
    xywh[0], xywh[2] = xywh[0] / width, xywh[2] / width
    xywh[1], xywh[3] = xywh[1] / height, xywh[3] / height
    return xywh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--class-txt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/")
    parser.add_argument("--image-patterns", type=str, nargs="+")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    with open(args.class_txt, encoding="utf-8") as f:
        classes = [x.strip() for x in f.readlines()]
    cls2idx = {cls: int(idx) for idx, cls in enumerate(classes)}

    image_paths = []
    for image_pattern in args.image_patterns or ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        image_paths += list(data_dir.glob(image_pattern))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]

        label_path = image_path.with_suffix(".json")
        if label_path.exists():
            with open(label_path, encoding="utf-8") as f:
                data = json.load(f)
            regions = data["shapes"]
        else:
            regions = []

        regions = list(filter(lambda r: r["label"] in classes, regions))
        label_strs = []
        for region in regions:
            cls_idx = cls2idx[region["label"]]
            xyxy = region2xyxy(region)
            xywh = xyxy2xywh(xyxy)
            xywh = normalize(xywh, width=image_width, height=image_height)
            
            label_str = " ".join(str(x) for x in [cls_idx] + xywh)
            label_strs.append(label_str)
        
        output_txt = output_dir.joinpath(image_path.relative_to(data_dir)).with_suffix(".txt")
        if not output_txt.parent.exists():
            output_txt.parent.mkdir(parents=True)

        with open(output_txt, "w") as f:
            f.write("\n".join(label_strs))
