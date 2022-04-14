import json
import argparse
from pathlib import Path
import shutil

import numpy as np


class_names = [
    "DEG_TYPE_1",
    "DEG_TYPE_2",
    "THPT_TYPE_1",
    "THPT_TYPE_2",
    "THPT_TYPE_3",
    "THPT_TYPE_4",
    "DOCUMENT"
]

def xyxy2xywh(xyxy):
    x = (xyxy[0] + xyxy[2]) / 2
    y = (xyxy[1] + xyxy[3]) / 2
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return [x, y, w, h]


def normalize(xywh, image_width, image_height):
    xywh[0] = xywh[0] / image_width
    xywh[1] = xywh[1] / image_height
    xywh[2] = xywh[2] / image_width
    xywh[3] = xywh[3] / image_height
    return xywh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--image-patterns", type=str, nargs="+")
    parser.add_argument("--output-dir", type=str, default="output/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    image_paths = []

    for image_pattern in args.image_patterns or ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        image_paths += list(data_dir.glob(image_pattern))
    
    for image_path in image_paths:
        label_path = image_path.with_suffix(".json")
        if label_path.exists():
            with open(label_path) as f:
                data = json.load(f)

            regions = list(filter(lambda x: x["label"] in class_names, data["shapes"]))

            output_path = output_dir.joinpath(image_path.parents[1].relative_to(data_dir), image_path.name)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)

            with open(output_path.with_suffix(".txt"), "w") as f:
                for region in regions:
                    xmin, ymin = np.array(region["points"]).min(axis=0)
                    xmax, ymax = np.array(region["points"]).max(axis=0)
                    xyxy = list(map(lambda x: float(x.item()), [xmin, ymin, xmax, ymax]))
                    xywh = xyxy2xywh(xyxy)
                    xywh = normalize(xywh, data["imageWidth"], data["imageHeight"])

                    f.write(" ".join(map(str, [class_names.index(region["label"])] + xywh)))
                    f.write("\n")

            shutil.copy(image_path, output_path)
