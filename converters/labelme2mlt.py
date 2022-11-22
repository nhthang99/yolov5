import json
import shutil
import argparse
from pathlib import Path

import numpy as np

ignored_fieldnames = [
    "STAMP",
    "V_SIGN",
    "PROFILE_IMG"
]


def get_poly_points(region):
    if region["shape_type"] == "rectangle":
        xmin, ymin = np.array(region["points"]).min(axis=0)
        xmax, ymax = np.array(region["points"]).max(axis=0)
        xmin, ymin, xmax, ymax = map(lambda x: float(round(x)), [xmin, ymin, xmax, ymax])
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    elif region["shape_type"] == "polygon":
        return region["points"]
    else:
        raise Exception(f"{region['shape_type']} is not supported")


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

    path_pairs = [(path, path.with_suffix(".json")) for path in image_paths
                    if path.with_suffix(".json").exists()]
    
    for image_path, label_path in path_pairs:
        with open(label_path) as f:
            data = json.load(f)
        
        regions = []
        for shape in data["shapes"]:
            if shape["label"] in ignored_fieldnames:
                continue

            points = get_poly_points(shape)
            flattened_points = np.array(points).reshape(-1).tolist()
            flattened_points = list(map(str, flattened_points))
            region = ",".join(flattened_points + [shape.get("value", ""), "Vietnamese"])
            regions.append(region)
        
        output_path = output_dir.joinpath(image_path.relative_to(data_dir))
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        shutil.copy(image_path, output_path)
        with open(output_path.with_suffix(".txt"), "w") as f:
            f.write("\n".join(regions))
