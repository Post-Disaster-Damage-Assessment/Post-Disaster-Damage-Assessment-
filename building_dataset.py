import json
import cv2
import numpy as np
import torch

from pathlib import Path
from shapely import wkt
from torch.utils.data import Dataset
from collections import Counter

LABELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

CLASS_NAMES = list(LABELS.keys())

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def collect_buildings(root):
    root = Path(root)
    label_dir = root / "labels"
    image_dir = root / "images"

    records = []

    for post_file in sorted(label_dir.glob("*_post_disaster.json")):

        pre_file = post_file.with_name(
            post_file.name.replace(
                "_post_disaster", "_pre_disaster"
            )
        )

        pre_image = image_dir / pre_file.name.replace(".json", ".png")
        post_image = image_dir / post_file.name.replace(".json", ".png")

        if not pre_file.exists():
            continue

        if not pre_image.exists() or not post_image.exists():
            continue

        with open(pre_file, encoding="utf-8") as f:
            pre_features = json.load(f)["features"]["xy"]

        with open(post_file, encoding="utf-8") as f:
            post_features = json.load(f)["features"]["xy"]

        pre_map = {
            x["properties"]["uid"]: x
            for x in pre_features
            if x.get("properties", {}).get("uid")
        }

        for post in post_features:

            properties = post.get("properties", {})
            uid = properties.get("uid")
            subtype = properties.get("subtype")

            if subtype not in LABELS or uid not in pre_map:
                continue

            try:
                pre_polygon = wkt.loads(pre_map[uid]["wkt"])
                post_polygon = wkt.loads(post["wkt"])

                if pre_polygon.is_empty or post_polygon.is_empty:
                    continue

                if not pre_polygon.is_valid or not post_polygon.is_valid:
                    continue

                bounds_a = pre_polygon.bounds
                bounds_b = post_polygon.bounds

                min_x = min(bounds_a[0], bounds_b[0])
                min_y = min(bounds_a[1], bounds_b[1])
                max_x = max(bounds_a[2], bounds_b[2])
                max_y = max(bounds_a[3], bounds_b[3])

                records.append({
                    "pre_image": str(pre_image),
                    "post_image": str(post_image),
                    "bounds": (min_x, min_y, max_x, max_y),
                    "label": LABELS[subtype],
                    "uid": uid,
                    "scene": post_file.stem.replace(
                        "_post_disaster", ""
                    )
                })

            except Exception:
                continue

    return records


class PairedBuildingDataset(Dataset):

    def __init__(self, records, output_size=128):
        self.records = records
        self.output_size = output_size

    def __len__(self):
        return len(self.records)

    def extract_crop(self, image, bounds):

        min_x, min_y, max_x, max_y = bounds

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        width = max_x - min_x
        height = max_y - min_y

        side = max(width, height) * 1.5
        side = max(side, 64)

        x1 = int(np.floor(center_x - side / 2))
        y1 = int(np.floor(center_y - side / 2))
        x2 = int(np.ceil(center_x + side / 2))
        y2 = int(np.ceil(center_y + side / 2))

        h, w = image.shape[:2]

        left = max(0, -x1)
        top = max(0, -y1)
        right = max(0, x2 - w)
        bottom = max(0, y2 - h)

        crop = image[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2)
        ]

        if crop.size == 0:
            raise ValueError("Empty building crop")

        if any((left, top, right, bottom)):
            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right,
                cv2.BORDER_REFLECT_101
            )

        crop = cv2.resize(
            crop,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA
        )

        return crop

    def preprocess(self, image):

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD

        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        return torch.from_numpy(image)

    def __getitem__(self, index):

        record = self.records[index]

        pre_image = cv2.imread(record["pre_image"])
        post_image = cv2.imread(record["post_image"])

        if pre_image is None or post_image is None:
            raise ValueError("Could not read paired satellite images")

        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        pre_crop = self.extract_crop(pre_image, record["bounds"])
        post_crop = self.extract_crop(post_image, record["bounds"])

        pre_tensor = self.preprocess(pre_crop)
        post_tensor = self.preprocess(post_crop)

        label = torch.tensor(record["label"], dtype=torch.long)

        return pre_tensor, post_tensor, label
