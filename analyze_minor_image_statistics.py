
import csv
import random
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict

from building_dataset import collect_buildings


# ---------------- CONFIGURATION ----------------

DATA_ROOT = "data/train"
OUTPUT_DIR = Path("event_visual_comparison")

TARGET_EVENTS = {
    "hurricane-matthew",
    "hurricane-michael",
    "hurricane-harvey"
}

SEED = 42
VAL_RATIO = 0.20

OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE_EVENTS = sorted(TARGET_EVENTS)


# ---------------- EXACT VALIDATION SPLIT ----------------

def create_scene_split(records):
    events = defaultdict(lambda: defaultdict(list))

    for record in records:
        scene = record["scene"]
        event = scene.rsplit("_", 1)[0]
        events[event][scene].append(record)

    rng = random.Random(SEED)

    train_records = []
    val_records = []

    for event in sorted(events):
        scenes = sorted(events[event])
        rng.shuffle(scenes)

        val_count = max(1, round(len(scenes) * VAL_RATIO))

        val_scenes = set(scenes[:val_count])
        train_scenes = set(scenes[val_count:])

        for scene in train_scenes:
            train_records.extend(events[event][scene])

        for scene in val_scenes:
            val_records.extend(events[event][scene])

    return train_records, val_records


# ---------------- RAW CROP EXTRACTION ----------------

def extract_raw_crop(image, bounds):

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
        raise ValueError("Empty crop")

    if any((left, top, right, bottom)):
        crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right,
            cv2.BORDER_REFLECT_101
        )

    return crop


# ---------------- IMAGE STATISTICS ----------------

def calculate_statistics(pre, post):

    pre_gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)

    pre_rgb_mean = np.mean(pre, axis=(0, 1))
    post_rgb_mean = np.mean(post, axis=(0, 1))

    return {
        "pre_brightness": float(np.mean(pre_gray)),
        "post_brightness": float(np.mean(post_gray)),

        "brightness_change": float(
            np.mean(post_gray) - np.mean(pre_gray)
        ),

        "pre_contrast": float(np.std(pre_gray)),
        "post_contrast": float(np.std(post_gray)),

        "pre_red": float(pre_rgb_mean[0]),
        "pre_green": float(pre_rgb_mean[1]),
        "pre_blue": float(pre_rgb_mean[2]),

        "post_red": float(post_rgb_mean[0]),
        "post_green": float(post_rgb_mean[1]),
        "post_blue": float(post_rgb_mean[2]),

        "mean_absolute_pixel_difference": float(
            np.mean(
                np.abs(
                    pre.astype(np.float32) -
                    post.astype(np.float32)
                )
            )
        )
    }


# ---------------- MAIN ----------------

def main():

    print("DEVICE: CPU image statistics")
    print("\nCollecting buildings...")

    records = collect_buildings(DATA_ROOT)

    print("Total buildings:", len(records))

    _, val_records = create_scene_split(records)

    selected = [
        record for record in val_records
        if record["label"] == 1
        and record["scene"].rsplit("_", 1)[0] in TARGET_EVENTS
    ]

    print("Selected validation buildings:", len(selected))

    rows = []

    cached_pre_path = None
    cached_post_path = None
    pre_image = None
    post_image = None

    for index, record in enumerate(selected, start=1):

        try:
            if (
                record["pre_image"] != cached_pre_path
                or record["post_image"] != cached_post_path
            ):

                pre_bgr = cv2.imread(record["pre_image"])
                post_bgr = cv2.imread(record["post_image"])

                if pre_bgr is None or post_bgr is None:
                    raise ValueError("Image could not be read")

                pre_image = cv2.cvtColor(
                    pre_bgr, cv2.COLOR_BGR2RGB
                )

                post_image = cv2.cvtColor(
                    post_bgr, cv2.COLOR_BGR2RGB
                )

                cached_pre_path = record["pre_image"]
                cached_post_path = record["post_image"]

            pre_crop = extract_raw_crop(
                pre_image, record["bounds"]
            )

            post_crop = extract_raw_crop(
                post_image, record["bounds"]
            )

            stats = calculate_statistics(
                pre_crop, post_crop
            )

            event = record["scene"].rsplit("_", 1)[0]

            row = {
                "event": event,
                "scene": record["scene"],
                "uid": record["uid"],
                "crop_width": pre_crop.shape[1],
                "crop_height": pre_crop.shape[0],
                **stats
            }

            rows.append(row)

        except Exception as error:
            print(
                f"Skipped {index}: {error}"
            )

        if index % 200 == 0:
            print(f"Processed {index}/{len(selected)}")

    # Save individual building measurements.

    detail_path = OUTPUT_DIR / "minor_image_statistics.csv"

    if rows:
        with open(
            detail_path, "w",
            newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    # Generate event-level summary.

    metric_names = [
        "pre_brightness",
        "post_brightness",
        "brightness_change",
        "pre_contrast",
        "post_contrast",
        "pre_red",
        "pre_green",
        "pre_blue",
        "post_red",
        "post_green",
        "post_blue",
        "mean_absolute_pixel_difference",
        "crop_width",
        "crop_height"
    ]

    summary = []

    for event in DEVICE_EVENTS:

        event_rows = [
            row for row in rows
            if row["event"] == event
        ]

        if not event_rows:
            continue

        result = {
            "event": event,
            "samples": len(event_rows)
        }

        for metric in metric_names:
            values = [
                float(row[metric])
                for row in event_rows
            ]

            result[f"{metric}_mean"] = float(
                np.mean(values)
            )

            result[f"{metric}_median"] = float(
                np.median(values)
            )

        summary.append(result)

    summary_path = (
        OUTPUT_DIR / "minor_event_statistics_summary.csv"
    )

    if summary:
        with open(
            summary_path, "w",
            newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file, fieldnames=list(summary[0].keys())
            )

            writer.writeheader()
            writer.writerows(summary)

    print("\n" + "=" * 75)
    print("MINOR-DAMAGE IMAGE STATISTICS")
    print("=" * 75)

    for result in summary:

        print(f"\n{result['event']}")
        print("Samples:", result["samples"])

        for metric in [
            "pre_brightness",
            "post_brightness",
            "brightness_change",
            "pre_contrast",
            "post_contrast",
            "mean_absolute_pixel_difference",
            "crop_width",
            "crop_height"
        ]:

            print(
                f"{metric:38s}: "
                f"{result[metric + '_mean']:.3f}"
            )

    print("\nSaved:", detail_path.resolve())
    print("Saved:", summary_path.resolve())
    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
