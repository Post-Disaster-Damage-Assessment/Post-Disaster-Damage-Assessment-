
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from shapely import wkt

from siamese_model import SiameseResNet18
from building_dataset import PairedBuildingDataset, CLASS_NAMES


COLORS = {
    "no-damage": (0, 200, 0),
    "minor-damage": (255, 220, 0),
    "major-damage": (255, 130, 0),
    "destroyed": (220, 0, 0),
}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = SiameseResNet18(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation Macro F1:", checkpoint["macro_f1"])

    return model


def load_scene(root, scene):
    image_dir = root / "images"
    label_dir = root / "labels"

    paths = {
        "pre_image": image_dir / f"{scene}_pre_disaster.png",
        "post_image": image_dir / f"{scene}_post_disaster.png",
        "pre_json": label_dir / f"{scene}_pre_disaster.json",
        "post_json": label_dir / f"{scene}_post_disaster.json",
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    pre_bgr = cv2.imread(str(paths["pre_image"]))
    post_bgr = cv2.imread(str(paths["post_image"]))

    if pre_bgr is None or post_bgr is None:
        raise RuntimeError("Could not read satellite images.")

    if pre_bgr.shape != post_bgr.shape:
        raise ValueError("Pre- and post-disaster image dimensions differ.")

    with open(paths["pre_json"], encoding="utf-8") as f:
        pre_data = json.load(f)

    with open(paths["post_json"], encoding="utf-8") as f:
        post_data = json.load(f)

    return (
        cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB),
        pre_data,
        post_data
    )


def collect_records(pre_data, post_data):
    pre_map = {}

    for feature in pre_data["features"]["xy"]:
        uid = feature.get("properties", {}).get("uid")
        if uid:
            pre_map[uid] = feature

    records = []
    skipped = 0

    for feature in post_data["features"]["xy"]:
        uid = feature.get("properties", {}).get("uid")

        if not uid or uid not in pre_map:
            skipped += 1
            continue

        try:
            pre_geom = wkt.loads(pre_map[uid]["wkt"])
            post_geom = wkt.loads(feature["wkt"])

            if (
                pre_geom.is_empty
                or post_geom.is_empty
                or not pre_geom.is_valid
                or not post_geom.is_valid
            ):
                skipped += 1
                continue

            a = pre_geom.bounds
            b = post_geom.bounds

            bounds = (
                min(a[0], b[0]),
                min(a[1], b[1]),
                max(a[2], b[2]),
                max(a[3], b[3])
            )

            records.append({
                "uid": uid,
                "bounds": bounds,
                "geometry": post_geom
            })

        except Exception:
            skipped += 1

    print("Matched valid buildings:", len(records))
    print("Skipped annotations:", skipped)

    return records


def get_polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]

    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)

    return []


def create_overlay(post_image, records, df, output_path):
    height, width = post_image.shape[:2]

    label_map = np.full(
        (height, width), -1, dtype=np.int8
    )

    for record, prediction in zip(records, df.itertuples()):
        class_index = CLASS_NAMES.index(
            prediction.predicted_damage
        )

        for polygon in get_polygons(record["geometry"]):
            points = np.asarray(
                polygon.exterior.coords,
                dtype=np.float32
            ).round().astype(np.int32)

            cv2.fillPoly(
                label_map,
                [points],
                int(class_index)
            )

    overlay = post_image.copy()

    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = label_map == class_index
        color = np.array(COLORS[class_name], dtype=np.float32)

        overlay[mask] = (
            0.50 * post_image[mask].astype(np.float32)
            + 0.50 * color
        ).astype(np.uint8)

    # Draw polygon boundaries
    for record, prediction in zip(records, df.itertuples()):
        color = COLORS[prediction.predicted_damage]

        for polygon in get_polygons(record["geometry"]):
            points = np.asarray(
                polygon.exterior.coords,
                dtype=np.float32
            ).round().astype(np.int32)

            cv2.polylines(
                overlay,
                [points],
                True,
                color,
                2
            )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    print("Overlay saved:", output_path)


def run_inference(model, records, pre_image, post_image,
                  batch_size, device):

    dataset = PairedBuildingDataset([])

    successful_records = []
    results = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        pre_tensors = []
        post_tensors = []
        valid_records = []

        for record in batch:
            try:
                pre_crop = dataset.extract_crop(
                    pre_image, record["bounds"]
                )

                post_crop = dataset.extract_crop(
                    post_image, record["bounds"]
                )

                pre_tensors.append(
                    dataset.preprocess(pre_crop)
                )

                post_tensors.append(
                    dataset.preprocess(post_crop)
                )

                valid_records.append(record)

            except Exception:
                continue

        if not valid_records:
            continue

        pre_batch = torch.stack(pre_tensors).to(device)
        post_batch = torch.stack(post_tensors).to(device)

        with torch.inference_mode():
            logits = model(pre_batch, post_batch)
            probabilities = torch.softmax(logits, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        for i, record in enumerate(valid_records):
            class_name = CLASS_NAMES[predictions[i].item()]
            probs = probabilities[i].cpu().numpy()

            result = {
                "building_uid": record["uid"],
                "predicted_damage": class_name,
                "confidence": round(float(confidence[i]), 4)
            }

            for j, name in enumerate(CLASS_NAMES):
                result[f"prob_{name}"] = round(
                    float(probs[j]), 4
                )

            results.append(result)
            successful_records.append(record)

        print(
            f"Processed {min(start + batch_size, len(records))}"
            f"/{len(records)} annotations"
        )

    return successful_records, results


def main():
    parser = argparse.ArgumentParser(
        description="xBD Siamese ResNet18 building damage inference"
    )

    parser.add_argument("--scene", required=True)
    parser.add_argument("--root", default="data/train")
    parser.add_argument(
        "--ckpt",
        default="checkpoints_siamese/best_model.pth"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="inference_results")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    root = Path(args.root)
    checkpoint_path = Path(args.ckpt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("SIAMESE RESNET18 DAMAGE INFERENCE")
    print("=" * 60)
    print("Scene:", args.scene)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_model(checkpoint_path, device)

    pre_image, post_image, pre_data, post_data = load_scene(
        root, args.scene
    )

    records = collect_records(pre_data, post_data)

    if not records:
        raise RuntimeError("No valid paired buildings found.")

    successful_records, results = run_inference(
        model,
        records,
        pre_image,
        post_image,
        args.batch_size,
        device
    )

    if not results:
        raise RuntimeError("No buildings were successfully classified.")

    df = pd.DataFrame(results)

    csv_path = output_dir / f"{args.scene}_predictions.csv"
    overlay_path = output_dir / f"{args.scene}_overlay.png"

    df.to_csv(csv_path, index=False)

    create_overlay(
        post_image,
        successful_records,
        df,
        overlay_path
    )

    print("\nDAMAGE DISTRIBUTION")
    print(
        df["predicted_damage"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print("\nAVERAGE CONFIDENCE")
    print(
        df.groupby("predicted_damage")["confidence"].mean()
        .reindex(CLASS_NAMES)
    )

    print("\nFINAL SUMMARY")
    print("Matched buildings:", len(records))
    print("Classified buildings:", len(df))
    print("Failed crops:", len(records) - len(df))
    print("CSV:", csv_path.resolve())
    print("Overlay:", overlay_path.resolve())


if __name__ == "__main__":
    main()
