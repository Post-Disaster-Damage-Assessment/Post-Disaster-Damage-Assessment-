
import csv
import json
from pathlib import Path
from datetime import date

BASE = Path(".")
OUTPUT = Path("FINAL_EXPERIMENTAL_RESULTS.md")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    return f"{float(value) * 100:.2f}%"


def dec(value):
    return f"{float(value):.4f}"


def table(headers, rows):
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for row in rows:
        result.append("| " + " | ".join(map(str, row)) + " |")

    return "\n".join(result)


def main():

    original = read_json(
        BASE / "comparison_results/original/validation_metrics.json"
    )

    augmented = read_json(
        BASE / "comparison_results/photo_aug/validation_metrics.json"
    )

    event_metrics = read_csv(
        BASE / "comparison_results/event_comparison/event_metrics_comparison.csv"
    )

    event_classes = read_csv(
        BASE / "comparison_results/event_comparison/event_class_metrics_comparison.csv"
    )

    size_metrics = read_csv(
        BASE / "building_size_comparison/building_size_metrics_comparison.csv"
    )

    global_comparison = read_csv(
        BASE / "comparison_results/event_comparison/global_metrics_comparison.csv"
    )

    prediction_changes_path = (
        BASE / "comparison_results/prediction_changes/"
        "building_prediction_changes.csv"
    )

    # --------------------------------------------------
    # Global metrics
    # --------------------------------------------------

    global_rows = []

    for name, metrics in [
        ("Original", original),
        ("Photometric augmentation", augmented)
    ]:
        global_rows.append([
            name,
            metrics["total_buildings"],
            pct(metrics["accuracy"]),
            dec(metrics["macro_f1"]),
            dec(metrics["weighted_f1"])
        ])

    global_table = table(
        ["Model", "Buildings", "Accuracy", "Macro F1", "Weighted F1"],
        global_rows
    )

    # --------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------

    class_rows = []

    for old, new in zip(
        original["class_metrics"],
        augmented["class_metrics"]
    ):
        class_rows.append([
            old["class"],
            old["support"],
            dec(old["f1"]),
            dec(new["f1"]),
            f"{new['f1'] - old['f1']:+.4f}",
            pct(old["recall"]),
            pct(new["recall"])
        ])

    class_table = table(
        [
            "Class", "Support", "Original F1",
            "Augmented F1", "F1 Change",
            "Original Recall", "Augmented Recall"
        ],
        class_rows
    )

    # --------------------------------------------------
    # Event-wise performance
    # --------------------------------------------------

    event_rows = []

    for row in event_metrics:
        event_rows.append([
            row["event"],
            row["buildings"],
            pct(row["original_accuracy"]),
            pct(row["photo_aug_accuracy"]),
            f"{float(row['accuracy_change']) * 100:+.2f} pp",
            dec(row["original_macro_f1"]),
            dec(row["photo_aug_macro_f1"]),
            f"{float(row['macro_f1_change']):+.4f}"
        ])

    event_table = table(
        [
            "Event", "Buildings",
            "Original Accuracy", "Augmented Accuracy",
            "Accuracy Change",
            "Original Macro F1", "Augmented Macro F1",
            "Macro F1 Change"
        ],
        event_rows
    )

    improved_events = sum(
        float(row["macro_f1_change"]) > 0
        for row in event_metrics
    )

    declined_events = sum(
        float(row["macro_f1_change"]) < 0
        for row in event_metrics
    )

    # --------------------------------------------------
    # Event-wise class metrics
    # --------------------------------------------------

    event_class_rows = []

    for row in event_classes:
        event_class_rows.append([
            row["event"],
            row["class"],
            row["support"],
            dec(row["original_f1"]),
            dec(row["photo_aug_f1"]),
            f"{float(row['f1_change']):+.4f}"
        ])

    event_class_table = table(
        [
            "Event", "Class", "Support",
            "Original F1", "Augmented F1", "Change"
        ],
        event_class_rows
    )

    # --------------------------------------------------
    # Building-size results
    # --------------------------------------------------

    size_lookup = {}

    for row in size_metrics:
        if row["class"] == "minor-damage":
            size_lookup[(row["model"], row["size_group"])] = row

    size_rows = []

    for group in [
        "Smallest",
        "Small-medium",
        "Large-medium",
        "Largest"
    ]:
        old = size_lookup[("original", group)]
        new = size_lookup[("photo_aug", group)]

        size_rows.append([
            group,
            old["support"],
            dec(old["f1"]),
            dec(new["f1"]),
            f"{float(new['f1']) - float(old['f1']):+.4f}",
            pct(old["group_accuracy"]),
            pct(new["group_accuracy"])
        ])

    size_table = table(
        [
            "Size Group", "Minor Support",
            "Original Minor F1", "Augmented Minor F1",
            "F1 Change", "Original Accuracy",
            "Augmented Accuracy"
        ],
        size_rows
    )

    # --------------------------------------------------
    # Prediction transition results
    # --------------------------------------------------

    transition_section = (
        "Prediction-transition analysis is available in "
        "`comparison_results/prediction_changes/`."
    )

    if prediction_changes_path.exists():
        changes = read_csv(prediction_changes_path)

        transition_section = (
            "Building-level prediction transitions were saved in "
            "`comparison_results/prediction_changes/"
            "building_prediction_changes.csv`. "
            "The analysis previously established: 27,561 buildings "
            "correct under both models, 3,656 incorrect under both, "
            "517 changed from incorrect to correct, and 417 changed "
            "from correct to incorrect. Net additional correct "
            "predictions: 100."
        )

    # --------------------------------------------------
    # Report content
    # --------------------------------------------------

    content = f"""# Post-Disaster Damage Assessment from Satellite Imagery

## Final Experimental Results

**Report date:** {date.today().isoformat()}

## 1. Project overview

This project investigates building-level damage classification using
paired pre-disaster and post-disaster satellite imagery from the
xView2/xBD dataset.

A Siamese ResNet18 architecture extracts features from both images
using a shared backbone. The resulting feature representations are
combined and passed through a classification head to predict one of
four building damage categories:

- No damage
- Minor damage
- Major damage
- Destroyed

The task is building-level damage classification. It is not
pixel-level damage segmentation.

## 2. Dataset and evaluation methodology

- Dataset: xView2/xBD training data.
- Total validation buildings: {original['total_buildings']:,}.
- Validation scenes: {original['validation_scenes']}.
- Validation split: scene-level, event-wise split.
- Random seed: 42.
- Validation ratio: 20%.
- Input: paired pre-disaster and post-disaster building crops.
- Model input resolution: 128 × 128 pixels per image.
- Number of damage classes: 4.

Both checkpoints were evaluated using the same validation records
and preprocessing pipeline.

## 3. Model architecture

The project uses a Siamese ResNet18 architecture with shared weights.

The two branches process pre-disaster and post-disaster building
crops. Each branch produces a 512-dimensional feature vector.

The feature fusion operation concatenates:

1. Pre-disaster features.
2. Post-disaster features.
3. Absolute feature difference.
4. Element-wise feature product.

The resulting 2,048-dimensional representation is passed through
a classification head:

- Linear: 2048 → 512
- Batch normalization
- ReLU
- Dropout (0.3)
- Linear: 512 → 128
- ReLU
- Dropout (0.2)
- Linear: 128 → 4

The model contains approximately 12.3 million parameters.

## 4. Experimental configurations

### 4.1 Original model

- Architecture: Siamese ResNet18.
- Optimizer: AdamW.
- Initial learning rate: 0.0001.
- Weight decay: 0.0001.
- Loss: weighted cross-entropy.
- Geometric augmentation: horizontal flip, vertical flip and
  rotations by multiples of 90 degrees.
- Best checkpoint: epoch 5.

Checkpoint:
`checkpoints_siamese/best_model.pth`

### 4.2 Photometric augmentation experiment

The experiment started from the original best checkpoint.

Additional synchronized brightness and contrast transformations
were applied to the paired pre-disaster and post-disaster crops.

- Brightness factor: 0.90–1.10.
- Contrast factor: 0.90–1.10.
- Learning rate: 0.00001.
- Optimizer: AdamW.
- Maximum epochs: 5.
- Early stopping: patience 2.
- Best checkpoint: epoch 1.

Checkpoint:
`checkpoints_photo_aug/best_photo_aug_model.pth`

The original checkpoint was preserved.

## 5. Global model comparison

{global_table}

### 5.1 Interpretation

The photometric augmentation experiment improved:

- Accuracy by approximately 0.31 percentage points.
- Macro F1 by approximately 0.0032.
- Weighted F1 by approximately 0.0023.

The improvement is measurable but modest.

## 6. Class-wise performance

{class_table}

### 6.1 Interpretation

- No-damage classification remains the strongest category.
- Destroyed-building classification performs comparatively well.
- Major-damage F1 improves from 0.6877 to 0.6950.
- Minor-damage remains the weakest category.

Minor-damage recall decreases from approximately 57.20% to 56.31%,
even though its precision increases.

Therefore, photometric augmentation does not resolve the
minor-damage classification difficulty.

## 7. Event-wise performance

{event_table}

### 7.1 Event-level findings

- {improved_events} of 10 events improved in macro F1.
- {declined_events} of 10 events declined in macro F1.
- The response to photometric augmentation varies by disaster event.
- Event-level performance should not be interpreted as proof of
  domain shift without further investigation.

The event-wise results indicate that global performance improvements
can conceal event-specific deterioration.

## 8. Event-wise class performance

{event_class_table}

This table provides class-level precision, recall and F1 comparisons
for every event. Small class supports should be interpreted cautiously.

## 9. Building-size analysis

Building size was calculated using the area of each building's
polygon bounding box:

`area = bounding_box_width × bounding_box_height`

The validation buildings were divided into four groups using fixed
area quartiles:

- Q1: 422.1419
- Q2: 968.7842
- Q3: 1849.5001

The same boundaries were used for both checkpoints.

### 9.1 Minor-damage performance by size

{size_table}

### 9.2 Interpretation

The original model shows declining minor-damage F1 across the
four building-size groups.

The augmented model improves minor-damage F1 in the large-medium
group but declines in the largest group.

Building-size subgroup performance is descriptive and does not
establish that building size causes classification difficulty.

## 10. Error analysis

The original validation model produced 4,173 incorrect predictions
out of 32,151 buildings, corresponding to an error rate of 12.98%.

Class-wise error rates:

| Class | Errors | Error rate |
|---|---:|---:|
| No damage | 1,894 | 7.83% |
| Minor damage | 1,157 | 42.80% |
| Major damage | 797 | 28.65% |
| Destroyed | 325 | 13.06% |

The most frequent confusion was no-damage buildings being classified
as minor damage.

Minor damage was the most difficult class in the original model.

## 11. Confidence and calibration analysis

Original model findings:

- Expected Calibration Error (ECE): approximately 0.0080.
- Brier score: approximately 0.1901.
- Log loss: approximately 0.3757.
- Incorrect predictions with confidence of at least 0.80: 870.

Although overall calibration error is low, calibration varies
between disaster events. Hurricane Matthew showed substantially
higher event-specific calibration error.

Confidence should therefore not be treated as a guarantee of
prediction correctness.

## 12. Selective prediction analysis

At a confidence threshold of 0.80, the original model retained:

- Coverage: 73.63%.
- Accuracy among accepted predictions: 96.32%.
- Accepted predictions: 23,673.
- Predictions requiring review: 8,478.

This is a retrospective validation analysis. It does not establish
a deployment-ready confidence threshold.

## 13. Prediction-transition analysis

{transition_section}

The augmentation experiment changes a subset of building predictions.
The overall improvement is modest rather than a complete change
in model behaviour.

## 14. Main findings

1. A Siamese ResNet18 model can classify four building damage
   categories from paired satellite imagery.
2. The original model achieves 87.02% validation accuracy and
   0.7442 macro F1.
3. Photometric augmentation improves validation accuracy to 87.33%
   and macro F1 to 0.7474.
4. Major-damage classification shows the largest global F1
   improvement.
5. Minor-damage classification remains the main weakness.
6. Performance varies considerably between disaster events.
7. Building-size subgroup analysis reveals different performance
   patterns across size groups.
8. Confidence-based selective prediction can reduce errors among
   accepted predictions, at the cost of reduced coverage.

## 15. Limitations

- No independent test dataset was available in the current project
  setup. Results are validation results, not independent test scores.
- The best checkpoint and augmentation experiment were selected
  using validation performance.
- The dataset contains four building-level damage categories.
- The system does not produce pixel-level damage segmentation.
- Event-level performance can vary substantially.
- The photometric augmentation improvement is modest.
- A single-scene inference demonstration cannot establish
  generalization to unseen disasters.
- Confidence scores do not guarantee prediction correctness.
- Building-size and image-statistics relationships are observational
  and do not establish causality.

## 16. Future improvements

- Evaluate on an independent disaster dataset or held-out test set.
- Investigate minor-damage classification using targeted experiments.
- Explore event-robust training and domain generalization.
- Investigate multi-scale building representations.
- Improve uncertainty estimation and confidence calibration.
- Develop a demonstration interface for paired satellite imagery.
- Evaluate performance on geographically and environmentally
  different disaster events.

## 17. Reproducibility and output files

Important project outputs:

- `comparison_results/original/validation_metrics.json`
- `comparison_results/photo_aug/validation_metrics.json`
- `comparison_results/event_comparison/event_metrics_comparison.csv`
- `comparison_results/event_comparison/event_class_metrics_comparison.csv`
- `comparison_results/event_comparison/event_confusion_comparison.csv`
- `building_size_comparison/building_size_metrics_comparison.csv`
- `comparison_results/prediction_changes/building_prediction_changes.csv`

The original and augmented checkpoints are stored separately.

---


## 18. Experimental visualizations

### 18.1 Overall model performance

![Global performance comparison](final_visualizations/01_global_performance.png)

### 18.2 Confusion matrices

![Original and augmented confusion matrices](final_visualizations/02_confusion_matrices.png)

### 18.3 Event-wise performance

![Event-wise macro F1 comparison](final_visualizations/03_event_wise_macro_f1.png)

### 18.4 Class-wise performance

![Class-wise F1 comparison](final_visualizations/04_class_wise_f1.png)

### 18.5 Building-size accuracy

![Accuracy across building-size groups](final_visualizations/05_building_size_accuracy.png)

### 18.6 Class-wise performance by building size

![Class-wise F1 across building-size groups](final_visualizations/06_class_f1_by_building_size.png)

### 18.7 Confidence calibration

![Confidence calibration curve](final_visualizations/07_confidence_calibration.png)

---

**End of experimental results report.**

"""

    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 65)
    print("FINAL EXPERIMENTAL REPORT GENERATED")
    print("=" * 65)
    print("File:", OUTPUT.resolve())
    print("Size:", OUTPUT.stat().st_size, "bytes")
    print("Events included:", len(event_metrics))
    print("Event-class rows:", len(event_classes))
    print("Building-size rows:", len(size_metrics))


if __name__ == "__main__":
    main()
