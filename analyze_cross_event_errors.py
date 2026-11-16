import csv
from pathlib import Path

import numpy as np


INPUT_FILE = Path(
    "event_analysis_v2/event_confusion_matrices_v2.csv"
)

OUTPUT_DIR = Path("event_analysis_v2")

CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

N = len(CLASSES)


def calculate_metrics(cm):
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    precision = np.divide(
        tp, predicted,
        out=np.zeros(N, dtype=float),
        where=predicted != 0,
    )

    recall = np.divide(
        tp, support,
        out=np.zeros(N, dtype=float),
        where=support != 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(N, dtype=float),
        where=(precision + recall) != 0,
    )

    accuracy = tp.sum() / cm.sum()

    return precision, recall, f1, accuracy, support, predicted


def main():
    print("=" * 75)
    print("CROSS-EVENT ERROR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    event_matrices = {}

    # --------------------------------------------------
    # Read event-wise confusion matrices
    # --------------------------------------------------

    with open(INPUT_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            event = row["event"]
            actual_class = row["actual_class"]

            actual_index = CLASSES.index(actual_class)

            if event not in event_matrices:
                event_matrices[event] = np.zeros(
                    (N, N), dtype=np.int64
                )

            for predicted_index, predicted_class in enumerate(CLASSES):
                key = f"predicted_{predicted_class}"

                event_matrices[event][
                    actual_index, predicted_index
                ] = int(row[key])

    if not event_matrices:
        raise ValueError("No event confusion matrices were found.")

    # --------------------------------------------------
    # Aggregate all events
    # --------------------------------------------------

    overall = np.zeros((N, N), dtype=np.int64)

    for cm in event_matrices.values():
        overall += cm

    precision, recall, f1, accuracy, support, predicted = (
        calculate_metrics(overall)
    )

    print("\nOVERALL RESULTS")
    print("-" * 75)
    print("Total samples:", int(overall.sum()))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1.mean():.4f}")

    print("\nCLASS-WISE PERFORMANCE")
    print("-" * 75)

    print(
        f"{'Class':15} {'Support':>9} {'Predicted':>10} "
        f"{'Precision':>11} {'Recall':>9} {'F1':>9}"
    )

    for i, name in enumerate(CLASSES):
        print(
            f"{name:15} "
            f"{support[i]:9d} "
            f"{predicted[i]:10d} "
            f"{precision[i]:11.4f} "
            f"{recall[i]:9.4f} "
            f"{f1[i]:9.4f}"
        )

    # --------------------------------------------------
    # Print overall confusion matrix
    # --------------------------------------------------

    print("\nOVERALL CONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("-" * 75)

    print(
        f"{'Actual':15} " +
        " ".join(f"{name:>15}" for name in CLASSES)
    )

    for i, name in enumerate(CLASSES):
        print(
            f"{name:15} " +
            " ".join(f"{overall[i, j]:15d}" for j in range(N))
        )

    # --------------------------------------------------
    # Identify most frequent errors
    # --------------------------------------------------

    errors = []

    for i in range(N):
        for j in range(N):
            if i != j:
                errors.append({
                    "actual": CLASSES[i],
                    "predicted": CLASSES[j],
                    "count": int(overall[i, j]),
                    "rate": (
                        overall[i, j] / support[i]
                        if support[i] else 0
                    ),
                })

    errors.sort(key=lambda x: x["count"], reverse=True)

    print("\nTOP 10 CLASSIFICATION ERRORS")
    print("-" * 75)

    for item in errors[:10]:
        print(
            f"{item['actual']:15} -> "
            f"{item['predicted']:15} | "
            f"{item['count']:6d} errors | "
            f"{item['rate']:.2%} of actual class"
        )

    # --------------------------------------------------
    # Minor damage analysis
    # --------------------------------------------------

    minor_index = CLASSES.index("minor-damage")

    print("\nMINOR-DAMAGE ERROR BREAKDOWN")
    print("-" * 75)

    minor_total = int(support[minor_index])

    for j, predicted_class in enumerate(CLASSES):
        count = int(overall[minor_index, j])

        print(
            f"Predicted as {predicted_class:15}: "
            f"{count:6d} ({count / minor_total:.2%})"
        )

    # --------------------------------------------------
    # Event-wise error analysis
    # --------------------------------------------------

    event_rows = []

    print("\nEVENT-WISE MINOR-DAMAGE PERFORMANCE")
    print("-" * 75)

    print(
        f"{'Event':25} {'Support':>9} "
        f"{'Correct':>9} {'Recall':>10} "
        f"{'Pred. no-damage':>16} {'Pred. major':>12}"
    )

    for event in sorted(event_matrices):
        cm = event_matrices[event]

        event_support = int(cm[minor_index].sum())
        correct = int(cm[minor_index, minor_index])

        no_damage_errors = int(
            cm[minor_index, CLASSES.index("no-damage")]
        )

        major_errors = int(
            cm[minor_index, CLASSES.index("major-damage")]
        )

        minor_recall = (
            correct / event_support
            if event_support else 0
        )

        print(
            f"{event:25} "
            f"{event_support:9d} "
            f"{correct:9d} "
            f"{minor_recall:10.4f} "
            f"{no_damage_errors:16d} "
            f"{major_errors:12d}"
        )

        event_rows.append({
            "event": event,
            "minor_support": event_support,
            "minor_correct": correct,
            "minor_recall": minor_recall,
            "predicted_no_damage": no_damage_errors,
            "predicted_major_damage": major_errors,
            "predicted_destroyed": int(
                cm[minor_index, CLASSES.index("destroyed")]
            ),
        })

    # --------------------------------------------------
    # Save reports
    # --------------------------------------------------

    with open(
        OUTPUT_DIR / "overall_confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + CLASSES)

        for i, name in enumerate(CLASSES):
            writer.writerow([name] + overall[i].tolist())

    with open(
        OUTPUT_DIR / "cross_event_error_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(event_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(event_rows)

    print("\nSaved reports:")
    print(OUTPUT_DIR / "overall_confusion_matrix.csv")
    print(OUTPUT_DIR / "cross_event_error_summary.csv")


if __name__ == "__main__":
    main()