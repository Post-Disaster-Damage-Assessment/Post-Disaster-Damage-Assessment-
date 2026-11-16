import time
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from building_dataset import collect_buildings
from train_siamese import create_scene_split, TrainingDataset
from siamese_model import SiameseResNet18


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading building records...")
    records = collect_buildings("data/train")
    train_records, _ = create_scene_split(records)

    # Select 200 samples from each class
    grouped = defaultdict(list)
    for record in train_records:
        grouped[record["label"]].append(record)

    rng = random.Random(42)
    selected = []

    for label in range(4):
        rng.shuffle(grouped[label])
        samples = grouped[label][:200]
        selected.extend(samples)
        print(f"Class {label}: {len(samples)} samples")

    if len(selected) != 800:
        raise RuntimeError("Could not collect 800 balanced training samples")

    dataset = TrainingDataset(selected, augment=True)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    print("\nLoading pretrained Siamese ResNet18...")
    model = SiameseResNet18(num_classes=4, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for batch_idx, (pre, post, labels) in enumerate(loader):
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pre, post)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Batch {batch_idx + 1}/50 | "
                f"Loss: {loss.item():.4f}"
            )

    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\nTRAINING TEST RESULTS")
    print("------------------------------")
    print("Samples:", 800)
    print("Batches:", batch_idx + 1)
    print("Time:", round(elapsed, 2), "seconds")
    print("Average:", round(elapsed / (batch_idx + 1), 2), "seconds/batch")
    print("Peak GPU memory:", round(peak_memory, 2), "GB")
    print("Final batch loss:", round(loss.item(), 4))
    print("\nSIAMESE TRAINING TEST PASSED")


if __name__ == "__main__":
    main()
