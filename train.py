import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import find_samples, split_samples, XView2
from model import UNet


DATA_ROOT = "data/train"
BATCH_SIZE = 2
EPOCHS = 15
LR = 1e-3
CROP_SIZE = 512
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("checkpoints", exist_ok=True)


def main():
    samples = find_samples(DATA_ROOT)
    train_samples, val_samples = split_samples(samples, val_frac=0.1, seed=42)

    print("Total samples :", len(samples))
    print("Train samples :", len(train_samples))
    print("Val samples   :", len(val_samples))
    print("Device        :", DEVICE)

    train_ds = XView2(train_samples, train=True, crop=CROP_SIZE)
    val_ds = XView2(val_samples, train=False, crop=CROP_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet(
        in_channels=6,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, (pre, post, mask) in enumerate(train_loader):
            pre = pre.to(DEVICE, non_blocking=True)
            post = post.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            x = torch.cat([pre, post], dim=1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type="cuda",
                enabled=(DEVICE == "cuda")
            ):
                output = model(x)
                loss = criterion(output, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for pre, post, mask in val_loader:
                pre = pre.to(DEVICE, non_blocking=True)
                post = post.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                x = torch.cat([pre, post], dim=1)

                with autocast(
                    device_type="cuda",
                    enabled=(DEVICE == "cuda")
                ):
                    output = model(x)
                    loss = criterion(output, mask)

                val_loss += loss.item()

                prediction = output.argmax(dim=1)

                valid = mask != 255

                correct += (
                    (prediction[valid] == mask[valid])
                    .sum()
                    .item()
                )

                total += valid.sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pixel Acc: {accuracy:.4f}"
        )
        print()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss
        }

        torch.save(
            checkpoint,
            f"checkpoints/epoch_{epoch + 1}.pth"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                checkpoint,
                "checkpoints/best_model.pth"
            )

            print("Saved new best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()
