import random
import torch

from building_dataset import (
    collect_buildings,
    PairedBuildingDataset,
    MEAN,
    STD
)
from train_siamese import create_scene_split, TrainingDataset

DATA_ROOT = "data/train"

mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)


def apply_photo_augmentation(pre, post):
    # Convert normalized tensors back to RGB [0, 1].
    pre = (pre * std + mean).clamp(0, 1)
    post = (post * std + mean).clamp(0, 1)

    # Same brightness and contrast factors for both images.
    brightness = random.uniform(0.90, 1.10)
    contrast = random.uniform(0.90, 1.10)

    pre_mean = pre.mean(dim=(1, 2), keepdim=True)
    post_mean = post.mean(dim=(1, 2), keepdim=True)

    pre = (pre_mean + contrast * (pre - pre_mean)) * brightness
    post = (post_mean + contrast * (post - post_mean)) * brightness

    pre = pre.clamp(0, 1)
    post = post.clamp(0, 1)

    # Restore ImageNet normalization.
    pre = (pre - mean) / std
    post = (post - mean) / std

    return pre, post


def main():
    print("Collecting records...")
    records = collect_buildings(DATA_ROOT)

    train_records, _ = create_scene_split(records)

    # Use actual training samples.
    base_dataset = TrainingDataset(
        train_records[:32],
        augment=True
    )

    print("Testing 32 paired samples...")

    for i in range(len(base_dataset)):
        pre, post, label = base_dataset[i]

        aug_pre, aug_post = apply_photo_augmentation(
            pre.clone(),
            post.clone()
        )

        assert aug_pre.shape == (3, 128, 128)
        assert aug_post.shape == (3, 128, 128)

        assert torch.isfinite(aug_pre).all()
        assert torch.isfinite(aug_post).all()

        if i == 0:
            print("Original pre range:", pre.min().item(), pre.max().item())
            print("Augmented pre range:", aug_pre.min().item(), aug_pre.max().item())
            print("Original post range:", post.min().item(), post.max().item())
            print("Augmented post range:", aug_post.min().item(), aug_post.max().item())

    print("\nPHOTO AUGMENTATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()