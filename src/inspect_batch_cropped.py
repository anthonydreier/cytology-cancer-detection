from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.dataset import CytologyDataset, IDX_TO_LABEL, IMAGENET_MEAN, IMAGENET_STD

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def unnormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std) + mean


def main():
    dataset = CytologyDataset("data/train_cropped.csv", split="train", image_size=224)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]

    print("Batch tensor shape:", images.shape)
    print("Label tensor shape:", labels.shape)
    print("First 16 labels:", [IDX_TO_LABEL[int(x)] for x in labels])

    images_vis = unnormalize(images).clamp(0, 1)
    grid = make_grid(images_vis, nrow=4)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    out_path = RESULTS_DIR / "phase3_train_batch_cropped.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved batch grid to {out_path}")


if __name__ == "__main__":
    main()