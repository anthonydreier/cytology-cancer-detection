from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_TO_IDX = {
    "NIL": 0,
    "LSIL": 1,
    "HSIL": 2,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(split: str, image_size: int = 224):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class CytologyDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, image_size: int = 224):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path).copy()

        # Safety patch
        self.df["label"] = self.df["label"].replace({"NILM": "NIL"})

        self.split = split
        self.transform = get_transforms(split=split, image_size=image_size)

        bad = self.df[~self.df["label"].isin(LABEL_TO_IDX.keys())]
        if not bad.empty:
            raise ValueError(f"Unknown labels found:\n{bad[['filename', 'label']].head()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label_str = row["label"]
        label = LABEL_TO_IDX[label_str]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "label_str": label_str,
            "filepath": img_path,
            "filename": row["filename"],
        }