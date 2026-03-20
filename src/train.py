from pathlib import Path
import json
import random

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import timm

from src.dataset import CytologyDataset, IDX_TO_LABEL


SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

EPOCHS_HEAD = 3
EPOCHS_FINETUNE = 7

LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"

RESULTS_DIR = Path("results/phase2_resnet50_full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone_resnet(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def make_loaders():
    train_ds = CytologyDataset(TRAIN_CSV, split="train", image_size=IMAGE_SIZE)
    val_ds = CytologyDataset(VAL_CSV, split="val", image_size=IMAGE_SIZE)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def make_model():
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=3,
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[IDX_TO_LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, report, cm


def save_confusion_matrix(cm, out_path: Path):
    labels = [IDX_TO_LABEL[i] for i in range(3)]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_history_plot(history_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader = make_loaders()
    model = make_model().to(device)

    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_report = None
    best_cm = None

    # Stage 1: train head only
    freeze_backbone_resnet(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )

    total_epochs = EPOCHS_HEAD + EPOCHS_FINETUNE
    epoch_num = 0

    for _ in range(EPOCHS_HEAD):
        epoch_num += 1

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, report, cm = evaluate(
            model, val_loader, criterion, device
        )

        row = {
            "epoch": epoch_num,
            "stage": "head",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)

        print(
            f"[Epoch {epoch_num}/{total_epochs}] "
            f"stage=head "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch_num
            best_report = report
            best_cm = cm

            torch.save(
                {
                    "epoch": epoch_num,
                    "model_name": "resnet50",
                    "image_size": IMAGE_SIZE,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                },
                RESULTS_DIR / "best_model.pt",
            )

    # Stage 2: fine-tune all layers
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_FINETUNE,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS_FINETUNE):
        epoch_num += 1

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, report, cm = evaluate(
            model, val_loader, criterion, device
        )

        row = {
            "epoch": epoch_num,
            "stage": "finetune",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)

        print(
            f"[Epoch {epoch_num}/{total_epochs}] "
            f"stage=finetune "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch_num
            best_report = report
            best_cm = cm

            torch.save(
                {
                    "epoch": epoch_num,
                    "model_name": "resnet50",
                    "image_size": IMAGE_SIZE,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                },
                RESULTS_DIR / "best_model.pt",
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULTS_DIR / "history.csv", index=False)
    save_history_plot(history_df, RESULTS_DIR / "history.png")

    if best_cm is not None:
        save_confusion_matrix(best_cm, RESULTS_DIR / "val_confusion_matrix.png")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "best_classification_report": best_report,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {RESULTS_DIR / 'best_model.pt'}")
    print(f"Saved history to: {RESULTS_DIR / 'history.csv'}")
    print(f"Saved history plot to: {RESULTS_DIR / 'history.png'}")
    print(f"Saved confusion matrix to: {RESULTS_DIR / 'val_confusion_matrix.png'}")
    print(f"Saved summary to: {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()