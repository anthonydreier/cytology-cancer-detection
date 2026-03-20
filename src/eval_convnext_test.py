from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import timm

from src.dataset import CytologyDataset, IDX_TO_LABEL


IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

TEST_CSV = "data/test.csv"
CHECKPOINT_PATH = "results/phase5_convnext_tiny_full/best_model.pt"
OUT_DIR = Path("results/phase6_convnext_tiny_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_model():
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=3,
    )
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []

    for batch in tqdm(loader, desc="test", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_filenames.extend(batch["filename"])

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_filenames,
    )


def save_confusion_matrix(cm, out_path: Path):
    labels = [IDX_TO_LABEL[i] for i in range(3)]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_ds = CytologyDataset(TEST_CSV, split="test", image_size=IMAGE_SIZE)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model().to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    y_true, y_pred, y_prob, filenames = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[IDX_TO_LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc_ovr = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")

    pred_df = pd.DataFrame({
        "filename": filenames,
        "true_idx": y_true,
        "pred_idx": y_pred,
        "true_label": [IDX_TO_LABEL[i] for i in y_true],
        "pred_label": [IDX_TO_LABEL[i] for i in y_pred],
        "prob_" + IDX_TO_LABEL[0]: y_prob[:, 0],
        "prob_" + IDX_TO_LABEL[1]: y_prob[:, 1],
        "prob_" + IDX_TO_LABEL[2]: y_prob[:, 2],
    })
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    save_confusion_matrix(cm, OUT_DIR / "test_confusion_matrix.png")

    summary = {
        "checkpoint_path": CHECKPOINT_PATH,
        "image_size": IMAGE_SIZE,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_macro_auc_ovr": float(auc_ovr),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(OUT_DIR / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTest evaluation complete.")
    print(f"Test accuracy:   {acc:.4f}")
    print(f"Test macro-F1:   {macro_f1:.4f}")
    print(f"Test macro AUC:  {auc_ovr:.4f}")
    print(f"Saved predictions to: {OUT_DIR / 'test_predictions.csv'}")
    print(f"Saved confusion matrix to: {OUT_DIR / 'test_confusion_matrix.png'}")
    print(f"Saved summary to: {OUT_DIR / 'test_summary.json'}")


if __name__ == "__main__":
    main()