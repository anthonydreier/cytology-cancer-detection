from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SPLITS_CSV = Path("data/splits.csv")
OUT_ROOT = Path("data/cropped_raw")

WHITE_THRESHOLD = 245
COLOR_SPREAD_THRESHOLD = 12
OPEN_KERNEL = 3
CLOSE_KERNEL = 9
DILATE_KERNEL = 5
MARGIN_FRAC = 0.08


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    # Conservative: keep anything not-near-white OR having some color spread.
    min_chan = rgb.min(axis=2)
    max_chan = rgb.max(axis=2)
    not_white = min_chan < WHITE_THRESHOLD
    colorful = (max_chan - min_chan) > COLOR_SPREAD_THRESHOLD

    mask = (not_white | colorful).astype(np.uint8) * 255

    open_kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    dilate_kernel = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


def crop_to_foreground(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mask = build_foreground_mask(rgb)

    ys, xs = np.where(mask > 0)

    # Fallback: if mask is empty, keep full image.
    if len(xs) == 0 or len(ys) == 0:
        return rgb

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    margin = int(MARGIN_FRAC * max(box_w, box_h))

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w - 1, x1 + margin)
    y1 = min(h - 1, y1 + margin)

    cropped = rgb[y0:y1 + 1, x0:x1 + 1]
    return cropped


def pad_to_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)

    canvas = np.full((side, side, 3), 255, dtype=np.uint8)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = rgb

    return canvas


def save_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")


def main():
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Missing {SPLITS_CSV}")

    df = pd.read_csv(SPLITS_CSV).copy()

    new_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping"):
        src_path = row["filepath"]
        label = row["label"]
        filename = row["filename"]

        rgb = load_rgb(src_path)
        cropped = crop_to_foreground(rgb)
        squared = pad_to_square(cropped)

        out_path = OUT_ROOT / label / filename
        save_rgb(out_path, squared)

        new_row = row.copy()
        new_row["filepath"] = str(out_path.resolve())
        new_rows.append(new_row)

    cropped_df = pd.DataFrame(new_rows)

    cropped_df.to_csv("data/splits_cropped.csv", index=False)
    cropped_df[cropped_df["split"] == "train"].to_csv("data/train_cropped.csv", index=False)
    cropped_df[cropped_df["split"] == "val"].to_csv("data/val_cropped.csv", index=False)
    cropped_df[cropped_df["split"] == "test"].to_csv("data/test_cropped.csv", index=False)

    print("Saved:")
    print("  data/splits_cropped.csv")
    print("  data/train_cropped.csv")
    print("  data/val_cropped.csv")
    print("  data/test_cropped.csv")
    print(f"  cropped images under {OUT_ROOT}")


if __name__ == "__main__":
    main()