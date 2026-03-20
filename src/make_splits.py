from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INDEX_CSV = Path("data/index.csv")
OUT_DIR = Path("data")
SEED = 42

def main():
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {INDEX_CSV}. Run src/make_index.py first.")

    df = pd.read_csv(INDEX_CSV)

    # Raw images only.
    df = df[df["is_annotated"] == False].copy()

    # Keep only known labels.
    df = df[df["label"].isin(["NILM", "LSIL", "HSIL"])].copy()

    # Clean columns.
    df = df[["filepath", "filename", "label", "is_annotated"]].reset_index(drop=True)

    print("Raw-only counts:")
    print(df["label"].value_counts().sort_index())
    print()

    # 60 / 20 / 20 split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    all_splits = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    all_splits.to_csv(OUT_DIR / "splits.csv", index=False)

    print("Split sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")
    print()

    print("Train counts:")
    print(train_df["label"].value_counts().sort_index())
    print()

    print("Val counts:")
    print(val_df["label"].value_counts().sort_index())
    print()

    print("Test counts:")
    print(test_df["label"].value_counts().sort_index())
    print()

    # Sanity checks
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["filepath"]).isdisjoint(set(val_df["filepath"]))
    assert set(train_df["filepath"]).isdisjoint(set(test_df["filepath"]))
    assert set(val_df["filepath"]).isdisjoint(set(test_df["filepath"]))

    print("Saved:")
    print("  data/train.csv")
    print("  data/val.csv")
    print("  data/test.csv")
    print("  data/splits.csv")

if __name__ == "__main__":
    main()
