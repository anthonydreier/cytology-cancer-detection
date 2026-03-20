from pathlib import Path
import pandas as pd

RAW_DIR = Path("cytology_data")
ANNOT_DIR = Path("cytology_data_annot")
OUT_CSV = Path("data/index.csv")

VLAID_EXTS = {".jpg", ".jpeg", ".png"}

def infer_label(name: str) -> str:
	upper = name.upper()
	if upper.startswith("NIL"):
		return "NILM"
	if upper.startswith("LSIL"):
		return "LSIL"
	if upper.startswith("HSIL"):
		return "HSIL"
	return "UNKNOWN"

def collect(folder: Path, is_annotated: bool):
	rows = []
	if not folder.exists():
		return rows

	for p in folder.rglob("*"):
		if p.suffix.lower() not in VLAID_EXTS:
			continue

		rows.append({
			"filepath": str(p.resolve()),
			"filename": p.name,
			"label": infer_label(p.name),
			"is_annotated": is_annotated,
			})

	return rows

def main():
	rows = []
	rows.extend(collect(RAW_DIR, is_annotated=False))
	rows.extend(collect(ANNOT_DIR, is_annotated=True))

	df = pd.DataFrame(rows)
	df = df.sort_values(["is_annotated", "label", "filename"]).reset_index(drop=True)

	OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUT_CSV, index=False)

	print(f"Saved {len(df)} rows to {OUT_CSV}")
	print("\nCounts by annotation + label:")
	print(df.groupby(["is_annotated", "label"]).size())

	unknown = df[df["label"] == "UNKNOWN"]
	if not unknown.empty:
		print("\nWARNING: Unknown label rows found:")
		print(unknown[["filename", "filepath"]].head(20).to_string(index=False))

if __name__ == "__main__":
	main()