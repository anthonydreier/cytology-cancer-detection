from pathlib import Path
from collections import Counter


RAW_DIR = Path("cytology_data")
ANNOT_DIR = Path("cytology_data_annot")

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def infer_label(name: str) -> str:
	upper = name.upper()
	if upper.startswith("NIL"):
		return "NIL"
	if upper.startswith("LSIL"):
		return "LSIL"
	if upper.startswith("HSIL"):
		return "HSIL"

	return "UNKNOWN"

def count_images(folder: Path):
	files = [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS]
	labels = Counter(infer_label(p.name) for p in files)
	return files, labels

def main():
	if not RAW_DIR.exists():
		print(f"Missing folder: {RAW_DIR}")
		return

	raw_files, raw_labels = count_images(RAW_DIR)
	print("RAW DATA")
	print(f"Total images: {len(raw_files)}")
	for i, j in sorted(raw_labels.items()):
		print(f"	{i}: {j}")

	if ANNOT_DIR.exists():
		annot_files, annot_labels = count_images(ANNOT_DIR)
		print("\nANNOTATED DATA")
		print(f"Total images: {len(annot_files)}")
		for i, j in sorted(annot_labels.items()):
			print(f"	{i}: {j}")
	else:
		print(f"Annotated folder not found: {ANNOT_DIR}")

if __name__ == "__main__":
	main()