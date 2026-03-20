# Cytology Cancer Detection

Classification of cervical cytology images into NILM, LSIL, and HSIL using transfer learning.

## Project structure

- `cytology_data/`: raw images
- `cytology_data_annot/`: annotated images (not used for baseline training)
- `src/`: scripts
- `data/`: generated CSV manifests
- `results/`: outputs, plots, checkpoints
- `report/`: short report PDF

## Setup

```bash
pip install -r requirements.txt
python src/sanity_check.py
python src/make_index.py