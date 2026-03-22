# Cervical Cytology Image Classification on the BMT Dataset Using Transfer Learning

Classification of cervical cytology images into NILM, LSIL, and HSIL using transfer learning.

## Project structure

- `cytology_data/`: raw images
- `cytology_data_annot/`: annotated images (not used for baseline training)
- `src/`: scripts
- `data/`: generated CSV manifests
- `results/`: outputs, plots, checkpoints
- `report/`: short report PDF

## Setup
Only tested on Python 3.12.3
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/sanity_check.py
python src/make_index.py
