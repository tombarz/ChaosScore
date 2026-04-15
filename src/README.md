# scFoundation Pipeline Scaffold

This repo uses one canonical scFoundation preprocessing contract:

- raw/count-like `.h5ad` in
- prepared scFoundation prefix out:
  - `<prefix>.counts_19264.npz`
  - `<prefix>.obs.csv.gz`
  - `<prefix>.var.csv`
  - `<prefix>.summary.json`
  - `<prefix>.manifest.json`

The fine-tuning path then consumes that prepared prefix directly.

The repo contains a separate fine-tuning entrypoint for cell-type-conditioned masked gene prediction:
- `scripts/train_scfoundation_conditioned_masked_task.py`
- `src/data/scfoundation_masked_dataset.py`
- `src/tasks/masked_gene_prediction.py`
- `src/models/scfoundation_conditioned_mgp.py`

## Canonical Workflow

1. Prepare aligned scFoundation-ready counts from raw/count-like AnnData:

```python
from src.prepare_scfoundation_input import prepare_scfoundation_input
```

2. Fine-tune on the prepared prefix:

```bash
python scripts/train_scfoundation_conditioned_masked_task.py \
  --prepared_prefix data/processed/lung/30_non_smoking_normal_healthy/scfoundation/healthy_ref \
  --cell_type_key ann_level_3 \
  --save_dir outputs/scfoundation_conditioned_mgp
```

## .env usage

The project root now contains:

- `.env`
- `.env.example`

Relevant settings are loaded once per process through `src.config.get_project_paths()`. Relative paths in `.env` are resolved against `PROJECT_ROOT`.
