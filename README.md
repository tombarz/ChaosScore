# ChaosScore

Docker setup:

1. Build the image:
   `docker compose build`
2. Start an interactive shell with the whole repo mounted:
   `docker compose run --rm chaosscore`
3. Inside the container, run your script:
   `python src/download_lung_reference_data.py --help`

Notes:
- Your local project folder is mounted at `/workspace` inside the container.
- Any edits on Windows are immediately visible inside the container.
- The compose service requests GPU access and exports:
  - `SCFOUNDATION_REPO=/workspace/external/scFoundation`
  - `SCFOUNDATION_CKPT_PATH=/workspace/external/scFoundation/model/models/models.ckpt`
  - `SCFOUNDATION_GENE_PANEL_PATH=/workspace/external/scFoundation/model/OS_scRNA_gene_index.19264.tsv`

## scFoundation preprocessing

Clone the upstream repo into `external/scFoundation` and download the checkpoint into `external/scFoundation/model/models/models.ckpt`.

Prepare a raw-count `.h5ad` for scFoundation:

```bash
python src/prepare_scfoundation_input.py \
  --input_h5ad data/lung_data/hlca_core.h5ad \
  --counts_source raw \
  --gene_symbol_field feature_name \
  --output_prefix data/processed/hlca_scfoundation/hlca_core \
  --dataset_role healthy
```

Score masked reconstruction error with healthy-depth residualization:

```bash
python src/score_scfoundation_abnormality.py \
  --reference_prefix data/processed/hlca_scfoundation/hlca_core \
  --output_dir data/processed/scfoundation_scores
```
