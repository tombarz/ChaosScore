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
  - `SCFOUNDATION_CKPT_PATH=/workspace/external/scFoundation/model/models.ckpt`
  - `SCFOUNDATION_GENE_PANEL_PATH=/workspace/external/scFoundation/model/OS_scRNA_gene_index.19264.tsv`

## scFoundation preprocessing

Clone the upstream repo into `external/scFoundation` and download the checkpoint into `external/scFoundation/model/models.ckpt`.

Step 1: prepare a raw-count `.h5ad` into the canonical scFoundation-ready prefix format:

```bash
python src/prepare_scfoundation_input.py \
  --input_h5ad data/lung_data/hlca_core.h5ad \
  --counts_source raw \
  --gene_symbol_field feature_name \
  --output_prefix data/processed/hlca_scfoundation/hlca_core \
  --dataset_role healthy
```

For the healthy reference subset created by `src/split_lung_groups.py`, prepare it the same way:

```bash
python src/prepare_scfoundation_input.py \
  --input_h5ad data/processed/lung/30_non_smoking_normal_healthy/subset.h5ad \
  --counts_source raw \
  --gene_symbol_field feature_name \
  --output_prefix data/processed/lung/30_non_smoking_normal_healthy/scfoundation/healthy_ref \
  --dataset_role healthy
```

Step 2a: fine-tune the conditioned masked-gene prediction head from the prepared prefix:

```bash
python scripts/train_scfoundation_conditioned_masked_task.py \
  --prepared_prefix data/processed/lung/30_non_smoking_normal_healthy/scfoundation/healthy_ref \
  --cell_type_key ann_level_3 \
  --save_dir outputs/scfoundation_conditioned_mgp
```

Step 2b: score masked reconstruction error with healthy-depth residualization:

```bash
python src/score_scfoundation_abnormality.py \
  --reference_prefix data/processed/hlca_scfoundation/hlca_core \
  --output_dir data/processed/scfoundation_scores
```

Split an HLCA `.h5ad` into a young healthy group and the complementary risk group:

```bash
python src/split_lung_groups.py \
  --input_h5ad data/lung_data/hlca_core.h5ad \
  --age_cutoff 30
```

This writes:
- `data/processed/lung/30_non_smoking_normal_healthy/subset.h5ad`
- `data/processed/lung/older_than_30_smoked_normal_risk_group/subset.h5ad`
