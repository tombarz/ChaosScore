# scFoundation Conditioned Masked Reconstruction Ablation Plan

Date: 2026-04-27

## Goal

Train a normal-reference model that reconstructs masked gene expression conditioned on cell type, sequencing depth, and visible gene context. The downstream use case is abnormality scoring: cells that violate normal conditional gene relationships should have higher masked reconstruction error.

The main question is not only "which model has the lowest reconstruction loss?", but also "which model learns normal biology without becoming overly sensitive to sequencing depth, batch, or cell type imbalance?"

## Baseline

Current model:

- Load scFoundation checkpoint branch `gene`.
- Reuse pretrained `token_emb`, `pos_emb`, and `encoder`.
- Pool encoder token outputs into one cell embedding.
- Predict each masked gene using:
  - pooled cell embedding
  - learned cell-type embedding
  - learned depth projection
  - newly initialized target-gene embedding
- Freeze most or all of the scFoundation encoder by default.
- Mask only nonzero, non-zero-padded genes.

This is a reasonable first baseline because the task is close to scFoundation's original masked gene expression prediction objective, but the custom head gives direct control over normal-only conditioning.

## Primary Ablations

### 1. Target Gene Embedding Source

Variants:

- Scratch target-gene embedding, current baseline.
- Pretrained scFoundation `pos_emb` used as target-gene feature, frozen.
- Pretrained scFoundation `pos_emb` used as target-gene feature, trainable.
- Pretrained `pos_emb` projected into a smaller target-gene feature, for example 768 -> 64 or 768 -> 128.
- Pretrained target-gene feature trainable with lower learning rate than the head.

Why:

The current head asks a new embedding table to learn gene identity from the normal reference only. With a few thousand to tens of thousands of cells, this may be weak for rare genes. scFoundation already learned broad gene context from large-scale pretraining. Reusing it may improve sample efficiency while the normal-only head still learns the normal conditional distribution.

Key risk:

Pretrained embeddings were learned from mixed normal and malignant data. They may carry broad disease-associated structure. This is not automatically bad, because the reconstruction head is trained on normal-only cells, but it should be tested.

Decision criteria:

- Lower validation masked MAE/MSE overall.
- Better per-gene stability, especially for rare or low-expression genes.
- Lower correlation between reconstruction error and total counts.
- Better separation of known abnormal/risk cells if an evaluation set is available.

### 2. Checkpoint Branch

Variants:

- `pretrained_key="gene"`, current baseline.
- `pretrained_key="cell"`.
- `pretrained_key="rde"`.

Why:

The `gene` branch is aligned with gene-level/gene-expression tasks in the scFoundation repo. The `cell` and `rde` branches are used for cell embeddings and read-depth-enhanced cell representations. Since our model currently uses only the encoder and then pools to a cell embedding, it is worth testing whether the cell-oriented branch produces better normal-reference cell context.

Expected outcome:

`gene` is likely best for masked gene reconstruction. `rde` may help when depth effects dominate. `cell` may help if cell-type structure matters more than exact expression reconstruction.

Decision criteria:

- Validation reconstruction metrics.
- Depth/error correlation.
- Cell-type-level calibration.
- Abnormality separation if available.

### 3. Encoder Freeze Schedule

Variants:

- Freeze scFoundation encoder completely, current safest setting.
- Unfreeze only the last encoder block.
- Unfreeze last two encoder blocks.
- Unfreeze embeddings only.
- Unfreeze embeddings plus last encoder block.

Why:

Normal-only fine-tuning should adapt the model to the reference distribution, but full fine-tuning on a small normal set may overfit or erase useful pretrained structure. A staged approach is safer.

Recommended order:

1. Train only the new head with frozen encoder.
2. If validation loss plateaus, unfreeze the last encoder block.
3. Use a smaller learning rate for pretrained parameters than for the new head.

Decision criteria:

- Validation metrics improve without widening train/validation gap.
- Per-cell-type performance improves rather than only common cell types improving.
- No increase in depth or batch dependence.

### 4. Conditioning Features

Variants:

- Cell type + depth, current baseline.
- Cell type only.
- Depth only.
- No explicit conditioning.
- Different cell-type resolutions, for example `ann_level_1`, `ann_level_2`, `ann_level_3`.
- Add batch/donor/study embedding if residual batch effects dominate.

Why:

Cell type should help define normal expected expression. Depth should absorb technical variation. However, too much conditioning can let the model explain away biological abnormality or learn dataset artifacts.

Decision criteria:

- Reconstruction improves within each cell type.
- Error is not mostly explained by depth, donor, batch, or study.
- Cell-type-specific abnormality scores remain comparable after calibration.

### 5. Masking Strategy

Variants:

- Mask only nonzero genes, current baseline.
- Mask both zero and nonzero genes, closer to scFoundation pretraining.
- Stratified masking by expression bin.
- Higher/lower mask ratios, for example 0.15, 0.30, 0.50.

Why:

Masking only nonzero genes focuses the model on reconstructing expressed genes, which is useful for abnormal expression magnitude. Masking zeros too teaches absence/dropout structure, but can make the objective dominated by easy zeros unless carefully balanced.

Decision criteria:

- Nonzero-gene reconstruction accuracy.
- False abnormality rate in low-depth cells.
- Stability across cell types with different sparsity.

## Secondary Ablations

### 6. Pooling Strategy

Variants:

- Max pooling, current default.
- Mean pooling.
- Max + mean pooling.
- Attention pooling over visible gene tokens.

Why:

Pooling determines what information survives from the visible genes. Max pooling may capture strong marker signals; mean pooling may better represent global cell state; max + mean may capture both.

Decision criteria:

- Validation reconstruction metrics.
- Performance in heterogeneous cell types.
- Compute and memory cost.

### 7. Use scFoundation Count Tokens More Directly

Variants:

- Current design: depth is passed only to the custom head.
- Append original-style total-count indicators to the scFoundation input path.
- Compare head-level depth conditioning vs backbone-level count tokens.

Why:

The scFoundation paper uses target and input total count indicators during pretraining. Our current backbone receives only the gene vector; depth is added later. This may be simpler and more inspectable, but it may not fully match the pretrained objective.

Decision criteria:

- Reduced depth dependence in errors.
- Better reconstruction in low-depth normal cells.
- No degradation in abnormality sensitivity.

### 8. Head Architecture

Variants:

- Current MLP head.
- Larger MLP head.
- Residual MLP head.
- Factorized head with separate cell-context and gene-context interactions.
- Bilinear interaction between cell embedding and target-gene embedding.

Why:

The model needs to represent gene-by-cell-context interactions. A simple concatenation MLP may be enough, but bilinear or factorized interactions may model conditional gene regulation more efficiently.

Decision criteria:

- Better validation reconstruction with similar overfitting profile.
- Better per-gene calibration.
- No excessive parameter growth.

### 9. Training Set Size Learning Curve

Variants:

- 1k cells.
- 5k cells.
- 10k cells.
- 25k cells.
- Full normal reference.

Why:

We need to know whether normal-only fine-tuning is data-limited and whether pretrained target-gene embeddings help most in the small-data regime.

Decision criteria:

- Validation loss vs training size.
- Per-cell-type coverage.
- Rare-gene stability.
- Runtime and memory requirements.

## Evaluation Metrics

Core reconstruction metrics:

- Masked MSE.
- Masked MAE.
- Per-cell reconstruction error.
- Per-gene reconstruction error.
- Per-cell-type reconstruction error.

Calibration checks:

- Correlation of error with total counts.
- Correlation of error with number of detected genes.
- Error by batch, donor, study, and sequencing platform.
- Error by cell type and cell-type frequency.

Abnormality checks, when labels are available:

- Error distribution for normal reference vs held-out abnormal/risk cells.
- AUROC/AUPRC for known abnormal labels.
- Cell-type-stratified abnormality separation.
- Whether top-error genes are biologically plausible rather than mostly depth or batch artifacts.

## Recommended First Experiment Set

Run a compact matrix first:

1. Scratch gene embedding + frozen encoder.
2. Pretrained projected gene embedding + frozen encoder.
3. Pretrained projected gene embedding + last encoder block unfrozen.
4. Scratch gene embedding + last encoder block unfrozen.

Use the same train/validation split, same mask seed, same cell-type key, and same training budget. This directly tests whether the pretrained gene identity prior helps and whether encoder adaptation is needed.

After that, test:

1. `pretrained_key="gene"` vs `"cell"` vs `"rde"`.
2. Depth on vs depth off.
3. `ann_level_2` vs `ann_level_3`.
4. Nonzero-only masking vs mixed zero/nonzero masking.

## Notes

The safest default for small normal references is to keep scFoundation mostly frozen and adapt the task head first. If the normal reference contains only a narrow biological region, aggressive fine-tuning may reduce generalization and make abnormality scores less meaningful.

Pretrained embeddings should be treated as a prior, not as the final definition of normal. The normal-only reconstruction objective is what should define the normal conditional distribution.
