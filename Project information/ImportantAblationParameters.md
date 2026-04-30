This file lists the information that should be recorded for every training
experiment or ablation run.

## Core Experiment Fields

1. Hyperparameters
   - Batch size
   - Learning rate
   - Weight decay
   - Number of epochs
   - Dropout
   - Mask ratio
   - Random seed
   - Optimizer and scheduler, if used

2. Pooling type
   - `max`
   - `mean`
   - `max_mean`
   - `attention`

3. Task
   - Example: masked gene reconstruction
   - Example: cell type classification
   - Example: abnormality scoring

4. Loss definition and reduction
   - Loss type: `huber`, `mse`, `mae`, cross entropy, etc.
   - State whether the loss is averaged per cell or summed globally.
   - For the current masked gene prediction training code, the reduction is:
     `mean_masked_genes_per_cell_then_mean_cells`.
   - This means every cell receives equal weight after averaging over its masked
     genes.

5. Architecture
   - Encoder source: scFoundation encoder, our own encoder, etc.
   - Encoder checkpoint path/version.
   - Frozen vs trainable encoder.
   - Which parts are trainable: head only, last block, embeddings, full encoder.
   - Prediction head dimensions.
   - Conditioning features: cell type embedding, depth covariate, batch covariate,
     etc.

## Additional Fields To Record

6. Dataset and cohort definition
   - Prepared data prefix.
   - Tissue/dataset source.
   - Inclusion criteria: healthy, age rule, smoking status, BMI rule, disease
     status.
   - Exact age rule, for example `age <= 30` vs `age < 30`.
   - Number of cells and genes used after filtering.

7. Train/validation/test split
   - Split assignment file.
   - Train split name.
   - Evaluation split names.
   - Number of cells assigned to each split.
   - Number of cells actually used after metadata filtering.
   - Whether test was used. Prefer keeping test unused until final evaluation.

8. Encoding/input representation
   - Raw counts, normalized counts, binned counts, scFoundation representation,
     or our own encoding.
   - Gene panel/order.
   - Whether zero-padded genes are excluded from masking.

9. Masking strategy
   - Mask ratio.
   - Mask seed.
   - Whether masking is deterministic per epoch.
   - Which genes can be masked.

10. Training scope and compute
    - Device: CPU or CUDA.
    - GPU name and VRAM.
    - PyTorch version and CUDA version.
    - Number of trainable parameters.
    - Runtime.

11. Reproducibility
    - Git commit hash.
    - Whether the worktree was dirty.
    - Python version.
    - Dependency versions when relevant.
    - Full CLI command.

12. Results summary
    - Final train loss.
    - Final validation loss.
    - Best validation loss and epoch.
    - Test metrics only if this is a final evaluation run.
    - Output artifact paths.

13. Notes and hypothesis
    - Short reason for running the experiment.
    - What changed compared with the previous experiment.
    - Expected result before running.
    - Observed result after running.

## Folder Naming Convention

Folder names should be short enough to scan, while the full details should be
stored inside the experiment folder in JSON/Markdown files.

Recommended pattern:

```text
outputs/experiments/<timestamp>__<task>__<cohort>__<encoder>_<pooling>__<loss-reduction>__bs<batch>_lr<lr>_s<seed>
```

Example:

```text
outputs/experiments/20260429_1625__mgp_recon__healthy30_ns__scfrozen_max__huber_cellmean__bs2_lr1e-3_s0
```

Meaning:

- `mgp_recon`: masked gene prediction / reconstruction task.
- `healthy30_ns`: healthy, age rule around 30, never smoker.
- `scfrozen_max`: frozen scFoundation encoder with max pooling.
- `huber_cellmean`: Huber loss, averaged per cell.
- `bs2_lr1e-3_s0`: batch size 2, learning rate 1e-3, seed 0.

More examples:

```text
outputs/experiments/20260429_1710__mgp_recon__healthy30_ns__scfrozen_attention__huber_cellmean__bs2_lr1e-3_s0
outputs/experiments/20260429_1745__mgp_recon__healthy30_ns__ownenc_mean__huber_cellmean__bs2_lr1e-3_s0
outputs/experiments/20260429_1815__celltype_cls__healthy30_ns__scfrozen_max__ce_cellmean__bs8_lr1e-4_s0
```

## Recommended Files Inside Each Experiment Folder

Each experiment folder should contain:

- `train_metrics.json`: metrics and run summary written by the training script.
- `model.pt`: trained model checkpoint.
- `feature_metadata.csv`: gene metadata used by the run.
- `experiment_notes.md`: human-written hypothesis, comments, and conclusion.
- `command.txt`: exact command used to launch the run.
- `environment.json`: Python, PyTorch, CUDA, GPU, git commit, dirty worktree flag.

The folder name is for human scanning. The JSON/Markdown files are the source of
truth for reproducibility.
