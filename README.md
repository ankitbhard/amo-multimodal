# AMO: Adaptive Multi-Objective Training for Multimodal Cancer Survival Prediction

Code for the paper *"AMO: Adaptive Multi-Objective Training for Multimodal Cancer Survival Prediction"* (CVPR 2027 submission).

## Overview

AMO decomposes multimodal survival training into four complementary objectives:
1. **Primary fused Cox loss** on the multimodal representation
2. **Per-modality auxiliary Cox losses** providing direct survival feedback to each encoder
3. **Censoring-aware pairwise ranking loss** capturing ordinal patient relationships
4. **EMA-based adaptive weight controller** that adjusts objective contributions per epoch

## Structure

```
configs/              # Per-cancer YAML configs (BRCA, KIRC, LUAD, LUSC, STAD)
experiments/
  clam_brca/
    train.py          # Main training loop (all methods + baselines)
    models.py         # CLAM attention MIL + aux heads + regularization baselines
    dataset.py        # Multimodal dataset (WSI patches + gene expression)
  samo/
    objectives.py     # LossWeightController (EMA adaptive weights)
    v4_controller.py  # AMO optimizer wrapper with gradient projection
models/
  joint_model.py      # Model factory
  encoders.py         # Gene encoder, image encoders
  fusion.py           # Cross-modal attention fusion
optimizers/
  ogm_wrapper.py      # OGM gradient modulation
  sam_optimizer.py     # SAM base optimizer
```

## Usage

```bash
# Train AMO on BRCA (5-fold CV)
python experiments/clam_brca/train.py --config configs/brca_clam.yaml --method clam_aux_adamw

# Train AMO+GP (with gradient projection)
python experiments/clam_brca/train.py --config configs/brca_clam.yaml --method clam_ra_samo

# Train Cox-only baseline
python experiments/clam_brca/train.py --config configs/brca_clam.yaml --method clam_adamw
```

### Methods

| Method | Config key | Description |
|--------|-----------|-------------|
| Cox-only | `clam_adamw` | Single fused Cox loss baseline |
| AMO | `clam_aux_adamw` | Aux heads + adaptive EMA weights |
| AMO+GP | `clam_ra_samo` | AMO + gradient projection |
| +Elastic Net | `clam_elasticnet` | Cox + L1/L2 regularization |
| +Contrastive | `clam_contrastive` | Cox + InfoNCE |
| +Consistency | `clam_consistency` | Cox + prediction agreement |
| +GradNorm | `clam_gradnorm` | Cox + GradNorm weight balancing |

## Data

The pipeline expects:
- **Histopathology**: Pre-extracted patch embeddings (UNI 1024-d or UNI2-h 1536-d) as `.pt` or `.h5` files
- **Gene expression**: CSV files with columns `patient_id`, `vital_status`, `survival_months`, and gene columns

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, pandas, h5py, PyYAML, scikit-learn
