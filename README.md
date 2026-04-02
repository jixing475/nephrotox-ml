# NephroTox ML — Nephrotoxicity Prediction Pipeline

Traditional ML, Graph Neural Networks, and Multimodal Fusion models for molecular nephrotoxicity prediction.

## Project Structure
```
nephrotox-ml/
├── src/                    # Traditional ML (RF/SVM/XGB/LGBM/ADA/QDA/LDA)
│   ├── dl/                # Deep learning (GCN/GAT/Weave/AttentiveFP/D-MPNN)
│   └── multimodal/        # Multimodal fusion (RDKit + ChemBERTa + Graph)
├── scripts/               # SHAP, UQ, and utility scripts
│   └── dl/                # DL training and tuning scripts
├── input/                 # Training and test data (see input/README.md)
├── output/                # Model outputs (generated during training)
├── batch_train.sh         # ML batch training script
├── pyproject.toml         # Dependencies (base + optional dl/multimodal)
└── README.md
```

## Reproducibility

All models are evaluated using 10-seed repeated random subsampling (seeds 0-9) with 80/10/10 train/validation/test splits. Reported metrics are mean ± SD across seeds. Hyperparameters are tuned once on the full training set via GridSearchCV or Optuna, then frozen for the 10-seed evaluation.

## Data
- Training: `input/cleaned_data_train_<descriptor>.csv`
- External test: `input/cleaned_data_test_<descriptor>.csv`
- Columns: `ID`, `label` (0/1), plus descriptor features (e.g., 206 RDKit descriptors).

## Environment
```bash
uv sync
# includes: pandas, numpy, scikit-learn, shap, matplotlib, seaborn,
#           xgboost, lightgbm, joblib, optuna
```

## Workflow
Hyperparameters are selected once via hyperparameter tuning on the full training set, then reused for the 10-seed evaluation.

### 1) Tune (saves best_params.json)

**GridSearchCV (default):**
```bash
uv run python -m src.train --config RF_RDKit --tune
```

**Optuna (Bayesian optimization):**
```bash
uv run python -m src.train --config RF_RDKit --tune --tuning-method optuna --optuna-trials 100
```

| Method | Use Case |
|--------|----------|
| `grid` | Small search space, exhaustive search |
| `optuna` | Large/continuous search space, faster convergence |

Outputs (under `output/RF_RDKit/`):
- `best_params.json`
- `tuning_cv_results.csv` (grid) or `optuna_study.json` + `optuna_importance.csv` (optuna)

### 2) Evaluate (10 seeds, 80/10/10 splits)
```bash
uv run python -m src.train --config RF_RDKit
```
Outputs:
- `cv_results_per_fold.csv`, `cv_summary.csv`
- `test_predictions.csv`, `roc_curve_data.csv`, `confusion_matrix.csv`
- `feature_importance.csv`, `shap_values.csv`
- `experiment_info.json`, `paper_targets.json`

### 3) Single final model (deployable bundle)
```bash
uv run python -m src.train --config RF_RDKit --save-final --final-seed 0
```
Outputs:
- `output/RF_RDKit/rf_rdkit_final.joblib` (model + fitted FeatureEngineer)
- `output/RF_RDKit/final_metrics.json`
- `output/RF_RDKit/final_predictions.csv`

## Using a saved model
```python
from joblib import load
import pandas as pd

bundle = load("output/RF_RDKit/rf_rdkit_final.joblib")
model = bundle["model"]
fe = bundle["fe"]

df_new = pd.read_csv("new_data.csv")
X_new = df_new.drop(columns=["ID", "label"], errors="ignore")
X_new_proc = fe.transform(X_new)
proba = model.predict_proba(X_new_proc)[:, 1]
pred = model.predict(X_new_proc)
```

## Batch Training

The `batch_train.sh` script automates training across multiple model-descriptor combinations. Each combination runs tuning (step 1) followed by evaluation (step 2).

### Full batch training (all models × all descriptors)

Train all 7 models × 4 descriptors = 28 combinations:

```bash
./batch_train.sh
```

### Selective retraining

Filter models and/or descriptors using environment variables:

**Single model (e.g., SVM only):**
```bash
MODELS=SVM ./batch_train.sh
```

**Multiple models:**
```bash
MODELS="RF,SVM,XGB" ./batch_train.sh
```

**Single descriptor:**
```bash
DESCRIPTORS=RDKit ./batch_train.sh
```

**Custom combination:**
```bash
MODELS="RF,SVM" DESCRIPTORS="RDKit,ChemoPy2d" ./batch_train.sh
```

### Tuning method selection

**GridSearchCV (default):**
```bash
./batch_train.sh
# or explicitly:
TUNING_METHOD=grid ./batch_train.sh
```

**Optuna (Bayesian optimization):**
```bash
TUNING_METHOD=optuna OPTUNA_TRIALS=100 ./batch_train.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | `RF,SVM,XGB,LGBM,ADA,QDA,LDA` | Comma-separated model list |
| `DESCRIPTORS` | `RDKit,ChemoPy2d,GraphOnly,KlekotaRoth` | Comma-separated descriptor list |
| `TUNING_METHOD` | `grid` | Tuning method: `grid` or `optuna` |
| `OPTUNA_TRIALS` | `50` | Number of Optuna trials (only used with `optuna`) |

### Outputs

- Log file: `batch_training_<timestamp>.log`
- Per-config outputs: same as manual training (see Workflow section)
- Summary statistics printed at the end

### SHAP Interpretability Analysis

Explain individual predictions using SHAP (SHapley Additive exPlanations):

```bash
# Compute SHAP values for a trained model
uv run python scripts/compute_shap.py --config RF_RDKit

# Plot beeswarm, bar, and dependence plots
uv run python scripts/plot_shap.py --config RF_RDKit
```

Outputs under `output/{CONFIG}/`:
- `shap_values.csv` — per-sample SHAP values
- `shap_beeswarm.png` — feature impact across samples
- `shap_bar.png` — mean |SHAP| per feature
- `shap_dependence_<feature>.png` — dependence plots for top features

### Uncertainty Quantification

Confidence scoring based on RF tree variance (four-tier system):

```bash
uv run python scripts/uq_prototype.py --config RF_RDKit
```

Produces per-sample confidence bands and a calibrated confidence tier (High / Medium / Low / Very Low). On the external validation set, filtering to High+Medium confidence predictions raised AUC from 72.7% to 90.2%.

## Notes
- Feature engineering: handle NaN/Inf; drop columns with >50% missing; variance filter; correlation filter (>|0.95|); StandardScaler.
- Random seeds: 10 iterations (0–9) for evaluation; `--final-seed` for the final bundle.
- Outputs are overwritten if they already exist; the tuned `best_params.json` is preserved.

---

# Deep Learning Models (GCN/GAT/Weave/AttentiveFP/D-MPNN)

This project includes a deep learning pipeline for molecular toxicity prediction using graph neural networks.

## Supported Models

| Backend | Algorithm | Config Name | Description |
|---------|-----------|-------------|-------------|
| DeepChem | GCN | `GCN_DeepChem_Graph` | Graph Convolutional Network |
| DGLlife | GCN | `GCN_DGLlife_Graph` | GCN with CanonicalAtomFeaturizer |
| DGLlife | GAT | `GAT_DGLlife_Graph` | Graph Attention Network |
| DGLlife | Weave | `Weave_DGLlife_Graph` | Weave model with atom-pair features |
| DGLlife | AttentiveFP | `AttentiveFP_DGLlife_Graph` | Attentive fingerprint with pooling |
| Chemprop | D-MPNN | `DMPNN_Chemprop_Graph` | Directed Message Passing Neural Network |
| Chemprop | D-MPNN+RDKit | `DMPNN_Chemprop_Graph+RDKit` | D-MPNN + RDKit descriptor fusion |
| Chemprop | D-MPNN+ChemoPy2d | `DMPNN_Chemprop_Graph+ChemoPy2d` | D-MPNN + ChemoPy2d descriptor fusion |

### Performance Summary (test AUC, mean ± SD, 10 seeds)

| Config | Test AUC |
|--------|----------|
| `RF_RDKit` (baseline) | 89.6 ± 1.4% |
| `GCN_DGLlife_Graph` | 86.2 ± 1.6% |
| `GAT_DGLlife_Graph` | 83.9 ± 2.9% |
| `Weave_DGLlife_Graph` | 83.2 ± 1.5% |
| `AttentiveFP_DGLlife_Graph` | 85.8 ± 2.1% |
| `DMPNN_Chemprop_Graph` | 84.5 ± 1.9% |
| `DMPNN_Chemprop_Graph+RDKit` | 85.0 ± 2.4% |

All DL models trained with the same 10-seed protocol. At the current dataset scale (~1,800 training compounds), RF_RDKit outperformed all DL variants.

## Installation

Requires GPU for training. Install the DL extra:

```bash
uv sync --extra dl
```

For CUDA-enabled DGL (manual install):

```bash
uv pip install "dgl>=1.1.0,<2.0" -f https://data.dgl.ai/wheels/cu118/repo.html
```

## Training

Single model training:

```bash
uv run python -m src.dl.train --config GCN_DGLlife_Graph
```

Batch training via shell script:

```bash
CONFIG=GAT_DGLlife_Graph ./scripts/dl/batch_train.sh
```

Parallel training (all models):

```bash
./scripts/dl/parallel_train_all.sh
```

## Model Configuration

All configurations are in `src/dl/config.py` under `MODEL_CONFIGS`. Naming convention: `{Algorithm}_{Backend}_{Feature}`.

## Hyperparameter Tuning

Uses Optuna (100 trials by default). Multi-GPU parallel tuning:

```bash
uv run python -m src.dl.parallel_tune --config GCN_DGLlife_Graph --n-trials 50 --n-gpus 2
```

## Feature Fusion

D-MPNN supports fusion with molecular descriptors:

```bash
CONFIG=DMPNN_Chemprop_Graph+RDKit ./scripts/dl/batch_train.sh
CONFIG=DMPNN_Chemprop_Graph+ChemoPy2d ./scripts/dl/batch_train.sh
```

Fusion preprocessing: variance filter → correlation filter (|r| > 0.95) → StandardScaler.

## Output

Results saved to `output/{CONFIG}/`:
- `best_params.json` — tuned hyperparameters
- `cv_results_per_fold.csv`, `cv_summary.csv` — per-seed and summary metrics
- `test_predictions.csv`, `roc_curve_data.csv`, `confusion_matrix.csv` — test set results
- `optuna_study.json`, `optuna_importance.csv` — tuning records

## Environment Variables (batch_train.sh)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `GCN_DeepChem_Graph` | Model config name |
| `OPTUNA_TRIALS` | `50` | Number of Optuna trials |
| `N_SEEDS` | `10` | Number of random seeds |
| `SKIP_TUNE` | `false` | Skip tuning, use existing best_params |
| `FORCE_CPU` | `false` | Force CPU mode |
| `CLASS_WEIGHTS` | `false` | Use class-weighted loss |

---

# Multimodal Fusion (RDKit + ChemBERTa + Graph)

Multimodal fusion combining RDKit descriptors, ChemBERTa embeddings, and molecular graph features.

## Architecture

```
RDKit (134D) ──────────────┐
                           ├──> Concat ──> MLP ──> Output
ChemBERTa (768D) ──> Proj ─┤
                           │
Graph ──> GCN ──> Pool ────┘
```

## Installation

```bash
uv sync --extra multimodal
```

Requires the `dl` extra plus `transformers>=4.30`.

## Training Scripts

Full dataset training with best hyperparameters:

```bash
uv run python scripts/dl/train_multimodal_full.py
```

Training with a specific config:

```bash
uv run python scripts/dl/train_multimodal_best.py
```

Hyperparameter tuning:

```bash
uv run python scripts/dl/tune_multimodal.py
```

Baseline comparisons:

```bash
uv run python scripts/dl/train_baselines.py
```

## Key Configuration (`src/multimodal/config.py`)

- `CHEMBERTA_CONFIG`: model name `seyonec/PubChem10M_SMILES_BPE_450k`, freeze=true
- `FEATURE_DIMS`: rdkit=134, chemberta=768, graph=256
- `OPTUNA_N_TRIALS`: 50

## Data Format

Same as ML pipeline: CSV with `ID`, `SMILES`, `label`, plus descriptor columns. Place data in `input/` directory.

## Glue Code Strategy

This module reuses existing functionality from `dl/` (graph features) and `ml/` (RDKit preprocessing). It only implements the fusion logic, ChemBERTa embedding extraction, and multimodal training loop.

---

## Citation

If you use this code in your research, please cite:

> Liu et al. NephroTox AI: An Explainable Machine Learning Framework with Uncertainty Quantification for Drug-Induced Nephrotoxicity Prediction. *Journal of Cheminformatics* (2026).
