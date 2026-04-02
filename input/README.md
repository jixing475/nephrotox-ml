# Data Format

This directory expects the following CSV files for training and evaluation.

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `ID` | string | Unique molecule identifier |
| `SMILES` | string | SMILES representation of the molecule |
| `label` | int (0/1) | Toxicity label (0 = non-toxic, 1 = toxic) |

## Descriptor Sets

### RDKit Descriptors

Files: `cleaned_data_train_rdkit_desc.csv`, `cleaned_data_test_rdkit_desc.csv`

Columns after `ID`, `SMILES`, `label`: ~200 RDKit molecular descriptors.

### ChemoPy2D Descriptors

Files: `cleaned_data_train_chemopy2d.csv`, `cleaned_data_test_chemopy2d.csv`

Columns after `ID`, `SMILES`, `label`: ~600 ChemoPy 2D molecular descriptors.

### GraphOnly

Files: `cleaned_data_train_graphonly.csv`, `cleaned_data_test_graphonly.csv`

Only `ID`, `SMILES`, `label` columns (no extra descriptors — graph features computed on-the-fly).

### KlekotaRoth Fingerprints

Files: `cleaned_data_train_klekota_roth.csv`, `cleaned_data_test_klekota_roth.csv`

Columns after `ID`, `SMILES`, `label`: Klekota-Roth molecular fingerprints.

### MACCS Keys

Files: `cleaned_data_train_maccs.csv`, `cleaned_data_test_maccs.csv`

Columns after `ID`, `SMILES`, `label`: MACCS molecular fingerprints.

## Data Source

The nephrotoxicity dataset is assembled from three public sources:

- **DIRIL_317**: 317 compounds from FDA drug labels (external validation only). Reference: https://doi.org/10.1016/j.drudis.2024.103931
- **Dataset_1831**: 1,831 compounds from literature mining. Reference: https://doi.org/10.1021/acs.jcim.5c01532
- **Dataset_1018**: 1,018 compounds from multiple toxicological databases. Reference: https://doi.org/10.1007/s11030-025-11376-3

After deduplication: **2,157 unique compounds** (1,111 nephrotoxic, 1,046 non-nephrotoxic).

- **Training set**: 1,829 compounds
- **External validation set**: 247 compounds (DIRIL_317)

Obtain the raw data from the original sources and place the processed CSV files in this directory before running training scripts.

## Quick Check

```bash
# Verify data format
python -c "
import pandas as pd
df = pd.read_csv('input/cleaned_data_train_rdkit_desc.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns[:5]), '...')
print('Label distribution:', df['label'].value_counts().to_dict())
"
```
