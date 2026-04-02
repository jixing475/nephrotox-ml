"""
B1 SHAP — Compute SHAP on external TEST set (247 samples).
Compare with training set SHAP for robustness validation.
"""

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import FeatureEngineer, split_features_labels

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output" / "RF_RDKit"

TRAIN_FILE = INPUT_DIR / "cleaned_data_train_rdkit_desc.csv"
TEST_FILE = INPUT_DIR / "cleaned_data_test_rdkit_desc.csv"
PARAMS_FILE = PROJECT_ROOT / "output" / "RF_RDKit" / "best_params.json"

SEED = 0

def main():
    print("=" * 60)
    print("SHAP on TEST set — Robustness validation")
    print("=" * 60)

    # Load data
    df_train_full = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    print(f"Train: {df_train_full.shape}, Test: {df_test.shape}")

    # Same split as training
    train_val_df, _ = train_test_split(
        df_train_full, test_size=0.1, stratify=df_train_full["label"], random_state=SEED
    )
    train_df, _ = train_test_split(
        train_val_df, test_size=1.0 / 9.0, stratify=train_val_df["label"], random_state=SEED
    )

    # Feature engineering (fit on train split only)
    _, X_train, y_train = split_features_labels(train_df)
    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)

    # Transform test set
    _, X_test, y_test = split_features_labels(df_test)
    X_test_proc = fe.transform(X_test)
    print(f"Features: {X_test_proc.shape[1]}")

    # Load params & train
    with open(PARAMS_FILE) as f:
        best_params = json.load(f)

    rf = RandomForestClassifier(**best_params, n_jobs=-1, random_state=SEED)
    rf.fit(X_train_proc, y_train)

    # SHAP on test set
    import shap
    print(f"\nComputing SHAP on test set ({X_test_proc.shape[0]} samples)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(rf)
    shap_explanation = explainer(X_test_proc)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    shap_vals = shap_explanation.values
    if shap_vals.ndim == 3:
        shap_values_cls1 = shap_vals[:, :, 1]
    else:
        shap_values_cls1 = shap_vals

    feature_names = list(X_test_proc.columns)

    # Feature importance (mean |SHAP|) — test set
    mean_abs_shap_test = np.abs(shap_values_cls1).mean(axis=0)
    fi_test = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_test": mean_abs_shap_test,
    }).sort_values("mean_abs_shap_test", ascending=False).reset_index(drop=True)
    fi_test["rank_test"] = range(1, len(fi_test) + 1)

    # Save test SHAP
    fi_test.to_csv(OUTPUT_DIR / "shap_feature_importance_test.csv", index=False)

    # Save full test SHAP values
    shap_test_df = pd.DataFrame(shap_values_cls1, columns=feature_names)
    shap_test_df.insert(0, "compound_idx", range(len(shap_test_df)))
    shap_test_df.insert(1, "label", y_test.values)
    shap_test_df.to_csv(OUTPUT_DIR / "shap_values_test.csv", index=False)

    # Load training SHAP importance for comparison
    fi_train = pd.read_csv(OUTPUT_DIR / "shap_feature_importance.csv")
    fi_train = fi_train.rename(columns={"mean_abs_shap": "mean_abs_shap_train"})
    fi_train["rank_train"] = range(1, len(fi_train) + 1)

    # Merge
    comparison = fi_train[["feature", "mean_abs_shap_train", "rank_train"]].merge(
        fi_test[["feature", "mean_abs_shap_test", "rank_test"]],
        on="feature"
    )
    comparison = comparison.sort_values("rank_train")
    comparison.to_csv(OUTPUT_DIR / "shap_train_vs_test_comparison.csv", index=False)

    # Spearman correlation of rankings
    from scipy.stats import spearmanr
    rho, pval = spearmanr(comparison["rank_train"], comparison["rank_test"])

    print(f"\n{'='*60}")
    print(f"Spearman correlation of rankings: ρ = {rho:.4f}, p = {pval:.2e}")
    print(f"{'='*60}")
    
    # Top 20 comparison
    print(f"\n{'Rank':>4} {'Feature':>25} {'Train SHAP':>12} {'Train Rank':>10} {'Test SHAP':>12} {'Test Rank':>10}")
    print("-" * 75)
    for _, row in comparison.head(20).iterrows():
        print(f"{int(row['rank_train']):4d} {row['feature']:>25s} {row['mean_abs_shap_train']:12.5f} {int(row['rank_train']):>10d} {row['mean_abs_shap_test']:12.5f} {int(row['rank_test']):>10d}")

    # Top 10 overlap
    train_top10 = set(comparison.head(10)["feature"])
    test_top10 = set(fi_test.head(10)["feature"])
    overlap10 = train_top10 & test_top10
    print(f"\nTop 10 overlap: {len(overlap10)}/10")
    print(f"Shared features: {overlap10}")

    train_top20 = set(comparison.head(20)["feature"])
    test_top20 = set(fi_test.head(20)["feature"])
    overlap20 = train_top20 & test_top20
    print(f"Top 20 overlap: {len(overlap20)}/20")

    # Direction consistency check
    print(f"\n{'='*60}")
    print("Direction consistency (top 10 training features):")
    print(f"{'='*60}")
    for feat in list(comparison.head(10)["feature"]):
        sv_train = pd.read_csv(OUTPUT_DIR / "shap_values_full.csv")[feat].values
        sv_test = shap_values_cls1[:, feature_names.index(feat)]
        
        dir_train = "↑ toxic" if np.mean(sv_train) > 0 else "↓ non-toxic"
        dir_test = "↑ toxic" if np.mean(sv_test) > 0 else "↓ non-toxic"
        consistent = "✅" if (np.mean(sv_train) > 0) == (np.mean(sv_test) > 0) else "❌ FLIP"
        
        print(f"  {feat:25s}  train: {dir_train:12s}  test: {dir_test:12s}  {consistent}")


if __name__ == "__main__":
    main()
