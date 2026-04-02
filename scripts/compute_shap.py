"""
B1 SHAP Analysis — Compute SHAP values for RF_RDKit model.

Workflow:
  1. Load training data (cleaned_data_train_rdkit_desc.csv)
  2. Apply FeatureEngineer (same pipeline as train.py) — fit on 80% train split
  3. Retrain RF with best_params.json (seed=0)
  4. Compute TreeExplainer SHAP values on FULL training set (1829 rows)
  5. Export: shap_values_full.csv, shap_feature_importance.csv, top_features_analysis.csv

Output dir: tasks/B1_shap_analysis/output/
"""

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add project root to path so we can import src modules
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import FeatureEngineer, split_features_labels

# ── Paths ──────────────────────────────────────────────────────────
INPUT_DIR = PROJECT_ROOT / "input"
MODEL_DIR = PROJECT_ROOT / "output" / "RF_RDKit"
OUTPUT_DIR = PROJECT_ROOT / "output" / "RF_RDKit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = INPUT_DIR / "cleaned_data_train_rdkit_desc.csv"
PARAMS_FILE = MODEL_DIR / "best_params.json"

SEED = 0


def main():
    # ── Step 1: Load data ──────────────────────────────────────────
    print("=" * 60)
    print("B1 SHAP Analysis — RF_RDKit")
    print("=" * 60)

    df_train_full = pd.read_csv(TRAIN_FILE)
    print(f"Training data loaded: {df_train_full.shape}")

    # ── Step 2: Reproduce train/val/test split (seed=0) ────────────
    # Same split logic as train.py:run_single_seed
    train_val_df, test_df = train_test_split(
        df_train_full, test_size=0.1, stratify=df_train_full["label"],
        random_state=SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1.0 / 9.0,
        stratify=train_val_df["label"], random_state=SEED
    )

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ── Step 3: Feature Engineering (fit on train only) ────────────
    _, X_train, y_train = split_features_labels(train_df)
    _, X_val, y_val = split_features_labels(val_df)

    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)
    X_val_proc = fe.transform(X_val)

    # Also transform full training data for SHAP computation
    _, X_full, y_full = split_features_labels(df_train_full)
    X_full_proc = fe.transform(X_full)

    print(f"Features after engineering: {X_train_proc.shape[1]}")
    print(f"Feature names: {list(X_train_proc.columns[:5])} ... ({len(X_train_proc.columns)} total)")

    # ── Step 4: Load best params & train RF ────────────────────────
    with open(PARAMS_FILE) as f:
        best_params = json.load(f)

    print(f"Best params: {best_params}")

    rf = RandomForestClassifier(
        **best_params,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train_proc, y_train)

    # Quick sanity check
    train_acc = rf.score(X_train_proc, y_train)
    val_acc = rf.score(X_val_proc, y_val)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")

    # ── Step 5: Compute SHAP values ────────────────────────────────
    import shap

    print("\nComputing SHAP values on full training set...")
    print(f"  Samples: {X_full_proc.shape[0]}, Features: {X_full_proc.shape[1]}")
    t0 = time.time()

    explainer = shap.TreeExplainer(rf)
    shap_explanation = explainer(X_full_proc)  # returns Explanation object

    elapsed = time.time() - t0
    print(f"  SHAP computation done in {elapsed:.1f}s")

    # shap_explanation.values shape: (n_samples, n_features, n_classes)
    # For binary classification, we want class 1 (toxic) SHAP values
    shap_vals = shap_explanation.values
    print(f"  SHAP values shape: {shap_vals.shape}")

    if shap_vals.ndim == 3:
        # (n_samples, n_features, 2) → take class 1
        shap_values_cls1 = shap_vals[:, :, 1]
    else:
        shap_values_cls1 = shap_vals

    print(f"  Class-1 SHAP values shape: {shap_values_cls1.shape}")

    # ── Step 6: Export SHAP values ─────────────────────────────────
    feature_names = list(X_full_proc.columns)

    # 6a. Full SHAP values matrix
    shap_df = pd.DataFrame(shap_values_cls1, columns=feature_names)
    shap_df.insert(0, "compound_idx", range(len(shap_df)))
    shap_df.insert(1, "label", y_full.values)
    shap_path = OUTPUT_DIR / "shap_values_full.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"\n  Saved: {shap_path} ({shap_df.shape})")

    # 6b. Feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values_cls1).mean(axis=0)
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    fi_df["rank"] = range(1, len(fi_df) + 1)
    fi_path = OUTPUT_DIR / "shap_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved: {fi_path} ({fi_df.shape})")

    # 6c. Top 20 features analysis
    top20 = fi_df.head(20).copy()
    # Add descriptive stats for each top feature
    stats_rows = []
    for _, row in top20.iterrows():
        feat = row["feature"]
        vals = X_full_proc[feat].values
        shap_v = shap_values_cls1[:, feature_names.index(feat)]
        stats_rows.append({
            "rank": row["rank"],
            "feature": feat,
            "mean_abs_shap": row["mean_abs_shap"],
            "feat_mean": np.mean(vals),
            "feat_std": np.std(vals),
            "feat_min": np.min(vals),
            "feat_max": np.max(vals),
            "shap_mean": np.mean(shap_v),
            "shap_std": np.std(shap_v),
            "shap_min": np.min(shap_v),
            "shap_max": np.max(shap_v),
            "correlation_feat_shap": np.corrcoef(vals, shap_v)[0, 1],
        })
    top_df = pd.DataFrame(stats_rows)
    top_path = OUTPUT_DIR / "top_features_analysis.csv"
    top_df.to_csv(top_path, index=False)
    print(f"  Saved: {top_path} ({top_df.shape})")

    # 6d. Also save feature values for visualization (needed for beeswarm)
    feat_vals_df = X_full_proc.copy()
    feat_vals_df.insert(0, "compound_idx", range(len(feat_vals_df)))
    feat_vals_df.insert(1, "label", y_full.values)
    feat_vals_path = OUTPUT_DIR / "feature_values_processed.csv"
    feat_vals_df.to_csv(feat_vals_path, index=False)
    print(f"  Saved: {feat_vals_path} ({feat_vals_df.shape})")

    # 6e. Save the Explanation object for Python plotting
    import pickle
    expl_path = OUTPUT_DIR / "shap_explanation.pkl"
    with open(expl_path, "wb") as f:
        pickle.dump({
            "shap_values_cls1": shap_values_cls1,
            "feature_names": feature_names,
            "feature_values": X_full_proc.values,
            "labels": y_full.values,
            "base_value": explainer.expected_value,
        }, f)
    print(f"  Saved: {expl_path}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Top 10 features by mean |SHAP|:")
    print("=" * 60)
    for _, row in fi_df.head(10).iterrows():
        print(f"  {int(row['rank']):2d}. {row['feature']:30s} {row['mean_abs_shap']:.6f}")

    # Compare with RF built-in feature importance
    rf_fi = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    print("\n\nTop 10 features by RF built-in importance (for comparison):")
    for i, (_, row) in enumerate(rf_fi.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:30s} {row['importance']:.6f}")

    print("\n✅ All SHAP outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
