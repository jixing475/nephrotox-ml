"""
Generate TEST SET predictions, ROC, confusion matrix, and calibration data for Fig.3.

The original training code saves only external set outputs.
This script re-trains with the same pipeline and saves internal test set (10% split) outputs.

Models: RF, SVM, XGB, LGBM (all with RDKit descriptors, tuned params)
Output: CSV files to output/fig3_data/
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import project modules
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_loader import load_data_full, split_features_labels, FeatureEngineer

warnings.filterwarnings("ignore")

# -- Paths --
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG3_DIR = OUTPUT_DIR / "fig3_data"
FIG3_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = INPUT_DIR / "cleaned_data_train_rdkit_desc.csv"

# -- Model configs --
MODELS = {
    "RF": {
        "class": RandomForestClassifier,
        "params_file": OUTPUT_DIR / "RF_RDKit" / "best_params.json",
        "extra": {"n_jobs": -1},
    },
    "SVM": {
        "class": SVC,
        "params_file": OUTPUT_DIR / "SVM_RDKit" / "best_params.json",
        "extra": {"probability": True},
    },
    "XGB": {
        "class": XGBClassifier,
        "params_file": OUTPUT_DIR / "XGB_RDKit" / "best_params.json",
        "extra": {"eval_metric": "logloss", "verbosity": 0},
    },
    "LGBM": {
        "class": LGBMClassifier,
        "params_file": OUTPUT_DIR / "LGBM_RDKit" / "best_params.json",
        "extra": {"verbose": -1},
    },
}

SEEDS = list(range(10))


def build_model(name, cfg, seed=0):
    """Build model with tuned hyperparams."""
    with open(cfg["params_file"]) as f:
        params = json.load(f)
    extra = cfg.get("extra", {})
    params.update(extra)
    params["random_state"] = seed
    return cfg["class"](**params)


def run_seed(seed, df_full, model_name, model_cfg):
    """Train on 80% train, validate on 10% val, predict on 10% test — return test set results."""
    # Same split logic as train.py
    train_val_df, test_df = train_test_split(
        df_full, test_size=0.1, stratify=df_full["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=1.0 / 9.0, stratify=train_val_df["label"],
        random_state=seed,
    )

    train_ids, X_train, y_train = split_features_labels(train_df)
    test_ids, X_test, y_test = split_features_labels(test_df)

    # Feature engineering
    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)
    X_test_proc = fe.transform(X_test)

    # Train
    model = build_model(model_name, model_cfg, seed=seed)
    model.fit(X_train_proc, y_train)

    # Predict on TEST set
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds, "seed": seed})

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"])
    cm_df["seed"] = seed

    # Predictions
    preds_df = pd.DataFrame({
        "seed": seed,
        "ID": test_ids.values,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "set": "test",
    })

    # Calibration (only compute once per model, using seed 0)
    cal_data = None
    if seed == 0:
        fraction_pos, mean_pred = calibration_curve(
            y_test, y_proba, n_bins=8, strategy="quantile"
        )
        cal_data = pd.DataFrame({
            "model": model_name,
            "bin_idx": range(len(fraction_pos)),
            "fraction_of_positives": fraction_pos,
            "mean_predicted_value": mean_pred,
        })

    return roc_df, cm_df, preds_df, cal_data


def generate_learning_curves(df_full):
    """Generate learning curves using the full training data with internal CV."""
    print("\n=== Generating Learning Curves (internal CV) ===")
    
    ids, X, y = split_features_labels(df_full)
    
    # Apply feature engineering on full data for learning curve
    fe = FeatureEngineer()
    X_proc = fe.fit_transform(X)
    
    train_sizes_frac = np.linspace(0.1, 1.0, 10)
    all_results = []

    for name, cfg in MODELS.items():
        print(f"  Processing {name}...")
        model = build_model(name, cfg, seed=42)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_proc, y,
            train_sizes=train_sizes_frac,
            cv=5, scoring="accuracy", n_jobs=-1, random_state=42,
        )

        for i, size in enumerate(train_sizes_abs):
            for fold_idx in range(train_scores.shape[1]):
                all_results.append({
                    "model": name,
                    "train_size": int(size),
                    "fold": fold_idx,
                    "train_score": train_scores[i, fold_idx],
                    "val_score": val_scores[i, fold_idx],
                })

    df = pd.DataFrame(all_results)
    df.to_csv(FIG3_DIR / "learning_curve_data.csv", index=False)

    summary = df.groupby(["model", "train_size"]).agg(
        train_mean=("train_score", "mean"), train_std=("train_score", "std"),
        val_mean=("val_score", "mean"), val_std=("val_score", "std"),
    ).reset_index()
    summary.to_csv(FIG3_DIR / "learning_curve_summary.csv", index=False)
    print(f"  Saved learning curve data ({len(df)} rows)")


def main():
    print("=" * 60)
    print("Fig.3 Data Generator v2 — Internal Test Set")
    print("=" * 60)

    df_full = load_data_full(TRAIN_FILE)
    print(f"Loaded training data: {len(df_full)} samples")

    # -- Generate test set predictions for all 4 models --
    print("\n=== Generating Test Set Predictions ===")
    
    all_roc = {}
    all_cm = {}
    all_preds = {}
    all_cal = []

    for model_name, model_cfg in MODELS.items():
        print(f"\n--- {model_name} ---")
        model_roc, model_cm, model_preds, model_cal_list = [], [], [], []

        for seed in SEEDS:
            roc_df, cm_df, preds_df, cal_data = run_seed(seed, df_full, model_name, model_cfg)
            model_roc.append(roc_df)
            model_cm.append(cm_df)
            model_preds.append(preds_df)
            if cal_data is not None:
                model_cal_list.append(cal_data)

        all_roc[model_name] = pd.concat(model_roc, ignore_index=True)
        all_cm[model_name] = pd.concat(model_cm)
        all_preds[model_name] = pd.concat(model_preds, ignore_index=True)
        if model_cal_list:
            all_cal.extend(model_cal_list)

        # Print summary
        preds_all = pd.concat(model_preds, ignore_index=True)
        acc = (preds_all["y_true"] == preds_all["y_pred"]).mean() * 100
        print(f"  Test ACC: {acc:.1f}% (avg over {len(SEEDS)} seeds)")

    # Save ROC data (per model)
    for model_name, roc_df in all_roc.items():
        roc_df.to_csv(FIG3_DIR / f"roc_test_{model_name}.csv", index=False)
    
    # Save combined ROC
    roc_combined = pd.concat(
        [df.assign(model=name) for name, df in all_roc.items()], ignore_index=True
    )
    roc_combined.to_csv(FIG3_DIR / "roc_test_all.csv", index=False)

    # Save confusion matrices (RF only for Panel D)
    all_cm["RF"].to_csv(FIG3_DIR / "confusion_matrix_test_RF.csv")

    # Save predictions (RF only for Panel E)
    all_preds["RF"].to_csv(FIG3_DIR / "predictions_test_RF.csv", index=False)

    # Save calibration (all models)
    cal_df = pd.concat(all_cal, ignore_index=True)
    cal_df.to_csv(FIG3_DIR / "calibration_test_all.csv", index=False)

    # -- Learning curves --
    generate_learning_curves(df_full)

    # -- Summary statistics --
    print("\n=== AUC Summary (Test Set) ===")
    for model_name, roc_df in all_roc.items():
        aucs = []
        for seed in SEEDS:
            seed_roc = roc_df[roc_df["seed"] == seed]
            auc = np.trapz(seed_roc["tpr"], seed_roc["fpr"])
            aucs.append(auc * 100)
        print(f"  {model_name}: AUC = {np.mean(aucs):.1f} ± {np.std(aucs):.1f}%")
    
    # Save AUC summary
    auc_summary = []
    for model_name, roc_df in all_roc.items():
        aucs = []
        for seed in SEEDS:
            seed_roc = roc_df[roc_df["seed"] == seed]
            auc = np.trapz(seed_roc["tpr"], seed_roc["fpr"])
            aucs.append(auc * 100)
        auc_summary.append({
            "model": model_name, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)
        })
    pd.DataFrame(auc_summary).to_csv(FIG3_DIR / "auc_test_summary.csv", index=False)

    print("\n✅ All Fig.3 data (test set) generated successfully!")
    print(f"Output directory: {FIG3_DIR}")


if __name__ == "__main__":
    main()
