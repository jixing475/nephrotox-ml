"""
UQ Prototype: RF Tree Variance Uncertainty Quantification

计算 RF 每棵树的预测标准差作为不确定性度量。
对内部 test set (10% holdout) 进行评估。

输出:
  - uq_prototype_data.csv: 每个样本的 y_true, y_pred, y_proba, uncertainty
  - uq_threshold_sweep.csv: 不同阈值下的 coverage 和性能
  - uq_summary.json: 汇总统计
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    matthews_corrcoef, roc_auc_score,
    recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats

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


def load_best_params():
    """Load tuned RF params."""
    params_path = OUTPUT_DIR / "RF_RDKit" / "best_params.json"
    with open(params_path) as f:
        return json.load(f)


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "AUC": roc_auc_score(y_true, y_proba) * 100,
        "ACC": accuracy_score(y_true, y_pred) * 100,
        "SE": recall_score(y_true, y_pred) * 100,  # sensitivity
        "SP": tn / (tn + fp) * 100,  # specificity
        "F1": f1_score(y_true, y_pred) * 100,
        "Kappa": cohen_kappa_score(y_true, y_pred) * 100,
        "MCC": matthews_corrcoef(y_true, y_pred) * 100,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def get_tree_uncertainty(model, X):
    """Get per-sample uncertainty from RF tree variance."""
    tree_preds = np.array([
        tree.predict_proba(X)[:, 1] for tree in model.estimators_
    ])
    # tree_preds shape: (n_trees, n_samples)
    uncertainty = np.std(tree_preds, axis=0)
    mean_proba = np.mean(tree_preds, axis=0)
    return uncertainty, mean_proba


def find_youden_threshold(y_true, y_pred, uncertainty):
    """Find optimal uncertainty threshold using nested CV on training data.
    
    Following Liu et al.:
    - Use Youden Index (J = SE + SP - 1) on the uncertainty-filtered predictions
    - The "positive" class here = correct prediction, "negative" = incorrect
    """
    correct = (y_true == y_pred).astype(int)
    
    # Try all unique uncertainty values as thresholds
    thresholds = np.sort(np.unique(uncertainty))
    best_j = -1
    best_theta = thresholds[0]
    
    results = []
    for theta in thresholds:
        mask = uncertainty < theta
        n_inside = mask.sum()
        if n_inside < 5:
            continue
        coverage = n_inside / len(y_true)
        
        # Accuracy within AD
        acc_inside = correct[mask].mean()
        results.append({
            "theta": theta,
            "coverage": coverage,
            "n_inside": n_inside,
            "acc_inside": acc_inside,
        })
        
        # Youden Index: maximize correct rate inside - wrong rate outside
        if (~mask).sum() > 0:
            acc_outside = correct[~mask].mean()
            j = acc_inside - acc_outside
        else:
            j = acc_inside - 0.5
        
        if j > best_j:
            best_j = j
            best_theta = theta
    
    return best_theta, results


def main():
    print("=" * 60)
    print("UQ Prototype: RF Tree Variance")
    print("=" * 60)
    
    # Load data
    df_full = load_data_full(TRAIN_FILE)
    print(f"Loaded training data: {len(df_full)} samples")
    
    params = load_best_params()
    print(f"RF params: {params}")
    
    # Run for multiple seeds to get robust results
    SEEDS = list(range(10))
    all_results = []
    
    for seed in SEEDS:
        # Same split as train.py
        train_val_df, test_df = train_test_split(
            df_full, test_size=0.1, stratify=df_full["label"], random_state=seed
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=1.0/9.0, stratify=train_val_df["label"],
            random_state=seed,
        )
        
        train_ids, X_train, y_train = split_features_labels(train_df)
        test_ids, X_test, y_test = split_features_labels(test_df)
        val_ids, X_val, y_val = split_features_labels(val_df)
        
        # Feature engineering
        fe = FeatureEngineer()
        X_train_proc = fe.fit_transform(X_train)
        X_test_proc = fe.transform(X_test)
        X_val_proc = fe.transform(X_val)
        
        # Train RF
        rf_params = {**params, "n_jobs": -1, "random_state": seed}
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train_proc, y_train)
        
        # Get predictions + uncertainty on TEST set
        uncertainty, mean_proba = get_tree_uncertainty(model, X_test_proc)
        y_pred = (mean_proba >= 0.5).astype(int)
        
        # Also compute on VAL set for threshold determination
        val_unc, val_proba = get_tree_uncertainty(model, X_val_proc)
        val_pred = (val_proba >= 0.5).astype(int)
        
        # Find threshold using VALIDATION set (prevent data leakage!)
        theta, _ = find_youden_threshold(y_val, val_pred, val_unc)
        
        # Apply threshold to TEST set
        for idx in range(len(X_test_proc)):
            all_results.append({
                "seed": seed,
                "ID": test_ids.iloc[idx],
                "y_true": int(y_test.iloc[idx]),
                "y_pred": int(y_pred[idx]),
                "y_proba": float(mean_proba[idx]),
                "uncertainty": float(uncertainty[idx]),
                "correct": int(y_test.iloc[idx] == y_pred[idx]),
                "theta": float(theta),
                "in_domain": int(uncertainty[idx] < theta),
            })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(FIG3_DIR / "uq_prototype_data.csv", index=False)
    print(f"\nSaved {len(results_df)} prediction rows")
    
    # ============================================
    # Analyze prototype results
    # ============================================
    print("\n" + "=" * 60)
    print("PROTOTYPE RESULTS")
    print("=" * 60)
    
    # --- Test 1: Correct vs Incorrect uncertainty distribution ---
    correct_unc = results_df[results_df["correct"] == 1]["uncertainty"]
    incorrect_unc = results_df[results_df["correct"] == 0]["uncertainty"]
    
    u_stat, p_value = stats.mannwhitneyu(
        correct_unc, incorrect_unc, alternative="less"
    )
    
    print(f"\n--- Test 1: Uncertainty Distribution ---")
    print(f"Correct predictions:   mean unc = {correct_unc.mean():.4f} ± {correct_unc.std():.4f}")
    print(f"Incorrect predictions: mean unc = {incorrect_unc.mean():.4f} ± {incorrect_unc.std():.4f}")
    print(f"Mann-Whitney U test:   p = {p_value:.2e}")
    print(f"Result: {'✅ SIGNIFICANT' if p_value < 0.01 else '❌ NOT SIGNIFICANT'}")
    
    # --- Test 2: In-domain performance ---
    print(f"\n--- Test 2: In-domain Performance ---")
    
    # Aggregate per-seed metrics
    all_metrics = {"all": [], "in_domain": [], "out_domain": []}
    coverage_list = []
    
    for seed in SEEDS:
        seed_data = results_df[results_df["seed"] == seed]
        theta = seed_data["theta"].iloc[0]
        
        # All
        m_all = calculate_metrics(
            seed_data["y_true"].values,
            seed_data["y_pred"].values,
            seed_data["y_proba"].values
        )
        all_metrics["all"].append(m_all)
        
        # In-domain
        in_dom = seed_data[seed_data["in_domain"] == 1]
        if len(in_dom) > 5:
            m_in = calculate_metrics(
                in_dom["y_true"].values,
                in_dom["y_pred"].values,
                in_dom["y_proba"].values
            )
            all_metrics["in_domain"].append(m_in)
            coverage_list.append(len(in_dom) / len(seed_data) * 100)
        
        # Out-domain
        out_dom = seed_data[seed_data["in_domain"] == 0]
        if len(out_dom) > 5:
            m_out = calculate_metrics(
                out_dom["y_true"].values,
                out_dom["y_pred"].values,
                out_dom["y_proba"].values
            )
            all_metrics["out_domain"].append(m_out)
    
    # Print summary table
    print(f"\n{'Subset':<15} {'N':>5} {'Coverage':>10} {'AUC':>8} {'ACC':>8} {'Kappa':>8} {'MCC':>8}")
    print("-" * 65)
    for subset in ["all", "in_domain", "out_domain"]:
        if not all_metrics[subset]:
            continue
        metrics = pd.DataFrame(all_metrics[subset])
        n_mean = metrics[["TP", "TN", "FP", "FN"]].sum(axis=1).mean()
        cov = np.mean(coverage_list) if subset == "in_domain" else (
            100 - np.mean(coverage_list) if subset == "out_domain" else 100.0
        )
        print(f"{subset:<15} {n_mean:>5.0f} {cov:>9.1f}% "
              f"{metrics['AUC'].mean():>7.1f}% {metrics['ACC'].mean():>7.1f}% "
              f"{metrics['Kappa'].mean():>7.1f}% {metrics['MCC'].mean():>7.1f}%")
    
    # --- Test 3: Kappa improvement ---
    kappa_all = np.mean([m["Kappa"] for m in all_metrics["all"]])
    kappa_in = np.mean([m["Kappa"] for m in all_metrics["in_domain"]]) if all_metrics["in_domain"] else 0
    kappa_diff = kappa_in - kappa_all
    coverage_mean = np.mean(coverage_list)
    
    print(f"\n--- Test 3: Success Criteria ---")
    print(f"Kappa improvement:  {kappa_diff:+.1f}% (threshold: ≥5%)")
    print(f"  {'✅ PASSED' if kappa_diff >= 5 else '❌ FAILED'}")
    print(f"Coverage:           {coverage_mean:.1f}% (threshold: ≥50%)")
    print(f"  {'✅ PASSED' if coverage_mean >= 50 else '❌ FAILED'}")
    print(f"Significance:       p = {p_value:.2e} (threshold: <0.01)")
    print(f"  {'✅ PASSED' if p_value < 0.01 else '❌ FAILED'}")
    
    passed = (kappa_diff >= 5) and (coverage_mean >= 50) and (p_value < 0.01)
    print(f"\n{'🎉 PROTOTYPE PASSED — Proceed with Panel F!' if passed else '⚠️ PROTOTYPE PARTIAL/FAILED — See details above'}")
    
    # --- Save threshold sweep for coverage-performance curve ---
    sweep_results = []
    # Use seed 0 data for sweep
    seed0 = results_df[results_df["seed"] == 0]
    unc_values = np.linspace(
        seed0["uncertainty"].min(),
        seed0["uncertainty"].max(),
        50
    )
    for theta_sweep in unc_values:
        mask = seed0["uncertainty"] < theta_sweep
        n_in = mask.sum()
        if n_in < 5:
            continue
        try:
            m = calculate_metrics(
                seed0[mask]["y_true"].values,
                seed0[mask]["y_pred"].values,
                seed0[mask]["y_proba"].values
            )
        except ValueError:
            continue
        sweep_results.append({
            "theta": theta_sweep,
            "coverage": n_in / len(seed0) * 100,
            "n_inside": n_in,
            **m
        })
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(FIG3_DIR / "uq_threshold_sweep.csv", index=False)
    
    # --- Save summary JSON ---
    summary = {
        "prototype_passed": passed,
        "criteria": {
            "significance": {"p_value": float(p_value), "passed": p_value < 0.01},
            "kappa_improvement": {"delta": float(kappa_diff), "passed": kappa_diff >= 5},
            "coverage": {"mean_pct": float(coverage_mean), "passed": coverage_mean >= 50},
        },
        "correct_uncertainty_mean": float(correct_unc.mean()),
        "incorrect_uncertainty_mean": float(incorrect_unc.mean()),
        "all_metrics": {k: float(np.mean([m[k] for m in all_metrics["all"]])) 
                       for k in ["AUC", "ACC", "Kappa", "MCC"]},
        "in_domain_metrics": {k: float(np.mean([m[k] for m in all_metrics["in_domain"]])) 
                             for k in ["AUC", "ACC", "Kappa", "MCC"]} if all_metrics["in_domain"] else {},
        "out_domain_metrics": {k: float(np.mean([m[k] for m in all_metrics["out_domain"]])) 
                              for k in ["AUC", "ACC", "Kappa", "MCC"]} if all_metrics["out_domain"] else {},
    }
    with open(FIG3_DIR / "uq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All UQ prototype data saved to {FIG3_DIR}")


if __name__ == "__main__":
    main()
